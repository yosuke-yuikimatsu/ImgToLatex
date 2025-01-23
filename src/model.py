import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn import CNN
from lstm import StackedLSTM
from attention import BahdanauAttention
from output_projector import OutputProjector
from criterion import create_criterion

class OCRModel(nn.Module):
    """
    Примерная «склейка» всех модулей в единое целое,
    """
    def __init__(self, config):
        super(OCRModel, self).__init__()
        # Параметры из config
        self.encoder_num_hidden = config["encoder_num_hidden"]
        self.encoder_num_layers = config["encoder_num_layers"]
        self.decoder_num_hidden = config["decoder_num_hidden"]
        self.decoder_num_layers = config["decoder_num_layers"]
        self.target_vocab_size = config["target_vocab_size"]
        self.target_embedding_size = config["target_embedding_size"]
        self.dropout = config.get("dropout", 0.0)
        self.ignore_index = config.get("ignore_index", 1)

        # 1) CNN (из cnn.py)
        self.cnn = CNN()

        # 2) "Row Encoder" LSTM: вход = 512 (число каналов на выходе из CNN),
        #    скрытое = encoder_num_hidden
        self.encoder = StackedLSTM(
            input_size=512,
            hidden_size=self.encoder_num_hidden,
            num_layers=self.encoder_num_layers,
            dropout=self.dropout
        )

        # 3) Внимание. Если хотим biLSTM, надо завести 2 энкодера,
        #    но для упрощения будем считать, что один.
        self.attention = BahdanauAttention(self.decoder_num_hidden)

        # 4) Decoder
        #   а) Встроим Embedding для target-символов
        self.embedding = nn.Embedding(self.target_vocab_size, self.target_embedding_size)
        #   б) Decoder LSTM, куда на вход подаётся concat(Emb, контекст).
        #   Но можно и через «input_feed», как в Lua.  
        self.decoder_rnn = StackedLSTM(
            input_size=self.target_embedding_size + self.encoder_num_hidden,
            hidden_size=self.decoder_num_hidden,
            num_layers=self.decoder_num_layers,
            dropout=self.dropout
        )

        # 5) Выходной проектор
        self.output_projector = OutputProjector(self.decoder_num_hidden, self.target_vocab_size)

        # 6) Критерий (NLLLoss)
        self.criterion = create_criterion(self.target_vocab_size, ignore_index=self.ignore_index)


    def forward(self, images, targets=None):
        """
        images: (batch, 1, H, W)
        targets: (batch, tgt_len) (опционально, если тренируем)
        """
        batch_size = images.size(0)

        # ---------------------
        # 1) Прогон через CNN
        # ---------------------
        # features.shape = (batch, H', W', 512)
        features = self.cnn(images)

        # Допустим, считаем H' это "кол-во строк",
        # а W' - "кол-во колонок". "Row encoder" будет идти по W'.
        b, Hp, Wp, C = features.shape

        # Сольём высоту (Hp) внутрь batch – чтобы для каждого "row" запустить LSTM
        # Но в оригинале (model.lua) была чуть иная логика.  
        # Здесь просто иллюстрация:
        # (batch * Hp, Wp, C) → (Wp, batch*Hp, C)
        row_enc_input = features.view(b*Hp, Wp, C).permute(1, 0, 2).contiguous()

        # ---------------------
        # 2) Encoder LSTM
        # ---------------------
        # Получаем (seq_len=Wp, batch*Hp, hidden)
        encoder_outputs, encoder_hidden = self.encoder(row_enc_input)

        # encoder_outputs обратно: (batch*Hp, Wp, hidden)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # При желании можно «разбить» обратно на (batch, Hp, Wp, hidden)
        # но тут оставим как есть.

        # ---------------------
        # 3) Если идёт тренировка (targets != None),
        #    делаем Teacher Forcing
        # ---------------------
        if targets is not None:
            # targets.shape = (batch, tgt_len)
            tgt_len = targets.size(1)

            # Инициализация decoder state нулями (упрощённо)
            dec_hidden = []
            for _ in range(self.decoder_num_layers):
                h = torch.zeros(batch_size, self.decoder_num_hidden, device=images.device)
                c = torch.zeros(batch_size, self.decoder_num_hidden, device=images.device)
                dec_hidden.append((h, c))

            # Запоминаем логи потерь
            loss = 0.0

            for t in range(tgt_len):
                # 1) Вытаскиваем очередной токен: (batch,)
                token_t = targets[:, t]

                # 2) Embedding: (batch, embed_dim)
                emb_t = self.embedding(token_t)

                # 3) Допустим, берём «суммарный» контекст через среднее и внимание:
                #    или как вариант, берём просто последний энкодер hidden
                #    В оригинале – своя логика внимания по encoder_outputs.
                #    Пример (возьмём h первого слоя decoder):
                h_dec = dec_hidden[-1][0]  # скрытый state верхнего слоя decoder
                context_t, attn_weights = self.attention(h_dec,  # (batch, dec_hidden)
                                                         # Превратим encoder_outputs в (batch, seq_len, hidden)
                                                         encoder_outputs.reshape(b, Wp*Hp, -1))

                # 4) Собираем input для Decoder LSTM
                dec_input = torch.cat([emb_t, context_t], dim=1)  # (batch, embed_dim + hidden)
                dec_input = dec_input.unsqueeze(0)  # сделаем (1, batch, ...), т.к. StackedLSTM ждёт seq_len первым

                # 5) Прогоняем через decoder_rnn
                dec_out, dec_hidden = self.decoder_rnn(dec_input, dec_hidden)
                # dec_out.shape = (1, batch, dec_hidden_size)
                dec_out = dec_out.squeeze(0)  # (batch, dec_hidden_size)

                # 6) выходной слой (лог-вероятности)
                log_probs = self.output_projector(dec_out)  # (batch, vocab_size)

                # 7) Считаем loss
                loss_t = self.criterion(log_probs, token_t)
                loss += loss_t

            return loss / batch_size  # усредним или вернём сумму

        else:
            # Режим инференса (greedy/beam search)
            # Здесь пример "жадного" шага, когда мы не знаем targets.
            # TODO: реализовать beam search при необходимости.
            max_length = 100
            generated = []
            dec_hidden = []
            for _ in range(self.decoder_num_layers):
                h = torch.zeros(batch_size, self.decoder_num_hidden, device=images.device)
                c = torch.zeros(batch_size, self.decoder_num_hidden, device=images.device)
                dec_hidden.append((h, c))

            # Начнём с символа <sos>=2, например
            inp_token = torch.zeros(batch_size, dtype=torch.long, device=images.device).fill_(2)

            for t in range(max_length):
                emb_t = self.embedding(inp_token)
                h_dec = dec_hidden[-1][0]
                context_t, _ = self.attention(h_dec, encoder_outputs.view(b*Hp, Wp, -1))
                dec_input = torch.cat([emb_t, context_t], dim=1).unsqueeze(0)
                dec_out, dec_hidden = self.decoder_rnn(dec_input, dec_hidden)
                dec_out = dec_out.squeeze(0)
                log_probs = self.output_projector(dec_out)
                # Берём argmax
                next_token = log_probs.argmax(dim=1)
                generated.append(next_token)
                inp_token = next_token

            # Склеим выход по шагам
            generated = torch.stack(generated, dim=1)  # (batch, max_length)
            return generated
