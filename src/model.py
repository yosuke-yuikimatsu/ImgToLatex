import torch
import torch.nn as nn

from cnn import CNN
from row_encoder import RowEncoder
from decoder import Decoder
from output_projector import OutputProjector
from criterion import create_criterion

class FullOCRModel(nn.Module):
    """
    Модель, повторяющая архитектуру:
      - CNN
      - biLSTM (fw/bw) построчно + positional embeddings
      - Decoder LSTM c input-feeding + Attention
      - Output projector
      - Criterion (NLL)
    """
    def __init__(self,
                 vocab_size,             # размер словаря
                 cnn_feature_size=512,   # выходное кол-во каналов CNN
                 encoder_num_hidden=256,
                 encoder_num_layers=1,
                 max_encoder_l_h=32,     # макс. кол-во строк
                 decoder_num_hidden=512,
                 decoder_num_layers=1,
                 target_embedding_size=256,
                 dropout=0.0,
                 input_feed=True,
                 pad_idx=1):
        super(FullOCRModel, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.decoder_num_layers = decoder_num_layers
        self.target_embedding_size = target_embedding_size
        self.dropout = dropout
        self.input_feed = input_feed
        self.pad_idx = pad_idx

        # 1) CNN
        self.cnn = CNN()

        # 2) RowEncoder (biLSTM + pos embeddings)
        self.row_encoder = RowEncoder(
            cnn_feature_size=cnn_feature_size,
            hidden_size=encoder_num_hidden,
            num_layers=encoder_num_layers,
            max_h=max_encoder_l_h,
            dropout=dropout
        )

        # 3) Decoder (LSTM + attention + input_feed)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=target_embedding_size,
            hidden_size=decoder_num_hidden,
            num_layers=decoder_num_layers,
            dropout=dropout,
            input_feed=input_feed,
            pad_idx=pad_idx
        )

        # 4) OutputProjector
        #   На выход декодера мы делаем concat(top_h, current_context), это 2*decoder_num_hidden
        #   Можно сделать доп. линейный слой → hidden_size, как в Lua, но тут напрямую:
        self.output_projector = OutputProjector(decoder_num_hidden*2, vocab_size)

        # 5) Criterion
        self.criterion = create_criterion(vocab_size, ignore_index=pad_idx)

    def forward(self, images, targets=None):
        """
        images: (batch, 1, H, W)
        targets: (batch, T) (опционально)
          Если targets=None, работает в режиме инференса (greedy).
          Иначе считает loss.
        """
        B = images.size(0)

        # === (1) CNN -> feature map ===
        feats = self.cnn(images)  # (B, H', W', 512)

        # === (2) biLSTM over rows ===
        #     Возвращаем context: (B, H'*W', 2*enc_hidden)
        context = self.row_encoder(feats)  # shape: (B, seq_len, 2*enc_hidden)

        # Для внимания удобно context сделать: (B, seq_len, dec_hidden)?
        # Но если encoder_num_hidden != decoder_num_hidden, можно
        # подогнать через линейный слой. Предположим, что dec_hidden = 2*enc_hidden.
        # Если нет, добавьте self.enc2dec = nn.Linear(2*enc_hidden, decoder_num_hidden).
        # context_dec = self.enc2dec(context) # (B, seq_len, dec_hidden)
        # А здесь для простоты допустим: decoder_num_hidden == 2*encoder_num_hidden
        seq_len = context.size(1)

        # === Если нет targets => инференс (greedy) ===
        if targets is None:
            max_len = 100
            preds = []

            # Инициализируем decoder hidden нулями
            # decoder.num_layers
            dec_hidden = []
            for _ in range(self.decoder_num_layers):
                h = torch.zeros(B, self.decoder_num_hidden, device=images.device)
                c = torch.zeros(B, self.decoder_num_hidden, device=images.device)
                dec_hidden.append((h,c))

            # prev_context для input_feed
            prev_context = torch.zeros(B, self.decoder_num_hidden, device=images.device)

            # Начнём с <sos> = 2 (пример)
            y_t = torch.full((B,), 2, dtype=torch.long, device=images.device)

            for t in range(max_len):
                dec_hidden, cur_context, combined = self.decoder(
                    y_prev=y_t,
                    prev_hidden=dec_hidden,
                    prev_context=prev_context,
                    context=context
                )
                # получаем лог-вероятности
                log_probs = self.output_projector(combined)  # (B, vocab_size)
                # берём argmax
                y_t = log_probs.argmax(dim=1)  # (B,)

                preds.append(y_t)
                prev_context = cur_context

            preds = torch.stack(preds, dim=1)
            return preds  # (B, max_len)

        else:
            # === Train mode: Teacher forcing по всем шагам targets. ===
            batch_loss = 0.0
            T = targets.size(1)

            # Decoder hidden init нулями
            dec_hidden = []
            for _ in range(self.decoder_num_layers):
                h = torch.zeros(B, self.decoder_num_hidden, device=images.device)
                c = torch.zeros(B, self.decoder_num_hidden, device=images.device)
                dec_hidden.append((h,c))
            prev_context = torch.zeros(B, self.decoder_num_hidden, device=images.device)

            # Проходим по каждому шагу в target
            for t in range(T):
                # y_{t-1}, но в Lua-коде "targets[t]" часто это "текущий"?
                # Допустим, берем targets[:, t] как вход.
                y_inp = targets[:, t]

                dec_hidden, cur_context, combined = self.decoder(
                    y_prev=y_inp,
                    prev_hidden=dec_hidden,
                    prev_context=prev_context,
                    context=context
                )
                # Вычисляем лог-вероятности
                log_probs = self.output_projector(combined)  # (B, vocab_size)

                # Считаем loss
                # На timestep t, целевой токен - это targets[:, t].
                # (B,)
                loss_t = self.criterion(log_probs, y_inp)
                batch_loss += loss_t

                prev_context = cur_context

            # Усредним по batch'у (как в Lua), можно ещё разделить на (B*T), зависит от желания
            avg_loss = batch_loss / B
            return avg_loss
