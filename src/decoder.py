import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class AttentionAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_length):
        super(AttentionAutoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Эмбеддинг для входных токенов
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Позиционное кодирование
        self.pos_encoding = nn.Embedding(max_length, embed_dim)

        # Трансформерный энкодер для анализа входной последовательности
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim,batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=6)

        # Трансформерный декодер для генерации исправленной последовательности
        decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim,batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=6)

        # Выходной слой
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_tokens, input_mask=None):
        B, T = input_tokens.shape
        device = input_tokens.device

        # Преобразуем токены в эмбеддинги
        src = self.embedding(input_tokens)  # (B, T, embed_dim)

        # Добавляем позиционное кодирование
        positions = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
        src = src + self.pos_encoding(positions)

        # Энкодер
        memory = self.encoder(src.transpose(0, 1), src_key_padding_mask=input_mask)  # (T, B, embed_dim)

        # Декодер (используем ту же последовательность как цель)
        tgt = src.transpose(0, 1)  # (T, B, embed_dim)
        output = self.decoder(tgt, memory, tgt_key_padding_mask=input_mask)  # (T, B, embed_dim)
        output = output.transpose(0, 1)  # (B, T, embed_dim)

        # Логиты исправленной последовательности
        logits = self.output_layer(output)  # (B, T, vocab_size)
        return logits

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        decoder_hidden: int,
        pad_idx: int = 0,
        sos_index: int = 0,
        eos_index: int = 130,
        max_length: int = 300
    ):
        super().__init__()

        self.pad_index = pad_idx
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_length = max_length

        # Основной декодер
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn_cell = nn.LSTMCell(embed_dim + decoder_hidden, decoder_hidden)
        self.W_h = nn.Linear(decoder_hidden, decoder_hidden, bias=False)
        self.W_v = nn.Linear(encoder_dim, decoder_hidden, bias=False)
        self.v = nn.Linear(decoder_hidden, 1, bias=False)
        self.W_c = nn.Linear(decoder_hidden + encoder_dim, decoder_hidden)
        self.W_out = nn.Linear(decoder_hidden, vocab_size)

        self.decoder_hidden = decoder_hidden
        self.encoder_dim = encoder_dim

        # Автоэнкодер для исправления ошибок
        self.autoencoder = AttentionAutoencoder(vocab_size, embed_dim, decoder_hidden, max_length)

    def _compute_attention(self, h_prev, encoder_outputs):
        B, H, W, C = encoder_outputs.shape
        enc_reshaped = encoder_outputs.contiguous().view(B, H*W, C)
        Wh = self.W_h(h_prev).unsqueeze(1)
        Wv = self.W_v(enc_reshaped)
        score = self.v(torch.tanh(Wh + Wv)).squeeze(-1)
        alpha = F.softmax(score, dim=1)
        context = torch.sum(enc_reshaped * alpha.unsqueeze(-1), dim=1)
        return alpha, context

    def forward(self, encoder_outputs, tgt_tokens=None, teacher_forcing_ratio=0.0):
        B, H, W, _ = encoder_outputs.shape
        device = encoder_outputs.device

        if tgt_tokens is not None:
            B_tgt, T = tgt_tokens.shape
            assert B == B_tgt
        else:
            T = self.max_length

        hx = torch.zeros(B, self.decoder_hidden, device=device)
        cx = torch.zeros(B, self.decoder_hidden, device=device)
        o_prev = torch.zeros(B, self.decoder_hidden, device=device)

        outputs = []
        alphas_all = []
        input_token = torch.full((B,), self.sos_index, dtype=torch.long, device=device)
        generated_tokens = torch.zeros(B, 0, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, T):
            emb = self.embedding(input_token)
            rnn_input = torch.cat([emb, o_prev], dim=-1)
            hx, cx = self.rnn_cell(rnn_input, (hx, cx))
            alpha, context = self._compute_attention(hx, encoder_outputs)
            alphas_all.append(alpha)
            concat = torch.cat([hx, context], dim=-1)
            o_t = torch.tanh(self.W_c(concat))
            logits_t = self.W_out(o_t)
            outputs.append(logits_t.unsqueeze(1))

            if tgt_tokens is not None:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                next_token = tgt_tokens[:, t] if use_teacher_forcing else logits_t.argmax(dim=-1)
            else:
                next_token = logits_t.argmax(dim=-1)

            finished = finished | (next_token == self.eos_index)
            if tgt_tokens is None:
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
            input_token = torch.where(finished, torch.tensor(self.pad_index, device=device), next_token)
            o_prev = o_t

            if tgt_tokens is None and finished.all():
                break

        if tgt_tokens is not None:
            logits = torch.cat(outputs, dim=1)  # (B, T-1, vocab_size)
            # Применяем автоэнкодер для исправления
            pred_tokens = logits.argmax(dim=-1)  # (B, T-1)
            corrected_logits = self.autoencoder(pred_tokens)  # (B, T-1, vocab_size)
            return corrected_logits, alphas_all
        else:
            # Применяем автоэнкодер для исправления сгенерированных токенов
            corrected_logits = self.autoencoder(generated_tokens)  # (B, generated_length, vocab_size)
            corrected_tokens = corrected_logits.argmax(dim=-1)  # (B, generated_length)
            return corrected_tokens, alphas_all