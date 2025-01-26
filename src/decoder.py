import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Декодер с LSTM + механизмом внимания,
    реализованный по формулам из условия:
      h_t = RNN(h_{t-1}, [y_{t-1}, o_{t-1}])
      c_t = sum(alpha_{t,h,w} * V_{h,w})
      o_t = tanh(W_c [h_t; c_t])
      p(y_{t+1} | ...) = softmax(W_out o_t)
    """
    def __init__(
        self,
        vocab_size: int,       # размер словаря выходных токенов
        embed_dim: int,        # размер эмбеддинга для входных (предыдущих) токенов
        encoder_dim: int,      # размерность выхода энкодера (C = 2*hidden_dim BiLSTM, например)
        decoder_hidden: int,   # размер скрытого состояния LSTM декодера
        pad_idx: int = 0,
        sos_index: int = 0,
        eos_index: int = 130,
        max_length: int = 300    # Максимальная длина генерируемой последовательности
    ):
        super().__init__()

        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_length = max_length

        # --- Эмбеддинг для входных токенов (y_{t-1}) ---
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        # --- LSTMCell как RNN-декодер (по формуле h_t = RNN(...)) ---
        # Обратите внимание: на вход LSTM подаём [emb(y_{t-1}), o_{t-1}],
        # значит размер входа = embed_dim + decoder_hidden (так как o_{t-1} имеет ту же размерность, что и h_t, см. ниже)
        self.rnn_cell = nn.LSTMCell(
            input_size=embed_dim + decoder_hidden,
            hidden_size=decoder_hidden
        )

        # --- Параметры для механизма внимания (Bahdanau-style) ---
        # e_{t,h,w} = beta^T tanh(W_h h_{t} + W_v V_{h,w})
        self.W_h = nn.Linear(decoder_hidden, decoder_hidden, bias=False)  # W_h
        self.W_v = nn.Linear(encoder_dim,    decoder_hidden, bias=False)  # W_v
        self.v   = nn.Linear(decoder_hidden, 1, bias=False)               # beta^T

        # --- Линейное преобразование [h_t; c_t] -> o_t ---
        # По формуле o_t = tanh(W_c [h_t; c_t]).
        # Размер [h_t; c_t] = decoder_hidden + encoder_dim.
        # Часто делают так, чтобы o_t тоже имел размер decoder_hidden (или другой).
        # Допустим, сделаем o_t такой же размерности, как h_t:
        self.W_c = nn.Linear(decoder_hidden + encoder_dim, decoder_hidden)

        # --- Выходной линейный слой: o_t -> logits по словарю ---
        self.W_out = nn.Linear(decoder_hidden, vocab_size)

        self.decoder_hidden = decoder_hidden
        self.encoder_dim = encoder_dim

    def _compute_attention(self, h_prev, encoder_outputs):
        """
        Считаем «выделение» (alignment) alpha_t и контекст c_t.

        encoder_outputs: (B, H, W, encoder_dim)
        h_prev: (B, decoder_hidden) -- это h_{t} (после обновления состояния)

        Вернёт:
          alpha (B, H*W)
          context (B, encoder_dim)
        """
        B, H, W, C = encoder_outputs.shape  # C = encoder_dim
        # Разворачиваем H,W в одно измерение:
        enc_reshaped = encoder_outputs.contiguous().view(B, H*W, C)  # (B, H*W, C)

        # W_h h_t: (B, decoder_hidden) -> (B, 1, decoder_hidden)
        Wh = self.W_h(h_prev).unsqueeze(1)             # (B, 1, decoder_hidden)
        # W_v V_{h,w}: (B, H*W, C) -> (B, H*W, decoder_hidden)
        Wv = self.W_v(enc_reshaped)                    # (B, H*W, decoder_hidden)

        # score = v^T tanh( Wh + Wv ), выходим скаляр (размер 1) на каждую позицию h,w
        score = self.v(torch.tanh(Wh + Wv))            # (B, H*W, 1)
        score = score.squeeze(-1)                      # (B, H*W)

        # alpha = softmax(score) (по оси H*W)
        alpha = F.softmax(score, dim=1)                # (B, H*W)

        # context = sum_i alpha_i * v_i
        context = torch.sum(enc_reshaped * alpha.unsqueeze(-1), dim=1)  # (B, C)

        return alpha, context

    def forward(self, encoder_outputs, tgt_tokens=None, teacher_forcing_ratio=1.0):
        """
        Запускаем декодер по всем шагам (T) или генерируем последовательность.

        encoder_outputs: (B, H, W, encoder_dim)
        tgt_tokens: (B, T) -- целевая последовательность (с [BOS], ..., [EOS])
                    Если None, выполняется генерация.

        Возврат:
          logits: (B, T-1, vocab_size)  — предсказанные распределения для шагов 1..T-1 (только при обучении)
                  Или список предсказанных токенов (B, generated_length) при генерации.
          alphas: список или тензор весов внимания по шагам (можно вернуть для анализа)
        """
        B, H, W, _ = encoder_outputs.shape
        device = encoder_outputs.device

        if tgt_tokens is not None:
            B_tgt, T = tgt_tokens.shape
            assert B == B_tgt, "Размер батча в encoder_outputs и tgt_tokens должен совпадать."
        else:
            # Определяем максимальную длину для генерации
            T = self.max_length

        # Инициализация скрытого состояния декодера
        hx = torch.zeros(B, self.decoder_hidden, device=device)
        cx = torch.zeros(B, self.decoder_hidden, device=device)

        # Начальное o_prev
        o_prev = torch.zeros(B, self.decoder_hidden, device=device)

        outputs = []
        alphas_all = []

        # Инициализация input_token с SOS токена
        input_token = torch.full((B,), self.sos_index, dtype=torch.long, device=device)  # (B,)

        # Для генерации: сохраняем предсказанные токены
        generated_tokens = torch.zeros(B, 0, dtype=torch.long, device=device)

        # Флаг завершения генерации для каждого элемента в батче
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, T):
            # 1) Эмбеддинг и подготовка входа для RNN
            emb = self.embedding(input_token)  # (B, embed_dim)
            rnn_input = torch.cat([emb, o_prev], dim=-1)  # (B, embed_dim + decoder_hidden)

            # 2) Обновление состояния LSTM
            hx, cx = self.rnn_cell(rnn_input, (hx, cx))
            # теперь hx = h_t

            # 3) Механизм внимания
            alpha, context = self._compute_attention(hx, encoder_outputs)
            alphas_all.append(alpha)

            # 4) Формирование o_t и предсказание логитов
            concat = torch.cat([hx, context], dim=-1)   # (B, decoder_hidden + encoder_dim)
            o_t = torch.tanh(self.W_c(concat))          # (B, decoder_hidden)

            logits_t = self.W_out(o_t)  # (B, vocab_size)
            outputs.append(logits_t.unsqueeze(1))

            # 5) Выбор следующего токена
            if tgt_tokens is not None:
                # Обучение с Teacher Forcing
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    next_token = tgt_tokens[:, t]
                else:
                    next_token = logits_t.argmax(dim=-1)
            else:
                # Генерация: всегда используем предсказанные токены
                next_token = logits_t.argmax(dim=-1)

            # 6) Обновление флага завершения генерации
            finished = finished | (next_token == self.eos_index)

            # 7) Сохранение предсказанных токенов (только при генерации)
            if tgt_tokens is None:
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)

            # 8) Обновление input_token: только для тех, кто ещё не закончил генерацию
            # Для тех, кто уже закончил, подаём PAD (или любой другой токен, который не влияет)
            input_token = torch.where(finished, torch.tensor(0, device=device), next_token)

            # 9) Обновление o_prev
            o_prev = o_t

            # 10) Проверка, завершена ли генерация для всех примеров
            if tgt_tokens is None and finished.all():
                break

        if tgt_tokens is not None:
            # Обучение: возвращаем логиты и веса внимания
            logits = torch.cat(outputs, dim=1)  # (B, T-1, vocab_size)
            return logits, alphas_all
        else:
            # Генерация: возвращаем предсказанные токены и веса внимания
            return generated_tokens, alphas_all
