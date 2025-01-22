import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import DecoderAttention

class Decoder(nn.Module):
    """
    Аналог декодера с 'input_feed':
      - Вход на каждом шаге: [embedding(y_{t-1}); prev_context].
      - LSTM над этим входом.
      - Далее считаем внимание "on top_h" (верхний слой LSTM).
      - Сохраняем контекст -> подаём на вход следующего шага.
    """
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 num_layers,
                 dropout=0.0,
                 input_feed=True,
                 pad_idx=1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_feed = input_feed
        self.pad_idx = pad_idx

        # Embedding для токенов
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)

        # LSTM (stacked). Вход:
        #   если input_feed=True, тогда (embed_size + hidden_size)
        #   иначе (embed_size)
        rnn_input_dim = embed_size + (hidden_size if input_feed else 0)
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = rnn_input_dim if i == 0 else hidden_size
            self.lstm_cells.append(self._build_lstm_cell(in_size, hidden_size))

        # attention
        self.attn = DecoderAttention(hidden_size)

    def _build_lstm_cell(self, input_size, hidden_size):
        """Возвращает обычную LSTMCell (как выше)."""
        return nn.LSTMCell(input_size, hidden_size)

    def forward(self, y_prev, prev_hidden, prev_context, context):
        """
        Выполняет один шаг декодера:
         - y_prev: (batch,) индексы предыдущего токена
         - prev_hidden: [(h_l, c_l), ..., ] список для num_layers
         - prev_context: (batch, hidden_size) контекст от предыдущего шага (input_feed)
         - context: (batch, source_len, hidden_size) от энкодера
        Возвращает:
         - next_hidden (обновлённые (h,c) для всех слоёв)
         - current_context (batch, hidden_size) — контекст внимания
         - log_probs (batch, vocab_size) — распределение на выход
        """
        # 1) embedding
        emb = self.embedding(y_prev)  # (batch, embed_size)

        # 2) input_feed?
        if self.input_feed:
            dec_input = torch.cat([emb, prev_context], dim=1)  # (batch, embed_size+hidden_size)
        else:
            dec_input = emb

        # 3) прогон через Stacked LSTM
        new_hidden = []
        x = dec_input
        for layer_idx, cell in enumerate(self.lstm_cells):
            (h_prev, c_prev) = prev_hidden[layer_idx]
            h_next, c_next = cell(x, (h_prev, c_prev))
            # dropout между слоями
            if layer_idx < self.num_layers-1 and self.dropout>0:
                h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            new_hidden.append((h_next, c_next))
            x = h_next

        top_h = new_hidden[-1][0]  # скрытое состояние верхнего слоя

        # 4) считаем внимание
        current_context, _ = self.attn(top_h, context)

        # 5) проецируем (h_t + c_t) -> log_probs
        #   вариант: сначала объединить top_h и current_context
        #   в Lua-коде: local out = tanh(Wc [ht; ct]) => далее Softmax(Wout out)
        #   Здесь для простоты сделаем: out = tanh([h; c]) -> linear -> logsoftmax
        #   Но "output_projector" обычно снаружи (как отдельный модуль). 
        #   Пока вернём просто скрытое состояние, а логи вероятностей — внешне.
        # Чтобы показать всю цепочку, можно сразу вернуть "combined" → линейный слой:
        combined = torch.tanh(torch.cat([top_h, current_context], dim=1))  # (batch, hidden_size*2)
        # Но в "model.py" мы ещё добавим линейный выход.

        return new_hidden, current_context, combined
