import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    """Одиночная LSTM-ячейка (как в LSTM.lua)."""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4*hidden_size)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size)

    def forward(self, x, hidden):
        h, c = hidden
        gates = self.i2h(x) + self.h2h(h)
        ingate, forgetgate, outgate, g_t = gates.chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        g_t = torch.tanh(g_t)
        c_next = forgetgate*c + ingate*g_t
        h_next = outgate*torch.tanh(c_next)
        return h_next, c_next

class StackedLSTM(nn.Module):
    """
    Многослойный LSTM "вверх-вниз" (без bidirectional).
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, inputs, hidden=None):
        """
        inputs: (seq_len, batch, input_size)
        hidden: список [(h_1, c_1), ..., (h_L, c_L)] (если None — инициализация нулями).
        Возвращаем:
          - выход (seq_len, batch, hidden_size)
          - новое hidden
        """
        seq_len, batch, _ = inputs.size()
        if hidden is None:
            hidden = []
            for _ in range(len(self.cells)):
                h = torch.zeros(batch, self.hidden_size, device=inputs.device)
                c = torch.zeros(batch, self.hidden_size, device=inputs.device)
                hidden.append((h, c))

        outputs = []
        for t in range(seq_len):
            x_t = inputs[t]
            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_new, c_new = cell(x_t, (h_prev, c_prev))
                x_t = h_new  # вход для следующего слоя
                new_hidden.append((h_new, c_new))
                if layer_idx < self.num_layers - 1 and self.dropout>0:
                    x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            outputs.append(x_t)
            hidden = new_hidden
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_size)
        return outputs, hidden


class RowEncoder(nn.Module):
    """
    Аналог модуля, который:
     - Берёт (batch, H', W', 512) фичи
     - Для каждой "строки" i=1..H' получает pos_embedding_fw и pos_embedding_bw
     - Прогоняет её слева-направо (forward LSTM) и справа-налево (backward LSTM)
     - Склеивает hidden-состояния [fwd; bwd]
     - Возвращает context (batch, H'*W', 2*hidden_size)
    """
    def __init__(self, cnn_feature_size, hidden_size, num_layers,
                 max_h, dropout=0.0):
        super(RowEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_feature_size = cnn_feature_size
        self.dropout = dropout

        # forward / backward LSTM
        self.encoder_fw = StackedLSTM(cnn_feature_size, hidden_size, num_layers, dropout)
        self.encoder_bw = StackedLSTM(cnn_feature_size, hidden_size, num_layers, dropout)

        # Позиционные эмбеддинги:
        # В lua-коде размер: (max_encoder_l_h, num_layers*hidden_size*2)
        # Чтобы хранить (h_1, c_1, h_2, c_2, ...) для forward/backward
        # Но будем хранить отдельно:
        self.pos_embedding_fw = nn.Embedding(max_h+1, num_layers*hidden_size*2)
        self.pos_embedding_bw = nn.Embedding(max_h+1, num_layers*hidden_size*2)

    def _init_state_from_pos(self, batch_size, pos_embedding, row_idx):
        """
        Достаём Embedding[row_idx], парсим по num_layers*2 (h_i, c_i).
        pos_embedding.shape = (num_layers*hidden_size*2,)
        Возвращаем список [(h_1, c_1), ..., (h_L, c_L)].
        """
        # pos_emb: (batch_size, num_layers*hidden_size*2)
        pos_emb = pos_embedding(row_idx)  # (batch_size, num_layers*hidden_size*2)

        hidden_states = []
        # Разбиваем на num_layers*2 блоков (каждый = hidden_size)
        # forward: L=1..num_layers => 2*(L-1) to 2*(L-1)+1
        offset = 0
        for _ in range(self.num_layers):
            # h
            h_part = pos_emb[:, offset : offset + self.hidden_size]
            offset += self.hidden_size
            # c
            c_part = pos_emb[:, offset : offset + self.hidden_size]
            offset += self.hidden_size
            hidden_states.append( (h_part, c_part) )
        return hidden_states

    def forward(self, features):
        """
        features: (batch, H', W', cnn_feature_size)
        Возвращаем context: (batch, H'*W', 2*hidden_size)
        """
        B, Hp, Wp, C = features.shape
        # Будем формировать "context" размера (B, Hp*Wp, 2*hidden_size)
        # Для каждой строки i=0..(Hp-1):
        #   1) берем slice: (B, Wp, C)
        #   2) init fwd-состояние из pos_embedding_fw[i+1], init bwd-состояние из pos_embedding_bw[i+1]
        #   3) Прогоняем fwd LSTM слева-направо => out_fwd: (Wp, B, hidden_size)
        #   4) Прогоняем bwd LSTM справа-налево => out_bwd: (Wp, B, hidden_size)
        #   5) Склеиваем out_fwd[t] и out_bwd[t] => (B, 2*hidden_size)
        #   6) Пишем в context[:, i*Wp + t, :]

        context = torch.zeros(B, Hp*Wp, 2*self.hidden_size, device=features.device)

        for i in range(Hp):
            row_slice = features[:, i, :, :]   # (B, Wp, C)
            # Меняем порядок измерений, чтоб было (Wp, B, C)
            row_slice_t = row_slice.permute(1,0,2).contiguous()

            # Создаём начальное состояние через pos_embedding_fw/bw
            # shape = [(h_1, c_1), (h_2, c_2), ...]
            row_idx_tensor = torch.full((B,), i+1, dtype=torch.long, device=features.device)
            # forward
            fwd_init = self._init_state_from_pos(B, self.pos_embedding_fw, row_idx_tensor)
            out_fw, _ = self.encoder_fw(row_slice_t, hidden=fwd_init)  # (Wp, B, hidden_size)

            # backward
            # Разворачиваем row_slice_t задом-наперед по "seq_len=Wp"
            row_slice_t_rev = torch.flip(row_slice_t, [0])  # (Wp->разворот)
            bwd_init = self._init_state_from_pos(B, self.pos_embedding_bw, row_idx_tensor)
            out_bw_rev, _ = self.encoder_bw(row_slice_t_rev, hidden=bwd_init)  # (Wp, B, hidden_size)
            # Вернём out_bw в прямом порядке
            out_bw = torch.flip(out_bw_rev, [0])

            # Теперь склеиваем out_fw[t], out_bw[t], (t=0..Wp-1)
            # out_fw[t] shape = (B, hidden_size)
            # => concat = (B, 2*hidden_size)
            for t in range(Wp):
                idx = i*Wp + t
                cat_val = torch.cat([out_fw[t], out_bw[t]], dim=1)  # (B, 2*hidden_size)
                context[:, idx, :] = cat_val

        return context
