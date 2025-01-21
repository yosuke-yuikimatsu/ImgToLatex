import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h, c = hidden
        gates = self.i2h(x) + self.h2h(h)
        ingate, forgetgate, outgate, g_t = gates.chunk(4, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        g_t = torch.tanh(g_t)

        next_c = forgetgate * c + ingate * g_t
        next_h = outgate * torch.tanh(next_c)

        return next_h, next_c


class StackedLSTM(nn.Module):
    """
    Многослойный LSTM "сверху вниз".
    По сути, это несколько LSTMCell друг над другом.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(StackedLSTM, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            cell = LSTMCell(in_size, hidden_size)
            self.layers.append(cell)

    def forward(self, x, hidden=None):
        """
        x: (seq_len, batch, input_size)
        hidden: [(h_1, c_1), ..., (h_L, c_L)] или None
        Возвращает:
          - выход (seq_len, batch, hidden_size),
          - новое hidden того же формата.
        """
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            # Инициализация нулями
            hidden = []
            for _ in range(self.num_layers):
                device = x.device
                h = torch.zeros(batch_size, self.hidden_size, device=device)
                c = torch.zeros(batch_size, self.hidden_size, device=device)
                hidden.append((h, c))

        outputs = []
        # Проходим по шагам "во времени" (seq_len)
        for t in range(seq_len):
            inp = x[t]
            new_hidden = []
            # Послойно
            for layer_idx, cell in enumerate(self.layers):
                h, c = hidden[layer_idx]
                h, c = cell(inp, (h, c))
                inp = h  # вход для следующего слоя
                new_hidden.append((h, c))
                # Дроп-аут между слоями
                if self.dropout > 0 and layer_idx < self.num_layers - 1:
                    inp = F.dropout(inp, p=self.dropout, training=self.training)
            outputs.append(inp)
            hidden = new_hidden

        # Склеиваем выходы по первому измерению
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_size)
        return outputs, hidden
