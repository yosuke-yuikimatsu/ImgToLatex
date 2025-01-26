import torch.nn as nn 

class EncoderBiLSTM(nn.Module):
    """
    Применяем BiLSTM по каждой строке (длина W).
    На вход ожидается тензор (batch, H, W, C), где C=512 (или любой input_dim).
    Выход: (batch, H, W, 2*hidden_dim)
    """
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, dropout=0.0):
        super(EncoderBiLSTM, self).__init__()
        # bidirectional=True для BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # Устанавливаем dropout=0.0, если num_layers=1
            bidirectional=True
        )

    def forward(self, x):
        # x: (batch, H, W, input_dim=512)
        b, H, W, C = x.shape
        # Склеиваем (batch, H) -> одна "партия" строк
        # Получим: (batch*H, W, C)
        x_reshaped = x.view(b * H, W, C)

        # Прогоняем через BiLSTM
        # Выход lstm_out будет (batch*H, W, 2 * hidden_dim),
        # тк bidirectional=True
        lstm_out, _ = self.lstm(x_reshaped)  # (_, (h_n, c_n)) можно получить state при необходимости

        # Возвращаем исходные измерения: (batch, H, W, 2*hidden_dim)
        lstm_out = lstm_out.view(b, H, W, -1)
        return lstm_out
