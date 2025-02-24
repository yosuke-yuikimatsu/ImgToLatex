import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderModule(nn.Module):
    def __init__(self, enc_hid_dim=512, num_heads=4, num_layers=2, ffn_dim=1024):
        """
        Параметры:
          enc_hid_dim: размерность выходных эмбеддингов (d_model в трансформере)
          num_heads: число голов внимания
          num_layers: число слоёв энкодера
          ffn_dim: размерность внутреннего слоя FFN
        """
        super(TransformerEncoderModule, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        # Позиционное кодирование для последовательностей длиной до 32768
        self.positional_encoding = nn.Embedding(32768, enc_hid_dim)
        encoder_layer = TransformerEncoderLayer(d_model=enc_hid_dim, nhead=num_heads, dim_feedforward=ffn_dim,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: тензор формы (B, H, W, enc_hid_dim)
        Сначала преобразуем в последовательность: (B, H*W, enc_hid_dim),
        затем применим позиционное кодирование и трансформер,
        и, наконец, вернём в форму (B, H, W, enc_hid_dim).
        """
        B, H, W, D = x.shape  # D должен быть равен enc_hid_dim (например, 512)
        seq_len = H * W
        # Преобразуем в (B, seq_len, D)
        x_flat = x.view(B, seq_len, D)
        # Создаем позиции (B, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(B, 1)
        pos_encoding = self.positional_encoding(positions)
        x_flat = x_flat + pos_encoding
        # Пропускаем через трансформер-энкодер
        x_enc = self.transformer_encoder(x_flat)  # (B, seq_len, enc_hid_dim)
        # Возвращаем исходную пространственную форму: (B, H, W, enc_hid_dim)
        x_out = x_enc.view(B, H, W, D)
        return x_out
