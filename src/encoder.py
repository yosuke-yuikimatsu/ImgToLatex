import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=128, width=128):
        super().__init__()
        self.pos_height = nn.Embedding(height, d_model)
        self.pos_width = nn.Embedding(width, d_model)

    def forward(self, x):
        B, H, W, D = x.shape
        h_pos = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W)
        w_pos = torch.arange(W, device=x.device).unsqueeze(0).repeat(H, 1)
        pos_enc = self.pos_height(h_pos) + self.pos_width(w_pos)
        return x + pos_enc.unsqueeze(0).repeat(B, 1, 1, 1)

class TransformerEncoderModule(nn.Module):
    def __init__(self, enc_hid_dim=1024, num_heads=16, num_layers=24, ffn_dim=8192, height=128, width=128):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.positional_encoding = PositionalEncoding2D(enc_hid_dim, height, width)
        encoder_layer = TransformerEncoderLayer(
            d_model=enc_hid_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, H, W, D = x.shape
        x = self.positional_encoding(x)  # Добавляем 2D позиционные эмбеддинги
        x_flat = x.view(B, H * W, D)  # Преобразуем в плоский формат для трансформера
        # Убрали checkpoint, используем прямой проход
        x_enc = self.transformer_encoder(x_flat)
        x_out = x_enc.view(B, H, W, D)  # Возвращаем исходную форму
        return x_out