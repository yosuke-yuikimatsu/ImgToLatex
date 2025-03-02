import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=256, width=256):
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
    def __init__(self, enc_hid_dim = 2176, num_heads = 16, num_layers= 12, ffn_dim = 2048, height = 256, width = 256):
        super().__init__()
        self.positional_encoding = PositionalEncoding2D(enc_hid_dim, height, width)
        self.transformer_encoder = nn.ModuleList([TransformerEncoderLayer(d_model=enc_hid_dim, nhead=num_heads, dim_feedforward=ffn_dim, batch_first=True) for _ in range(num_layers)])

    def forward(self, x):
        B, H, W, D = x.shape
        x = self.positional_encoding(x)
        x_flat = x.view(B, H * W, D)
        num_segments = 4
        segment_size = len(self.transformer_encoder) // num_segments
        for i in range(0, len(self.transformer_encoder), segment_size):
            x_flat = checkpoint(self._forward_segment, x_flat, i, segment_size,use_reentrant=False)
        x_out = x_flat.view(B, H, W, D)
        return x_out

    def _forward_segment(self, x, start, segment_size):
        for i in range(start, start + segment_size):
            x = self.transformer_encoder[i](x)
        return x