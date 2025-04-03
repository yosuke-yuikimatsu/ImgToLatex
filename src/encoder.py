import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class OptimizedPositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=324, width=324, dropout=0.1):
        super().__init__()
        # Use parameter buffers instead of Embedding layers for better efficiency
        pos_height = torch.arange(height).unsqueeze(1).expand(height, width).float()
        pos_width = torch.arange(width).unsqueeze(0).expand(height, width).float()
        
        # Create 2D position encodings directly
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe_h = torch.zeros(height, width, d_model)
        pe_w = torch.zeros(height, width, d_model)
        
        pe_h[:, :, 0::2] = torch.sin(pos_height.unsqueeze(-1) * div_term)
        pe_h[:, :, 1::2] = torch.cos(pos_height.unsqueeze(-1) * div_term)
        pe_w[:, :, 0::2] = torch.sin(pos_width.unsqueeze(-1) * div_term)
        pe_w[:, :, 1::2] = torch.cos(pos_width.unsqueeze(-1) * div_term)
        
        self.register_buffer('pe', pe_h + pe_w)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        B, H, W, D = x.shape
        # Use pre-computed positional encodings with proper slicing
        pe = self.pe[:H, :W, :].unsqueeze(0)
        return self.dropout(x + pe)

class TransformerEncoderModule(nn.Module):
    def __init__(self, enc_hid_dim=1536, num_heads=12, num_layers=6, ffn_dim=3072, 
                 height=324, width=324, dropout=0.1, activation="gelu",training=True):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.trainining = training
        
        # More efficient positional encoding
        self.positional_encoding = OptimizedPositionalEncoding2D(enc_hid_dim, height, width, dropout)
        
        # Use more efficient activation function
        encoder_layer = TransformerEncoderLayer(
            d_model=enc_hid_dim, 
            nhead=num_heads, 
            dim_feedforward=ffn_dim, 
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm architecture for better training stability
        )
        
        # Add layer norm before the encoder for better gradient flow
        self.pre_norm = nn.LayerNorm(enc_hid_dim)
        
        # Use gradient checkpointing for memory efficiency
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.use_checkpoint = True
        
    def forward(self, x):
        B, H, W, D = x.shape
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Reshape for transformer
        x_flat = x.view(B, H * W, D)
        
        # Apply pre-norm
        x_flat = self.pre_norm(x_flat)
        
        # Use gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            x_enc = torch.utils.checkpoint.checkpoint(
                self.transformer_encoder, x_flat,use_reentrant=False
            )
        else:
            x_enc = self.transformer_encoder(x_flat)
            
        return x_enc
