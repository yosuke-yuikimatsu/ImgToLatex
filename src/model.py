import torch
import torch.nn as nn
from cnn import CNN
from encoder import TransformerEncoderModule
from decoder import TransformerDecoderModule

class ImageToLatexModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        enc_hidden_dim: int = 2176,
        pad_idx: int = 0,
        sos_index: int = 1,
        eos_index: int = 2,
        max_length: int = 300
    ):
        super().__init__()
        self.cnn = CNN(output_channels=enc_hidden_dim)
        self.encoder = TransformerEncoderModule(enc_hid_dim=enc_hidden_dim, num_layers=24, ffn_dim=8192)
        self.decoder = TransformerDecoderModule(
            vocab_size=vocab_size,
            embed_dim=enc_hidden_dim,
            num_heads=16,
            num_layers=12,
            ffn_dim=8192,
            max_length=max_length,
            sos_index=sos_index,
            eos_index=eos_index
        )
        self.pad_idx = pad_idx
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_length = max_length

    def forward(self, images, tgt_tokens=None):
        features = self.cnn(images)
        encoder_outputs = self.encoder(features)
        B, H, W, D = encoder_outputs.shape
        memory = encoder_outputs.view(B, H * W, D)
        if tgt_tokens is not None:
            logits = self.decoder(tgt_tokens, memory)
            return logits  # В режиме обучения возвращаем только логиты
        else:
            logits, predicted_tokens = self.decoder(tgt_tokens, memory)
            return logits, predicted_tokens  # В режиме инференса возвращаем логиты и токены