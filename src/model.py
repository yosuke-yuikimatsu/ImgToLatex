import torch
import torch.nn as nn
from cnn import CNN
from encoder import TransformerEncoderModule
from decoder import TransformerDecoderModule

class ImageToLatexModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        enc_hidden_dim: int = 1536,
        pad_idx: int = 0,
        sos_index: int = 1,
        eos_index: int = 2,
        max_length: int = 50,
        beam_width: int = 5
    ):
        super().__init__()
        self.cnn = CNN(output_channels=enc_hidden_dim)
        self.encoder = TransformerEncoderModule(enc_hid_dim=enc_hidden_dim, num_layers=10, ffn_dim=8192, num_heads=8)
        self.decoder = TransformerDecoderModule(
            vocab_size=vocab_size,
            embed_dim=enc_hidden_dim,
            num_heads=8,
            num_layers=10,
            ffn_dim=8192,
            max_length=max_length,
            sos_index=sos_index,
            eos_index=eos_index,
            pad_idx=pad_idx  # Добавляем pad_idx в декодер
        )
        self.pad_idx = pad_idx
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_length = max_length
        self.beam_width = beam_width

    def forward(self, images, tgt_tokens=None, train=False):
        features = self.cnn(images)
        encoder_outputs = self.encoder(features)
        B, H, W, D = encoder_outputs.shape
        memory = encoder_outputs.view(B, H * W, D)

        if train and tgt_tokens is None:
            # RL режим с REINFORCE
            predicted_tokens, rewards, loss = self.decoder(tgt_tokens=None, memory=memory, train=True)
            return predicted_tokens, rewards, loss
        elif tgt_tokens is not None:
            # Supervised режим
            logits = self.decoder(tgt_tokens=tgt_tokens, memory=memory)
            return logits  # В режиме обучения возвращаем только логиты
        else:
            # Инференс режим
            logits, predicted_tokens = self.decoder(tgt_tokens=None, memory=memory, beam_width=self.beam_width)
            return logits, predicted_tokens  # В режиме инференса возвращаем логиты и токены