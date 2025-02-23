import torch
import torch.nn as nn
from cnn import CNN
from decoder import Decoder
from encoder import TransformerEncoderModule

class ImageToLatexModel(nn.Module):
    """
    Модель, объединяющая:
      1) CNN для извлечения признаков из изображения (B, H', W', 512)
      2) Transformer-энкодер, который принимает вход (B, H', W', 512),
         преобразует его в последовательность и возвращает в исходную форму
         (B, H', W', 512)
      3) Декодер, который принимает выход энкодера (B, H', W', 512)
         и генерирует последовательность токенов.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        enc_hidden_dim: int = 512,
        dec_hidden_dim: int = 512,
        pad_idx: int = 0,
        sos_index: int = 129,
        eos_index: int = 130,
        max_length: int = 300
    ):
        super().__init__()

        # 1) CNN – выдаёт (B, H', W', 512)
        self.cnn = CNN()

        # 2) Transformer-энкодер, который теперь работает с размерностью 512
        self.encoder = TransformerEncoderModule(enc_hid_dim=enc_hidden_dim)

        # 3) Декодер, который ожидает вход (B, H', W', 512)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=enc_hidden_dim,
            decoder_hidden=dec_hidden_dim,
            pad_idx=pad_idx,
            sos_index=sos_index,
            eos_index=eos_index,
            max_length=max_length
        )

    def forward(self, images, tgt_tokens=None, teacher_forcing_ratio=0.0):
        # 1) Изображение -> CNN: (B, H', W', 512)
        features = self.cnn(images)
        # 2) Пропуск через Transformer-энкодер:
        encoder_outputs = self.encoder(features)  # (B, H', W', 512)
        # 3) Передаем в декодер:
        outputs = self.decoder(
            encoder_outputs,
            tgt_tokens=tgt_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        return outputs
