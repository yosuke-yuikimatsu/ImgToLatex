
import torch
import torch.nn as nn

from cnn import CNN
from encoder import EncoderBiLSTM
from decoder import Decoder


class ImageToLatexModel(nn.Module):
    """
    Модель, объединяющая:
      1) CNN для извлечения признаков из картинки (размер выхода: B, H', W', 512)
      2) BiLSTM-Encoder для преобразования признаков (размер выхода: B, H', W', 2*enc_hidden_dim)
      3) Decoder (LSTM с механизмом внимания), который генерирует выходную последовательность.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        enc_hidden_dim: int = 256,
        dec_hidden_dim: int = 512,
        pad_idx: int = 0,
        sos_index: int = 129,
        eos_index: int = 130,
        max_length: int = 300
    ):
        """
        Параметры:
          vocab_size    : размер словаря (количество выходных токенов)
          embed_dim     : размер эмбеддингов в декодере
          enc_hidden_dim: размер скрытого состояния в BiLSTM энкодера
          dec_hidden_dim: размер скрытого состояния декодера
          pad_idx       : индекс токена PAD
          sos_index     : индекс токена начала последовательности (SOS)
          eos_index     : индекс токена конца последовательности (EOS)
          max_length    : максимальная длина генерируемой последовательности
        """
        super().__init__()

        # 1) Сверточная сеть (CNN)
        self.cnn = CNN()

        # 2) Энкодер на базе BiLSTM
        # Выходная размерность = 2*enc_hidden_dim
        self.encoder = EncoderBiLSTM(
            input_dim=512,
            hidden_dim=enc_hidden_dim,
            num_layers=1,        # можно увеличить при необходимости
            dropout=0.0          # если num_layers > 1, можно добавить дропаут
        )

        # 3) Декодер с вниманием
        # Здесь encoder_dim = 2*enc_hidden_dim
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=2 * enc_hidden_dim,
            decoder_hidden=dec_hidden_dim,
            pad_idx=pad_idx,
            sos_index=sos_index,
            eos_index=eos_index,
            max_length=max_length
        )

    def forward(self, images, tgt_tokens=None, teacher_forcing_ratio=0.0):
        """
        Параметры:
          images: тензор изображений (B, 3, H, W)
          tgt_tokens: (B, T) целевые токены (при обучении) или None (при генерации)
          teacher_forcing_ratio: вероятность teacher forcing

        Возвращает:
          - При обучении (если tgt_tokens не None): (logits, alphas)
            * logits: (B, T-1, vocab_size)
            * alphas: список (или тензор) весов внимания
          - При генерации (если tgt_tokens=None): (generated_tokens, alphas)
            * generated_tokens: (B, генерируемая_длина)
            * alphas: аналогично, для анализа внимания
        """
        # 1) Получаем признаки из картинки через CNN
        features = self.cnn(images)  # (B, H', W', 512)

        # 2) Пропускаем через BiLSTM-энкодер
        encoder_outputs = self.encoder(features)  # (B, H', W', 2*enc_hidden_dim)

        # 3) Декодируем в выходную последовательность
        outputs = self.decoder(
            encoder_outputs,
            tgt_tokens=tgt_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return outputs