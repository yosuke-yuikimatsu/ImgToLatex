import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, output_channels=512):
        """
        Параметры:
          output_channels: число каналов на выходе, например 512 или другое значение.
          pretrained: использовать ли предобученную ResNet50.
        """
        super(CNN, self).__init__()
        # Загружаем предобученную ResNet50
        resnet = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
        # Убираем слои глобального усреднения и классификации:
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Если требуется изменить число каналов (ResNet50 выдает 2048), добавляем 1x1 свертку.
        self.conv_reduction = None
        if output_channels != 2048:
            self.conv_reduction = nn.Conv2d(2048, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, x):
        """
        x: входной тензор с размерностью (B, 3, H, W)
        Возвращает: (B, H', W', output_channels)
        """
        x = self.features(x)  # (B, 2048, H', W')
        if self.conv_reduction is not None:
            x = self.conv_reduction(x)  # (B, output_channels, H', W')
        # Меняем порядок осей: (B, output_channels, H', W') -> (B, H', W', output_channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
