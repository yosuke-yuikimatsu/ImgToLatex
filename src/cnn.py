import torch
import torch.nn as nn
import timm

class CNN(nn.Module):
    def __init__(self, output_channels=2152):
        super().__init__()
        # Новая архитектура: EfficientNetV2-L вместо более легкой версии
        self.efficientnet = timm.create_model('efficientnetv2_rw_m', pretrained=True, num_classes=0)
        # Уменьшаем выходные каналы с 1280 до output_channels через 1x1 свертку
        self.conv_reduction = nn.Conv2d(2152, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, x):
        # Извлекаем признаки с помощью EfficientNetV2-L
        x = self.efficientnet.forward_features(x)  # (B, 2152, H', W')
        x = self.conv_reduction(x)  # (B, output_channels, H', W')
        # Меняем порядок осей для совместимости с трансформером
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', output_channels)
        return x