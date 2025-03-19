import torch
import torch.nn as nn
import timm

class CNN(nn.Module):
    def __init__(self, output_channels=1536):  # Оставляем 1536, но используем convnext_base
        super().__init__()
        self.convnext = timm.create_model('convnext_base.fb_in22k', pretrained=False, num_classes=0)
        in_channels = 1024  # ConvNeXt Base выдаёт 1024 канала
        self.conv_reduction = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, x):
        x = self.convnext.forward_features(x)  # (B, 1024, H', W')
        x = self.conv_reduction(x)  # (B, output_channels, H', W')
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', output_channels)
        return x