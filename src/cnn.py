import torch
import torch.nn as nn
import timm

class CNN(nn.Module):
    def __init__(self, output_channels=1536):
        super().__init__()
        self.convnext = timm.create_model('convnext_large.fb_in22k', pretrained=True, num_classes=0)
        self.conv_reduction = nn.Conv2d(1536, output_channels, kernel_size=1)  # ConvNeXt Large выдает 1536 каналов
        self.output_channels = output_channels

    def forward(self, x):
        x = self.convnext.forward_features(x)  # (B, 1536, H', W')
        x = self.conv_reduction(x)  # (B, output_channels, H', W')
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', output_channels)
        return x
