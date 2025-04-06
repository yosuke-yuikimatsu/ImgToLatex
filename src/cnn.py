import torch.nn as nn
import timm

class CNN(nn.Module):
    def __init__(self, output_channels=1536):
        super().__init__()
        
        self.convnext = timm.create_model('convnext_large.fb_in22k', pretrained=False, num_classes=0)
        
        self.output_channels = output_channels
        
        self.conv_reduction = nn.Conv2d(1536, self.output_channels, kernel_size=1)
        
        # Add batch normalization for more stable training
        self.batch_norm = nn.BatchNorm2d(output_channels)
        
        # Optional activation for better feature representation
        self.activation = nn.SiLU()  # SiLU (Swish) activation often performs better than ReLU

    def forward(self, x):
        # Use memory-efficient operations
        x = self.convnext.forward_features(x)  # (B, 1536, H', W')
        x = self.conv_reduction(x)  # (B, output_channels, H', W')
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # More efficient permute operation
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H', W', output_channels)
        return x
