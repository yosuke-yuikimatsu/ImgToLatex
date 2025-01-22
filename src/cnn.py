import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Аналог createCNNModel из cnn.lua на PyTorch:
      - Вход: (batch_size, 1, H, W)
      - Выход: (batch_size, H', W', 512)
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.sub_mean = -128.0
        self.div_val = 128.0

        # Блок 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Блок 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) # высоту не уменьшаем, уменьшаем ширину

        # Блок 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)) # уменьшаем высоту, ширину не трогаем

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Нормируем: (x - 128)/128
        x = x + self.sub_mean
        x = x / self.div_val

        x = self.relu1(self.conv1(x))  # (batch,64,H,W)
        x = self.pool1(x)             # (batch,64,H/2,W/2)

        x = self.relu2(self.conv2(x))  # (batch,128,H/2,W/2)
        x = self.pool2(x)             # (batch,128,H/4,W/4)

        x = self.relu3(self.bn3(self.conv3(x)))  # (batch,256,H/4,W/4)
        x = self.relu4(self.conv4(x))            # (batch,256,H/4,W/4)
        x = self.pool3(x)                        # (batch,256,H/4, W/8)

        x = self.relu5(self.bn5(self.conv5(x)))  # (batch,512,H/4,W/8)
        x = self.pool4(x)                        # (batch,512,H/8, W/8)

        x = self.relu6(self.bn6(self.conv6(x)))  # (batch,512,H'/W')

        # Перестановка: (batch, 512, H', W') -> (batch, H', W', 512)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
