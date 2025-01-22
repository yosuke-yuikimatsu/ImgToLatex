import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputProjector(nn.Module):
    """
    Аналог createOutputUnit:
      - Linear(input_size, output_size) + LogSoftMax
    """
    def __init__(self, input_size, output_size):
        super(OutputProjector, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)         # (batch, output_size)
        x = self.log_softmax(x)    # (batch, output_size)
        return x
