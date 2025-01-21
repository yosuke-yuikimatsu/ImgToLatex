import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputProjector(nn.Module):
    """
    Аналог createOutputUnit из output_projector.lua:
    просто линейный слой + LogSoftMax.
    """
    def __init__(self, input_size, output_size):
        super(OutputProjector, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.log_softmax(x)
        return x
