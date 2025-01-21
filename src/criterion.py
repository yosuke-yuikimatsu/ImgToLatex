import torch
import torch.nn as nn

def create_criterion(vocab_size, ignore_index=1):
    """
    Аналог nn.ClassNLLCriterion, но с ignore_index
    = 1 (подразумевая, что <pad>=1).
    """
    criterion = nn.NLLLoss(
        weight=None,
        ignore_index=ignore_index,
        reduction='sum'
    )
    return criterion
