import torch
import torch.nn as nn

def create_criterion(vocab_size, ignore_index=1):
    """
    Аналог ClassNLLCriterion.
    ignore_index=1 (например, если <pad>=1).
    """
    crit = nn.NLLLoss(ignore_index=ignore_index, reduction='sum')
    return crit
