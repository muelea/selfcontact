import torch
import torch.nn as nn

class L2Prior(nn.Module):
    def __init__(self):
        super(L2Prior, self).__init__()

    def forward(self, module_input):
        return torch.sum(module_input.pow(2))