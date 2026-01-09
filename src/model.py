import torch
import torch.nn as nn


class FCNModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.dropout = cfg['train']['dropout_rate']
        self.fcn = nn.Linear(10, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fcn(x)
        x = self.activation(x)
        return x
