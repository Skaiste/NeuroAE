from __future__ import annotations

import torch
from torch import nn


class ShallowMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 64]
        layers = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, int(num_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

