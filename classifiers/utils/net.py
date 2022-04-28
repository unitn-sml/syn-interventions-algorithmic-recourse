import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dimensions, layers=4):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(dimensions//i, dimensions//(i+1))
            for i in range(1, layers)
        ])
        self.output = nn.Linear(dimensions//layers, 1)

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
        x = torch.sigmoid(self.output(x))
        return x