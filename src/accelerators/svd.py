import torch
from torch import nn


class SVDWrapper(nn.Module):
    def __init__(self, layer, k):
        super().__init__()
        self.layer = layer
        self.k = k

        # Apply SVD to the layer's weight matrix
        u, s, v = torch.svd(layer.weight)
        self.u = nn.Parameter(u[:, :k])
        self.s = nn.Parameter(s[:k])
        self.v = nn.Parameter(v[:k, :])

        # Replace the layer's weights with the reduced weights
        weight = self.u @ torch.diag(self.s) @ self.v
        self.layer.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        output = self.layer(input)
        return output
