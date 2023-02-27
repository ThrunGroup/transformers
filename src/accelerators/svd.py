import torch
from torch import nn


class SVDWrapper(nn.Module):
    def __init__(self, layer, k):
        super().__init__()
        self.layer = layer
        self.k = k

        # Apply SVD to the layer's weight matrix
        U, S, V = torch.svd(layer.weight)
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:, :k].t()

        # Replace the layer's weights with the reduced weights
        weight = U_k @ torch.diag(S_k) @ V_k
        self.layer.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        output = self.layer(input)
        return output
