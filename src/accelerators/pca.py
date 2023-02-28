import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA


class PCAWrapper(nn.Module):
    def __init__(self, layer: nn.Linear, k: int):
        super().__init__()
        self.layer = layer
        self.k = k
        self.pca = PCA(n_components=k)

        # Apply PCA to the layer's weight matrix
        weight = self.layer.weight.detach().numpy()

        # Note that the shape of PyTorch nn.Linear is (out_features, in_features)
        do_transpose = weight.shape[0] > weight.shape[1]

        if do_transpose:
            weight = weight.T

        reduced_weights = self.pca.fit_transform(weight)

        if do_transpose:
            reduced_weights = reduced_weights.T

        # Replace the layer's weights with the reduced weights
        self.layer.weight = torch.nn.Parameter(torch.from_numpy(reduced_weights))

        bias = self.layer.bias.detach().numpy()
        reduce_bias = len(bias) == max(weight.shape)
        if do_transpose:
            reduced_bias = bias[:k]
            self.layer.bias = torch.nn.Parameter(torch.from_numpy(reduced_bias))

        # print(self.layer.weight.shape)

    def forward(self, input):
        output = self.layer(input)
        return output
