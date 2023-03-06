import torch
from torch import nn
from sklearn.decomposition import TruncatedSVD


class SVDWrapper(nn.Module):
    def __init__(self, layer, k):
        super().__init__()
        self.layer = layer
        self.k = k

        # Apply SVD to the layer's weight matrix
        # svd = TruncatedSVD(n_components=k)
        # svd.fit(self.layer.weight)

        # U, S, V = svd.components_
        # self.U_k = U[:, :k]
        # self.S_k = S[:k]
        # self.V_k = V[:, :k].t()

        U, S, V = torch.svd(layer.weight)
        self.U_k = U[:, :k]
        self.S_k = S[:k]
        self.V_k = V[:, :k].t()

        # Replace the layer's weights with the reduced weights
        # weight = U_k @ torch.diag(S_k) @ V_k
        # self.layer.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        print("input: ", input.shape)  # 4, 1, 1024
        print("u: ", self.U_k.shape)  # 4096, 4
        print("s: ", self.S_k.shape)  # 4, 4
        print("v: ", self.V_k.shape)  # 4, 1024
        output = self.U_k @ (torch.diag(self.S_k) @ (self.V_k @ input.permute(0, 2, 1)))
        # output = self.layer(input)  # torch.Size([4, 1, 4096])

        """
        4096, 1, 4
        
        (4, s)
        """

        """
        ((4, 1, 1024) (1024 @ 4) (4, 4) @ (4, 4096))))
        """
        return output
