import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils.extmath import randomized_svd
import os


class SVDWrapper(nn.Module):
    def __init__(self, layer, k, checkpoint_path, is_conv1d: bool = True):
        super().__init__()
        self.layer = layer
        self.k = k
        self.accelerator_checkpoint_path = os.path.join(checkpoint_path, "svd", f"k_{k}")
        self.is_conv1d = is_conv1d

        # Apply SVD to the layer's weight matrix
        if os.path.exists(self.accelerator_checkpoint_path):
            accelerator_checkpoint = torch.load(self.accelerator_checkpoint_path)
            self.U = accelerator_checkpoint["U"]
            self.S = accelerator_checkpoint["S"]
            self.V_T = accelerator_checkpoint["V_T"]
        else:
            self.U, self.S, self.V_T = torch.linalg.svd(layer.weight, full_matrices=False)
            self.U = self.U[:, :self.k]
            self.S = self.S[:self.k]
            self.V_T = self.V_T[:self.k, :]
            os.makedirs(os.path.dirname(self.accelerator_checkpoint_path), exist_ok=True)
            torch.save(
                {"U": self.U, "S": self.S, "V_T": self.V_T}, self.accelerator_checkpoint_path)

    def forward(self, x):
        if self.is_conv1d:
            size_out = x.size()[:-1] + (self.V_T.size()[-1],)
            xU = torch.mm(x.view(-1, x.size(-1)), self.U)
            xUSV_T = torch.addmm(self.layer.bias, xU, self.S[:, None] * self.V_T)
            xUSV_T = xUSV_T.view(*size_out)
            return xUSV_T
        else:
            xV = F.linear(x, self.V_T)
            xVSU_T = F.linear(xV, self.S[None, :] * self.U)
            return xVSU_T
