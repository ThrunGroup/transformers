from torch import nn
import torch


class QuantizationWrapper(nn.Module):
    def __init__(self): #, layer, k, checkpoint_path, is_conv1d: bool = True):
        super().__init__()
        # self.layer = layer
        # matrix_max = layer.weight.max()
        # matrix_min = layer.weight.min()
        # self.scaling_factor = 128 / (matrix_max - matrix_min)
        # self.weight = layer.weight.data * self.scaling_factor
        # self.weight = self.weight.int().float()
        # self.bias = self.layer.bias.data
        # print(self.weight[:3, :3] / self.scaling_factor)
        # print(self.layer.weight.size(), self.weight.size(), self.layer.bias.size(), self.bias.size())

    def forward(self, x):

        # size_out = x.size()[:-1] + (self.layer.nf,)
        # x = x * self.scaling_factor
        # x = torch.addmm(self.layer.bias, x.view(-1, x.size(-1)), self.weight)
        # x = x.view(size_out).float()
        return x
