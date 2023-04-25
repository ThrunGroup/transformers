import torch
from torch import nn
from torch.nn import functional as F
import os


class PruningWrapper(nn.Module):
    def __init__(self, layer, k: int, is_conv1d: bool=False, pruning_type: str = "random"):
        super().__init__()
        self.layer = layer
        # num_input_features = self.layer.weight.size()[-1]
        # self.k = int(self.layer.weight.size()[-1] * 0.40)
        # self.idx = torch.range(0, 1, step=2/self.layer.weight.size()[-1])[:-1] * num_input_features
        # self.idx = self.idx.to(torch.int32)
        # self.pruning_type = pruning_type
        # if self.pruning_type == "random":
        #     self.layer.weight = nn.Parameter(layer.weight[..., self.idx])
    def forward(self, x):
        idcs = torch.randint(0, x.size()[-1], size=(200, ))
        output = F.linear(x[..., idcs], self.layer.weight[..., idcs], self.layer.bias)
        # output = self.layer(x)

        sampled_activated_idcs = output > -0.01
        true_output = self.layer(x)
        true_activated_idcs = true_output > 0
        # print(torch.histogram(true_output[131].cpu(), 10))
        print("Accuracy: ", torch.sum(sampled_activated_idcs[true_activated_idcs])/torch.sum(true_activated_idcs))
        print("Used_dims proportion: ", torch.sum(sampled_activated_idcs)/torch.prod(torch.tensor(sampled_activated_idcs.size())))
        output = torch.zeros_like(output)
        for idx in range(x.size()[0]):
            sample_idcs = sampled_activated_idcs[idx, :]
            output[idx, sample_idcs] += F.linear(x[idx, :], self.layer.weight[sample_idcs, :]) + self.layer.bias[sample_idcs]
        return output