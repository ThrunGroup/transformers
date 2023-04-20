import torch

from torch import nn
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn import QuantLinear


class QuantLinearWrapper(nn.Module):
    def __init__(self, layer: torch.nn.Linear, use_cuda: bool = False):
        super().__init__()
        self.layer = layer
        self.quant_linear = QuantLinear(layer.in_features, layer.out_features)
        self.quant_linear.weight = self.layer.weight
        self.quant_linear.bias = self.layer.bias
        if use_cuda:
            print("ang")
            self.quant_linear = self.quant_linear.cuda()

    def forward(self, x):
        return self.quant_linear(x.cuda())
