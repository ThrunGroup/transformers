from torch import nn


class VanillaWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input):
        output = self.layer(input)
        return output
