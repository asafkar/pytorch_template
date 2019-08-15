# This is an example file - such a custom user model will hold the model architecture
# TODO define your architecture

import torch.nn as nn
from tools import model_tools


class SomeModule(nn.Module):
    def __init__(self, batch_norm=False, c_in=3, other={}):
        super(SomeModule, self).__init__()
        self.other = other
        self.conv = model_tools.output_conv(batch_norm, c_in, 64)

    def forward(self, x):
        out = self.conv(x)
        return out

