# This is an example file - such a custom user model will hold the model architecture
# TODO define your architecture

import torch.nn as nn
from tools import model_tools


class SomeModule(nn.Module):
    def __init__(self, args, c_in=32, other={}):
        super(SomeModule, self).__init__()
        self.other = other
        self.fc = model_tools.fc(args, 3*c_in**2)
        self.c_in = c_in

    def forward(self, x):
        x_flat = x.view((-1, 3*self.c_in**2))
        out = self.fc(x_flat)
        return out

