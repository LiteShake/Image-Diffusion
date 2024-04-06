import torch
import torch.nn as nn

class UpScale(nn.Module):

    def __init__(self, inplanes, planes) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(inplanes, planes, 3)

    def forward(self, x):

        return self.up(x)
