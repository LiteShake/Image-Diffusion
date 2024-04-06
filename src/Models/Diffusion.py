
import torch
import torch.nn as nn

from Models.Components.UNet import *

class Diffusion(nn.Module):

    def __init__(self, device, steps = 100) -> None:
        super().__init__()
