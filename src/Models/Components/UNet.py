
import torch
import torch.nn as nn

from .UNetComponents.UNetEncoder import *
from .UNetComponents.UNetDecoder import *

class UNet(nn.Module):

    def __init__(self, device) -> None:
        super().__init__()

        self.PosEmb = nn.Embedding(30, 36_608)
        self.PosEmb.load_state_dict(torch.load("./Models/Saves/PositionEmbedding.pt"))
        self.PosEmb.to(device)

        self.enc = UNetEncoder()
        self.dec = UNetDecoder()

    def forward(self, img, idx) :

        pos = self.PosEmb(torch.tensor(idx).to(torch.device("cuda")))
        pos = pos.view(208, 176)
        # print(pos.shape)

        r1, r2, r3, r4, r5 = self.enc(img, pos)
        out = self.dec(r1, r2, r3, r4, r5, pos)

        return out
