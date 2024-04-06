
import torch
import torch.nn as nn

from .ResNet import ResNet

class UNetEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.rn1 = ResNet( 3,   8)
        self.rn2 = ResNet( 8,  16)
        self.rn3 = ResNet(16,  32)
        self.rn4 = ResNet(32,  64)
        self.rn5 = ResNet(64, 128)

    def forward(self, img, pos):


        ps = pos[:img.shape[2], :img.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * img.shape[1], dim = 0)

        # print`(f"Encoder img {img.shape} {ps.shape}")
        img = img + ps

        res1 = self.rn1( img)

        p1 = self.pool(res1)
        # print(f"Encoder pool 1 {p1.shape}")

        ps = pos[:p1.shape[2], :p1.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * p1.shape[1], dim = 0)

        p1 = p1 + ps

        res2 = self.rn2( p1 )

        p2 = self.pool(res2)

        ps = pos[:p2.shape[2], :p2.shape[3]]
        # print(f"Encoder pool 2 {p2.shape}")
        ps = ps[None, ::]
        ps = torch.cat([ps] * p2.shape[1], dim = 0)

        p2 = p2 + ps

        res3 = self.rn3( p2 )

        p3 = self.pool(res3)
        # print(f"Encoder pool 3 {p3.shape}")

        ps = pos[:p3.shape[2], :p3.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * p3.shape[1], dim = 0)

        p3 = p3 + ps

        res4 = self.rn4( p3 )

        p4 = self.pool(res4)

        # print(f"Encoder pool 4 {p4.shape}")

        ps = pos[:p4.shape[2], :p4.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * p4.shape[1], dim = 0)

        res5 = self.rn5( p4 )

        return (res1, res2, res3, res4, res5)