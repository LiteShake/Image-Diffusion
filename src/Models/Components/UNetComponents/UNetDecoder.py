
import torch
import torch.nn as nn

from .ResNet import ResNet

class UNetDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.rn1 = ResNet(64 +128, 64)
        self.rn2 = ResNet(32 + 64, 32)
        self.rn3 = ResNet(16 + 32, 16)
        self.rn4 = ResNet( 8 + 16,  8)

        self.rn5 = ResNet( 8,  3)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")


    def forward(self, res1, res2, res3, res4, res5, pos):

        out = self.up(res5)
        # print(f"Decoder res 5 | {out.shape}")

        res4 = res4.split( out.shape[2], dim=2 )[0]
        res4 = res4.split( out.shape[3], dim=3 )[0]

        out = torch.cat( [out,res4], dim = 1 )

        ps = pos[:out.shape[2], :out.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * out.shape[1], dim = 0)

        out = out + ps

        out = self.rn1(out)

        out = self.up(out)
        # print(f"Decoder res 4 | {out.shape}")

        res3 = res3.split( out.shape[2], dim=2 )[0]
        res3 = res3.split( out.shape[3], dim=3 )[0]

        out = torch.cat( [out,res3], dim = 1 )

        ps = pos[:out.shape[2], :out.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * out.shape[1], dim = 0)

        out = out + ps

        out = self.rn2(out)

        out = self.up(out)

        res2 = res2.split( out.shape[2], dim=2 )[0]
        res2 = res2.split( out.shape[3], dim=3 )[0]

        # print(f"Decoder res 3 | {out.shape}")
        # # print(f"Out {out.shape} Res {res2.shape}")
        out = torch.cat( [out,res2], dim = 1 )


        ps = pos[:out.shape[2], :out.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * out.shape[1], dim = 0)

        out = out + ps

        out = self.rn3(out)

        out = self.up(out)
        # print(f"Decoder res 2 | {out.shape}")

        res1 = res1.split( out.shape[2], dim=2 )[0]
        res1 = res1.split( out.shape[3], dim=3 )[0]

        out = torch.cat( [out,res1], dim = 1 )

        ps = pos[:out.shape[2], :out.shape[3]]
        ps = ps[None, ::]
        ps = torch.cat([ps] * out.shape[1], dim = 0)

        out = out + ps

        out = self.rn4(out)

        # print(f"Decoder res 1 | {out.shape}")
        out = self.rn5(out)

        return out