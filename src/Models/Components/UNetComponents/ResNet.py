import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        residual = out
        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = out + residual

        return out
