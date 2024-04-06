
import os
import torch

from Data.LoadCeleb import *
from Models.Components.UNet import *
from Trainer import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = GetData()
unet = UNet(device)
print(data[0].shape)
trainer = Trainer(data, unet)

trainer.train(device)
