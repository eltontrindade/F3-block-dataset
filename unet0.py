import torch
import torch.nn as nn
import torch.nn.functional as F

import acsconv

from acsconv.converters.base_converter import BaseConverter
from acsconv.operators import Conv2_5d
#from ..utils import _pair_same

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels,groups=2, kernel_size=[1,7,7], padding=[0,3,3]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels,groups=2, kernel_size=[1,7,7], padding=[0,3,3]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels,groups=2, kernel_size=[1,7,7], padding=[0,3,3]),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, out_channels,groups=2, kernel_size=[1,7,7], padding=[0,3,3]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = _EncoderBlock(2, 64*2)
        self.enc2 = _EncoderBlock(64*2, 128*2)
        self.dec1 = _DecoderBlock(192*2, 64*2, 32*2)
        self.interpolate = nn.Upsample(scale_factor=2, mode='trilinear')
        self.final = nn.Conv3d(32*2, num_classes,groups=2, kernel_size=[1,1,1],padding=[0, 0, 0])

    def forward(self, x):
        x = self.enc1(x)
        x1 = x.clone()
        x = self.enc2(x)
        x = self.dec1(torch.cat([x1, self.interpolate(x)], 1))
        x = self.final(self.interpolate(x))
        
        return x