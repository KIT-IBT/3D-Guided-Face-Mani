import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator G"""
    def __init__(self, conv_dim=64, in_num=6, repeat_num=6, out_num=3):
        super().__init__()

        self.out_num = out_num
        self.in_num = in_num
        layers = []
        self.first_conv = nn.Conv2d(in_num, conv_dim,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3,
                                    bias=False)
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(curr_dim, out_num,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3,
                                    bias=False)
        self.activate = nn.Tanh()


    def forward(self, x, p=None):
        if p is not None:
            p, mask = p
            x = torch.cat((x, p), dim=1)
        x = self.first_conv(x)
        y = self.main(x)
        y = self.final_conv(y)
        y = self.activate(y)
        if self.out_num == 8:
            return y[:, :3], y[:, 3:]
        else:
            return y

class Discriminator(nn.Module):
    """Discriminator D"""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        h = self.main(x)
        out = self.conv1(h)
        return out

class RoiDiscriminator(nn.Module):
    """Discriminator D_ROI"""
    def __init__(self, image_size=64, conv_dim=64, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            stride = 1
            kernel_size = 3
            if i % 2 == 0:
                stride = 2
                kernel_size = 4
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        h = self.main(x)
        out = self.conv1(h)
        return out
