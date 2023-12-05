import torch.nn as nn

from config import cfg


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        n_classes = cfg.MODEL.latent_len
        norm2d = nn.InstanceNorm2d
        self.feature = nn.Sequential(
            ConvLayer(n_classes, 512, 5, stride=2),
            norm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            ConvLayer(512, 512, 5, stride=2),
            norm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            ConvLayer(512, 512, 5, stride=2),
            norm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            ConvLayer(512, 1, 1, stride=1),
        )

    def forward(self, x):
        x = self.feature(x)
        return x

