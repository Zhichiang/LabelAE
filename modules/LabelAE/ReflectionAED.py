import torch.nn as nn
from utils.selecting_layer import SelectionLayer
from torchvision.models.resnet import resnet18


class SelectReLU(nn.Module):
    def __init__(self, inplace=True):
        super(SelectReLU, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.selector = SelectionLayer(keep_layers=1, keep_percent=0.2, fix_layers=1, overlap=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.selector(x)
        return x


class ConvLayer(nn.Module):
    reflection_padding = False

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        if self.reflection_padding:
            self.reflection_pad = nn.ReflectionPad2d(padding)
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # , padding)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        if self.reflection_padding:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm, select=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = norm(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = norm(channels, affine=True)

        if select:
            self.srelu = SelectReLU()
        else:
            self.srelu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.srelu(out)
        return out

    # Image Transform Network


class ReflectionEncoder(nn.Module):
    Norm = nn.BatchNorm2d
    # Norm = nn.InstanceNorm2d

    def __init__(self, num_classes, num_latent, select_latent=False, select_layer=False):
        super(ReflectionEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.select_latent = select_latent

        if select_layer:
            self.ReLU = SelectReLU
        else:
            self.ReLU = nn.ReLU

        self.relu = SelectReLU() if select_latent else nn.ReLU()

        # encoding layers
        self.layer1 = nn.Sequential(
            ConvLayer(num_classes, 64, kernel_size=1, stride=1),
            self.Norm(64, affine=True),
            nn.ReLU(),
            ConvLayer(64, 64, kernel_size=7, stride=4),
            self.Norm(64, affine=True),
            self.ReLU(),
        )

        self.layer2 = nn.Sequential(
            ConvLayer(64, 128, kernel_size=1, stride=1),
            self.Norm(128, affine=True),
            nn.ReLU(),
            ConvLayer(128, 128, kernel_size=7, stride=4),
            self.Norm(128, affine=True),
            self.ReLU(),
        )

        self.layer3 = nn.Sequential(
            ConvLayer(128, 128, kernel_size=1, stride=1),
            self.Norm(128, affine=True),
            nn.ReLU(),
            ConvLayer(128, 128, kernel_size=3, stride=2),
            self.Norm(128, affine=True),
            self.ReLU(),
        )

        # self.layer4 = nn.Sequential(
        #     ConvLayer(128, 128, kernel_size=3, stride=2),
        #     self.Norm(128, affine=True),
        #     self.ReLU(),
        # )
        #
        # self.layer5 = nn.Sequential(
        #     ConvLayer(128, 128, kernel_size=3, stride=2),
        #     self.Norm(128, affine=True),
        #     self.ReLU(),
        # )

        # self.layer6 = nn.Sequential(
        #     ConvLayer(128, 128, kernel_size=3, stride=2),
        #     self.Norm(128, affine=True),
        #     nn.ReLU(),
        #     ConvLayer(128, 128, kernel_size=1, stride=1),
        #     self.Norm(128, affine=True),
        #     nn.ReLU(),
        # )

        self.out_layer = ConvLayer(128, num_latent, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        x = self.out_layer(x)

        if self.select_latent:
            x = self.relu(x)

        return x


class ReflectionDecoder(nn.Module):
    Norm = nn.BatchNorm2d
    # Norm = nn.InstanceNorm2d

    def __init__(self, num_classes, num_latent, select_latent=False, select_layer=False):
        super(ReflectionDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.select_latent = select_latent

        if select_layer:
            self.ReLU = SelectReLU
        else:
            self.ReLU = nn.ReLU

        self.in_layer = nn.Sequential(
            ConvLayer(num_latent, 128, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        # self.layer6 = nn.Sequential(
        #     ConvLayer(128, 128, kernel_size=1, stride=1),
        #     self.Norm(128, affine=True),
        #     nn.ReLU(),
        #     UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2),
        #     self.Norm(128, affine=True),
        #     self.ReLU(),
        # )

        # self.layer5 = nn.Sequential(
        #     UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2),
        #     self.Norm(128, affine=True),
        #     self.ReLU(),
        # )
        #
        # self.layer4 = nn.Sequential(
        #     UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2),
        #     self.Norm(128, affine=True),
        #     self.ReLU(),
        # )

        self.layer3 = nn.Sequential(
            UpsampleConvLayer(128, 128, kernel_size=3, stride=1, upsample=2),
            self.Norm(128, affine=True),
            nn.ReLU(),
            ConvLayer(128, 128, kernel_size=1, stride=1),
            self.Norm(128, affine=True),
            self.ReLU(),
        )

        self.layer2 = nn.Sequential(
            UpsampleConvLayer(128, 128, kernel_size=7, stride=1, upsample=4),
            self.Norm(128, affine=True),
            nn.ReLU(),
            ConvLayer(128, 64, kernel_size=1, stride=1),
            self.Norm(64, affine=True),
            self.ReLU(),
        )

        self.layer1 = nn.Sequential(
            UpsampleConvLayer(64, 64, kernel_size=7, stride=1, upsample=4),
            self.Norm(64, affine=True),
            nn.ReLU(),
            ConvLayer(64, 64, kernel_size=1, stride=1),
            self.Norm(64, affine=True),
            self.ReLU(),
        )

        self.out_layer = ConvLayer(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.in_layer(x)
        # x = self.layer6(x)
        # x = self.layer5(x)
        # x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.out_layer(x)

        return x


if __name__ == "__main__":
    import torch
    decoder = ReflectionDecoder(10, 20)
    encoder = ReflectionEncoder(10, 20)

    in_image = torch.randn(2, 10, 64, 64)
    out_code = encoder(in_image)
    rec_in_image = decoder(out_code)
    pass
