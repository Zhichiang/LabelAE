import torch.nn as nn
from utils.selecting_layer import SelectionLayer
from torchvision.models.resnet import resnet18


class SelectReLU(nn.Module):
    def __init__(self, inplace=True, use_relu=False):
        super(SelectReLU, self).__init__()
        self.use_relu = use_relu
        self.relu = nn.ReLU(inplace=inplace)
        self.selector = SelectionLayer(keep_layers=1, keep_percent=0.1, fix_layers=0, overlap=True)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)
        x = self.selector(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_c, out_c, kernel_size=3, stride=stride)
        self.in1 = nn.BatchNorm2d(out_c, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(out_c, out_c, kernel_size=3, stride=1)
        self.in2 = nn.BatchNorm2d(out_c, affine=True)
        self.srelu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.srelu(out)
        return out


class LabelEncoder(nn.Module):
    def __init__(self, num_classes, num_latent, select_latent=False, *args, **kwargs):
        super(LabelEncoder, self).__init__()
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.select_latent = select_latent

        # encoding layers
        self.in_layer = nn.Sequential(
            ConvLayer(num_classes, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )

        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 128, 2)
        self.layer4 = self.make_layer(128, 128, 2)
        self.layer5 = self.make_layer(128, 128, 2)

        self.out_layer = ConvLayer(128, num_latent, kernel_size=1, stride=1)

    @staticmethod
    def make_layer(in_c, out_c, stride):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = nn.Sequential(
                ConvLayer(in_c, out_c, 1, stride)
            )
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride, downsample),
            ResidualBlock(out_c, out_c),
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.out_layer(x)

        return x


class LabelDecoder(nn.Module):
    def __init__(self, num_classes, num_latent, select_latent=False, *args, **kwargs):
        super(LabelDecoder, self).__init__()
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.select_latent = select_latent
        self.select_relu = SelectReLU()

        self.in_layer = nn.Sequential(
            ConvLayer(num_latent, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer5 = self.make_layer(128, 128, stride=2)
        self.layer4 = self.make_layer(128, 128, stride=2)
        self.layer3 = self.make_layer(128, 128, stride=2)
        self.layer2 = self.make_layer(128, 64, stride=2)

        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvLayer(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
        )

        self.out_layer = ConvLayer(64, num_classes, kernel_size=1, stride=1)

    def make_layer(self, in_c, out_c, stride):
        downsample = None
        layers = []
        if in_c != out_c:
            downsample = nn.Sequential(
                ConvLayer(in_c, out_c, 1, 1)
            )
        if stride != 1:
            layers.append(nn.Upsample(scale_factor=stride, mode='bilinear'))
        layers.append(ResidualBlock(in_c, out_c, downsample=downsample))
        layers.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.select_latent:
            x = self.select_relu(x)

        x = self.in_layer(x)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.out_layer(x)

        return x


if __name__ == "__main__":
    import torch
    decoder = LabelDecoder(10, 20)
    encoder = LabelEncoder(10, 20)

    in_image = torch.randn(2, 10, 64, 64)
    out_code = encoder(in_image)
    rec_in_image = decoder(out_code)
    pass
