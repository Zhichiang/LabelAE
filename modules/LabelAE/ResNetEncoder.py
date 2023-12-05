import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101

from config import cfg


class ResNetEncode(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNetEncode, self).__init__()

        model = resnet101(pretrained=pretrained, **kwargs)

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.out_layer = nn.Conv2d(2048, cfg.MODEL.latent_len, kernel_size=1, stride=1)

        del model

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_layer(x)

        return x


if __name__ == "__main__":
    resnet18_encode = ResNetEncode(num_classes=10, num_latent=10)
    in_image = torch.randn(2, 10, 32, 32)
    out_image = resnet18_encode(in_image)
    pass
