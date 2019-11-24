from torch import nn
from torchsummary import summary
import torch
from torch.nn import functional as F
from torchvision.models import resnet50


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ResNet50(nn.Module):

    def __init__(self, num_classes=6, dropout_p=0.1):

        super(ResNet50, self).__init__()

        backbone = resnet50(pretrained=True)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.concat_pool = AdaptiveConcatPool2d()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * 4 * 2, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.concat_pool(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        x = self.activation(x)

        return x


if __name__ == '__main__':

    model = ResNet50()
    summary(model, (3, 256, 256), device='cpu')
