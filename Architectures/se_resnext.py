from pretrainedmodels import se_resnext50_32x4d
from torch import nn
from torchsummary import summary
import torch
from torch.nn import functional as F
from fastai.vision import requires_grad
from Architectures.utils import AdaptiveConcatPool2d, ASPP, SPP


class SEResNeXt50(nn.Module):

    def __init__(self, num_classes=6, dropout_p=0.1):

        super(SEResNeXt50, self).__init__()

        backbone = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')

        self.layer0 = backbone.layer0
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(inplanes=2048, output_stride=16, dropout=None)
        # self.spp = SPP(input_channels=256)
        # self.spp = SPP(input_channels=2048)

        self.concat_pool = AdaptiveConcatPool2d()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.fc1 = nn.Linear(512, 256)
        # self.fc1 = nn.Linear(2048 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        # x = self.spp(x)

        x = self.concat_pool(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x


if __name__ == '__main__':

    model = SEResNeXt50()
    summary(model, (3, 512, 512), device='cpu')

    # 31, 806, 198
    # 41, 178, 870
