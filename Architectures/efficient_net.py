from torch import nn
from torchsummary import summary
import torch
from torch.nn import functional as F
from fastai.vision import requires_grad
from Architectures.utils import AdaptiveConcatPool2d, ASPP, SPP
from efficientnet_pytorch import EfficientNet


class EfficientNetB4(nn.Module):

    def __init__(self, num_classes=6, dropout_p=0.1):

        super(EfficientNetB4, self).__init__()

        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.aspp = ASPP(inplanes=1792, output_stride=16, dropout=None)
        # self.spp = SPP(input_channels=1792)

        self.concat_pool = AdaptiveConcatPool2d()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.fc1 = nn.Linear(1792 * 4, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.backbone.extract_features(x)
        x = self.aspp(x)

        x = self.concat_pool(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x


if __name__ == '__main__':

    model = EfficientNetB4()
    summary(model, (3, 512, 512), device='cpu')
