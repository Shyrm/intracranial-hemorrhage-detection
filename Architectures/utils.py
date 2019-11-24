from torch import nn
import torch
from torch.nn import functional as F
from fastai.vision import requires_grad
from torchsummary import summary


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, output_stride, batchnorm=nn.BatchNorm2d, dropout=None):

        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=batchnorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=batchnorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=batchnorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=batchnorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             batchnorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = batchnorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x) if self.dropout is not None else x

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SPP(nn.Module):

    def __init__(self, input_channels):

        super(SPP, self).__init__()

        self.pooling = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 4, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 4, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 4, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=6, stride=1, padding=0),
                nn.Conv2d(in_channels=input_channels, out_channels=input_channels // 4, kernel_size=1, stride=1)
            )
        ])

    def forward(self, x):

        input_size = (x.size(2), x.size(3))
        res = [x]
        for plg in self.pooling:
            res.append(F.interpolate(plg(x), input_size, mode='bilinear', align_corners=True))

        return torch.cat(res, 1)


if __name__ == '__main__':

    spp_layer = SPP(input_channels=4)
    summary(spp_layer, (4, 32, 32), device='cpu')




