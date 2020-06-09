""" VoVNet as per https://arxiv.org/pdf/1904.09730.pdf (v1) and
https://arxiv.org/pdf/1911.06667.pdf (v2). """

import collections
import torch
from torch import nn

__all__ = ['VoVNet']

# The paper is unclear as to where to downsample, so the downsampling was
# derived from the pretrained model graph as visualized by Netron. V2 simply
# enables ESE and identity connections here, nothing else changes.
CONFIG = {
    # Introduced in V2. Difference is 3 repeats instead of 5 within each block.
    "vovnet19": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 64, 3, 128, True],
        [3, 80, 3, 192, True],

        [3, 96, 5, 192, False],
        # [3, 96, 3, 192, False],
    ],
    "vovnet27_slim": [
        [3, 64, 5, 128, True],
        [3, 80, 5, 256, True],
        [3, 96, 5, 348, True],
        [3, 112, 5, 512, True],
    ],
    "vovnet39": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x2
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x2
        [3, 224, 5, 1024, False],
    ],
    "vovnet57": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],
        [3, 192, 5, 768, True],  # x4
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x3
        [3, 224, 5, 1024, False],
        [3, 224, 5, 1024, False],
    ],
    "vovnet99": [
        [3, 128, 5, 256, True],
        [3, 160, 5, 512, True],  # x3
        [3, 160, 5, 512, False],
        [3, 160, 5, 512, False],
        [3, 192, 5, 768, True],  # x9
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 192, 5, 768, False],
        [3, 224, 5, 1024, True],  # x3
        [3, 224, 5, 1024, False],
        [3, 224, 5, 1024, False],
    ],
}


class _ESE(nn.Module):
    def __init__(self, channels: int) -> None:
        # TODO: Might want to experiment with bias=False. At least for
        # MobileNetV3 it leads to better accuracy on detection.
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = self.conv(y)
        # Hard sigmoid multiplied by input.
        return x * (nn.functional.relu6(y + 3, inplace=True) / 6.0)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _OSA(nn.Module):
    def __init__(
            self,
            in_ch: int,
            inner_ch: int,
            out_ch: int,
            repeats: int = 5,
            kernel_size: int = 3,
            stride: int = 1,
            downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList(
            [
                _ConvBnRelu(
                    in_ch if r == 0 else inner_ch,
                    inner_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                for r in range(repeats)
            ]
        )
        self.exit_conv = _ConvBnRelu(in_ch + repeats * inner_ch, out_ch, kernel_size=1)
        self.ese = _ESE(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all modules, but retain outputs.
        input = x
        if self.downsample:
            x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        features = [x]
        for l in self.layers:
            features.append(l(x))
            x = features[-1]
        x = torch.cat(features, dim=1)
        x = self.exit_conv(x)
        x = self.ese(x)
        # All non-downsampling V2 layers have a residual. They also happen to
        # not change the number of channels.
        if not self.downsample:
            x += input
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out, kernel_size=kernel_size, stride=stride, padding=padding),
                                  nn.BatchNorm2d(out),
                                  nn.ReLU(inplace=True))
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # return F.relu(x, inplace=True)
        return x

class GoogLeNetV4Stem(nn.Module):
    def __init__(self):
        super(GoogLeNetV4Stem, self).__init__()
        self.conv1 = BasicConv2d(3, 48, kernel_size=3, stride=1, padding=0)
        self.conv2 = BasicConv2d(48, 48, kernel_size=3, stride=2, padding=0)
        # self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4_2 = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1)

        # self.conv5_1_1 = BasicConv2d(128, 48, kernel_size=1, stride=1, padding=0)
        self.conv5_1_2 = BasicConv2d(96, 48, kernel_size=(3, 1), stride=1, padding=(1, 0))#7
        self.conv5_1_3 = BasicConv2d(48, 48, kernel_size=(1, 3), stride=1, padding=(0, 1))#7
        self.conv5_1_4 = BasicConv2d(48, 48, kernel_size=(3, 3), stride=1, padding=0)

        self.conv5_2_1 = BasicConv2d(96, 48, kernel_size=1, stride=1, padding=0)
        self.conv5_2_2 = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=0)

        # self.pool6_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv6_2 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # print x.shape
        x = self.conv2(x)
        # print x.shape
        # x = self.conv3(x)
        # print x.shape

        # x1 = self.pool4_1(x)
        # print x1.shape
        x2 = self.conv4_2(x)
        # print x2.shape
        x = torch.cat((x, x2), dim=1)
        # x1 = self.conv5_1_1(x)
        x1 = self.conv5_1_2(x)
        x1 = self.conv5_1_3(x1)
        x1 = self.conv5_1_4(x1)
        # print x1.shape
        x2 = self.conv5_2_1(x)
        x2 = self.conv5_2_2(x2)
        # print x2.shape
        x = torch.cat((x1, x2), dim=1)
        # print(x.shape)
        x1 = self.conv6_2(x)
        # x2 = self.pool6_1(x)
        x = torch.cat((x1, x), dim=1)
        # print(x.shape)

        return x


class VoVNet(nn.Module):
    def __init__(
            self,
            in_ch: int = 3,
            num_classes: int = 100,
            model_type: str = "vovnet19",
            has_classifier: bool = True,
            dropout: float = 0.4,
    ):

        super().__init__()

        # Input stage.
        # self.stem = nn.Sequential(
        #     _ConvBnRelu(in_ch, 64, kernel_size=3, stride=1),
        #     _ConvBnRelu(64, 64, kernel_size=3, stride=2),
        #     _ConvBnRelu(64, 96, kernel_size=3, stride=1),
        # )

        self.stem = GoogLeNetV4Stem()

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = 192
        for idx, block in enumerate(conf):
            kernel_size, inner_ch, repeats, out_ch, downsample = block
            body_layers[f"osa{idx}"] = _OSA(
                in_ch,
                inner_ch,
                out_ch,
                repeats=repeats,
                kernel_size=kernel_size,
                downsample=downsample,
            )
            in_ch = out_ch
        self.body = nn.Sequential(body_layers)
        self.has_classifier = has_classifier

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_ch, num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        # print(y.shape)
        y = self.body(y)
        # print(y.shape)
        if self.has_classifier:
            y = self.classifier(y)
        return y



net = VoVNet(3, 100)
net = net.eval()
with torch.no_grad():
    y = net(torch.rand(2, 3, 32, 32))
    print(list(y.shape))
