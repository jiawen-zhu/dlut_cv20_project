import collections
import torch
import math
from torch import nn

__all__ = ['dutcvcnet']

CONFIG = {
    "dutcvcnet": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        # [3, 64, 3, 128, True],
        # [3, 80, 3, 192, True],
        # [3, 96, 3, 192, False],

        [3, 16, 1, 128, True],
        [3, 16, 1, 128, False],
        [3, 24, 1, 256, True],
        [3, 32, 1, 256, False],
        [3, 64, 3, 512, True],  #63.4
    ],
}


class _ESE(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean([2, 3], keepdim=True)
        y = self.conv(y)
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
        if not self.downsample:
            x += input
        return x



class dutcvcnet(nn.Module):
    def __init__(
            self,
            in_ch: int = 3,
            num_classes: int = 100,
            model_type: str = "dutcvcnet",
            has_classifier: bool = True,
            dropout: float = 0.4,
    ):

        super().__init__()

        self.stem = nn.Sequential(
            _ConvBnRelu(in_ch, 16, kernel_size=3, stride=1),
            _ConvBnRelu(16, 16, kernel_size=3, stride=1),
            _ConvBnRelu(16, 24, kernel_size=3, stride=1),
            _ConvBnRelu(24, 36, kernel_size=3, stride=1), #63.4
        )

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = 36
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

        # self._initialize_weights()
        self._initialize_weights2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        # print(y.shape)
        y = self.body(y)
        # print(y.shape)
        if self.has_classifier:
            y = self.classifier(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _initialize_weights2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



# net = dutcvcnet(3, 100)
# net = net.eval()
# with torch.no_grad():
#     y = net(torch.rand(2, 3, 32, 32))
#     print(list(y.shape))
