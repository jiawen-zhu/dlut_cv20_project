import collections
import torch
import math
from torch import nn

__all__ = ['dutcvcnet']


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


class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                # nn.Conv2d(features, features, kernel_size=3, dilation=i+1, stride=stride, padding=i+1, groups=G),
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),

            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

CONFIG = {
    "dutcvcnet": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 16, 5, 128, True],
        [3, 24, 8, 384, True],
        [1, 64, 2, 512, True]  # 66.2

        # [3, 16, 8, 128, True],
        # [3, 24, 8, 384, True],
        # [1, 64, 2, 512, True] #66.2

        # [3, 32, 5, 128, True],
        # [3, 32, 5, 256, True],
        # [1, 48, 3, 384, True]  # 66.2

        # [3, 16, 5, 128, True],
        # [3, 24, 8, 384, True],
        # [1, 64, 5, 512, True]  # 66.2
    ],
}

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
            # _ConvBnRelu(in_ch, 24, kernel_size=3, stride=1),
            # SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            # nn.ReLU(),
            # SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            # nn.ReLU(),
            # SKUnit(24, 36, 32, 2, 6, 2, stride=1),
            # nn.ReLU(),
            # SKUnit(36, 64, 32, 2, 8, 2, stride=1),
            # nn.ReLU()     #66.2 head1

            _ConvBnRelu(in_ch, 16, kernel_size=3, stride=1),
            SKUnit(16, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 36, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(36, 64, 32, 2, 8, 2, stride=1),
            nn.ReLU()  # 66.2 head2
        )

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = 64
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
