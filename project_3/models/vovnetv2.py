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
    "vovnet19_0": [
        [3, 16, 5, 128, True],
        [3, 24, 8, 384, True],
        [1, 64, 2, 512, True] #66.2

        # [3, 16, 8, 128, True],
        # [3, 24, 8, 384, True],
        # [1, 64, 2, 512, True] #66.2
    ],
    "vovnet19_1": [
        [3, 32, 5, 128, True],
        [3, 32, 5, 256, True],
        [1, 48, 3, 384, True]   #66.2
    ],
    "vovnet19_2": [
        [3, 16, 5, 128, True],
        [3, 24, 8, 384, True],
        [1, 64, 5, 512, True]   #66.2
    ],
    "vovnet19_3": [
    ],
    "vovnet19_4": [
    ],
    "vovnet19_5": [
    ],
    "vovnet19_6": [

    ],
    "vovnet19_7": [

    ],
    "vovnet19_8": [

    ],


    "vovnet19": [
        # kernel size, inner channels, layer repeats, output channels, downsample
        [3, 16, 1, 128, True],
        [3, 16, 1, 128, False],
        [3, 24, 1, 256, True],
        [3, 32, 1, 256, False],
        [3, 64, 3, 512, True],    #62.1 #63.4
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

    def forward(self, x):
        x = self.conv(x)
        return x

class GoogLeNetV4Stem(nn.Module):
    def __init__(self):
        super(GoogLeNetV4Stem, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = BasicConv2d(64, 48, kernel_size=3, stride=2, padding=0)

        self.conv4_2 = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=1)



        self.conv5_1_2 = BasicConv2d(96, 48, kernel_size=(3, 1), stride=1, padding=(1, 0))#7
        self.conv5_1_3 = BasicConv2d(48, 48, kernel_size=(1, 3), stride=1, padding=(0, 1))#7
        self.conv5_1_4 = BasicConv2d(48, 48, kernel_size=(3, 3), stride=1, padding=0)

        self.conv5_2_1 = BasicConv2d(96, 48, kernel_size=1, stride=1, padding=0)
        self.conv5_2_2 = BasicConv2d(48, 48, kernel_size=3, stride=1, padding=0)

        self.conv6_2 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x2 = self.conv4_2(x)
        x = torch.cat((x, x2), dim=1)
        x1 = self.conv5_1_2(x)
        x1 = self.conv5_1_3(x1)
        x1 = self.conv5_1_4(x1)
        x2 = self.conv5_2_1(x)
        x2 = self.conv5_2_2(x2)
        x = torch.cat((x1, x2), dim=1)
        x1 = self.conv6_2(x)
        x = torch.cat((x1, x), dim=1)

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

class VoVNet(nn.Module):
    def __init__(
            self,
            in_ch: int = 3,
            num_classes: int = 100,
            model_type: str = "vovnet19",
            has_classifier: bool = True,
            dropout: float = 0.4,
            head: int = 0,
    ):

        super().__init__()

        # # Input stage.
        # self.stem = nn.Sequential(
        #     _ConvBnRelu(in_ch, 64, kernel_size=3, stride=1),
        #     _ConvBnRelu(64, 64, kernel_size=3, stride=2),
        #     _ConvBnRelu(64, 96, kernel_size=3, stride=1),
        # )

        self.head = head

        self.connect = 64
        self.stem0 = nn.Sequential(
            _ConvBnRelu(in_ch, 24, kernel_size=3, stride=1),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 36, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(36, 64, 32, 2, 8, 2, stride=1),
            nn.ReLU()     #66.2
        )

        self.stem1 = nn.Sequential(
            _ConvBnRelu(in_ch, 16, kernel_size=3, stride=1),
            SKUnit(16, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 36, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(36, 64, 32, 2, 8, 2, stride=1),
            nn.ReLU()   #66.2
        )
        self.stem2 = nn.Sequential(
            _ConvBnRelu(in_ch, 36, kernel_size=3, stride=1),
            SKUnit(36, 64, 32, 2, 16, 2, stride=1),
            SKUnit(64, 64, 32, 2, 16, 2, stride=1),
            nn.ReLU(),  #65.2
        )
        self.stem3 = nn.Sequential(
            _ConvBnRelu(in_ch, 24, kernel_size=3, stride=1),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 24, 32, 2, 6, 2, stride=1),
            nn.ReLU(),
            SKUnit(24, 36, 32, 2, 9, 2, stride=1),
            nn.ReLU(),
            SKUnit(36, 64, 32, 2, 16, 2, stride=1),
            nn.ReLU()   #65.8
        )
        self.stem4 = nn.Sequential(

        )
        # self.connect = 64
        self.stem5 = nn.Sequential(

        )

        # self.connect = 64
        self.stem6 = nn.Sequential(

        )
        self.stem7 = nn.Sequential(

        )
        self.stem8 = nn.Sequential(

        )


        self.stem = nn.Sequential(

        )

        # self.stem = GoogLeNetV4Stem()

        body_layers = collections.OrderedDict()
        conf = CONFIG[model_type]
        in_ch = self.connect #36
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

        self._initialize_weights2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = self.stem(x)

        if self.head == 0:
            y = self.stem0(x)
        elif self.head == 1:
            y = self.stem1(x)
        elif self.head == 2:
            y = self.stem2(x)
        elif self.head == 3:
            y = self.stem3(x)
        elif self.head == 4:
            y = self.stem4(x)
        elif self.head == 5:
            y = self.stem5(x)
        elif self.head == 6:
            y = self.stem6(x)
        elif self.head == 7:
            y = self.stem7(x)
        elif self.head == 8:
            y = self.stem8(x)

        # print(y.shape)
        y = self.body(y)
        # print(y.shape)
        if self.has_classifier:
            y = self.classifier(y)
        return y

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



# net = VoVNet(3, 100, head=1, model_type='vovnet19_0')
# net = net.eval()
# with torch.no_grad():
#     y = net(torch.rand(2, 3, 32, 32))
#     print(list(y.shape))
