"""
VGG11/13/16/19 in Pytorch.

Adapted from: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

Overview: https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/


"""

import torch.nn as nn


cfg = {
    "VGG8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "VGG9": [64, "M", 128, "M", 256, "M", 512, "M", 512, 512, "M"],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, input_shape, n_class=10):
        super(VGG, self).__init__()
        self.in_channels = input_shape[0]
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        # self.classifier = nn.Linear(512, 10)
        # https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/9c5da95297d2e49d547c92cb8f0ffdd448262c8a/vgg.py#L22
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, n_class),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def name(self):
        return f"{self.vgg_name}"
