from torch import nn, Tensor
from typing import OrderedDict

__all__ = ['MobileNetV1']


class MobileNetV1(nn.Module):
    """
    This version of mobilenet is from https://github.com/gilshm/mlperf-pytorch/blob/master/models/mobilenet_v1.py,
    what they said it is migrated from MLPerf.
    """

    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
                nn.BatchNorm2d(oup, eps=0.001),
                nn.ReLU6()
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.ZeroPad2d((0, stride - 1, 0, stride - 1)),
                nn.Conv2d(inp, inp, 3, stride, 1 if stride ==
                          1 else 0, groups=inp, bias=False),
                nn.BatchNorm2d(inp, eps=0.001),
                nn.ReLU6(),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=0.001),
                nn.ReLU6(),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, 1001)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        super().load_state_dict(state_dict['state_dict'], strict=strict)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
