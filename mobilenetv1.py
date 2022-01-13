from pytorchcv.models.mobilenet import mobilenet_w1
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
        self.model = mobilenet_w1(pretrained=True, root='./model/mlperf/pretrained')

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
