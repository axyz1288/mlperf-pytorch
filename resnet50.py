from torch import nn, Tensor
from torchvision.models import resnet50
from typing import OrderedDict

__all__ = ['ResNet50']


class ResNet50(nn.Module):
    """
    The model is from torch official, and the pretrained model is from
    https://github.com/gilshm/mlperf-pytorch/blob/master/models/mobilenet_v1.py
    """

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__()
        self.model = resnet50(num_classes=1001, **kwargs)

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        self.model.load_state_dict(state_dict['state_dict'], strict=strict)

    def forward(self, x):
        return self.model(x)
