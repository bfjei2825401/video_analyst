# -*- coding: utf-8 -*

import torch.nn as nn
import torch.nn.functional as F

from videoanalyst.model.backbone.backbone_impl.resnet import ResNet50_M
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.pyramid.pyramid_base import (TRACK_PYRAMIDS,
                                                     VOS_PYRAMIDS)


@VOS_PYRAMIDS.register
@TRACK_PYRAMIDS.register
class PANet(ModuleBase):
    r"""
    PAN
    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, pyramid):
        super(PANet, self).__init__()
        self.pyramid = pyramid
        # self.down1 =

    def forward(self, x):
        features = self.pyramid(x)
        len(features)
        return x


class CBL(ModuleBase):
    r"""
    CBL: Convolution + Batch Norm + Leaky ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, inplace=True)


class PyramidRes50(ResNet50_M):
    r"""
    PyramidRes50: Feature Pyramid based on ResNet-50 (5 stages)
    """

    def __init__(self):
        super(PyramidRes50, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x1, x2, x3, x4, x5
