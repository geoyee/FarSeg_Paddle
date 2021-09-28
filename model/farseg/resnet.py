import paddle.nn as nn
from functools import partial
from ._resnet import (resnet18, resnet34, resnet50, resnet101, resnet152)
from .layers_lib import freeze_params, freeze_modules


class ResNetEncoder(nn.Layer):
    def __init__(self, 
                 resnet_type=resnet50,
                 norm_layer=nn.BatchNorm2D,
                 include_conv5=True,
                 batchnorm_trainable=True,
                 pretrained=True,
                 freeze_at=0,
                 output_stride=32):
        super(ResNetEncoder, self).__init__()
        self.norm_layer = norm_layer
        self.include_conv5 = include_conv5
        if all([output_stride != 16,
                output_stride != 32,
                output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')
        self.resnet = resnet_type(pretrained=pretrained, norm_layer=norm_layer, drop_fc=True)
        if not batchnorm_trainable:
            self._frozen_res_bn()
        self._freeze_at(at=freeze_at)
        if output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    def _frozen_res_bn(self):
        freeze_modules(self.resnet, self.norm_layer)
        for m in self.resnet.sublayers():
            if isinstance(m, self.norm_layer):
                m.eval()

    def _freeze_at(self, at=2):
        if at >= 1:
            freeze_params(self.resnet.conv1)
            freeze_params(self.resnet.bn1)
        if at >= 2:
            freeze_params(self.resnet.layer1)
        if at >= 3:
            freeze_params(self.resnet.layer2)
        if at >= 4:
            freeze_params(self.resnet.layer3)
        if at >= 5:
            freeze_params(self.resnet.layer4)

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y
        return _function

    def forward(self, inputs):
        x = inputs
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # os 4, #layers/outdim: 18,34/64; 50,101,152/256
        c2 = self.resnet.layer1(x)
        # os 8, #layers/outdim: 18,34/128; 50,101,152/512
        c3 = self.resnet.layer2(c2)
        # os 16, #layers/outdim: 18,34/256; 50,101,152/1024
        c4 = self.resnet.layer3(c3)
        # os 32, #layers/outdim: 18,34/512; 50,101,152/2048
        if self.include_conv5:
            c5 = self.resnet.layer4(c4)
            return [c2, c3, c4, c5]
        return [c2, c3, c4]

    def _nostride_dilate(self, m, dilate):
        # ref:
        # https://github.com/CSAILVision/semantic-segmentation-pypaddle/blob/1235deb1d68a8f3ef87d639b95b2b8e3607eea4c/models/models.py#L256
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)