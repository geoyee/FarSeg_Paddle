import paddle.nn as nn


GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2D(1)


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def freeze_params(module):
    for name, p in module.named_parameters():
        p.stop_gradient = True
        if isinstance(module, nn.BatchNorm2D):
            module.eval()


def freeze_modules(module, specific_class=None):
    for m in module.sublayers():
        if specific_class is not None:
            if not isinstance(m, specific_class):
                continue
        freeze_params(m)