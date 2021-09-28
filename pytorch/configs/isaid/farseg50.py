import torch.nn as nn
from simplecv.module import fpn


model_cfg = dict(
    type='FarSeg',
    params=dict(
        resnet_encoder=dict(
            resnet_type='resnet50',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 8, 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            stem3_3x3=False,
        ),
        fpn=dict(
            in_channels_list=(256, 512, 1024, 2048),
            out_channels=256,
            conv_block=fpn.default_conv_block,
            top_blocks=None,
        ),
        scene_relation=dict(
            in_channels=2048,
            channel_list=(256, 256, 256, 256),
            out_channels=256,
            scale_aware_proj=True,
        ),
        decoder=dict(
            in_channels=256,
            out_channels=128,
            in_feat_output_strides=(4, 8, 16, 32),
            out_feat_output_stride=4,
            norm_fn=nn.BatchNorm2d,
            num_groups_gn=None
        ),
        num_classes=16,
    )
)
