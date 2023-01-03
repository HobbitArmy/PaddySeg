import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..backbones.gbinet import GhostModule


@HEADS.register_module()
class GCNHead(BaseDecodeHead):
    """Ghost Convolution Networks for Semantic Segmentation.
    This head is implemented of `GBiNet `_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        ghost_ratio: expanding ratio of GhostConv
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 ghost_ratio=2,
                 concat_input=False,
                 **kwargs):

        assert num_convs > 0

        self.num_convs = num_convs
        self.kernel_size = kernel_size
        self.concat_input = concat_input

        super(GCNHead, self).__init__(**kwargs)

        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(GhostModule(_in_channels,
                                     self.channels,
                                     kernel_size=kernel_size,
                                     ratio=ghost_ratio,
                                     stride=1))
        self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels,
                                       kernel_size=kernel_size,
                                       padding=kernel_size // 2,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
