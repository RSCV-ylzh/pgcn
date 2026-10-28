from typing import Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from pynvml import c_nvmlConfComputeGetKeyRotationThresholdInfo_t
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import ConvFCBBoxHead

import torch # ylzh

@MODELS.register_module()
class DFusROIS2cHead(ConvFCBBoxHead):

    def __init__(self,
                 conv_out_channels: int = 256,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            num_shared_convs=2,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=1,
            conv_out_channels=conv_out_channels,
            with_cls = True,
            with_reg = True,
            *args,
            **kwargs)

        self.conv_d0 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        dilation=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
        self.conv_d1 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        dilation=2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
        self.conv_d2 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        dilation=3,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)

        # add cls specific branch
        self.cls_fc = nn.Linear(conv_out_channels*4, self.fc_out_channels)
        # add reg specific branch
        self.reg_fc = nn.Linear(conv_out_channels*4, self.fc_out_channels)

        self.sigmoid = nn.functional.sigmoid

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # ylzh: conv with dilation of d0 d1 d2, making different resolution
        x0 = x
        x1 = x
        x2 = x
        x0 = self.conv_d0(x0)
        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)

        # print(x0.size(),x1.size(),x2.size())
        x_add = (x0 + x1 + x2)/3
        x_d01 = self.sigmoid(x0-x1)
        x_d02 = self.sigmoid(x0-x2)
        x_d12 = self.sigmoid(x1-x2)
        x = torch.cat((x_add,x_d01,x_d02,x_d12),dim=1)
        x=x.squeeze()
        # print(x.size())

        x_cls = self.cls_fc(x)
        x_reg = self.reg_fc(x)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@MODELS.register_module()
class DFusROI2S2cHead(ConvFCBBoxHead):

    def __init__(self,
                 conv_out_channels: int = 256,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            num_shared_convs=2,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=1,
            conv_out_channels=conv_out_channels,
            with_cls = True,
            with_reg = True,
            *args,
            **kwargs)

        self.conv_d0 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=7,
                        stride=1,
                        padding=0,
                        dilation=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
        self.conv_d1 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        dilation=2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)
        self.conv_d2 = ConvModule(
                        conv_out_channels,
                        conv_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        dilation=3,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg)

        # add cls specific branch
        self.cls_fc = nn.Linear(conv_out_channels*4, self.fc_out_channels)
        # add reg specific branch
        self.reg_fc = nn.Linear(conv_out_channels*4, self.fc_out_channels)

        self.sigmoid = nn.functional.sigmoid

    def forward(self, x: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """

        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # ylzh: conv with dilation of d0 d1 d2, making different resolution
        x0 = x
        x1 = x
        x2 = x
        x0 = self.conv_d0(x0)
        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)

        # print(x0.size(),x1.size(),x2.size())
        x_add = (x0 + x1 + x2)/3
        x_d01 = self.sigmoid(x0-x1)
        x_d02 = self.sigmoid(x0-x2)
        x_d12 = self.sigmoid(x1-x2)
        x_new = torch.cat((x_add,x_d01,x_d02,x_d12),dim=1)
        x_new=x_new.squeeze()
        # print(x.size())


        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        x_cls = self.cls_fc(x_new)+x_cls
        x_reg = self.reg_fc(x_new)+x_reg

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred
