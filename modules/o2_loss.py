from typing import Optional

import torch
import torch.nn as nn
from sympy import false

from torch import Tensor
import numpy as np
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

import sys

@weighted_loss
def l1_loss_o2_xy(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss with 2nd order of dxdy + dwdh.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """

    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    # loss = torch.abs(pred - target)
    # return loss

    # 提取 dx, dy, dw, dh
    pred_dx = pred[:, 0]  # shape: [N]
    pred_dy = pred[:, 1]
    pred_dw = pred[:, 2]
    pred_dh = pred[:, 3]

    target_dx = target[:, 0]
    target_dy = target[:, 1]
    target_dw = target[:, 2]
    target_dh = target[:, 3]

    # 计算 dx 与 dy 的绝对误差乘积
    loss_dx_dy_1 = torch.abs(pred_dx * pred_dy - target_dx * target_dy) # shape: [N]
    loss_dx_dy_2 = torch.abs(pred_dx - target_dx) * torch.abs(pred_dy - target_dy)  # shape: [N]
    # print('loss_dx_dy_1\n',loss_dx_dy_1)
    # print('loss_dx_dy_2\n',loss_dx_dy_2)

    # 计算 dw 与 dh 的绝对误差乘积
    loss_dw_dh_1 = torch.abs(pred_dw * pred_dh - target_dw * target_dh) # shape: [N]
    loss_dw_dh_2 = torch.abs(pred_dw - target_dw) * torch.abs(pred_dh - target_dh)  # shape: [N]
    # print('loss_dw_dh_1\n',loss_dw_dh_1)
    # print('loss_dw_dh_2\n',loss_dw_dh_2)

    # loss_dx_dw_mean = np.mean(pred.detach().cpu().numpy(), axis=0)
    # print('loss_dx_dw_mean\n',loss_dx_dw_mean)

    # 总损失为两部分的均值之和
    # epsilon = torch.tensor(0.01, device=loss_dx_dw_mean.device)   # 可调整的极小值
    # epsilon = torch.full_like(loss_dx_dw_mean, 0.01)[0]
    # total_loss = (loss_dx_dy_1 + loss_dx_dy_2).mean()/torch.max((loss_dx_dw_mean[0] + loss_dx_dw_mean[1]), epsilon) + (loss_dw_dh_1 + loss_dw_dh_2).mean()/torch.max((loss_dx_dw_mean[2]+loss_dx_dw_mean[3]), epsilon)
    total_loss = (loss_dx_dy_1 + loss_dx_dy_2).mean()/2 + (loss_dw_dh_1 + loss_dw_dh_2).mean()/2


    print('total_loss\n',total_loss)

    # sys.exit("调试模式：程序已终止")  # 终止并输出提示

    # 返回总和
    return total_loss

@weighted_loss
def l1_loss_o2_xw(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss with 2nd order of dxdw + dydh

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """

    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    # loss = torch.abs(pred - target)
    # return loss

    # 提取 dx, dy, dw, dh
    pred_dx = pred[:, 0]  # shape: [N]
    pred_dy = pred[:, 1]
    pred_dw = pred[:, 2]
    pred_dh = pred[:, 3]

    target_dx = target[:, 0]
    target_dy = target[:, 1]
    target_dw = target[:, 2]
    target_dh = target[:, 3]

    # 计算 dx 与 dw 的绝对误差乘积
    loss_dx_dw_1 = torch.abs(pred_dx * pred_dw - target_dx * target_dw) # shape: [N]
    loss_dx_dw_2 = torch.abs(pred_dx - target_dx) * torch.abs(pred_dw - target_dw)  # shape: [N]

    # 计算 dy 与 dh 的绝对误差乘积
    loss_dy_dh_1 = torch.abs(pred_dy * pred_dh - target_dy * target_dh) # shape: [N]
    loss_dy_dh_2 = torch.abs(pred_dy - target_dy) * torch.abs(pred_dh - target_dh)  # shape: [N]


    # loss_dx_dw_mean = np.mean(pred.detach().cpu().numpy(), axis=0)

    # 总损失为两部分的均值之和
    # epsilon = torch.tensor(0.01, device=loss_dx_dw_mean.device) # 可调整的极小值
    # epsilon = torch.full_like(loss_dx_dw_mean, 0.01)[0]
    # total_loss = (loss_dx_dw_1 + loss_dx_dw_2).mean()/torch.max((loss_dx_dw_mean[0]+loss_dx_dw_mean[2]),epsilon) + (loss_dy_dh_1 + loss_dy_dh_2).mean()/torch.max((loss_dx_dw_mean[1] + loss_dx_dw_mean[3]), epsilon)
    total_loss = (loss_dx_dw_1 + loss_dx_dw_2).mean()/2 + (loss_dy_dh_1 + loss_dy_dh_2).mean()/2



    # 返回总和
    return total_loss

@weighted_loss
def l1_loss_o2_xy_xw(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss with 2nd order of (dxdy + dwdh) + (dxdw + dydh)

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """

    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    # loss = torch.abs(pred - target)
    # return loss

    # 提取 dx, dy, dw, dh
    pred_dx = pred[:, 0]  # shape: [N]
    pred_dy = pred[:, 1]
    pred_dw = pred[:, 2]
    pred_dh = pred[:, 3]

    target_dx = target[:, 0]
    target_dy = target[:, 1]
    target_dw = target[:, 2]
    target_dh = target[:, 3]

    # 计算 dx 与 dy 的绝对误差乘积
    loss_dx_dy_1 = torch.abs(pred_dx * pred_dy - target_dx * target_dy) # shape: [N]
    loss_dx_dy_2 = torch.abs(pred_dx - target_dx) * torch.abs(pred_dy - target_dy)  # shape: [N]

    # 计算 dw 与 dh 的绝对误差乘积
    loss_dw_dh_1 = torch.abs(pred_dw * pred_dh - target_dw * target_dh) # shape: [N]
    loss_dw_dh_2 = torch.abs(pred_dw - target_dw) * torch.abs(pred_dh - target_dh)  # shape: [N]

    # 计算 dx 与 dw 的绝对误差乘积
    loss_dx_dw_1 = torch.abs(pred_dx * pred_dw - target_dx * target_dw) # shape: [N]
    loss_dx_dw_2 = torch.abs(pred_dx - target_dx) * torch.abs(pred_dw - target_dw)  # shape: [N]

    # 计算 dy 与 dh 的绝对误差乘积
    loss_dy_dh_1 = torch.abs(pred_dy * pred_dh - target_dy * target_dh) # shape: [N]
    loss_dy_dh_2 = torch.abs(pred_dy - target_dy) * torch.abs(pred_dh - target_dh)  # shape: [N]

    total_loss = (loss_dx_dw_1 + loss_dx_dw_2).mean()/4 + (loss_dy_dh_1 + loss_dy_dh_2).mean()/4 + (loss_dx_dy_1 + loss_dx_dy_2).mean()/4 + (loss_dw_dh_1 + loss_dw_dh_2).mean()/4

    # 返回总和
    return total_loss

@weighted_loss
def l1_loss_o2_xy_xw_var(pred: Tensor, target: Tensor, var_list: list) -> Tensor:
    """L1 loss with 2nd power, weights for different items variable.

    - o2: 2 次幂
    - o2_xy: xy方式的 交叉二次幂
    - o2_xw: xw方式的 交叉二次幂
    - o2_xyxw: xy+xw方式的 交叉二次幂

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """

    if target.numel() == 0:
        return pred.sum() * 0

    # assert pred.size() == target.size()
    # loss = torch.abs(pred - target)
    # return loss

    # 提取 dx, dy, dw, dh
    pred_dx = pred[:, 0]  # shape: [N]
    pred_dy = pred[:, 1]
    pred_dw = pred[:, 2]
    pred_dh = pred[:, 3]

    target_dx = target[:, 0]
    target_dy = target[:, 1]
    target_dw = target[:, 2]
    target_dh = target[:, 3]

    # 计算 dx 与 dy 的绝对误差乘积
    ldxy = torch.abs(pred_dx * pred_dy - target_dx * target_dy) # shape: [N]
    ldxdy = torch.abs(pred_dx - target_dx) * torch.abs(pred_dy - target_dy)  # shape: [N]

    # 计算 dw 与 dh 的绝对误差乘积
    ldwh = torch.abs(pred_dw * pred_dh - target_dw * target_dh) # shape: [N]
    ldwdh = torch.abs(pred_dw - target_dw) * torch.abs(pred_dh - target_dh)  # shape: [N]

    # 计算 dx 与 dw 的绝对误差乘积
    ldxw = torch.abs(pred_dx * pred_dw - target_dx * target_dw) # shape: [N]
    ldxdw = torch.abs(pred_dx - target_dx) * torch.abs(pred_dw - target_dw)  # shape: [N]

    # 计算 dy 与 dh 的绝对误差乘积
    ldyh = torch.abs(pred_dy * pred_dh - target_dy * target_dh) # shape: [N]
    ldydh = torch.abs(pred_dy - target_dy) * torch.abs(pred_dh - target_dh)  # shape: [N]

    # total_loss = (loss_dx_dw_1 + loss_dx_dw_2).mean()/4 + (loss_dy_dh_1 + loss_dy_dh_2).mean()/4 + (loss_dx_dy_1 + loss_dx_dy_2).mean()/4 + (loss_dw_dh_1 + loss_dw_dh_2).mean()/4

    total_loss = ldxy.mean() * var_list[0] + \
                 ldxdy.mean() * var_list[1] + \
                 ldwh.mean() * var_list[2] + \
                 ldwdh.mean() * var_list[3] + \
                 ldxw.mean() * var_list[4] + \
                 ldxdw.mean() * var_list[5] + \
                 ldyh.mean() * var_list[6] + \
                 ldydh.mean() * var_list[7]
    # 返回总和
    # return ldxy.mean(), ldxdy.mean(), ldwh.mean(), ldwdh.mean(), ldxw.mean(), ldxdw.mean(), ldyh.mean(), ldydh.mean()
    return total_loss

@MODELS.register_module()
class L1LossOrder2XY(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss_o2_xy(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@MODELS.register_module()
class L1LossOrder2XW(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss_o2_xw(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@MODELS.register_module()
class L1LossOrder2XYXW(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss_o2_xy_xw(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@MODELS.register_module()
class L1LossOrder2XYXWVar(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 var_list: list = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                 var_sqrt: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # vars
        self.var_sqrt = var_sqrt

        if len(var_list)==1:
            self.var_list = [ var_list[0] ] * 8
        elif len(var_list)==2:
            self.var_list = [ var_list[0] ] *4 + [var_list[1]] *4
        elif len(var_list)==4:
            self.var_list = [ var_list[0] ] *2 + [var_list[1]] *2 + [ var_list[0] ] *2 + [var_list[1]] *2
        elif len(var_list)==8:
            self.var_list = var_list
        else:
            print('Illegal count of list, please input 1, 2, 4, or 8 float numbers.')

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        total_loss = l1_loss_o2_xy_xw_var(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor,
            var_list=self.var_list)

        if self.var_sqrt is True:
            total_loss = torch.sqrt(total_loss)

        loss_bbox = self.loss_weight * total_loss

        return loss_bbox
