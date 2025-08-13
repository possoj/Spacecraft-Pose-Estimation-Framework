"""
Copyright (c) 2024 Julien Posso
"""

import torch
from torch.nn.modules import Module
from typing import Union


class PosRegLoss(Module):
    """
    Loss function used for the Position branch in regression configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
        norm_distance (bool): Whether to normalize the distance. Default is True.
    """
    def __init__(self, reduction: str = 'mean', norm_distance: bool = True):
        super().__init__()
        assert reduction in ('mean', 'sum'), "reduction must be 'mean' or 'sum'"
        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred (torch.Tensor): Estimated position.
            target (torch.Tensor): Ground truth position.

        Returns:
            torch.Tensor: Position regression loss.
        """
        loss = torch.linalg.norm(pred - target)
        if self.norm_distance:
            loss = loss / torch.linalg.norm(target)
        return self.reduction(loss)


class OriRegLoss(Module):
    """
    Loss function used for the Orientation branch in regression configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
        norm_distance (bool): Whether to normalize the distance. Default is True.
    """
    def __init__(self, reduction: str = 'mean', norm_distance: bool = True):
        super(OriRegLoss, self).__init__()

        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum
        self.norm_distance = norm_distance

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, target_pos: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred (torch.Tensor): Estimated orientation.
            target (torch.Tensor): Ground truth orientation.
            target_pos (torch.Tensor, optional): Ground truth position. Default is None.

        Returns:
            torch.Tensor: Orientation regression loss.
        """
        inter_sum = torch.abs(torch.sum(pred * target, dim=1, keepdim=True))
        # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1. See scoring for more details
        if True in inter_sum[inter_sum > 1.01]:
            raise ValueError("Error while computing orientation Loss")

        inter_sum[inter_sum > 1] = 0
        loss = torch.arccos(inter_sum)
        if self.norm_distance:
            loss = loss / torch.linalg.norm(target_pos, dim=1, keepdim=True)
        return self.reduction(loss)


class SoftClassLoss(Module):
    """
    Loss function used for the Orientation branch in classification configuration.

    Args:
        reduction (str): Type of reduction to apply to the loss. Must be 'mean' or 'sum'.
    """
    def __init__(self, reduction: str = 'mean'):
        super(SoftClassLoss, self).__init__()
        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.reduction = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            pred (torch.Tensor): Predicted orientation probabilities.
            target (torch.Tensor): Ground truth orientation.

        Returns:
            torch.Tensor: Orientation classification loss.
        """
        loss = self.reduction(torch.sum(-(target * torch.log(pred)), dim=1))
        if True in torch.isnan(loss):
            raise ValueError("Error while computing orientation Loss")
        return loss


class SPELoss:
    """
    Spacecraft Pose Estimation Loss class.

    Args:
        ori_mode (str): Orientation mode ('regression', 'classification', or 'keypoints').
        pos_mode (str): Position mode ('regression', 'classification', or 'keypoints').
        beta (float): Weight parameter for the orientation loss. Default is 1.
        norm_distance (bool): Whether to normalize the distance for regression ori and pos loss. Default is False.
    """
    def __init__(self, ori_mode: str, pos_mode: str, beta: float = 1, norm_distance: bool = False):
        assert ori_mode in ['regression', 'classification', 'keypoints']
        assert pos_mode in ['regression', 'classification', 'keypoints']
        self.ori_mode = ori_mode
        self.pos_mode = pos_mode
        self.beta = beta
        self.ori_criterion = OriRegLoss(norm_distance=norm_distance) if ori_mode == 'regression' else SoftClassLoss()
        self.pos_criterion = PosRegLoss(norm_distance=norm_distance) if pos_mode == 'regression' else SoftClassLoss()
        self.keypoints_regression_criterion = torch.nn.MSELoss(reduction='mean')

    def compute_loss(self, prediction: dict, target: dict) -> torch.Tensor:
        """
        Computes the overall loss.

        Args:
            prediction (dict): Dictionary containing the estimated orientation and position (or keypoints).
            target (dict): Dictionary containing the ground truth orientation and position (or keypoints).

        Returns:
            torch.Tensor: Overall loss.
        """
        if self.ori_mode == 'keypoints' and self.pos_mode == 'keypoints':
            return self.keypoints_regression_criterion(prediction['keypoints'], target['keypoints'])

        ori_loss = (self.ori_criterion(prediction['ori'], target['ori'], target['pos'])
                    if self.ori_mode == 'regression'
                    else self.ori_criterion(prediction['ori_soft'], target['ori_soft']))

        pos_loss = (self.pos_criterion(prediction['pos'], target['pos'])
                    if self.pos_mode == 'regression'
                    else self.pos_criterion(prediction['pos_soft'], target['pos_soft']))

        return self.beta * ori_loss + pos_loss
