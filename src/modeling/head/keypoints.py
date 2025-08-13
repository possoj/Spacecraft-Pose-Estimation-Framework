"""
Copyright (c) 2023 Julien Posso
"""
import torch
import torch.nn as nn
import brevitas.nn as qnn
from src.modeling.common.quantizers import *


class KeypointRegressionHead(nn.Module):
    """A PyTorch module that defines a simple FP32 pose estimation head with a single layer to perform keypoints
    regression."""
    def __init__(self, n_feature_maps=1280, n_outputs=24, bias=True, dropout_rate=0.2):
        super().__init__()

        # Position branch
        self.layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            # nn.Linear(in_features=n_feature_maps, out_features=n_outputs, bias=bias)
            nn.Linear(in_features=122880, out_features=n_outputs, bias=bias)
        )

    def forward(self, x):
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = x.reshape(x.size(0), x.size(1))
        x = torch.flatten(x, start_dim=1)
        x = self.layer(x)
        return x