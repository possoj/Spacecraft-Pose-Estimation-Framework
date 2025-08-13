"""
Copyright (c) 2025 Julien Posso
"""

import torch
import gc
import time
from typing import Tuple, Dict

from src.spe.spe_utils import SPEUtils

class SPETorch:
    """
    Class for Spacecraft Pose Estimation using a PyTorch model.

    This class wraps a PyTorch model to perform inference for pose estimation.
    It also provides methods for updating, deleting, and moving the model between CPU and GPU.

    Attributes:
        model (torch.nn.Module): The PyTorch model used for inference.
        device (torch.device): The device on which the model runs.
        spe_utils (SPEUtils): Utility functions for post-processing predictions (e.g., activation functions, decoding).
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, spe_utils: SPEUtils) -> None:
        """
        Initialize the SPETorch instance.

        Moves the provided model to the specified device and sets it to evaluation mode.

        Args:
            model (torch.nn.Module): The PyTorch model to be used.
            device (torch.device): The device on which to run the model.
            spe_utils (SPEUtils): Utilities for post-processing predictions.
        """
        self.model = model
        self.device = device
        self.spe_utils = spe_utils
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images: torch.Tensor) -> Tuple[Dict, float]:
        """
        Perform inference on a batch of images.

        Expects a batch of images in the NCHW format.

        Args:
            images (torch.Tensor): A batch of images as a torch tensor.

        Returns:
            Tuple[Any, float]: A tuple containing:
                - The predicted pose (after post-processing).
                - The inference latency in milliseconds.
        """
        assert hasattr(self, 'model')

        t1 = time.time()
        with torch.no_grad():
            pred = self.model(images.to(self.device))
        t2 = time.time()
        inference_latency_ms: float = (t2 - t1) * 1000

        if self.spe_utils.ori_mode == 'keypoints' and self.spe_utils.pos_mode == 'keypoints':
            pose = {'keypoints': pred.detach().cpu().numpy()}
        else:
            ori_key = 'ori_soft' if self.spe_utils.ori_mode == 'classification' else 'ori'
            pos_key = 'pos_soft' if self.spe_utils.pos_mode == 'classification' else 'pos'
            pose = {
                ori_key: pred[0].detach().cpu().numpy(),
                pos_key: pred[1].detach().cpu().numpy()
            }

        pose = self.spe_utils.last_activ(pose)
        pose = self.spe_utils.decode(pose)

        return pose, inference_latency_ms

    def update_model(self, model: torch.nn.Module, device: torch.device) -> None:
        """
        Update the current model with a new one and move it to the specified device.

        Releases resources held by the current model before updating.

        Args:
            model (torch.nn.Module): The new model to use.
            device (torch.device): The device on which the new model should run.
        """
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(torch.device("cpu"))
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.model = None

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def delete_model(self) -> None:
        """
        Delete the current model to free up resources.

        Moves the model to CPU, deletes it, and clears the CUDA cache if applicable.
        """
        self.model.to(torch.device("cpu"))
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None

    def move_to_cpu(self) -> None:
        """
        Move the current model to CPU and clear the CUDA cache.
        """
        self.model.to(torch.device("cpu"))
        gc.collect()
        torch.cuda.empty_cache()

    def move_to_gpu(self) -> None:
        """
        Move the current model to GPU (if self.device is a GPU).
        """
        self.model.to(self.device)
