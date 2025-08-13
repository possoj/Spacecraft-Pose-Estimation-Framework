"""
Copyright (c) 2025 Julien Posso
"""

import torch
import time
import numpy as np
from typing import Tuple, Optional, List, Dict, Tuple, Union, Any

# from src.temporal.moving_average import MovingAverage
# from src.tools.utils import RunningAverage
# from src.temporal.filter import LowPassFilter
# from src.temporal.pdf_compare import PDFComparer, TemporalPDF
from src.temporal.pdf_compare import TemporalPDF
from src.spe.spe_utils import SPEUtils
from src.spe.spe_torch import SPETorch
from src.nvidia.jetson_deploy import deploy_to_jetson


class Inference:
    def __init__(self, model: torch.nn.Module, inference_device: str, spe_utils: SPEUtils):
        """
        Initialize the inference engine.

        Args:
            model (torch.nn.Module): The neural network model for pose estimation.
            inference_device (str): The inference device (host CPU, GPU, or Jetson GPU / ARM CPU Ultra96).
            spe_utils (SPEUtils): Utility object for pose processing and decoding.
        """
        self.model = model
        self.inference_device = inference_device
        self.spe_utils = spe_utils
        self.inference_engine = None

        self.prev_still_ori = None
        self.prev_video_ori = None

        self.pdf_adapt_ori = TemporalPDF(n=0.8, alpha=16.49, distance_metric='l2')
        self.pdf_adapt_pos = TemporalPDF(n=0.5, alpha=48.64, distance_metric='l2')

        self.ssh_jetson = None
        self.img_size = None

        self.select_inference_engine(self.inference_device)

    def select_inference_engine(self, device: str, model_name: str = None):
        assert device in ('gpu_host', 'cpu_host', 'gpu_jetson', 'cpu_ultra96')

        # Close ssh connection with Jetson board if it was previously used
        if self.inference_device == 'gpu_jetson':
            self.close_jetson()

        self.inference_device = device

        if 'host' in device:
            if device == 'gpu_host':
                assert torch.cuda.is_available()
                inference_device = torch.device('cuda:0')
            else:
                inference_device = torch.device('cpu')
            self.inference_engine = SPETorch(self.model, inference_device, self.spe_utils)

        elif device == "gpu_jetson":

            model = {
            'murso_speed': "murso_nvidia_qat_speed",
            'mursop_dspeed': "mursop_nvidia_qat_dspeed",
            }
            assert model_name is not None
            if not model_name in model.keys():
                raise ValueError(f"Model {model_name} not implemented on Nvidia Jetson, check experiments/nvidia folder")

            if not self.ssh_jetson and self.img_size:
                self.inference_engine, self.ssh_jetson = deploy_to_jetson(
                    f"experiments/build/nvidia/{model[model_name]}",
                    self.spe_utils,
                    self.img_size
                )
        else:
            raise ValueError("Not implemented yet")

    def close_jetson(self):
        if self.ssh_jetson is not None:
            print(f"close ssh connection with Jetson")
            self.inference_engine.close()
            for i in range(3):
                print(f"Waiting for execution on Jetson to complete... {i + 1}/3")
                time.sleep(1)
            self.ssh_jetson.close_ssh_thread()
            self.ssh_jetson = None

    def reset(self) -> None:
        """
        Reset the state of the inference engine.
        """
        self.prev_still_ori = None
        self.prev_video_ori = None
        self.pdf_adapt_ori.reset()
        self.pdf_adapt_pos.reset()

    def update(self, model: torch.nn.Module, spe_utils: object) -> None:
        """
        Update the model and utilities.

        Args:
            model (torch.nn.Module): The new model.
            spe_utils (object): The updated utility object.
        """
        self.model = model
        self.spe_utils = spe_utils
        self.select_inference_engine(self.inference_device)
        self.reset()

    def predict(self, image: torch.Tensor, video_type: str = None) -> Tuple[dict, float, Optional[dict]]:
        """
        Perform inference and optional video-based filtering on a single image.

        Args:
            image (torch.Tensor): Input image tensor NCHW.
            video_type (str, optional): Type of video filtering (e.g., "Adaptative"). Defaults to None.

        Returns:
            tuple[dict, float, dict | None]:
                - Pose estimation results for still frames.
                - Inference latency in milliseconds.
                - Pose estimation results for video frames (if applicable).
        """
        if not self.img_size or self.img_size != tuple(image.size()):
            self.img_size = tuple(image.size())

        # Inference
        pose_still, inference_latency_ms = self.inference_engine.predict(image)
        pose_still = {key: value.squeeze(0) for key, value in pose_still.items()}

        # Ensure no discontinuities in pose_still
        if self.prev_still_ori is not None:
            dot_val = np.dot(self.prev_still_ori, pose_still['ori'])
            if dot_val < 0:
                pose_still['ori'] = -pose_still['ori']
            # Do not update the quaternion pole if an outlier is detected
            if np.abs(dot_val) > 0.5:
                self.prev_still_ori = pose_still['ori']
        else:
                self.prev_still_ori = pose_still['ori']

        # Create bounding box and keypoints for visualization
        if self.spe_utils.keypoints is not None:
            if self.spe_utils.ori_mode == 'keypoints' and self.spe_utils.pos_mode == 'keypoints':
                pose_still['bbox'] = self.spe_utils.keypoints.create_bbox_from_keypoints(pose_still['keypoints'])

            elif (self.spe_utils.pos_mode in ('classification', 'regression')
                  and self.spe_utils.ori_mode in ('classification', 'regression')):
                pose_still['keypoints'] = self.spe_utils.keypoints.create_keypoints2d(pose_still['ori'],
                                                                                      pose_still['pos'])
                pose_still['bbox'] = self.spe_utils.keypoints.create_bbox_from_keypoints(pose_still['keypoints'])

        # Optional temporal post-processing
        if video_type is not None:
            if video_type == 'Adaptative':
                assert self.spe_utils.ori_mode == 'classification'
                assert self.spe_utils.pos_mode == 'classification'

                pose_video = {}
                pose_video['ori_soft'], pose_video['ori_distance'] = self.pdf_adapt_ori.update_pdf(pose_still['ori_soft'])
                pose_video['pos_soft'], pose_video['pos_distance'] = self.pdf_adapt_pos.update_pdf(pose_still['pos_soft'])

                pose_video['ori'], _ = self.spe_utils.orientation.decode(pose_video['ori_soft'])
                pose_video['pos'] = self.spe_utils.position.decode(pose_video['pos_soft'])
            else:
                raise ValueError(f'type of video filtering not implemented: {video_type}')

            # Ensure no discontinuity in quaternions
            if self.prev_video_ori is not None and pose_video is not None:
                dot_val = np.dot(self.prev_video_ori, pose_video['ori'])
                if dot_val < 0:
                    pose_video['ori'] = -pose_video['ori']
                if np.abs(dot_val) > 0.5:
                    self.prev_video_ori = pose_video['ori']
            else:
                self.prev_video_ori = pose_video['ori']

            # Create bounding box and keypoints for visualization
            if self.spe_utils.keypoints is not None:
                if self.spe_utils.ori_mode == 'keypoints' and self.spe_utils.pos_mode == 'keypoints':
                    pose_video['bbox'] = self.spe_utils.keypoints.create_bbox_from_keypoints(pose_video['keypoints'])

                elif (self.spe_utils.pos_mode in ('classification', 'regression')
                      and self.spe_utils.ori_mode in ('classification', 'regression')):
                    pose_video['keypoints'] = self.spe_utils.keypoints.create_keypoints2d(pose_video['ori'],
                                                                                          pose_video['pos'])
                    pose_video['bbox'] = self.spe_utils.keypoints.create_bbox_from_keypoints(pose_video['keypoints'])
        else:
            pose_video = None

        return pose_still, inference_latency_ms, pose_video
