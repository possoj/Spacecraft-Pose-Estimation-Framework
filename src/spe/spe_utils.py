"""
Copyright (c) 2024 Julien Posso
"""

import numpy as np
from src.spe.keypoints_utils import KeyPoints
from src.spe.classification_utils import OrientationSoftClassification, PositionSoftClassification


class SPEUtils:
    """
    Spacecraft Pose Estimation Utils.
    """

    def __init__(
        self,
        camera,
        ori_mode: str = 'regression',
        n_ori_bins_per_dim: int = 12,
        ori_smooth_factor: int = 3,
        ori_delete_unused_bins: bool = True,
        pos_mode: str = 'regression',
        n_pos_bins_per_dim: int = 10,
        pos_smooth_factor: int = 100,
        keypoints_path: str = None
    ):
        """
        Initialize the SPEUtils object.

        Args:
            camera: The dataset-specific camera object.
            ori_mode: Mode of orientation (e.g., 'regression', 'classification', 'keypoints').
            n_ori_bins_per_dim: Number of orientation bins per dimension.
            ori_smooth_factor: Smoothing factor for orientation.
            ori_delete_unused_bins: weather to delete unused bins in orientation histogram
            (see OrientationSoftClassification for more details).
            pos_mode: Mode of position (e.g., 'regression', 'classification', 'keypoints').
            keypoints_path: path to the 3D keypoints.
        """
        assert ori_mode in ['regression', 'classification', 'keypoints']
        assert pos_mode in ['regression', 'classification', 'keypoints']
        if pos_mode == 'keypoints' or ori_mode == 'keypoints':
            assert keypoints_path is not None

        self.ori_mode = ori_mode
        self.pos_mode = pos_mode
        self.camera = camera
        self.orientation = OrientationSoftClassification(n_ori_bins_per_dim, ori_smooth_factor, ori_delete_unused_bins)
        # min_lim=np.array([-16, -12, -2]), max_lim=np.array([16, 12, 40]
        # These values correspond to a 5 meters margin on the orientation histogram
        self.position = PositionSoftClassification(
            n_pos_bins_per_dim, pos_smooth_factor, min_lim=np.array([-16, -12, -2]), max_lim=np.array([16, 12, 40])
        )
        self.keypoints = KeyPoints(camera, keypoints_path) if keypoints_path is not None else None

    def last_activ(self, pose: dict) -> dict:
        """
        Optionally add an activation function at the end of the neural network.

        Args:
            pose: Dictionary containing pose information.

        Returns:
            Updated pose dictionary with applied activation function.
        """
        if self.ori_mode == 'keypoints' and self.pos_mode == 'keypoints':
            # Sigmoid
            pose['keypoints'] = 1 / (1 + np.exp(-pose['keypoints']))
        else:
            if self.ori_mode == 'regression':
                # L2 norm
                pose['ori'] = pose['ori'] / np.linalg.norm(pose['ori'], ord=2, axis=1, keepdims=True)
            else:
                # Softmax
                e_ori = np.exp(pose['ori_soft'] - np.max(pose['ori_soft'], axis=1, keepdims=True))
                pose['ori_soft'] = e_ori / np.sum(e_ori, axis=1, keepdims=True)
            if self.pos_mode == 'classification':
                e_pos = np.exp(pose['pos_soft'] - np.max(pose['pos_soft'], axis=1, keepdims=True))
                pose['pos_soft'] = e_pos / np.sum(e_pos, axis=1, keepdims=True)

        return pose

    def decode(self, pose: dict) -> dict:
        """
        Decode the output of the neural network and compute the spacecraft pose.

        Args:
            pose: Dictionary containing pose information.

        Returns:
            Decoded pose dictionary.
        """
        if self.ori_mode == 'keypoints' and self.pos_mode == 'keypoints' and self.keypoints is not None:
            pose.update(self.keypoints.decode_batch(pose['keypoints']))
        else:
            if self.ori_mode == 'classification':
                pose['ori'], _ = self.orientation.decode_batch(pose['ori_soft'])
            if self.pos_mode == 'classification':
                pose['pos'] = self.position.decode_batch(pose['pos_soft'])

        return pose

    @staticmethod
    def get_score(true_pose: dict, pred_pose: dict) -> dict:
        """
        Compute the score for the pose estimation based on true and predicted poses.
        Expects numpy arrays with batch dimension
        Score definition:
        https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
        https://arxiv.org/abs/1911.02050

        Args:
            true_pose: Dictionary containing true pose information.
            pred_pose: Dictionary containing predicted pose information.

        Returns:
            Dictionary with evaluation metrics including ESA score, orientation error, and position error.
        """
        ori_target = true_pose['ori']
        pos_target = true_pose['pos']
        ori_pred = pred_pose['ori']
        pos_pred = pred_pose['pos']

        # 1. Position error (e_t):
        pos_error = np.linalg.norm(pos_target - pos_pred, axis=1)
        mean_pos_error = np.mean(pos_error)

        # 2. Normalized position error (e_t/):
        norm_pos_error = pos_error / np.linalg.norm(pos_target, axis=1)
        mean_norm_pos_error = np.mean(norm_pos_error)

        # 3. Orientation error (e_q):
        inter_sum = np.abs(np.sum(ori_pred * ori_target, axis=1, keepdims=True))
        # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1 (seems it is what ESA does for scoring):
        # Set it to one when the sum is just above 1: I guess the overflow is due to numerical errors.
        # Raise ValueError when it is greater than 1.01: the overflow is due to errors in model prediction.
        if True in inter_sum[inter_sum > 1.01]:
            raise ValueError("Intermediate sum issue due to error in model prediction (orientation)")
            # Remark: it seems that ESA scoring (website) is not Raising error but scaling down to zero.
            # In practice this condition is true only when there are issues with the model or solver function.
            # With the current code this error was never raised.
            # inter_sum[inter_sum > 1.01] = 0
        inter_sum[inter_sum > 1] = 1

        mean_ori_error = np.mean(2 * np.arccos(inter_sum))
        mean_ori_error_deg = mean_ori_error * 180 / np.pi

        # 4. ESA score
        esa_score = mean_ori_error + mean_norm_pos_error

        eval_metrics = {
            'esa_score': esa_score,
            'ori_score': mean_ori_error,
            'pos_score': mean_norm_pos_error,
            'ori_error': mean_ori_error_deg,
            'pos_error': mean_pos_error,
        }

        return eval_metrics
