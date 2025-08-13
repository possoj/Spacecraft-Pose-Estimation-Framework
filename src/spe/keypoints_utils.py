"""
Copyright (c) 2024 Julien Posso
Copyright (c) 2022 SLAB Group
"""

import numpy as np
import cv2
from scipy.io import loadmat

from src.spe.utils import quat2dcm, dcm2quat


class KeyPoints:
    """
    This class provides Keypoints utils spacecraft pose estimation.

    It is based on Park's code and uses it's 3d keypoints by default: https://github.com/tpark94/spnv2
    """
    def __init__(self, camera, keypoints_dir: str = None):
        """
        Initializes the KeyPoints object.

        Args:
            camera: The dataset-specific camera object.
            keypoints_dir (str): The directory containing 3D keypoints data.
        """
        assert keypoints_dir is not None
        self.camera = camera
        self.keypoints3d = self.load_3d_keypoints(keypoints_dir)

    @staticmethod
    def load_3d_keypoints(mat_path: str, name: str = 'tango3Dpoints') -> np.ndarray:
        """
        Loads 3D keypoints from a MAT file.

        Args:
            mat_path (str): The path to the MAT file.
            name (str): The name of the variable containing 3D keypoints.

        Returns:
            np.ndarray: An array of 3D keypoints.
        """
        vertices = loadmat(mat_path)[name]  # [3 x N]
        corners3d = np.transpose(np.array(vertices, dtype=np.float32))  # [N x 3]
        return corners3d

    def project(self, ori: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Project 3D keypoints to 2D using camera parameters.

        Args:
            ori (np.ndarray): Orientation as a quaternion.
            pos (np.ndarray): Position.

        Returns:
            np.ndarray: 2D keypoints.
        """
        # Extract camera parameters
        k = self.camera.K
        dist = self.camera.distCoeffs if hasattr(self.camera, 'distCoeffs') else None

        keypoints = np.concatenate((np.zeros((3, 1), dtype=np.float32),
                                    np.transpose(self.keypoints3d)), axis=1)

        # Make sure keypoints are 3 x N
        if keypoints.shape[0] != 3:
            keypoints = np.transpose(keypoints)

        # Keypoints into 4 x N homogenous coordinates
        keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

        # Transformation to image frame
        pose_mat = np.hstack((quat2dcm(ori), np.expand_dims(pos, 1)))
        xyz = np.dot(pose_mat, keypoints)  # [3 x N]
        x0, y0 = xyz[0, :] / xyz[2, :], xyz[1, :] / xyz[2, :]  # [1 x N] each

        if dist is not None:
            # Apply distortion
            r2 = x0 * x0 + y0 * y0
            cdist = 1 + dist[0] * r2 + dist[1] * r2 * r2 + dist[4] * r2 * r2 * r2
            x = x0 * cdist + dist[2] * 2 * x0 * y0 + dist[3] * (r2 + 2 * x0 * x0)
            y = y0 * cdist + dist[2] * (r2 + 2 * y0 * y0) + dist[3] * 2 * x0 * y0
        else:
            # No distortion coefficients, use undistorted x0, y0
            x, y = x0, y0

        # Apply camera matrix
        keypoints2d = np.vstack((k[0, 0] * x + k[0, 2], k[1, 1] * y + k[1, 2]))

        return keypoints2d

    def create_keypoints2d(self, ori: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Create 2D normalized (0-1) keypoints from Pose and 3D keypoints.

        Args:
            ori (np.ndarray): Orientation tensor.
            pos (np.ndarray): Position tensor.

        Returns:
            np.ndarray: 2D keypoints.
        """
        # Create 2d keypoints from Pose and 3d keypoints
        keypoints2d = self.project(ori, pos)
        # Normalize w.r.t. Image size
        keypoints2d[0] = keypoints2d[0] / self.camera.nu
        keypoints2d[1] = keypoints2d[1] / self.camera.nv
        # Create a 1d vector that contains the origin (x0, y0) of the spacecraft reference frame + keypoints
        keypoints2d = np.reshape(np.transpose(keypoints2d), (np.prod(keypoints2d.shape),)).astype(dtype=np.float32)
        return keypoints2d

    def pnp(self, keypoints2d: np.ndarray, rvec=None, tvec=None, useExtrinsicGuess=False) -> dict:
        """
        Solve Perspective-n-Point (PnP) problem to estimate orientation and position.

        Args:
            keypoints2d (np.ndarray): Keypoints normalized (0-1).
            rvec: Rotation vector (optional).
            tvec: Translation vector (optional).
            useExtrinsicGuess: Whether to use extrinsic guess (optional).

        Returns:
            dict: Dict containing the Pose: orientation as quaternion (np.ndarray) and position (np.ndarray).
        """

        # Convert keypoints to pixels
        x = keypoints2d[::2] * self.camera.nu
        y = keypoints2d[1::2] * self.camera.nv
        keypoints2d = np.vstack([x, y])

        keypoints2d = np.transpose(keypoints2d[:, 1:])

        assert self.keypoints3d.shape[0] == keypoints2d.shape[0], \
            'points 3D and points 2D must have the same number of vertices'

        dist = self.camera.distCoeffs if hasattr(self.camera, 'distCoeffs') else np.zeros((5, 1), dtype=np.float32)

        keypoints3d = np.ascontiguousarray(self.keypoints3d).reshape((-1, 1, 3))
        keypoints2d = np.ascontiguousarray(keypoints2d).reshape((-1, 1, 2))

        _, r_exp, pos = cv2.solvePnP(keypoints3d, keypoints2d, self.camera.K, dist, rvec, tvec, useExtrinsicGuess,
                                     flags=cv2.SOLVEPNP_EPNP)

        r_pr, _ = cv2.Rodrigues(r_exp)

        ori = dcm2quat(r_pr).astype(dtype=np.float32)
        pos = np.squeeze(pos).astype(dtype=np.float32)
        pose = {'ori': ori, 'pos': pos}

        return pose

    def decode_batch(self, keypoints2d: np.ndarray) -> dict:
        """
        Decode batch of keypoint predictions

        Args:
            keypoints2d (np.ndarray): Keypoints normalized (0-1).

        Returns:
            dict: Dict containing the Pose: orientation as quaternion (np.ndarray) and position (np.ndarray).
        """

        batch_size = keypoints2d.shape[0]
        pose_batch = {
            'ori': np.zeros((batch_size, 4), dtype=np.float32),
            'pos': np.zeros((batch_size, 3), dtype=np.float32)
        }

        for i in range(batch_size):
            pose = self.pnp(keypoints2d[i])
            pose_batch['ori'][i] = pose['ori']
            pose_batch['pos'][i] = pose['pos']

        return pose_batch

    def create_bbox_from_keypoints(self, keypoints2d: np.ndarray) -> np.ndarray:
        """
        Create a bounding box from a set of keypoints.

        Args:
            keypoints2d: A 1D numpy array of keypoints in the form [x1, y1, x2, y2, ...].

        Returns:
            A numpy array representing the bounding box in the form [x_min, y_min, x_max, y_max].
        """
        # Scale keypoints from 0-1 to pixel coordinates
        keypoint_x = keypoints2d[::2] * self.camera.nu
        keypoint_y = keypoints2d[1::2] * self.camera.nv

        # Compute the bounding box coordinates
        x_min = np.min(keypoint_x) / self.camera.nu
        x_max = np.max(keypoint_x) / self.camera.nu
        y_min = np.min(keypoint_y) / self.camera.nv
        y_max = np.max(keypoint_y) / self.camera.nv

        bbox = np.array([x_min, y_min, x_max, y_max])

        return bbox
