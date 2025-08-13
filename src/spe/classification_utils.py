"""
Copyright (c) 2024 Julien Posso
Copyright (c) 2019 Pedro F. Proença
"""

import numpy as np
from src.spe.utils import euler2quat


class OrientationSoftClassification:
    """
    This class provides soft-classification for attitude (orientation) estimation in spacecraft pose estimation domain.
    Based on Proença's URSONet: https://github.com/pedropro/UrsoNet
    Improvements over Proença's URSONet:
        - Accelerate encoding and decoding (60 times faster).
        - Reducing the number of parameters by deleting unused bins from the orientation histogram
          instead of masking unused bins.
    """

    def __init__(self, n_bins_per_dim: int, smooth_factor: int, delete_unused_bins: bool = False):
        """
        Initialize the OrientationSoftClassification instance.

        Args:
            n_bins_per_dim (int): Number of orientation bins per dimension (i.e., Euler angles: yaw, pitch, roll).
            smooth_factor (int): Smoothing factor for orientation encoding.
            delete_unused_bins (bool): 
        """
        self.n_bins_per_dim = n_bins_per_dim
        self.smooth_factor = smooth_factor
        self.delete_unused_bins = delete_unused_bins
        self.histogram, self.redundant_flags = self.build_histogram(
            np.array([-180, -90, -180]), np.array([180, 90, 180])
        )
        self.n_bins = self.histogram.shape[0]

        self.b = self.pre_compute_ori_decode()

    def build_histogram(self, min_lim: np.ndarray, max_lim: np.ndarray):
        """
        Build the histogram of all possible orientation bins.
        The histogram is built only once to save time during execution compared to Proença's code.

        Args:
            min_lim (np.ndarray): Minimum limits on Z (yaw), Y (pitch), and X (roll) axis.
            max_lim (np.ndarray): Maximum limits on Z (yaw), Y (pitch), and X (roll) axis.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the orientation bins and redundant flags.
        """

        d = 3
        n_bins = self.n_bins_per_dim ** d

        # Construct histogram structure
        bins_per_dim = np.linspace(0.0, 1.0, self.n_bins_per_dim)
        bins_all_dims = np.stack(np.meshgrid(bins_per_dim, bins_per_dim, bins_per_dim, indexing='ij'),
                                 axis=-1).reshape(-1, 3)
        euler_bins = bins_all_dims * (max_lim - min_lim) + min_lim

        quaternions_bins = np.zeros((n_bins, 4))

        for i in range(n_bins):
            quaternions_bins[i, :] = euler2quat(
                euler_bins[i, 0], euler_bins[i, 1], euler_bins[i, 2], gymbal_check=False
            )

        # Mark redundant boundary bins because of the circular nature of Euler angles: yaw = 180° and roll = 180°
        boundary_flags = np.logical_or(euler_bins[:, 0] == max_lim[0],
                                       euler_bins[:, 2] == max_lim[2])
        # Mark gimbal lock bins due to the two singularities at pitch = +/- 90 deg
        # Except bins at yaw = -180 deg and pitch = -90 deg
        gimbal_flags = np.logical_and(np.abs(euler_bins[:, 1]) == max_lim[1],
                                      euler_bins[:, 0] != min_lim[0])
        redundant_flags = np.logical_or(boundary_flags, gimbal_flags)

        if self.delete_unused_bins:
            quaternions_bins = quaternions_bins[~redundant_flags]
            print(f'removed {sum(redundant_flags)} unused bins in the orientation '
                  f'soft-classification histogram')
            print(f'Number of bins in orientation histogram: {quaternions_bins.shape[0]}\n')

        return quaternions_bins, redundant_flags

    def encode(self, ori: np.ndarray):
        """
        Encode orientation (true orientation from the dataset).
        This code is optimized compared to Proença's code: vectorization using NumPy.

        Args:
            ori (np.ndarray): True orientation.

        Returns:
            np.ndarray: Encoded orientation.
        """
        variance = (self.smooth_factor / self.n_bins_per_dim) ** 2 / 12

        # Compute Kernel function (equation 3 in Proença's article https://arxiv.org/pdf/1907.04298.pdf)
        kernel_fct = np.exp(-((2 * np.arccos(np.minimum(1.0, np.abs(np.sum(
            ori * self.histogram, axis=1)))) / np.pi) ** 2) / (2 * variance))

        # Mask unused bins if not deleted from histogram
        if not self.delete_unused_bins:
            kernel_fct[self.redundant_flags] = 0

        ori_encoded = kernel_fct / np.sum(kernel_fct)
        if np.any(np.isnan(ori_encoded)):
            raise ValueError('NaN found in encoded orientation')
        ori_encoded = ori_encoded.astype(np.float32)

        return ori_encoded

    def decode(self, ori: np.ndarray):
        """
        Decode orientation predicted by the neural network.
        This code is optimized compared to Proença's code: vectorization and pre-compute.

        See equation 4 in Proença's URSONet: https://arxiv.org/pdf/1907.04298.pdf
        For more solutions check:
        F. Landis Markley et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics (2007)
        https://ntrs.nasa.gov/api/citations/20070017872/downloads/20070017872.pdf

        Args:
            ori (np.ndarray): Predicted orientation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the solution and the uncertainty in the
            maximum likelihood sense.
        """

        n_bins = self.b.shape[0]
        a = np.sum(self.b * np.reshape(ori, (n_bins, 1, 1)), axis=0)

        if np.any(np.isnan(a)):
            raise ValueError("Error during orientation decoding")

        s, v = np.linalg.eig(a)
        idx = np.argsort(s)
        q_avg = v[:, idx[-1]]
        # Due to numerical errors, we need to enforce normalization
        q_avg = q_avg / np.linalg.norm(q_avg)
        h_inv = np.linalg.inv(a)

        q_avg = q_avg.astype(np.float32)
        h_inv = h_inv.astype(np.float32)

        return q_avg, h_inv

    def decode_batch(self, ori_batch: np.ndarray):
        """
        Decode a batch of orientations using the pre-computed orientation decode variable (b).

        Args:
            ori_batch (np.ndarray): Batch of predicted orientations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the averaged orientations and uncertainties.
        """
        batch_size = ori_batch.shape[0]
        ori_avg = np.zeros((batch_size, 4), dtype=np.float32)
        h_avg = np.zeros((batch_size, 4, 4), dtype=np.float32)

        for i in range(batch_size):
            ori_avg[i], h_avg[i] = self.decode(ori_batch[i])

        return ori_avg, h_avg

    def pre_compute_ori_decode(self):
        """
        Pre-compute orientation decode to save time during training/inference using NumPy.

        Returns:
            np.ndarray: Pre-computed orientation decode variable.
        """
        n_bins = self.histogram.shape[0]
        return np.reshape(self.histogram, (n_bins, 4, 1)) * np.reshape(self.histogram, (n_bins, 1, 4))


class PositionSoftClassification:
    """
    This class provides soft-classification for position estimation in 3D space.
    """

    def __init__(self, n_bins_per_dim: int, smooth_factor: int, min_lim: np.ndarray, max_lim: np.ndarray):
        """
        Initialize the PositionSoftClassification instance.

        Args:
            n_bins_per_dim (int): Number of position bins per dimension (i.e., x, y, z).
            smooth_factor (int): Smoothing factor for position encoding.
            min_lim (np.ndarray): Minimum limits on the x, y, and z axes.
            max_lim (np.ndarray): Maximum limits on the x, y, and z axes.
        """
        self.n_bins_per_dim = n_bins_per_dim
        self.smooth_factor = smooth_factor
        self.min_lim = min_lim
        self.max_lim = max_lim
        self.histogram = self.build_histogram()
        self.n_bins = self.histogram.shape[0]

    def build_histogram(self):
        """
        Build the histogram of all possible position bins.
        The histogram is built only once to save time during execution.

        Returns:
            np.ndarray: A numpy array containing the position bins.
        """

        # Construct histogram structure
        bins_per_dim = np.linspace(0.0, 1.0, self.n_bins_per_dim)
        bins_all_dims = np.stack(np.meshgrid(bins_per_dim, bins_per_dim, bins_per_dim, indexing='ij'),
                                 axis=-1).reshape(-1, 3)
        position_bins = bins_all_dims * (self.max_lim - self.min_lim) + self.min_lim

        return position_bins

    def encode(self, pos: np.ndarray):
        """
        Encode position (true position from the dataset).
        This code is optimized using NumPy for vectorization.

        Args:
            pos (np.ndarray): True position.

        Returns:
            np.ndarray: Encoded position.
        """
        variance = (self.smooth_factor / self.n_bins_per_dim) ** 2 / 12

        # Compute Kernel function
        kernel_fct = np.exp(-np.sum((pos - self.histogram) ** 2, axis=1) / (2 * variance))

        pos_encoded = kernel_fct / np.sum(kernel_fct)

        if np.any(np.isnan(pos_encoded)):
            raise ValueError('NaN found in encoded position')
        pos_encoded = pos_encoded.astype(np.float32)

        return pos_encoded

    def decode(self, pos: np.ndarray):
        """
        Decode position predicted by the neural network.

        Args:
            pos (np.ndarray): Predicted position.

        Returns:
            np.ndarray: Decoded position.
        """
        # Ensure the soft-classification vector is normalized
        if np.sum(pos) == 0:
            raise ValueError("Encoded position vector sum is zero, cannot decode.")

        # Compute the weighted sum of the position bins
        weighted_sum = np.sum(self.histogram * np.reshape(pos, (-1, 1)), axis=0)

        # Normalize the weighted sum to get the average position
        pos_avg = weighted_sum / np.sum(pos)

        if np.any(np.isnan(pos_avg)):
            raise ValueError("Error during position decoding, NaN found in decoded position.")

        pos_avg = pos_avg.astype(np.float32)

        return pos_avg

    def decode_batch(self, pos_batch: np.ndarray):
        """
        Decode a batch of positions using the pre-computed position decode variable (b).

        Args:
            pos_batch (np.ndarray): Batch of predicted positions.

        Returns:
            np.ndarray: A numpy array containing the averaged positions.
        """
        batch_size = pos_batch.shape[0]
        pos_avg = np.zeros((batch_size, 3), dtype=np.float32)

        for i in range(batch_size):
            pos_avg[i] = self.decode(pos_batch[i])

        return pos_avg
