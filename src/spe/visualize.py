"""
Copyright (c) 2024 Julien Posso
"""

import numpy as np
import cv2
import copy

from typing import Tuple

from src.spe.utils import quat2dcm


class VisualizePose:
    def __init__(self, spe_utils, resolution: Tuple[int, int] = None):
        """
        Initialize the VisualizePose object.

        Args:
            spe_utils: Utility functions for satellite pose estimation.
            resolution: Optional resolution for resizing the output image.
        """
        self.spe_utils = spe_utils
        self.resolution = resolution

    def set_spe_utils(self, spe_utils):
        """
        Set the utility functions for satellite pose estimation.

        Args:
            spe_utils: Utility functions for satellite pose estimation.
        """
        self.spe_utils = spe_utils

    def add_visualization(
        self,
        image: np.ndarray = None,
        true_pose: dict = None,
        pred_pose: dict = None,
        temp_pose: dict = None,
        show_true_pose: bool = False,
        show_pred_pose: bool = False,
        show_temp_pose: bool = False,
        show_true_bbox: bool = False,
        show_pred_bbox: bool = False,
        show_temp_bbox: bool = False,
        show_true_keypoints: bool = False,
        show_pred_keypoints: bool = False,
        show_temp_keypoints: bool = False,
    ) -> np.ndarray:
        """
        Add visualizations such as arrows, bounding boxes, and keypoints to the image.

        Args:
            image: Input image in numpy array format.
            true_pose: Dictionary containing true pose information.
            pred_pose: Dictionary containing predicted pose information.
            temp_pose: Dictionary containing temporary pose information.
            show_true_pose: Flag to show true pose.
            show_pred_pose: Flag to show predicted pose.
            show_temp_pose: Flag to show temporary pose.
            show_true_bbox: Flag to show true bounding box.
            show_pred_bbox: Flag to show predicted bounding box.
            show_temp_bbox: Flag to show temporary bounding box.
            show_true_keypoints: Flag to show true keypoints.
            show_pred_keypoints: Flag to show predicted keypoints.
            show_temp_keypoints: Flag to show temporary keypoints.

        Returns:
            Image with added visualizations.
        """
        assert image is not None, "Image must be provided."

        image = copy.deepcopy(image)

        if true_pose is not None:
            if show_true_pose:
                image = self.draw_arrow(image, true_pose, color='RGB')
            if 'keypoints' in true_pose and show_true_keypoints:
                image = self.draw_keypoints(image, true_pose['keypoints'], color='red')
            if 'bbox' in true_pose and show_true_bbox:
                image = self.draw_bbox(image, true_pose['bbox'], color='red')

        if pred_pose is not None:
            if show_pred_pose:
                image = self.draw_arrow(image, pred_pose, color='MYC')
            if 'keypoints' in pred_pose and show_pred_keypoints:
                image = self.draw_keypoints(image, pred_pose['keypoints'], color='green')
            if 'bbox' in pred_pose and show_pred_bbox:
                image = self.draw_bbox(image, pred_pose['bbox'], color='green')

        if temp_pose is not None:
            if show_temp_pose:
                image = self.draw_arrow(image, temp_pose, color='RCA')
            if 'keypoints' in temp_pose and show_temp_keypoints:
                image = self.draw_keypoints(image, temp_pose['keypoints'], color='blue')
            if 'bbox' in temp_pose and show_temp_bbox:
                image = self.draw_bbox(image, temp_pose['bbox'], color='blue')

        if self.resolution is not None:
            image = cv2.resize(image, self.resolution)

        return image

    def project(self, q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project points to the image frame to draw axes.

        Args:
            q: Orientation quaternion.
            r: Translation vector.

        Returns:
            Tuple of x and y coordinates in the image plane.
        """
        # Reference points in satellite frame for drawing axes (homogeneous coordinates)
        p_axes = np.array([
            [0, 0, 0, 1],  # Origin of object's local coordinate system
            [1, 0, 0, 1],  # Head of the arrow along the local x-axis
            [0, 1, 0, 1],  # Head of the arrow along the local y-axis
            [0, 0, 1, 1],  # Head of the arrow along the local z-axis
        ])
        points_body = np.transpose(p_axes)

        # Transformation to camera frame
        pose_mat = np.hstack((quat2dcm(q), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # Getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # Projection to image plane
        points_image_plane = self.spe_utils.camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y

    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray, color: str = 'red', diameter: int = 8) -> np.ndarray:
        """
        Draw keypoints on the image.

        Args:
            image: Input image in numpy array format.
            keypoints: Keypoints to draw.
            color: Color for the keypoints ('red', 'green', 'blue').
            diameter: Diameter of the keypoint circles.

        Returns:
            Image with keypoints drawn.
        """
        dot_colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255)
        }

        line_colors = {
            'red': (255, 0, 255),  # Magenta
            'green': (255, 255, 0),  # Yellow
            'blue': (0, 255, 255),  # Cyan
        }
        if color not in dot_colors:
            raise ValueError("Color must be 'red', 'green', or 'blue'")

        dot_color = dot_colors[color]
        line_color = line_colors[color]

        # Scale keypoints from 0-1 to pixel coordinates
        keypoint_x = keypoints[::2] * self.spe_utils.camera.nu
        keypoint_y = keypoints[1::2] * self.spe_utils.camera.nv

        # Draw circles at keypoints
        for x, y in zip(keypoint_x[1:], keypoint_y[1:]):
            cv2.circle(image, (int(x), int(y)), diameter, dot_color, -1)

        # Exclude origin keypoint to trace lines
        x = keypoint_x[1:].astype(int).tolist()
        y = keypoint_y[1:].astype(int).tolist()

        # Draw lines between keypoints
        lines = [
            (0, 1), (0, 3), (1, 2), (2, 3),  # Big rectangle (solar panel)
            (5, 6), (4, 5), (4, 7), (6, 7),  # Small rectangle (opposite face)
            (1, 5), (0, 4), (3, 7), (2, 6)  # Vertices
        ]
        for start, end in lines:
            image = cv2.line(image, (x[start], y[start]), (x[end], y[end]), line_color, 2)

        # Draw specific calculated lines (antennas)
        specific_lines = [
            (1, 5, 8), (2, 6, 9), (3, 7, 10)
        ]
        for a, b, c in specific_lines:
            x1 = int(0.8 * x[a] + 0.2 * x[b])
            y1 = int(0.8 * y[a] + 0.2 * y[b])
            image = cv2.line(image, (x1, y1), (x[c], y[c]), line_color, 2)

        return image

    def draw_arrow(self, image: np.ndarray, pose: dict, color: str = 'RGB') -> np.ndarray:
        """
        Draw arrows representing the pose on the image.

        Args:
            image: Input image in numpy array format.
            pose: Pose dictionary containing 'pos' and 'ori'.
            color: Color scheme for the arrows ('RGB', 'MYC', 'OSV', 'RCA').

        Returns:
            Image with arrows drawn.
        """
        # Project to Euclidean space
        pos = pose['pos']
        ori = pose['ori']
        xa, ya = self.project(ori, pos)
        x = [int(xi) for xi in xa]
        y = [int(yi) for yi in ya]

        colors = {
            'RGB': [
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
            ],
            'MYC': [
                (255, 0, 255),  # Magenta
                (255, 255, 0),  # Yellow
                (0, 255, 255),  # Cyan
            ],
            'OSV': [
                (255, 127, 0),  # Orange
                (0, 255, 127),  # Spring Green
                (127, 0, 255),  # Violet
            ],
            'RCA': [
                (255, 0, 127),  # Rose
                (127, 255, 0),  # Chartreuse Green
                (0, 127, 255),  # Azure
            ]
        }

        # Draw arrowed line
        image = cv2.arrowedLine(image, (x[0], y[0]), (x[1], y[1]), colors[color][0], 6)
        image = cv2.arrowedLine(image, (x[0], y[0]), (x[2], y[2]), colors[color][1], 6)
        image = cv2.arrowedLine(image, (x[0], y[0]), (x[3], y[3]), colors[color][2], 6)

        return image

    def draw_bbox(self, image: np.ndarray, bbox: np.ndarray, color: str = 'red') -> np.ndarray:
        """
        Draw a bounding box on the image.

        Args:
            image: Input image in numpy array format.
            bbox: Bounding box in the form [x_min, y_min, x_max, y_max].
            color: Color for the bounding box ('red', 'green', 'blue').

        Returns:
            Image with bounding box drawn.
        """
        colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
        }

        # Convert float 0-1 labels to int pixel coordinates
        x_min = int(bbox[0] * self.spe_utils.camera.nu)
        y_min = int(bbox[1] * self.spe_utils.camera.nv)
        x_max = int(bbox[2] * self.spe_utils.camera.nu)
        y_max = int(bbox[3] * self.spe_utils.camera.nv)

        # Draw bbox
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[color], 2)

        return image
