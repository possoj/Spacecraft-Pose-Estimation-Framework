"""
Dataset Generator for D-SPEED: Still and Video Sequences

This script generates a synthetic dataset for spacecraft pose estimation, consisting of:
- Still samples with randomized 6-DoF poses (position and attitude) and visibility filtering
- Video sequences with smooth, parameterized 6-DoF motion, including optional accelerations

Each generated pose (orientation and position) is saved in JSON format, and corresponding figures
are generated for both still and video data to illustrate orientation and position trajectories.

The dataset is created using the SPEUtils module, which handles 3D keypoint projection
and enforces minimum visibility constraints in the image plane.

A graphical user interface (GUI) is also provided to visualize each pose using a wireframe spacecraft model.
Note: this script only generates numerical 6-DoF pose data; it does not render or export synthetic images.

Author: Julien Posso
Year: 2025
"""

import sys
import random
import json
from tqdm import tqdm
import numpy as np
import os
from typing import Tuple, Dict

from src.spe.utils import generate_orientation, euler2quat, multiply_quaternions, euler2dcm, dcm2quat, conjugate_quaternion, quat2dcm
from src.data.datasets.dspeed import Camera
from src.create_dspeed.create_figures import create_figures_still, create_figures_video
from src.create_dspeed.visualize import SpacecraftPoseGui
from src.spe.spe_utils import SPEUtils


def count_visible_keypoints(x_keypoints: np.ndarray, y_keypoints: np.ndarray) -> int:
    """
    Counts how many keypoints are visible in the image plane.

    Args:
        x_keypoints (list): X pixel coordinates of keypoints (including origin at index 0).
        y_keypoints (list): Y pixel coordinates of keypoints.

    Returns:
        int: Number of keypoints (excluding the origin) within the image bounds.
    """
    cnt = 0
    # Remove origin from keypoints
    for x, y in zip(x_keypoints[1:], y_keypoints[1:]):
        if 0 <= x <= Camera.nu and 0 <= y <= Camera.nv:
            cnt += 1

    return cnt


def generate_positions(orientations: np.ndarray, spe_utils: SPEUtils) -> np.ndarray:
    """
    Generates 3D positions for each given orientation such that the spacecraft is visible
    in the image (i.e., at least 8 or 11 keypoints are visible depending on distance).

    Args:
        orientations (np.ndarray): Array of quaternion orientations.
        spe_utils (SPEUtils): Object for projecting 3D keypoints into the image plane.

    Returns:
        list: List of 3D position vectors.
    """

    min_z, max_z = 3, 35

    positions = []
    loop = tqdm(orientations, desc='still positions', file=sys.stdout)

    for orientation in loop:

        # Generate position
        z = random.uniform(min_z, max_z)
        loop.set_postfix_str(f"z = {z:.1f} m")

        while True:
            y = random.uniform(-0.3 * z, 0.3 * z)
            x = random.uniform(-0.3 * z, 0.3 * z)

            position = np.array([x, y, z])

            # Project position in image plane
            x_keypoints, y_keypoints = spe_utils.keypoints.project(orientation, position)

            # Check satellite visibility
            cnt = count_visible_keypoints(x_keypoints, y_keypoints)
            minimum_visible_keypoints = 8 if z < 5 else 11
            if cnt >= minimum_visible_keypoints:
                positions.append(position)
                break

    return np.array(positions)


def create_json(orientations: np.ndarray, positions: np.ndarray, path: str) -> Dict[str, Dict[np.ndarray, np.ndarray]]:
    """
    Saves orientation and position pairs to a JSON file, one entry per image.

    Args:
        orientations (list or np.ndarray): List of quaternions.
        positions (list or np.ndarray): List of 3D positions.
        path (str): Path to the output JSON file.

    Returns:
        dict: Dictionary of the JSON content that was written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pose = {}
    for i in range(len(orientations)):
        # Round the orientations and positions to 6 decimal places
        rounded_ori = np.round(orientations[i], 6).tolist()
        rounded_pos = np.round(positions[i], 6).tolist()
        pose[f"{os.path.dirname(path)}/{i}.png"] = {'ori': rounded_ori, 'pos': rounded_pos}

    with open(path, 'w') as json_file:
        json.dump(pose, json_file)

    return pose


def generate_sequence(
        n_samples: int,
        initial_pose: dict,
        evolution: dict,
        spe_utils: SPEUtils,
        desc: str = 'sequential',
        camera_movement: bool = False
) -> tuple:
    """
    Generates a sequence of orientations and positions for a spacecraft, considering potential
    accelerations in both position and orientation across specified frames and axes.
    This function updates the position and orientation of the spacecraft at each frame, based on the specified
    constant velocities and any defined accelerations. It also checks the visibility of the spacecraft by ensuring a
    minimum number of keypoints are visible in the projected image plane at each frame.
    The function uses quaternions to represent and update the current orientation. Rotation matrix is an alternative.
    Avoid using Euler angles because of gimbal lock.

    Args:
        n_samples (int): The number of samples in the sequence.
        initial_pose (dict): A dictionary with keys 'ori' for initial orientation (quaternion) and 'pos' for initial position (3D vector).
        evolution (dict): A dictionary containing 'ori' for orientation velocity and 'pos' for position velocity. It may also contain
                          'pos_acceleration' and 'ori_acceleration' as lists of tuples, each representing an acceleration phase
                          with elements (axis, start_frame, stop_frame, acceleration_value).
        spe_utils (SPEUtils): Contains 3D keypoints projection functions
        desc (str): Description for the tqdm progress bar.
        camera_movement (bool): Whether the camera (chaser) or the target moves (rotation and/or translation).

    Returns:
        tuple: A tuple of two numpy arrays. The first array contains the sequence of orientations as quaternions,
               and the second contains the sequence of positions as 3D vectors.
    """

    # Initial orientation and position
    current_ori = initial_pose['ori']
    current_pos = initial_pose['pos']

    # Rotation and translation increment to apply at every frame, ZYX (yaw, pitch, roll) convention
    # May change during the sequence if acceleration is applied
    current_ori_velocity = evolution['ori']
    current_pos_velocity = evolution['pos']

    ori_sequence = np.array([current_ori])
    pos_sequence = np.array([current_pos])

    for frame in tqdm(range(n_samples), desc=desc, file=sys.stdout):
        # Apply position acceleration based on the frame number and specified acceleration periods
        for acc in evolution.get('pos_acceleration', []):
            axis, start, stop, value = acc
            current_pos_velocity[axis] += value if start <= frame < stop else 0

        # Apply orientation acceleration based on the frame number and specified acceleration periods
        for acc in evolution.get('ori_acceleration', []):
            axis, start, stop, value = acc
            current_ori_velocity[axis] += value if start <= frame < stop else 0

        if camera_movement:
            # The camera moves and the target is fixed
            # Update position and orientation based on velocity and angular velocity
            current_pos -= current_pos_velocity
            # The quaternion multiplication is flipped and conjugated: rotation is applied from the camera perspective
            current_ori = multiply_quaternions(
                conjugate_quaternion(euler2quat(*current_ori_velocity)),
                current_ori
            )
            # The world position coordinates does not change but the relative camera-target position changes
            # when the camera rotates
            current_pos = np.dot(np.linalg.inv(euler2dcm(*current_ori_velocity)), current_pos)

        else:
            # The target moves and the camera is fixed
            # Update position and orientation based on velocity and angular velocity
            # Updating the position does not take into account the orientation of the satellite
            current_pos += current_pos_velocity
            # Apply current_ori_velocity rotation to the current target orientation
            current_ori = multiply_quaternions(current_ori, euler2quat(*current_ori_velocity))

        # Project position in image plane
        x_keypoints, y_keypoints = spe_utils.keypoints.project(current_ori, current_pos)

        # Check satellite visibility
        cnt = count_visible_keypoints(x_keypoints, y_keypoints)
        minimum_visible_keypoints = 8 if current_pos[-1] < 5 else 11
        assert cnt >= minimum_visible_keypoints, (f"Satellite is not visible:\n"
                                                  f"pos = {current_pos}\n"
                                                  f"ori = {current_ori}\n"
                                                  f"number of visible keypoints = {cnt}")

        ori_sequence = np.vstack((ori_sequence, current_ori))
        pos_sequence = np.vstack((pos_sequence, current_pos))

    return ori_sequence, pos_sequence


def create_data(save_path: str, spe_utils: SPEUtils):
    """
    Generates the full D-SPEED 6-DoF positions and attitudes with one still-image set and several video sequences.

    It creates multiple folders:
    - still/
    - S/, TIR/, TIT/, TITR/, TAR/, TAT/, TATR/, CIR/, CAR/, CATR/, Docking/

    Each folder contains a `pose.json` file and figures of the ground truth.

    Args:
        save_path (str): Root directory where the dataset and figures are saved.
        spe_utils (SPEUtils): Utility object for 3D keypoint projections.

    Returns:
        None
    """

    os.makedirs(save_path, exist_ok=True)

    data = {}

    # Generate still data
    n_data = 60000
    ori_q = generate_orientation(n_data)
    pos = generate_positions(ori_q, spe_utils)
    create_figures_still(ori_q, pos, os.path.join(save_path, 'still'))
    data['still'] = create_json(ori_q, pos, os.path.join(save_path, 'still', 'pose.json'))

    # GENERATE VIDEOS.
    # Videos will start with the same orientation
    # Multiply by 1 to ensure full precision initial orientation
    initial_q = multiply_quaternions(
        np.array([0.0, -0.7071, 0.7071, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
    )

    n_data = 1499

    # S(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([0.0, 0.0, 15.0])
    }
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0])  # Position evolution in meters per frame
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='stationary')
    data['s'] = create_json(ori_q, pos, os.path.join(save_path, 'S', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'S', ))

    # TIR(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([0.0, 0.0, 15.0])
    }
    pose_evolution = {
        # 30 times the maximum of docking (IDSS)
        # 'ori': np.array([0.24, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'ori': np.array([0.24, 0.10733, 0.21466]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0])  # Position evolution in meters per frame
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TIR')
    data['tir'] = create_json(ori_q, pos, os.path.join(save_path, 'TIR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TIR', ))

    # TIT(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([-7.0, -4.5, 30.0])
    }
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        # 3, 2, 4 (x, y z) times the maximum of docking (IDSS)
        'pos': np.array([3 * 0.0016, 2 * 0.0016, -4 * 0.004])  # Position evolution in meters per frame

    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TIT')
    data['tit'] = create_json(ori_q, pos, os.path.join(save_path, 'TIT', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TIT', ))

    # TITR(EB/ST)
    init_pose = {
        'ori': np.array([0.0000, -0.7071, 0.7071, 0.0000]),
        'pos': np.array([-7.0, -4.5, 30.0])
    }
    # Same evolution as TIT and TIR
    pose_evolution = {
        'ori': np.array([0.24, 0.10733, 0.21466]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([3 * 0.0016, 2 * 0.0016, -4 * 0.004])  # Position evolution in meters per frame
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TITR')
    data['titr'] = create_json(ori_q, pos, os.path.join(save_path, 'TITR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TITR', ))

    # TAR(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([0.0, 0.0, 15.0])
    }
    # Max velocity = TIR scenario
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution per frame
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution per frame
        'ori_acceleration': [
            (0, 50, 150, 0.24 / 100),  # Accelerate for 4 seconds on yaw axis
            (1, 250, 350, 0.10733 / 100),  # Accelerate for 4 seconds on pitch axis
            (2, 450, 550, 0.21466 / 100),  # Accelerate for 4 seconds on roll axis
            (1, 650, 850, -0.10733 / 200),  # Decelerate for 8 seconds on pitch axis
            (2, 950, 1150, -0.21466 / 200),  # Decelerate for 8 seconds on roll axis
            (0, 1250, 1450, -0.24 / 200),  # Decelerate for 8 seconds on yaw axis
        ],
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TAR')
    data['tar'] = create_json(ori_q, pos, os.path.join(save_path, 'TAR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TAR', ))

    # TAT(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([5.5, 3, 20.0])
    }
    # Max velocity = TIT scenario
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution in meters per frame
        'pos_acceleration': [
            (0, 50, 150, -3 * 0.0016 / 100),  # Accelerate for 4 seconds on x-axis
            (1, 250, 350, -2 * 0.0016 / 100),  # Accelerate for 4 seconds on y-axis
            (2, 450, 550, -4 * 0.004 / 100),  # Accelerate for 4 seconds on z-axis
            (1, 1050, 1250, 2 * 0.0016 / 200),  # Decelerate for 8 seconds on y-axis
            (0, 1150, 1350, 3 * 0.0016 / 200),  # Decelerate for 8 seconds on x-axis
            (2, 1250, 1450, 4 * 0.004 / 200),  # Decelerate for 8 seconds on z-axis
        ]
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TAT')
    data['tat'] = create_json(ori_q, pos, os.path.join(save_path, 'TAT', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TAT', ))

    # TATR(EB/ST)
    # Max velocity = TITR scenario
    init_pose = {
        'ori': initial_q,
        'pos': np.array([5.5, 3, 20.0])
    }
    # Max velocity = TITR scenario
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution in meters per frame
        'ori_acceleration': [
            (0, 50, 150, 0.24 / 100),  # Accelerate for 4 seconds on yaw axis
            (1, 250, 350, 0.10733 / 100),  # Accelerate for 4 seconds on pitch axis
            (2, 450, 550, 0.21466 / 100),  # Accelerate for 4 seconds on roll axis
            (1, 650, 850, -0.10733 / 200),  # Decelerate for 8 seconds on pitch axis
            (2, 950, 1150, -0.21466 / 200),  # Decelerate for 8 seconds on roll axis
            (0, 1250, 1450, -0.24 / 200),  # Decelerate for 8 seconds on yaw axis
        ],
        'pos_acceleration': [
            (0, 50, 150, -3 * 0.0016 / 100),  # Accelerate for 4 seconds on x-axis
            (1, 250, 350, -2 * 0.0016 / 100),  # Accelerate for 4 seconds on y-axis
            (2, 450, 550, -4 * 0.004 / 100),  # Accelerate for 4 seconds on z-axis
            (1, 1050, 1250, 2 * 0.0016 / 200),  # Decelerate for 8 seconds on y-axis
            (0, 1150, 1350, 3 * 0.0016 / 200),  # Decelerate for 8 seconds on x-axis
            (2, 1250, 1450, 4 * 0.004 / 200),  # Decelerate for 8 seconds on z-axis
        ]
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='TATR')
    data['tatr'] = create_json(ori_q, pos, os.path.join(save_path, 'TATR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'TATR', ))

    # CIR(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([0.0, 0.0, 15.0])
    }
    pose_evolution = {
        # 30 times the maximum of docking (IDSS)
        'ori': np.array([0.24, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution in meters per frame
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='CIR', camera_movement=True)
    data['cir'] = create_json(ori_q, pos, os.path.join(save_path, 'CIR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'CIR', ))

    # CAR(EB/ST)
    init_pose = {
        'ori': initial_q,
        'pos': np.array([2.0, 0.0, 15.0])
    }
    pose_evolution = {
        # 30 times the maximum of docking (IDSS)
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution in meters per frame
        'ori_acceleration': [
            (0, 50, 150, 0.24 / 100),  # Accelerate for 4 seconds on yaw axis
            (0, 750, 950, -0.24 / 200),  # Decelerate for 8 seconds on yaw axis
            (1, 1100, 1150, -0.10733 / 50),  # Accelerate for 2 seconds on pitch axis (negative)
            (1, 1170, 1220, 0.10733 / 50),  # Decelerate for 8 seconds on pitch axis
        ],
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='CAR', camera_movement=True)
    data['car'] = create_json(ori_q, pos, os.path.join(save_path, 'CAR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'CAR', ))

    # CATR
    init_pose = {
        'ori': initial_q,
        'pos': np.array([-3.0, 0.0, 15.0])
    }
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.0]),  # Position evolution in meters per frame
        'ori_acceleration': [
            (1, 25, 50, -0.24 / 25),  # Accelerate for 1 seconds on pitch axis
            (1, 125, 150, 0.24 / 25),  # Decelerate for 1 seconds on pitch axis
            (1, 400, 425, -0.24 / 25),  # Accelerate for 1 seconds on pitch axis
            (1, 500, 525, 0.24 / 25),  # Decelerate for 1 seconds on pitch axis
            (1, 775, 800, -0.24 / 25),  # Accelerate for 1 seconds on pitch axis
            (1, 875, 900, 0.24 / 25),  # Decelerate for 1 seconds on pitch axis
            (1, 1150, 1175, -0.24 / 25),  # Accelerate for 1 seconds on pitch axis
            (1, 1250, 1275, 0.24 / 25),  # Decelerate for 1 seconds on pitch axis
        ],
        'pos_acceleration': [
            (0, 175, 200, +0.037 / 25),  # Accelerate for 1 seconds on x-axis
            (0, 350, 375, -0.037 / 25),  # Decelerate for 1 seconds on x-axis
            (0, 550, 575, +0.037 / 25),  # Accelerate for 1 seconds on x-axis
            (0, 725, 750, -0.037 / 25),  # Decelerate for 1 seconds on x-axis
            (0, 925, 950, +0.037 / 25),  # Accelerate for 1 seconds on x-axis
            (0, 1100, 1125, -0.037 / 25),  # Decelerate for 1 seconds on x-axis
            (0, 1300, 1325, +0.037 / 25),  # Accelerate for 1 seconds on x-axis
            (0, 1475, 1500, -0.037 / 25),  # Decelerate for 1 seconds on x-axis
        ]
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='CATR', camera_movement=True)
    data['catr'] = create_json(ori_q, pos, os.path.join(save_path, 'CATR', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'CATR', ))

    # Docking(EB/ST)
    init_pose = {
        'ori': euler2quat(0, 0, 0),
        'pos': np.array([0.0, 0.0, 9.0])
    }
    pose_evolution = {
        'ori': np.array([0.0, 0.0, 0.0]),  # Orientation evolution in degrees per frame (ZYX convention)
        'pos': np.array([0.0, 0.0, 0.1 / 25]),  # Position evolution in meters per frame
    }
    ori_q, pos = generate_sequence(n_data, init_pose, pose_evolution, spe_utils, desc='Docking',
                                   camera_movement=True)

    data['docking'] = create_json(ori_q, pos, os.path.join(save_path, 'Docking', 'pose.json'))
    create_figures_video(ori_q, pos, os.path.join(save_path, 'Docking', ))


if __name__ == "__main__":

    utils = SPEUtils(camera=Camera(), keypoints_path="models/3d_models/tangoPoints.mat")

    create_data(
        save_path="experiments/create_dspeed",
        spe_utils=utils,
    )

    SpacecraftPoseGui(
        splits_path="experiments/create_dspeed",
        spe_utils=utils,
    ).run()
