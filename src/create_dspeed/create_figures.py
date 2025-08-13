"""
Copyright (c) 2025 Julien Posso
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from src.spe.utils import quat2euler


def create_figures_still(
    ori: np.ndarray = None,
    pos: np.ndarray = None,
    save_path: str = None,
    language: str = 'english'
) -> None:
    """
    Creates and saves static (still image) plots of orientation and position distributions.

    This function is typically used to analyze a dataset of individual annotated images,
    producing histograms and scatter plots of orientation angles (Euler) and Cartesian positions.

    Args:
        ori (np.ndarray, optional): Array of orientation quaternions of shape (N, 4).
        pos (np.ndarray, optional): Array of positions of shape (N, 3) in meters.
        save_path (str): Output directory where all plots will be saved (required).
        language (str): Language for axis labels. Must be 'english' or 'french'.

    Returns:
        None
    """

    assert save_path is not None
    os.makedirs(save_path, exist_ok=True)

    ori = [quat2euler(q) for q in ori]

    if ori is not None:
        # Define labels based on the selected language
        labels = {
            'english': {
                'z_axis': 'Z-Axis rotation - Yaw (deg)',
                'y_axis': 'Y-Axis rotation - Pitch (deg)',
                'x_axis': 'X-Axis rotation - Roll (deg)',
                'distance': 'Satellite distance (m)',
                'images': 'Number of images',
            },
            'french': {
                'z_axis': 'Rotation axe Z - Lacet (degrés)',
                'y_axis': 'Rotation axe Y - Tangage (degrés)',
                'x_axis': 'Rotation axe X - Roulis (degrés)',
                'distance': 'Distance au satellite (mètres)',
                'images': "Nombre d'images",
            }
        }

        plt.figure(figsize=(9, 7))
        plt.hist([x[0] for x in ori], bins=36, color='skyblue', edgecolor='royalblue')
        # plt.xlabel(labels[language]['z_axis'], fontsize=22)
        # plt.ylabel(labels[language]['images'], fontsize=22)
        plt.xticks([-180, -90, 0, 90, 180], fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.savefig(os.path.join(save_path, f'Yaw.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.hist([x[1] for x in ori], bins=36, color='lightcoral', edgecolor='indianred')
        # plt.xlabel(labels[language]['y_axis'], fontsize=22)
        # plt.ylabel(labels[language]['images'], fontsize=22)
        plt.xticks([-180, -90, 0, 90, 180], fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.savefig(os.path.join(save_path, f'Pitch.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.hist([x[2] for x in ori], bins=36, color='lightgreen', edgecolor='limegreen')
        # plt.xlabel(labels[language]['x_axis'], fontsize=22)
        # plt.ylabel(labels[language]['images'], fontsize=22)
        plt.xticks([-180, -90, 0, 90, 180], fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.savefig(os.path.join(save_path, f'Roll.png'), bbox_inches='tight')
        plt.close()

    if pos is not None:
        plt.figure(figsize=(9, 7))
        plt.scatter([x[2] for x in pos], [x[0] for x in pos], marker='.', color='indianred', s=10)
        plt.xlabel('t$_{BC,z}$ (m)' if language == 'english' else 't$_{BC,z}$ (m)', fontsize=24)
        plt.ylabel('t$_{BC,x}$ (m)' if language == 'english' else 't$_{BC,x}$ (m)', fontsize=24)
        plt.grid()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.savefig(os.path.join(save_path, f'x.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.scatter([x[2] for x in pos], [x[1] for x in pos], marker='.', color='limegreen', s=10)
        plt.xlabel('t$_{BC,z}$ (m)' if language == 'english' else 't$_{BC,z}$ (m)', fontsize=24)
        plt.ylabel('t$_{BC,y}$ (m)' if language == 'english' else 't$_{BC,y}$ (m)', fontsize=24)
        plt.grid()
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.savefig(os.path.join(save_path, f'y.png'), bbox_inches='tight')
        plt.close()

        # Plot distribution of the distance with the target
        plt.figure(figsize=(9, 7))
        plt.hist([np.linalg.norm(x) for x in pos], bins=45, color='skyblue', edgecolor='royalblue')
        plt.xlabel(labels[language]['distance'], fontsize=24)
        plt.xticks(fontsize=22)
        # plt.ylabel(labels[language]['images'], fontsize=24)
        plt.yticks(fontsize=22)
        plt.grid()
        plt.savefig(os.path.join(save_path, f'Distance.png'), bbox_inches='tight')
        plt.close()

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(projection='3d')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.set_box_aspect([1, 1, 4])

        ax.scatter([x[0] for x in pos], [x[1] for x in pos], [x[2] for x in pos],
                   marker='.', color='royalblue', s=10)

        # Customize the plot
        ax.set_xlabel('t$_{BC,x}$ (m)' if language == 'english' else 't$_{BC,x}$ (m)', fontsize=17)
        ax.set_ylabel('t$_{BC,y}$ (m)' if language == 'english' else 't$_{BC,y}$ (m)', fontsize=17)
        ax.set_zlabel('t$_{BC,z}$ (m)' if language == 'english' else 't$_{BC,z}$ (m)', fontsize=17)

        ax.view_init(elev=15., azim=-50)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        plt.savefig(os.path.join(save_path, f'z.png'))
        plt.close(fig)


def create_figures_video(
    ori: np.ndarray = None,
    pos: np.ndarray = None,
    path: str = None,
    language: str = 'english'
) -> None:
    """
    Creates and saves temporal plots (as a function of frame index) of orientation and position.

    This function is typically used for sequences or video-like datasets. It visualizes:
    - Quaternion components across time
    - Orientation angles (Euler) across time
    - Position along x, y, and z across time

    Args:
        ori (np.ndarray, optional): Array of orientation quaternions of shape (T, 4), T = number of frames.
        pos (np.ndarray, optional): Array of positions of shape (T, 3), in meters.
        path (str): Output directory where all plots will be saved.
        language (str): Language for labels ('english' or 'french').

    Returns:
        None
    """

    # Define labels based on the selected language
    labels = {
        'english': {
            'index': 'Image index',
            'quat_index': 'Orientation quaternion index',
            'angle': 'Orientation angle',
            'position': 'Position',
            'yaw': 'Yaw',
            'pitch': 'Pitch',
            'roll': 'Roll',
            'meters': 'meters',
        },
        'french': {
            'index': "Index d'image",
            'quat_index': 'Indice quaternion orientation',
            'angle': "Angle d'orientation",
            'position': 'position',
            'yaw': 'Lacet',
            'pitch': 'Tangage',
            'roll': 'Roulis',
            'meters': 'mètres',
        }
    }

    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(45, 40))
    for i in range(4):
        plt.subplot(4, 1, i + 1)  # 4 subplots in a row
        plt.plot([x[i] for x in ori], marker='.', color='limegreen', linestyle='-', markersize=8)
        plt.xlabel(labels[language]['index'], fontsize=24)
        plt.ylabel(f'{labels[language]["quat_index"]} {i}', fontsize=24)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(os.path.join(path, f'ori_quat_indices.png'))
    plt.close()

    plt.figure(figsize=(45, 40))
    angles = [labels[language]['yaw'], labels[language]['pitch'], labels[language]['roll']]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot([quat2euler(x)[i] for x in ori], marker='.', color='limegreen', linestyle='-', markersize=8)
        plt.xlabel(labels[language]['index'], fontsize=24)
        plt.ylabel(
            f'{labels[language]["angle"]} {angles[i]} (degrees)' if language == 'english' else f'{labels[language]["angle"]} {angles[i]} (degrés)',
            fontsize=24)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(os.path.join(path, f'ori_euler_indices.png'))
    plt.close()

    plt.figure(figsize=(45, 30))
    positions = ["x", "y", "z"]
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot([x[i] for x in pos], marker='.', color='limegreen', linestyle='-', markersize=8, label='true')
        plt.xlabel(labels[language]['index'], fontsize=24)
        plt.ylabel(
            f'{labels[language]["position"]} {positions[i]} axis (m)' if language == 'english' else
            f'Axe {positions[i]} {labels[language]["position"]} (m)',
            fontsize=24)
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(os.path.join(path, f'pos_xyz_indices.png'))
    plt.close()
