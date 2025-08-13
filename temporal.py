"""
Copyright (c) 2025 Julien Posso
"""

import copy
import os
import sys
import torch
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List
import cv2

from src.tools.utils import RunningAverage
from src.config.train.config import load_config, save_config
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.modeling.model import import_model
from src.spe.utils import quat2euler, euler_angle_difference
from src.temporal.inference import Inference
from src.spe.visualize import VisualizePose


def compute_statistics(data: List) -> Tuple[float, float, float, float, float]:
    """
    Calculate basic statistics for a dataset.

    Args:
        data (list): Array of numerical data.

    Returns:
        tuple[float, float, float, float, float]:
            - Minimum value of the data.
            - Maximum value of the data.
            - Median value of the data.
            - Mean value of the data.
            - Standard deviation of the data.
    """
    # Calculate the statistics
    min_value = np.min(data)
    max_value = np.max(data)
    median_value = np.median(data)
    mean_value = np.mean(data)
    std_value = np.std(data)
    return min_value, max_value, median_value, mean_value, std_value


def main(
    model_path: str = "models/mursop_fp32_dspeed_large_image",
    video_type: str = None,
    language: str = 'english'
) -> None:

    ori_error_label = {
        'english': 'Orientation Error (degrees)',
        'french': 'Erreur angulaire (degrés)'
    }

    pos_error_label = {
        'english': 'Position Error (meters)',
        'french': 'Erreur de position (mètres)'
    }

    index_image = {
        'english': 'Index of the Frame in the Video',
        'french': 'Index de l\'image dans la vidéo'
    }

    labels = {
        'english': {'still': 'still prediction', 'video': 'temporal prediction', 'true': 'ground truth'},
        'french': {'still': 'prédiction statique', 'video': 'prédiction temporelle', 'true': 'vérité terrain'}
    }

    angles = {
        'english': ['Yaw', 'Pitch', 'Roll'],
        'french': ['Lacet', 'Tangage', 'Roulis'],
    }

    positions = ["x", "y", "z"]

    exp_dir = f'{os.path.basename(model_path)}_{video_type}'
    exp_path = os.path.join('experiments', 'temporal', exp_dir)

    shutil.rmtree(exp_path, ignore_errors=True)
    os.makedirs(exp_path, exist_ok=True)

    config = load_config(os.path.join(model_path, 'config.yaml'))
    save_config(config, os.path.join(exp_path, 'config.yaml'))

    camera = load_camera(config.DATA.PATH)
    spe_utils = SPEUtils(
        camera, config.MODEL.HEAD.ORI, config.MODEL.HEAD.N_ORI_BINS_PER_DIM, config.DATA.ORI_SMOOTH_FACTOR,
        config.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, config.MODEL.HEAD.POS, config.MODEL.HEAD.N_POS_BINS_PER_DIM,
        config.DATA.POS_SMOOTH_FACTOR, config.MODEL.HEAD.KEYPOINTS_PATH
    )
    rot_augment = False
    other_augment = False
    data_shuffle = False
    batch_size = 1  # The code only works with a batch size of 1
    data_path = config.DATA.PATH.replace('still', 'video')
    # data_path = config.DATA.PATH
    data, splits = load_dataset(spe_utils, data_path, batch_size, config.DATA.IMG_SIZE,
                               rot_augment, other_augment, data_shuffle)

    params_path = os.path.join(model_path, 'model', 'parameters.pt')
    bw_path = os.path.join(model_path, 'model', 'bit_width.json')
    bit_width_path = bw_path if os.path.exists(bw_path) else None
    model, bit_width = import_model(
        data, config.MODEL.BACKBONE.NAME, config.MODEL.HEAD.NAME, params_path, bit_width_path,
        manual_copy=False, residual=config.MODEL.BACKBONE.RESIDUAL, quantization=config.MODEL.QUANTIZATION,
        ori_mode=config.MODEL.HEAD.ORI, n_ori_bins=spe_utils.orientation.n_bins,
        pos_mode=config.MODEL.HEAD.POS, n_pos_bins=spe_utils.position.n_bins,
    )
    device = "gpu_host" if torch.cuda.is_available() else "cpu_host"
    inference = Inference(model, device, spe_utils)

    still_metrics = {}
    video_metrics = {}
    distances = {}

    for split, dataloader in data.items():

        # if split != "CATR":
        #     continue

        base_dir_save = os.path.join(exp_path, split)
        os.makedirs(base_dir_save, exist_ok=True)

        print("base_dir_save =", base_dir_save)
        assert os.path.exists(base_dir_save), f"Dossier inexistant: {base_dir_save}"

        # SAVE VIDEO
        # visu = VisualizePose(spe_utils)
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # video1 = cv2.VideoWriter(os.path.join(base_dir_save, 'video.avi'), fourcc, 25, (1920, 1200))
        # video2 = cv2.VideoWriter(os.path.join(base_dir_save, 'video_true.avi'), fourcc, 25, (1920, 1200))
        # video3 = cv2.VideoWriter(os.path.join(base_dir_save, 'pred_sill.avi'), fourcc, 25, (1920, 1200))
        # video4 = cv2.VideoWriter(os.path.join(base_dir_save, 'pred_video.avi'), fourcc, 25, (1920, 1200))

        running_avg = RunningAverage(keys=('esa_score', 'ori_score', 'pos_score', 'ori_error', 'pos_error'))
        inference.reset()

        ori_error_still = []
        pos_error_still = []
        ori_error_video = []
        pos_error_video = []
        true_ori = []
        true_pos = []
        still_ori = []
        still_pos = []
        video_ori = []
        video_pos = []
        pos_distance = []
        ori_distance = []

        # prev_pred_ori_soft = None  # DEBUG

        loop = tqdm(dataloader, desc=f"{split}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=130, file=sys.stdout)

        for i, (image, true_pose) in enumerate(loop):
            # image_name = os.path.basename(image['path'][0])
            still_pose, _, video_pose = inference.predict(image["torch"], video_type)

            # Compute evaluation metrics
            still_batch = {key: np.expand_dims(value, axis=0) for key, value in still_pose.items()}
            true_pose = {key: value.detach().cpu().numpy() for key, value in true_pose.items()}
            eval_metrics_still = spe_utils.get_score(true_pose, still_batch)
            if video_type is not None:
                video_batch = {key: np.expand_dims(value, axis=0) for key, value in video_pose.items()}
                eval_metrics_video = spe_utils.get_score(true_pose, video_batch)

            # Store pred and true pose
            true_ori.append(true_pose['ori'].squeeze(0).tolist())
            true_pos.append(true_pose['pos'].squeeze(0).tolist())
            still_ori.append(still_pose['ori'].tolist())
            still_pos.append(still_pose['pos'].tolist())
            ori_error_still.append(eval_metrics_still['ori_error'].squeeze(0))
            pos_error_still.append(eval_metrics_still['pos_error'].squeeze(0))
            if video_type is not None:
                video_ori.append(video_pose['ori'].tolist())
                video_pos.append(video_pose['pos'].tolist())
                ori_error_video.append(eval_metrics_video['ori_error'].squeeze(0))
                pos_error_video.append(eval_metrics_video['pos_error'].squeeze(0))
                if 'ori_distance' in video_pose:
                    ori_distance.append(video_pose['ori_distance'])
                if 'pos_distance' in video_pose:
                    pos_distance.append(video_pose['pos_distance'])

            # TQDM print
            running_avg.update(eval_metrics_still, image['torch'].size(0))
            loop.set_postfix(running_avg.get_multiple(keys=('ori_error', 'pos_error', 'esa_score')))

            # DEBUG ORIENTATION OUTLIERS:
            # if video_type is not None:
            #     if eval_metrics_video['ori_error'] > 100:
            #         if prev_pred_ori_soft is not None:
            #             plt.figure(figsize=(45, 40))
            #             plt.subplot(3, 1, 1)
            #             plt.title(f"{split}: ORI PDF comparison at frame {i}. "
            #                       f"Distance = {video_pose['ori_distance']}"
            #                       f"Ori error = {eval_metrics_video['ori_error']}", fontsize=35)
            #             plt.plot(prev_pred_ori_soft, marker='.', color='indianred', linestyle='-', markersize=8,
            #                      label='prev pred pdf')
            #             plt.legend(fontsize=25)
            #             plt.subplot(3, 1, 2)
            #             plt.plot(video_pose['ori_soft'], marker='.', color='royalblue', linestyle='-', markersize=8,
            #                      label='current pred pdf')
            #             plt.legend(fontsize=25)
            #             plt.subplot(3, 1, 3)
            #             plt.plot(true_pose['ori_soft'].squeeze(0), marker='.', color='limegreen', linestyle='-', markersize=8,
            #                      label='current gt pdf')
            #             plt.legend(fontsize=25)
            #             plt.tight_layout()  # Adjust subplots to fit into figure area.
            #             plt.show()
            #             plt.close()
            #     prev_pred_ori_soft = video_pose['ori_soft']

            # SAVE VIDEO
        #     img = copy.deepcopy(image['original'].cpu().numpy().squeeze(0))
        #     video1.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #
        #     pose = {key: value[0] for key, value in true_pose.items()}
        #     img = visu.add_visualization(image['original'].cpu().numpy().squeeze(0), pose, show_true_pose=True)
        #     video2.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #
        #     # pose = {key: value[0] for key, value in still_pose.items()}
        #     img = visu.add_visualization(image['original'].cpu().numpy().squeeze(0), still_pose, show_true_pose=True)
        #     video3.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #
        #     # pose = {key: value[0] for key, value in video_pose.items()}
        #     img = visu.add_visualization(image['original'].cpu().numpy().squeeze(0), video_pose, show_true_pose=True)
        #     video4.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #
        # video1.release()
        # video2.release()
        # video3.release()
        # video4.release()

        # Ensure true and predicted quaternions are in the same pole for visualization purposes
        still_ori = [[-elem for elem in q] if np.dot(true_ori[0], still_ori[0]) < 0 else q for q in still_ori]
        video_ori = [[-elem for elem in q] if np.dot(true_ori[0], video_ori[0]) < 0 else q for q in video_ori]

        # Store evaluation metrics
        still_metrics[split] = {
            "ori_err": compute_statistics(ori_error_still),
            "ori_err_yaw": compute_statistics(
                [abs(euler_angle_difference(quat2euler(true_ori[i])[0], quat2euler(still_ori[i])[0])) for i in
                 range(len(true_ori))]
            ),
            "ori_err_pitch": compute_statistics(
                [abs(euler_angle_difference(quat2euler(true_ori[i])[1], quat2euler(still_ori[i])[1])) for i in
                 range(len(true_ori))]
            ),
            "ori_err_roll": compute_statistics(
                [abs(euler_angle_difference(quat2euler(true_ori[i])[2], quat2euler(still_ori[i])[2])) for i in
                 range(len(true_ori))]
            ),
            "pos_err": compute_statistics(pos_error_still),
            "pos_err_x": compute_statistics([abs(true_pos[i][0] - still_pos[i][0]) for i in range(len(true_pos))]),
            "pos_err_y": compute_statistics([abs(true_pos[i][1] - still_pos[i][1]) for i in range(len(true_pos))]),
            "pos_err_z": compute_statistics([abs(true_pos[i][2] - still_pos[i][2]) for i in range(len(true_pos))]),
        }

        if video_type is not None:
            video_metrics[split] = {
                "ori_err_video": compute_statistics(ori_error_video),
                "ori_err_yaw_video": compute_statistics(
                    [abs(euler_angle_difference(quat2euler(true_ori[i])[0], quat2euler(video_ori[i])[0])) for i in
                     range(len(true_ori))]
                ),
                "ori_err_pitch_video": compute_statistics(
                    [abs(euler_angle_difference(quat2euler(true_ori[i])[1], quat2euler(video_ori[i])[1])) for i in
                     range(len(true_ori))]
                ),
                "ori_err_roll_video": compute_statistics(
                    [abs(euler_angle_difference(quat2euler(true_ori[i])[2], quat2euler(video_ori[i])[2])) for i in
                     range(len(true_ori))]
                ),
                "pos_err_video": compute_statistics(pos_error_video),
                "pos_err_x_video": compute_statistics(
                    [abs(true_pos[i][0] - video_pos[i][0]) for i in range(len(true_pos))]),
                "pos_err_y_video": compute_statistics(
                    [abs(true_pos[i][1] - video_pos[i][1]) for i in range(len(true_pos))]),
                "pos_err_z_video": compute_statistics(
                    [abs(true_pos[i][2] - video_pos[i][2]) for i in range(len(true_pos))]),
            }

            distances[split] = {
                "ori_distance": compute_statistics(ori_distance),
                "pos_distance": compute_statistics(pos_distance),
            }

        # Orientation error
        if ori_distance:
            plt.figure(figsize=(45, 20))
            plt.subplot(2, 1, 1)
            plt.plot(ori_error_still, marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot(ori_error_video, marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            plt.ylabel(ori_error_label[language], fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)

            plt.subplot(2, 1, 2)
            plt.plot(ori_distance, marker='.', color='indianred', linestyle='-', markersize=8)
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = 'orientation distance' if language == 'english' else 'distance sur l\'attitude'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)

        else:
            plt.figure(figsize=(45, 10))
            plt.plot(ori_error_still, marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot(ori_error_video, marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            plt.ylabel(ori_error_label[language], fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'ori_error.png'))
        plt.close()

        # Orientation error per axis (euler)
        plt.figure(figsize=(45, 30))
        for i in range(3):
            plt.subplot(3, 1, i + 1)  # 3 subplots in a row
            angle_diff = [euler_angle_difference(quat2euler(true_ori[j])[i], quat2euler(still_ori[j])[i]) for j in
                          range(len(true_ori))]
            plt.plot(angle_diff, marker='.', color='indianred', linestyle='-', markersize=8,
                 label=labels[language]['still'])
            if video_type is not None:
                angle_diff = [euler_angle_difference(quat2euler(true_ori[j])[i], quat2euler(video_ori[j])[i]) for j in
                              range(len(true_ori))]
                plt.plot(angle_diff, marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = f'{angles[language][i].capitalize()} Error (degrees)' if language == 'english' else \
                f'Erreur de {angles[language][i]} (degrés)'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'ori_error_per_axis.png'))
        plt.close()

        # Orientation error histogram per euler angle
        plt.figure(figsize=(30, 10))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.hist([euler_angle_difference(quat2euler(true_ori[j])[i], quat2euler(still_ori[j])[i]) for j in range(len(true_ori))],
                     bins=36, color='lightcoral', edgecolor='indianred', alpha=0.5, label=labels[language]['still'])
            if video_type is not None:
                plt.hist([euler_angle_difference(quat2euler(true_ori[j])[i], quat2euler(video_ori[j])[i]) for j in range(len(true_ori))],
                         bins=36, color='skyblue', edgecolor='royalblue', alpha=0.5, label=labels[language]['video'])
            plt.legend(fontsize=25)
            xlabel = f'{angles[language][i].capitalize()} Error (degrees)' if language == 'english' else \
                f'Erreur de {angles[language][i]} (degrés)'
            plt.xlabel(xlabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'ori_histogram.png'))
        plt.close()

        # Orientation quaternion elements: pred vs true quat
        plt.figure(figsize=(45, 40))
        for i in range(4):
            plt.subplot(4, 1, i + 1)  # 4 subplots in a row
            plt.plot([x[i] for x in true_ori], marker='.', color='limegreen', linestyle='-', markersize=8,
                     label=labels[language]['true'])
            plt.plot([x[i] for x in still_ori], marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot([x[i] for x in video_ori], marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = f'Quaternion Element {i}' if language == 'english' else f'Élément {i} du quaternion'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'ori_quat_elements.png'))
        plt.close()

        # Orientation euler elements: pred vs true angles
        plt.figure(figsize=(45, 30))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot([quat2euler(x)[i] for x in true_ori], marker='.', color='limegreen', linestyle='-',
                     markersize=8, label=labels[language]['true'])
            plt.plot([quat2euler(x)[i] for x in still_ori], marker='.', color='indianred', linestyle='-',
                     markersize=8, label=labels[language]['still'])
            if video_type is not None:
                plt.plot([quat2euler(x)[i] for x in video_ori], marker='.', color='royalblue', linestyle='-',
                         markersize=8, label='video')
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = f'{angles[language][i]} Angle' if language == 'english' else f'Angle de {angles[language][i]}'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'ori_euler_elements.png'))
        plt.close()

        # Position error
        if pos_distance:
            plt.figure(figsize=(45, 20))
            plt.subplot(2, 1, 1)
            plt.plot(pos_error_still, marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot(pos_error_video, marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            plt.ylabel(pos_error_label[language], fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)

            plt.subplot(2, 1, 2)
            plt.plot(pos_distance, marker='.', color='indianred', linestyle='-', markersize=8)
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = 'Position distance' if language == 'english' else 'distance sur la position'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)

        else:
            plt.figure(figsize=(45, 10))
            plt.plot(pos_error_still, marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot(pos_error_video, marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            plt.ylabel(pos_error_label[language], fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'pos_error.png'))
        plt.close()

        # Position error per axis
        plt.figure(figsize=(45, 30))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot([still_pos[j][i] - true_pos[j][i] for j in range(len(true_pos))],
                     marker='.', color='indianred', linestyle='-', markersize=8, label=labels[language]['still'])
            if video_type is not None:
                plt.plot([video_pos[j][i] - true_pos[j][i] for j in range(len(true_pos))],
                         marker='.', color='royalblue', linestyle='-', markersize=8, label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = f'{positions[i].capitalize()} Axis' if language == 'english' else f'Axe {positions[i]}'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'pos_error_per_axis.png'))
        plt.close()

        # Histogram of position errors
        plt.figure(figsize=(30, 10))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.hist([true_pos[j][i] - still_pos[j][i] for j in range(len(true_pos))],
                     bins=36, color='lightcoral', edgecolor='indianred', alpha=0.5, label=labels[language]['still'])
            if video_type is not None:
                plt.hist([true_pos[j][i] - video_pos[j][i] for j in range(len(true_pos))],
                         bins=36, color='skyblue', edgecolor='royalblue', alpha=0.5, label=labels[language]['video'])
            plt.legend(fontsize=25)
            xlabel = f'{positions[i].capitalize()} Axis Error (meters)' if language == 'english' else \
                f"Erreur axe {positions[i]} (mètres)"
            plt.xlabel(xlabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'pos_histogram.png'))
        plt.close()

        # Position elements: true vs pred
        plt.figure(figsize=(45, 30))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot([x[i] for x in true_pos], marker='.', color='limegreen', linestyle='-', markersize=8,
                     label=labels[language]['true'])
            plt.plot([x[i] for x in still_pos], marker='.', color='indianred', linestyle='-', markersize=8,
                     label=labels[language]['still'])
            if video_type is not None:
                plt.plot([x[i] for x in video_pos], marker='.', color='royalblue', linestyle='-', markersize=8,
                         label=labels[language]['video'])
            plt.xlabel(index_image[language], fontsize=25)
            ylabel = f'{positions[i]} axis' if language == 'english' else f'Axe {positions[i]}'
            plt.ylabel(ylabel, fontsize=25)
            plt.yticks(fontsize=21)
            plt.xticks(fontsize=21)
            plt.legend(fontsize=25)
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.savefig(os.path.join(base_dir_save, f'pos_elements.png'))
        plt.close()

    with pd.ExcelWriter(os.path.join(exp_path, 'still_metrics.xlsx'), engine='xlsxwriter') as writer:
        for split in list(still_metrics.keys()):
            pd.DataFrame(data=still_metrics[split], index=["min", "max", "median", "mean",
                                                    "std"]).to_excel(writer, sheet_name=f'{split}')

    with pd.ExcelWriter(os.path.join(exp_path, 'video_metrics.xlsx'), engine='xlsxwriter') as writer:
        for split in list(video_metrics.keys()):
            pd.DataFrame(data=video_metrics[split], index=["min", "max", "median", "mean",
                                                    "std"]).to_excel(writer, sheet_name=f'{split}')

    with pd.ExcelWriter(os.path.join(exp_path, 'distances.xlsx'), engine='xlsxwriter') as writer:
        for split in list(distances.keys()):
            pd.DataFrame(data=distances[split], index=["min", "max", "median", "mean",
                                                "std"]).to_excel(writer, sheet_name=f'{split}')


if __name__ == "__main__":
    main(
        model_path="models/mursop_fp32_dspeed",
        video_type='Adaptative',
        language='english'
    )

    # main(
    #     model_path="models/mursop_fp32_dspeed",
    #     video_type='Adaptative',
    #     language='french'
    # )
