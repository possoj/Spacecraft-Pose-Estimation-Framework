"""
Copyright (c) 2025 Julien Posso
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.spe.classification_utils import OrientationSoftClassification, PositionSoftClassification
from src.spe.utils import euler2quat, quat2euler, multiply_quaternions, euler_angle_difference, generate_orientation


def soft_class_experiment(
        n_bins: list,
        smooth_factors: list,
        pose_range: dict,
        step: int = 1,
        pose_type: str = 'position',
        save_path: str = None,
        delete_unused_bins=True,
):
    assert pose_type in ['position', 'orientation']

    trans = {
        'position': {'z': 'z', 'y': 'y', 'x': 'x'},
        'orientation': {'z': 'yaw', 'y': 'pitch', 'x': 'roll'}
    }

    z_values = np.arange(pose_range[trans[pose_type]['z']][0], pose_range[trans[pose_type]['z']][1] + step, step)
    y_values = np.arange(pose_range[trans[pose_type]['y']][0], pose_range[trans[pose_type]['y']][1] + step, step)
    x_values = np.arange(pose_range[trans[pose_type]['x']][0], pose_range[trans[pose_type]['x']][1] + step, step)

    errors = np.zeros((len(n_bins), len(smooth_factors), len(z_values), len(y_values), len(x_values)))

    def calculate_errors(i, j, k, z):
        if pose_type == 'orientation':
            # Whatever the value of delete_unused_bins it has not impact on this experience
            soft_class = OrientationSoftClassification(n_bins[i], smooth_factors[j], delete_unused_bins=delete_unused_bins)
        else:
            soft_class = PositionSoftClassification(
                n_bins[i], smooth_factors[j],
                np.array([values[0] for values in pose_range.values()]),
                np.array([values[1] for values in pose_range.values()])
            )
        for l, y in enumerate(y_values):
            for m, x in enumerate(x_values):
                pose_original = np.array([x, y, z]) if pose_type == 'position' else euler2quat(z, y, x,
                                                                                               gymbal_check=False)
                pose_soft = soft_class.encode(pose_original)
                pose_decoded = soft_class.decode(pose_soft)
                if pose_type == 'position':
                    errors[i, j, k, l, m] = np.linalg.norm(pose_original - pose_decoded)
                else:
                    dot_product = np.clip(np.sum(pose_decoded[0] * pose_original), -1.0, 1.0)
                    errors[i, j, k, l, m] = 2 * np.arccos(np.abs(dot_product)) * 180 / np.pi

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(n_bins)):
            for j in range(len(smooth_factors)):
                for k, z in enumerate(z_values):
                    futures.append(executor.submit(calculate_errors, i, j, k, z))

        for future in tqdm(as_completed(futures), total=len(futures), desc='Calculating errors'):
            future.result()

    if save_path is not None:
        np.save(save_path, errors)

    return errors


def filter_errors_within_usable_range(errors, pose_range, usable_range, step):
    # Create masks for the usable ranges
    z_mask = (np.arange(pose_range['z'][0], pose_range['z'][1] + step, step) >= usable_range['z'][0]) & \
             (np.arange(pose_range['z'][0], pose_range['z'][1] + step, step) <= usable_range['z'][1])
    y_mask = (np.arange(pose_range['y'][0], pose_range['y'][1] + step, step) >= usable_range['y'][0]) & \
             (np.arange(pose_range['y'][0], pose_range['y'][1] + step, step) <= usable_range['y'][1])
    x_mask = (np.arange(pose_range['x'][0], pose_range['x'][1] + step, step) >= usable_range['x'][0]) & \
             (np.arange(pose_range['x'][0], pose_range['x'][1] + step, step) <= usable_range['x'][1])

    # Apply masks to filter the errors array
    filtered_errors = errors[:, :, z_mask, :, :][:, :, :, y_mask, :][:, :, :, :, x_mask]

    return filtered_errors


def calculate_error_statistics(errors):
    # Calculate min, max, mean, standard deviation, and median errors across the last three dimensions (positions)
    error_statistics = {
        'min': np.min(errors, axis=(2, 3, 4)),
        'max': np.max(errors, axis=(2, 3, 4)),
        'mean': np.mean(errors, axis=(2, 3, 4)),
        'std': np.std(errors, axis=(2, 3, 4)),
        'median': np.median(errors, axis=(2, 3, 4)),
    }

    return error_statistics


def plot_heatmap_stats(
        errors,
        n_bins,
        smooth_factors,
        error_type='mean',
        pose_type='position',
        pos_range=None,
        figsize=(10, 8),
        fontsize=16,
        language='fr',
        title=True,
        save_path=None,
):
    assert error_type in ['mean', 'median', 'min', 'max', 'std'], \
        "Invalid error_type. Must be 'mean', 'median', 'min', 'max', or 'std'."
    assert pose_type in ['position', 'orientation']
    if pose_type == 'position':
        assert pos_range in ['full', 'usable']
    assert language in ['fr', 'en']

    # Define units and labels in both languages
    units = {
        'en': {'position': 'meters', 'orientation': 'degrees'},
        'fr': {'position': 'm', 'orientation': '°'}
    }

    error_print = {
        'fr': {'mean': 'moyenne', 'median': 'médiane', 'min': 'minimale', 'max': 'maximale', 'std': '(écart-type)'},
        'en': {'mean': 'mean', 'median': 'median', 'min': 'minimum', 'max': 'maximum', 'std': 'standard'},
    }

    err_type = 'angulaire' if pose_type == 'orientation' else 'de position'  # For french

    labels = {
        'en': {
            'xlabel': 'Smoothing factor',
            'ylabel': 'Number of bins per dimension',
            'title': f'{error_print[language][error_type].capitalize()} {pose_type} error ({units[language][pose_type]})',
            'range_note': ' on position range'
        },
        'fr': {
            'xlabel': 'Facteurs de lissage',
            'ylabel': 'Nombre de classes par angle d\'Euler' if pose_type == 'orientation' else 'Nombre de classes par '
                                                                                                'axe de position',
            # 'title': f'Erreur {err_type} {error_print[language][error_type]} ({units[language][pose_type]})',
            'title': f'Erreur {err_type} ({units[language][pose_type]})',
            'range_note': ' sur la plage de position'
        }
    }

    desc = labels[language]['title']

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        errors, annot=True, fmt=".2f", cmap='viridis',
        xticklabels=smooth_factors, yticklabels=n_bins,
        # cbar_kws={'label': desc.capitalize(), 'fontsize': fontsize}
        annot_kws={"size": int(fontsize - 4)}
    )

    colorbar = ax.collections[0].colorbar
    colorbar.set_label(desc.capitalize(), fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=int(fontsize - 2))  # Font size for colorbar ticks

    plt.xlabel(labels[language]['xlabel'], fontsize=fontsize)
    plt.ylabel(labels[language]['ylabel'], fontsize=fontsize)
    plt.xticks(fontsize=int(fontsize-2))
    plt.yticks(fontsize=int(fontsize-2))

    if pos_range and pose_type == 'position':
        desc += labels[language]['range_note']

    desc += ' vs. number of bins per dimension and smoothing factor'
    if title:
        plt.title(desc.capitalize(), fontsize=fontsize)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_errors_for_smooth_factors(
        errors: np.ndarray,
        smooth_factors: list,
        n_bins: list,
        selected_n_bins: int,
        pose_range: dict,
        step_range: int,
        axis: str,
        pose_type: str = 'position',
        pos_range: str = None,
        figsize: tuple = (10, 8),
        save_path: str = None,
):
    assert pose_type in ['position', 'orientation']
    if pose_type == 'position':
        assert axis in ['x', 'y', 'z']
    else:
        assert axis in ['yaw', 'pitch', 'roll']

    if pos_range:
        assert pos_range in ['full', 'usable']

    unit = {'position': 'meters', 'orientation': 'degrees'}

    plt.figure(figsize=figsize)

    if pose_type == 'position':
        trans = {'z': 'z', 'y': 'y', 'x': 'x'}
    else:
        trans = {'yaw': 'z', 'pitch': 'y', 'roll': 'x'}

    pose_range = {trans[key]: list(range(value[0], value[1] + step_range, step_range))
                  for key, value in pose_range.items()}

    other_axes_val = {
        'position': {'x': 0, 'y': 0, 'z': 15},  # x, y, z position
        'orientation': {'z': 0, 'y': 0, 'x': 0}  # Yaw, pitch roll rotation
    }

    if axis in ['z', 'yaw']:
        filtered_error = errors[
                         n_bins.index(selected_n_bins), :, :,
                         pose_range['y'].index(other_axes_val[pose_type]['y']),
                         pose_range['x'].index(other_axes_val[pose_type]['x'])
                         ]
    elif axis in ['y', 'pitch']:
        filtered_error = errors[
                         n_bins.index(selected_n_bins), :, pose_range['z'].index(other_axes_val[pose_type]['z']), :,
                         pose_range['x'].index(other_axes_val[pose_type]['x'])
                         ]
    else:
        filtered_error = errors[
                         n_bins.index(selected_n_bins), :, pose_range['z'].index(other_axes_val[pose_type]['z']),
                         pose_range['y'].index(other_axes_val[pose_type]['y']), :
                         ]

    for sf, error in zip(smooth_factors, filtered_error):
        plt.plot(pose_range[trans[axis]], error, label=f'smoothing factor: {sf}')

    plt.xlabel(f'{axis} axis'.capitalize())
    plt.ylabel(f'{pose_type} error ({unit[pose_type]})'.capitalize())

    other_axes_val[pose_type].pop(trans[axis])
    title = (f'{pose_type} error vs {axis} axis \nwith {selected_n_bins} bins and '
             f'{other_axes_val[pose_type]}')
    if pos_range and pose_type == 'position':
        title += f'\non {pos_range} position range'
    plt.title(title.capitalize())
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_heatmap_for_selected_values(
        errors: np.ndarray,
        n_bins: list,
        smooth_factors: list,
        selected_n_bins: int,
        selected_smooth_factor: int,
        selected_z: int,
        pose_range: dict,
        step_range: int,
        pose_type: str = 'position',
        pos_range: str = None,
        figsize: tuple = (10, 8),
        fontsize: int = 16,
        language: str = 'en',
        save_path: str = None,
):
    assert pose_type in ['position', 'orientation']
    assert language in ['en', 'fr']

    plt.figure(figsize=figsize)

    if pose_type == 'position':
        assert pos_range is not None
        x_values = np.arange(pose_range['x'][0], pose_range['x'][1] + step_range, step_range)
        y_values = np.arange(pose_range['y'][0], pose_range['y'][1] + step_range, step_range)
        z_index = np.where(np.arange(pose_range['z'][0], pose_range['z'][1] + step_range, step_range) == selected_z)[0][
            0]
        heatmap_data = errors[
                       n_bins.index(selected_n_bins), smooth_factors.index(selected_smooth_factor), z_index, :, :
                       ]

        xlabel = 'x axis' if language == 'en' else 'Axe x'
        ylabel = 'y axis' if language == 'en' else 'Axe y'
        title = f'Position error heatmap at z = {selected_z} on {pos_range} range' if language == 'en' else \
            f"Carte thermique de l'erreur de position à z = {selected_z} sur la plage {pos_range}"
        colorbar_label = 'Error' if language == 'en' else 'Erreur'

    else:
        x_values = np.arange(pose_range['yaw'][0], pose_range['yaw'][1] + step_range, step_range)
        y_values = np.arange(pose_range['roll'][0], pose_range['roll'][1] + step_range, step_range)
        pitch_index = np.where(np.arange(pose_range['pitch'][0], pose_range['pitch'][1] + step_range, step_range) ==
                               selected_z)[0][0]
        heatmap_data = errors[
                       n_bins.index(selected_n_bins), smooth_factors.index(selected_smooth_factor), :, pitch_index, :
                       ]

        xlabel = 'yaw axis' if language == 'en' else 'Axe de lacet (yaw)'
        ylabel = 'roll axis' if language == 'en' else 'Axe de roulis (roll)'
        title = f'Orientation error heatmap at pitch = {selected_z}' if language == 'en' else \
            f"Carte thermique de l'erreur d'orientation à tangage = {selected_z}"
        colorbar_label = 'Orientation error (degrees)' if language == 'en' else 'Erreur d\'orientation (degrés)'

    plt.figure(figsize=figsize)
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis', xticklabels=x_values, yticklabels=y_values)
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(colorbar_label, fontsize=fontsize)
    colorbar.ax.tick_params(labelsize=int(fontsize-2))
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=int(fontsize-2))
    plt.yticks(fontsize=int(fontsize-2))
    # plt.legend(fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_3d_scatter_for_selected_values(
        errors: np.ndarray,
        n_bins: list,
        smooth_factors: list,
        selected_n_bins: int,
        selected_smooth_factor: int,
        pose_range: dict,
        step_range: int,
        pose_type: str = 'position',
        pos_range: str = None,
):
    assert pose_type in ['position', 'orientation']

    trans = {'position': ['x', 'y', 'z'], 'orientation': ['yaw', 'pitch', 'roll']}
    axis_labels = trans[pose_type]

    x_values = np.arange(pose_range[axis_labels[0]][0], pose_range[axis_labels[0]][1] + step_range, step_range)
    y_values = np.arange(pose_range[axis_labels[1]][0], pose_range[axis_labels[1]][1] + step_range, step_range)
    z_values = np.arange(pose_range[axis_labels[2]][0], pose_range[axis_labels[2]][1] + step_range, step_range)

    selected_n_bins_index = n_bins.index(selected_n_bins)
    selected_smooth_factor_index = smooth_factors.index(selected_smooth_factor)

    error_data = errors[selected_n_bins_index, selected_smooth_factor_index, :, :, :]

    Z, Y, X = np.meshgrid(z_values, y_values, x_values, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    error_flat = error_data.flatten()

    fig = go.Figure(data=[go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode='markers',
        marker=dict(
            size=3,
            color=error_flat,
            colorscale='Viridis',
            colorbar=dict(title=f'{pose_type.capitalize()} Error'),
            opacity=0.8
        )
    )])

    title = f'3D Scatter Plot of {pose_type.capitalize()} Error<br>' \
            f'n_bins={selected_n_bins}, smoothing_factor={selected_smooth_factor}'
    if pose_type == 'position':
        title += f'\non {pos_range} position range'

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2]
        )
    )

    fig.show()


# def angle_difference(angle1, angle2, range_min, range_max):
#     """Calculate the minimum difference between two angles in a specified range."""
#     range_width = range_max - range_min
#     diff = (angle1 - angle2 + range_width / 2) % range_width - range_width / 2
#     return diff


def create_unit_quaternion(set_value, index):
    if set_value < -1.0 or set_value > 1.0:
        raise ValueError("The set value must be in the range [-1, 1]")

    quaternion = np.array([0.0, 0.0, 0.0, 0.0])
    quaternion[index] = set_value

    remaining_sum = 1.0 - set_value ** 2
    if remaining_sum < 0.0:
        raise ValueError("The set value is too large to create a unit quaternion")

    remaining_value = np.sqrt(remaining_sum)

    for i in range(4):
        if i != index:
            quaternion[i] = remaining_value / np.sqrt(3)  # Distributing equally among the remaining three elements

    return quaternion


def main(load_ori: bool = False, load_pos: bool = False, language: str = 'en') -> None:

    assert language in ('en', 'fr')

    ###############################################################
    # ORIENTATION TEMPORAL EXPERIMENT #############################
    ###############################################################

    soft_class = OrientationSoftClassification(12, 3, delete_unused_bins=True)
    offset_per_s = np.array([6, 3, 6])
    offset_per_frame = offset_per_s / 25  # D-SEED dataset contains 25 FPS videos
    q = euler2quat(0, 0, 0)
    q_velocity = euler2quat(*offset_per_frame)

    ori_distances = []
    for i in range(1500):
        q_new = multiply_quaternions(q, q_velocity)
        pdf1 = soft_class.encode(q)
        pdf2 = soft_class.encode(q_new)
        ori_distances.append(np.linalg.norm(pdf1 - pdf2))
        q = q_new


    print(f"Orientation distance between consecutive PDFs:\n"
          f"min: {np.min(ori_distances):.6f}\n"
          f"max: {np.max(ori_distances):.6f}\n"
          f"mean: {np.mean(ori_distances):.6f}\n"
          f"median: {np.median(ori_distances):.6f}\n"
          f"std: {np.std(ori_distances):.6f}")

    ###############################################################
    # POSITION TEMPORAL EXPERIMENT ################################
    ###############################################################

    position_range = {
        'min': np.array([-16, -12, -2]),
        'max': np.array([16, 12, 40]),
    }
    position_range_usable = {
        'min': np.array([-11, -7, 3]),
        'max': np.array([11, 7, 35]),
    }
    soft_class = PositionSoftClassification(10, 100,
                                            position_range['min'], position_range['max'])
    # Position and velocity of TIT scenario (the one with maximum speed)
    pos = np.array([-11, -7, 35])
    # pos_velocity = np.array([0.12, 0.08, -0.4]) / 25  # D-SEED dataset contains 25 FPS videos
    pos_velocity = np.array([0.925, 0, 0]) / 25  # D-SEED dataset contains 25 FPS videos
    # pos_velocity = np.array([0.92, 0, -0.4]) / 25  # D-SEED dataset contains 25 FPS videos

    pos_distances = []
    for i in range(1500):
        pos_new = pos + pos_velocity
        pdf1 = soft_class.encode(pos)
        pdf2 = soft_class.encode(pos_new)
        pos_distances.append(np.linalg.norm(pdf1 - pdf2))
        pos = pos_new
        if np.any(pos < position_range_usable['min']) or np.any(pos > position_range_usable['max']):
            print("Break before end of experiment. Position outside")
            break

    print(f"Position distance between consecutive PDFs:\n"
          f"min: {np.min(pos_distances):.6f}\n"
          f"max: {np.max(pos_distances):.6f}\n"
          f"mean: {np.mean(pos_distances):.6f}\n"
          f"median: {np.median(pos_distances):.6f}\n"
          f"std: {np.std(pos_distances):.6f}")

    save_dir = 'experiments/soft_classification/'
    fig_size = (10, 8)
    fontsize = 23

    ###############################################################
    # ORIENTATION STILL EXPERIMENT ################################
    ###############################################################

    # POSITION EXPERIMENTS
    ori_range = {
        "yaw": [-180, 180],
        "pitch": [-90, 90],
        "roll": [-180, 180],
    }
    step = 10

    # Variables to test (EXP 1 and 2)
    ori_bins_per_dim = [10, 11, 12, 13, 14]
    ori_smoothing_factors = [2, 3, 4, 5, 6]

    # Selected values for EXP 2, 3 and 4
    selected_n_bins = 12
    selected_smooth_factor = 3

    # RUN/LOAD Orientation experiment
    if load_ori:
        orientation_errors = np.load(os.path.join(save_dir, 'orientation', 'raw_data_errors.npy'))
    else:
        orientation_errors = soft_class_experiment(
            ori_bins_per_dim,
            ori_smoothing_factors,
            ori_range,
            step=step,
            pose_type='orientation',
            save_path=os.path.join(save_dir, 'orientation', 'raw_data_errors.npy'),
            delete_unused_bins=False,
        )

    # Compute error stats: min, max, mean, std, median
    ori_error_stats = calculate_error_statistics(orientation_errors)

    # EXP 1: HEATMAP ORI ERROR VS. SMOOTH FACTOR AND NUMBER OF BINS
    for error_type, error_values in ori_error_stats.items():
        plot_heatmap_stats(
            error_values, ori_bins_per_dim, ori_smoothing_factors, error_type, 'orientation', ori_range,
            fig_size, fontsize, language,
            title=False,
            save_path=os.path.join(save_dir, 'orientation', f'heatmap_{error_type}.png')
        )

    # EXP 2: PLOT ORI YAW PITCH ROLL FOR VARIOUS SMOOTH FACTOR WITH SELECTED AMOUNT OF BINS
    for axis in ('yaw', 'pitch', 'roll'):
        plot_errors_for_smooth_factors(
            orientation_errors, ori_smoothing_factors, ori_bins_per_dim, selected_n_bins, ori_range, step, axis,
            'orientation',
            save_path=os.path.join(save_dir, 'orientation', f'error_vs_sf_on_{axis}_axis.png')
        )

    # EXP 3: HEATMAP ORI ERROR
    selected_z = 0
    plot_heatmap_for_selected_values(
        orientation_errors, ori_bins_per_dim, ori_smoothing_factors, selected_n_bins, selected_smooth_factor,
        selected_z, ori_range, step, 'orientation', 'full',
        tuple(x * 2 for x in fig_size), fontsize, language,
        save_path=os.path.join(save_dir, 'orientation', f'heatmap_vs_xy_full_{selected_z}.png')

    )
    selected_z = 40
    plot_heatmap_for_selected_values(
        orientation_errors, ori_bins_per_dim, ori_smoothing_factors, selected_n_bins, selected_smooth_factor,
        selected_z, ori_range, step, 'orientation', 'full',
        tuple(x * 2 for x in fig_size), fontsize, language,
        save_path=os.path.join(save_dir, 'orientation', f'heatmap_vs_xy_full_{selected_z}.png')

    )

    # EXP 4: 3D interactive plot
    plot_3d_scatter_for_selected_values(
        orientation_errors, ori_bins_per_dim, ori_smoothing_factors, selected_n_bins, selected_smooth_factor,
        ori_range, step, pose_type='orientation', pos_range='full'
    )

    # EXP5: Orientation error histogram
    n_q = 10000
    sf_list = [2.5, 3, 3.5]
    error = np.zeros((len(sf_list), n_q))
    yaw_error = np.zeros((len(sf_list), n_q))
    pitch_error = np.zeros((len(sf_list), n_q))
    roll_error = np.zeros((len(sf_list), n_q))
    q_list = generate_orientation(n_q)

    for i, sf in enumerate(sf_list):
        soft_class = OrientationSoftClassification(selected_n_bins, sf)

        for j, q in enumerate(q_list):
            soft_ori = soft_class.encode(q)
            qd, _ = soft_class.decode(soft_ori)
            dot_product = np.clip(np.sum(qd * q), -1.0, 1.0)
            error[i, j] = 2 * np.arccos(np.abs(dot_product)) * 180 / np.pi

            yaw_d, pitch_d, roll_d = quat2euler(qd)
            yaw, pitch, roll = quat2euler(q)
            yaw_error[i, j] = euler_angle_difference(yaw_d, yaw)
            pitch_error[i, j] = euler_angle_difference(pitch_d, pitch)
            roll_error[i, j] = euler_angle_difference(roll_d, roll)

    for error_type, error_name in zip((error, yaw_error, pitch_error, roll_error),
                                      ('total error', 'yaw error', 'pitch error', 'roll error')):
        plt.figure(figsize=(10, 8))
        for i, sf in enumerate(sf_list):
            plt.hist(error_type[i], bins=36, alpha=0.5, label=f'sf {sf}')
            plt.legend()
        plt.xlabel(f'{error_name} error (degrees)', fontsize=24)
        # plt.xticks([-180, -90, 0, 90, 180], fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid()
        plt.show()
        plt.close()

    ###############################################################
    # POSITION EXPERIMENT #########################################
    ###############################################################

    # POSITION EXPERIMENTS
    # 6 meters margin
    # full_pos_range = {
    #     "x": [-17, 17],
    #     "y": [-13, 13],
    #     "z": [-3, 41],
    # }
    # We choose 5 meters margin
    full_pos_range = {
        "x": [-16, 16],
        "y": [-12, 12],
        "z": [-2, 40],
    }
    usable_pos_range = {
        "x": [-11, 11],  # On dspeed dataset: x_min = -10.49, x_max = 10.43
        "y": [-7, 7],  # On dspeed dataset: y_min = -6.72, y_max = 6.63
        "z": [3, 35],  # On dspeed dataset, z_min = 3, z_max = 35 (and only 16 positions where z > 35 on speed dataset)
    }

    step = 1

    # Variables to test (EXP 1 and 2)
    pos_bins_per_dim = [8, 9, 10, 11, 12]
    pos_smoothing_factors = [80, 90, 100, 110, 120]

    # Selected values for EXP 2, 3 and 4
    selected_n_bins = 10
    selected_smooth_factor = 100

    # RUN/LOAD Position experiment
    # Will generate a lot of "impossible" positions where the satellite is outside the image which increases the
    # average/median/std position error. The maximum and minimum position error are a more suitable metric for this
    # experiment
    if load_pos:
        full_position_errors = np.load(os.path.join(save_dir, 'position', 'raw_data_errors.npy'))
    else:
        full_position_errors = soft_class_experiment(
            pos_bins_per_dim,
            pos_smoothing_factors,
            full_pos_range,
            step=step,
            pose_type='position',
            save_path=os.path.join(save_dir, 'position', 'raw_data_errors.npy')
        )
    usable_position_errors = filter_errors_within_usable_range(full_position_errors, full_pos_range, usable_pos_range, step)

    # Compute error stats: min, max, mean, std, median
    full_pos_error_stats = calculate_error_statistics(full_position_errors)
    usable_pos_error_stats = calculate_error_statistics(usable_position_errors)

    # EXP 1: HEATMAP POS ERROR VS. SMOOTH FACTOR AND NUMBER OF BINS
    for error_stats, position_range in zip((full_pos_error_stats, usable_pos_error_stats), ('full', 'usable')):
        for error_type, error_values in error_stats.items():
            plot_heatmap_stats(
                error_values, pos_bins_per_dim, pos_smoothing_factors, error_type, 'position', position_range,
                fig_size, fontsize, language,
                title=False,
                save_path=os.path.join(save_dir, 'position', f'heatmap_{position_range}_{error_type}.png')
            )

    # EXP 2: PLOT POS X, Y AND Z POSITION FOR VARIOUS SMOOTH FACTOR WITH SELECTED AMOUNT OF BINS
    for axis in ('x', 'y', 'z'):
        plot_errors_for_smooth_factors(
            full_position_errors, pos_smoothing_factors, pos_bins_per_dim, selected_n_bins, full_pos_range, step, axis,
            'position', 'full',
            save_path=os.path.join(save_dir, 'position', f'error_vs_sf_on_{axis}_axis.png')
        )
    for axis in ('x', 'y', 'z'):
        plot_errors_for_smooth_factors(
            usable_position_errors, pos_smoothing_factors, pos_bins_per_dim, selected_n_bins, usable_pos_range, step,
            axis, 'position', 'usable',
            save_path=os.path.join(save_dir, 'position', f'error_vs_sf_usable_on_{axis}_axis.png')
        )

    # EXP 3: HEATMAP POS ERROR VS POSITION OF SATELLITE IN IMAGE AT FIXED DISTANCE
    selected_z = 15

    # Need to increase the size of the fig
    plot_heatmap_for_selected_values(
        full_position_errors, pos_bins_per_dim, pos_smoothing_factors, selected_n_bins, selected_smooth_factor,
        selected_z, full_pos_range, step, 'position', 'full',
        tuple(x * 2 for x in fig_size), fontsize, language,
        save_path=os.path.join(save_dir, 'position', f'heatmap_vs_xy_full_{selected_z}.png')
    )

    plot_heatmap_for_selected_values(
        usable_position_errors, pos_bins_per_dim, pos_smoothing_factors, selected_n_bins, selected_smooth_factor,
        selected_z, usable_pos_range, step, 'position', 'full',
        tuple(x * 2 for x in fig_size), fontsize, language,
        save_path=os.path.join(save_dir, 'position', f'heatmap_vs_xy_usable_{selected_z}.png')
    )

    # EXP 4: 3D interactive plot
    plot_3d_scatter_for_selected_values(
        full_position_errors, pos_bins_per_dim, pos_smoothing_factors, selected_n_bins, selected_smooth_factor,
        full_pos_range, step, pose_type='position', pos_range='full'
    )

    plot_3d_scatter_for_selected_values(
        usable_position_errors, pos_bins_per_dim, pos_smoothing_factors, selected_n_bins, selected_smooth_factor,
        usable_pos_range, step, pose_type='position', pos_range='usable'
    )


if __name__ == '__main__':

    # main(load_ori=False, load_pos=False, language='en')

    # load orientation and position errors to accelerate plots
    main(load_ori=True, load_pos=True, language='fr')
