"""
Copyright (c) 2025 Julien Posso
"""

import os
import time
import pandas as pd
from typing import Dict

import src.config.train.config as train_cfg
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.tools.utils import save_score_error
from src.tools.evaluation import evaluation
from src.boards.boards_cfg import import_board
from src.nvidia.spe_nvidia import SPEJetson
from src.nvidia.ssh_deploy import JetsonSSHDeployer
from src.nvidia.jetson_deploy import deploy_to_jetson


def save_latency(latency: Dict[str, list], path: str, name: str = "latency.xlsx") -> None:
    """
    Save latency measurements to an Excel (xlsx) file.

    Args:
        latency (Dict[str, list]): Dictionary containing latency values with keys as model names.
        path (str): Directory path where the file will be saved.
        name (str, optional): Filename for the Excel file. Defaults to "latency.xlsx".

    Returns:
        None
    """
    writer = pd.ExcelWriter(os.path.join(path, name), engine='xlsxwriter')
    pd.DataFrame(data=latency).to_excel(writer, sheet_name='latency')
    writer.close()


def main():
    """
    Deploys a spacecraft pose estimation model on an NVIDIA Jetson board,
    handles remote execution via SSH, and evaluates model performance.

    The process includes:
    - Selecting and loading an experiment
    - Deploying files and starting an inference server on the Jetson board
    - Running evaluations and measuring performance (latency, throughput)
    - Closing connections and cleaning up after execution
    """

    # Select and validate experiment
    experiment_name = input('Select the experiment name you want to deploy on a board: ')
    experiment_path = os.path.join('experiments', 'build', 'nvidia', experiment_name)
    print(f'Loading experiment {experiment_path}')
    assert os.path.exists(experiment_path), f'Path {experiment_path} does not exist.'

    # Load experiment configuration
    cfg_fp32 = train_cfg.load_config(os.path.join(experiment_path, 'config_fp32.yaml'))
    camera = load_camera(cfg_fp32.DATA.PATH)
    spe_utils = SPEUtils(
        camera, cfg_fp32.MODEL.HEAD.ORI, cfg_fp32.MODEL.HEAD.N_ORI_BINS_PER_DIM, cfg_fp32.DATA.ORI_SMOOTH_FACTOR,
        cfg_fp32.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, cfg_fp32.MODEL.HEAD.POS, cfg_fp32.MODEL.HEAD.N_POS_BINS_PER_DIM,
        cfg_fp32.DATA.POS_SMOOTH_FACTOR, cfg_fp32.MODEL.HEAD.KEYPOINTS_PATH
    )

    # Load dataset
    batch_size = 1  # Inference is usually done with batch size 1
    data, split = load_dataset(
        spe_utils, cfg_fp32.DATA.PATH, batch_size, cfg_fp32.DATA.IMG_SIZE,
        rot_augment=False, other_augment=False, shuffle=False
    )

    # --- Deploy model and inference server on Jetson ---
    # Send image size to Jetson for TensorRT compilation
    img, _ = next(iter(data[split['eval'][0]]))
    img = img['torch']

    spe_jetson, ssh_deploy = deploy_to_jetson(
        experiment_path,
        spe_utils,
        tuple(img.size())
    )

    # --- Evaluation ---
    print("Starting evaluation...")
    score, error = evaluation(spe_jetson, data, spe_utils, split['eval'])
    save_score_error(
        score, error, path=os.path.join(experiment_path, 'on_board'),
        name=f'eval.xlsx'
    )

    # --- Throughput Test ---
    print('Starting throughput test...')
    _, lat = spe_jetson.predict(img, num_predict=1000)
    latency = {'jetson_orin_nano_int8': [lat]}
    save_latency(latency, path=os.path.join(experiment_path, 'on_board'),
                 name=f'latency_ms.xlsx')
    print(f"Nvidia Jetson Orin Nano = {lat:.2f} ms")

    # Measure power consumption
    # _, lat = spe_jetson.predict(img, num_predict=500000)

    # Close connections and cleanup
    spe_jetson.close()
    for i in range(3):
        print(f"Waiting for execution on Jetson to complete... {i+1}/3")
        time.sleep(1)
    ssh_deploy.close_ssh_thread()


if __name__ == "__main__":
    main()
