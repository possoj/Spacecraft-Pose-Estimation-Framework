"""
Copyright (c) 2025 Julien Posso
"""

import os
import pandas as pd
from typing import Any, Dict, Tuple

import src.config.train.config as train_cfg
import src.config.build.tvm.config as tvm_cfg
from src.data.import_dataset import load_dataset, load_camera
from src.spe.spe_utils import SPEUtils
from src.tools.utils import set_seed
from src.tools.utils import save_score_error
from src.tools.evaluation import evaluation
from src.tvm.rpc_handler import RPCHandler
from src.boards.boards_cfg import import_board
from src.tvm.spe_tvm import SPETVMARM


def save_latency(latency: Dict, path: str, name: str = "latency.xlsx") -> None:
    """
    Save latency information to an Excel (xlsx) file.

    Args:
        latency (Dict[Any, Any]): Dictionary containing latency data.
        path (str, optional): Directory path where the file will be saved. Defaults to "../results/".
        name (str, optional): Filename for the Excel file. Defaults to "latency.xlsx".

    Returns:
        None
    """
    writer = pd.ExcelWriter(os.path.join(path, name), engine='xlsxwriter')
    pd.DataFrame(data=latency).to_excel(writer, sheet_name='latency')
    writer.save()


def main():

    experiment_name = input('Select the experiment name you want to deploy on a board: ')
    experiment_path = os.path.join('experiments', 'build', 'tvm', experiment_name)
    print(f'Loading experiment {experiment_path}')
    assert os.path.exists(experiment_path), f'path {experiment_path} does not exists'

    # Load TVM configuration: QAT and compilation
    cfg_fp32 = train_cfg.load_config(os.path.join(experiment_path, 'config_fp32.yaml'))
    cfg = tvm_cfg.load_config(os.path.join(experiment_path, 'config.yaml'))

    camera = load_camera(cfg_fp32.DATA.PATH)
    spe_utils = SPEUtils(
        camera, cfg_fp32.MODEL.HEAD.ORI, cfg_fp32.MODEL.HEAD.N_ORI_BINS_PER_DIM, cfg_fp32.DATA.ORI_SMOOTH_FACTOR,
        cfg_fp32.MODEL.HEAD.ORI_DELETE_UNUSED_BINS, cfg_fp32.MODEL.HEAD.POS, cfg_fp32.MODEL.HEAD.N_POS_BINS_PER_DIM,
        cfg_fp32.DATA.POS_SMOOTH_FACTOR, cfg_fp32.MODEL.HEAD.KEYPOINTS_PATH
    )

    seed = 1001
    set_seed(seed)

    # Load data for evaluation and TorchScript compilation
    data, split = load_dataset(spe_utils, cfg_fp32.DATA.PATH, 1, cfg_fp32.DATA.IMG_SIZE,
                               False, False, False)

    board = import_board(cfg.COMPILE.BOARD)

    rpc_handler = RPCHandler(board, start_tracker=True, start_ssh=True, print_ssh=True)
    host_ip = rpc_handler.get_host_ip()

    input_name = "input0"
    spe_tvm = SPETVMARM(os.path.join(experiment_path, 'model', 'lib.tar'), input_name, board, host_ip, spe_utils)

    print("Evaluation on board")
    score, error = evaluation(spe_tvm, data, spe_utils, split['eval'])
    save_score_error(
        score, error, path=os.path.join(experiment_path, 'on_board'),
        name=f'eval_{cfg.COMPILE.BOARD}.xlsx'
    )

    print('Throughput test....')
    # Get example image
    img, _ = next(iter(data[split['eval'][0]]))
    img = img['torch']
    _, lat = spe_tvm.predict(img, num_predict=1000)
    latency = {'tvm_arm_int8': [lat]}
    save_latency(latency, path=os.path.join(experiment_path, 'on_board'),
                 name=f'arm_tvm_{cfg.COMPILE.BOARD}_latency_ms.xlsx')
    print(f"TVM ARM latency = {lat:.2f} ms")

    # Close SSH session
    rpc_handler.close_ssh_thread()


if __name__ == "__main__":
    main()
