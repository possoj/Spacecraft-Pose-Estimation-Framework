"""
Copyright (c) 2025 Julien Posso
"""

from dataclasses import dataclass
from typing import Any, Tuple

from src.boards.boards_cfg import import_board
from src.nvidia.ssh_deploy import JetsonSSHDeployer
from src.nvidia.spe_nvidia import SPEJetson


@dataclass
class JetsonDeployConfig:
    """
    Configuration structure for Jetson deployment.
    Modify the attributes below if you need to change default paths or ports.
    """
    spe_folder: str = "spacecraft_pose_estimation"
    docker_img: str = "torch_tensorrt:r35.3.1"
    inference_server: str = "jetson_inference_server.py"
    model_filename: str = "jit_model.pt"
    port: int = 50009
    board_name: str = "Jetson"


def deploy_to_jetson(
        experiment_path: str,
        spe_utils: Any,
        img_shape: Tuple[int, int, int, int],
        cfg: JetsonDeployConfig = JetsonDeployConfig(),
) -> Tuple[SPEJetson, JetsonSSHDeployer]:
    """
    Deploys a model and inference server on a Jetson device via SSH.

    This function performs the following steps:
    1. Creates the remote folder on the Jetson.
    2. Uploads the TorchScript model and the inference server script.
    3. Starts the Docker container running the inference server.
    4. Sends image shape information to the server for TensorRT compilation.
    5. Waits for the Jetson server to signal readiness.

    Parameters
    ----------
    experiment_path : str
        Path to the local experiment folder containing the model subfolder.
    spe_utils : Any
        Utility object needed to initialize SPEJetson.
    img_shape : Tuple[int, int, int, int]
        Shape of the input image tensor (e.g., from `img.shape` in PyTorch).
    cfg : JetsonDeployConfig, optional
        Configuration object to customize paths, ports, filenames, etc.

    Returns
    -------
    Tuple[SPEJetson, JetsonSSHDeployer]
        - The initialized `SPEJetson` object (for inference).
        - The active `JetsonSSHDeployer` object (for optional teardown/logs).
    """

    # Load the board configuration
    board = import_board(cfg.board_name)

    # Set up SSH deployer
    ssh = JetsonSSHDeployer(board, cfg.spe_folder)

    # 1. Create the remote folder on the Jetson device
    ssh.create_remote_directory(f"/home/{board.username}/{cfg.spe_folder}")

    # 2. Upload the model file
    print(f"Uploading {cfg.model_filename} to Jetson...", end="")
    ssh.upload_file(
        f"{experiment_path}/model/{cfg.model_filename}",
        f"/home/{board.username}/{cfg.spe_folder}/{cfg.model_filename}",
    )
    print("OK")

    # Upload the inference server script
    print(f"Uploading {cfg.inference_server} to Jetson...", end="")
    ssh.upload_file(
        f"src/nvidia/{cfg.inference_server}",
        f"/home/{board.username}/{cfg.spe_folder}/{cfg.inference_server}",
    )
    print("OK")

    # 3. Start the Docker-based inference server on the Jetson
    ssh.start_remote_server(cfg.docker_img, cfg.inference_server, cfg.port)

    # 4. Set up the socket client (SPEJetson)
    # spe_jetson = SPEJetson(spe_utils, board, inference_port=cfg.port)
    spe_jetson = SPEJetson(spe_utils, board, inference_port=cfg.port, img_size=img_shape)

    # Send image size to Jetson so it can compile with the correct input dimensions
    # spe_jetson.send_data_to_server(list(img_shape), b"<IMAGE_SIZE>")

    # Wait for confirmation that the inference server is ready
    # spe_jetson.wait_for_server(b"<SERVER_READY>")

    # Return objects for further use (inference, closing, logs, etc.)
    return spe_jetson, ssh
