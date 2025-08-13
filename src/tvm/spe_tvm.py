"""
Copyright (c) 2025 Julien Posso
"""

import os
from typing import Any, Tuple

from tvm import auto_scheduler
from tvm.contrib import utils, graph_executor


class SPETVMARM:
    """
    Spacecraft Pose Estimation with TVM.

    This class connects to a remote board to upload the TVM runtime module and provides
    a method to predict the pose of a spacecraft using the loaded model.

    Attributes:
        input_name (str): Name of the input tensor.
        spe_utils (Any): Utilities for post-processing predictions (e.g., activation functions, decoding).
        board (Any): Board configuration object with attributes such as 'name' and 'port'.
        host_ip (str): IP address of the host for the RPC connection.
        graph (graph_executor.GraphModule): Graph executor module loaded on the remote board.
        device (Any): Device context (e.g., CPU) obtained from the remote board.
    """

    def __init__(self, lib_path: str, input_name: str, board: Any, host_ip: str, spe_utils: Any) -> None:
        """
        Initialize the SPETVMARM instance by connecting to the remote board and uploading the TVM runtime module.

        Args:
            lib_path (str): Path to the compiled TVM library.
            input_name (str): Name of the input tensor.
            board (Any): Board configuration object.
            host_ip (str): Host IP address for the RPC connection.
            spe_utils (Any): Utilities for post-processing predictions.
        """
        self.input_name = input_name
        self.spe_utils = spe_utils
        self.board = board
        self.host_ip = host_ip
        self.graph, self.device = self.connect_remote(lib_path)

    def predict(self, image: Any, num_predict: int = 1) -> Tuple[Any, float]:
        """
        Predict the pose of a target spacecraft.

        Expects a PyTorch image tensor with shape NCHW.

        Args:
            image (Any): Input image as a PyTorch tensor.
            num_predict (int, optional): Number of runs for inference to measure average latency. Defaults to 1.

        Returns:
            Tuple[Any, float]: A tuple containing:
                - The predicted pose (post-processed output).
                - The mean latency in milliseconds.
        """
        # Convert the PyTorch tensor to a NumPy array.
        image = image.numpy()

        # Set the input tensor.
        self.graph.set_input(self.input_name, image)

        # Run inference and measure latency.
        timer = self.graph.module.time_evaluator("run", self.device, number=num_predict)
        timing = timer()

        # Retrieve outputs.
        ori = self.graph.get_output(0)
        pos = self.graph.get_output(1)

        # Post-processing: determine keys based on the mode.
        ori_key = 'ori_soft' if self.spe_utils.ori_mode == 'classification' else 'ori'
        pos_key = 'pos_soft' if self.spe_utils.pos_mode == 'classification' else 'pos'
        pose = {ori_key: ori.numpy(), pos_key: pos.numpy()}

        # Apply final activation and decode to get the predicted pose.
        pose = self.spe_utils.last_activ(pose)
        pred_pose = self.spe_utils.decode(pose)

        # Convert latency to milliseconds.
        mean_latency_ms = timing.mean * 1000

        return pred_pose, mean_latency_ms

    def connect_remote(self, lib_path: str) -> Tuple[graph_executor.GraphModule, Any]:
        """
        Connect to the remote board via RPC and upload the TVM runtime module.

        Args:
            lib_path (str): Path to the compiled TVM library file.

        Returns:
            Tuple[graph_executor.GraphModule, Any]: A tuple containing:
                - The graph executor module.
                - The device context from the remote board.
        """
        # Connect to the RPC server on the board.
        remote = auto_scheduler.utils.request_remote(
            self.board.name.lower(), self.host_ip, self.board.port, timeout=10000
        )
        print("Successful connection to the remote")

        # Upload the compiled library.
        remote.upload(lib_path)
        remote_lib = remote.load_module(os.path.basename(lib_path))
        print("Successful module upload")

        # Retrieve the device context (e.g., CPU) and create a GraphModule.
        device = remote.cpu()
        graph = graph_executor.GraphModule(remote_lib["default"](device))
        print("Done getting graphModule from remote_lib")
        return graph, device
