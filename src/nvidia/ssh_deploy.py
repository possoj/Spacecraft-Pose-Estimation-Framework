"""
Copyright (c) 2025 Julien Posso
"""

import paramiko
import time
from threading import Thread

from src.boards.boards_cfg import Jetson


class JetsonSSHDeployer:
    """
    Manages SSH-based deployment on a Jetson board, including directory creation,
    file transfers, and launching the remote inference server.

    This class ensures that SSH connections are properly handled using context managers.
    """

    def __init__(self, board: Jetson, deployment_folder: str):
        """
        Initializes the SSH deployer for the Jetson board.

        Args:
            board (Jetson): Board configuration containing:
                - ip (str): Jetson's IP address.
                - port (int): SSH port.
                - username (str): SSH username.
                - password (str): SSH password.
            deployment_folder (str): Directory on the Jetson board where the model will be deployed.
        """
        self.board = board
        self.deployment_folder = deployment_folder
        self.execute_ssh_thread = False
        self.ssh_thread = None

    def create_remote_directory(self, remote_dir: str) -> None:
        """
        Creates a directory on the Jetson board if it does not already exist.

        Args:
            remote_dir (str): Path to the directory to be created on the Jetson board.
        """
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.board.ip,
                port=self.board.port,
                username=self.board.username,
                password=self.board.password
            )
            ssh.exec_command(f"mkdir -p {remote_dir}")

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Uploads a file from the host machine to the Jetson board.

        Args:
            local_path (str): Path to the file on the host machine.
            remote_path (str): Destination path on the Jetson board.
        """
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.board.ip,
                port=self.board.port,
                username=self.board.username,
                password=self.board.password
            )
            with ssh.open_sftp() as sftp:
                sftp.put(local_path, remote_path)

    def start_remote_server_thread(self, docker_image: str, server_script: str, port: int) -> None:
        """
        Starts the inference server on the Jetson board inside a Docker container.

        Args:
            docker_image (str): Name of the Docker image used for running the script.
            server_script (str): Filename of the script to execute inside the container.
            port (int): Jetson TCP port to listen on. Must match the port used by the client when connecting.

        The method assumes usage of jetson-containers with an appropriately tagged image (e.g., torch_tensorrt).
        """

        docker_cmd = (
            "jetson-containers run "
            f"-v /home/{self.board.username}/{self.deployment_folder}:/workspace/{self.deployment_folder} "
            f"-w /workspace/{self.deployment_folder} "
            f"dustynv/{docker_image} python3 -u {server_script} --port {port}"
        )

        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.board.ip,
                port=self.board.port,
                username=self.board.username,
                password=self.board.password
            )

            # Enable a pseudo-terminal to capture real-time output
            stdin, stdout, stderr = ssh.exec_command(docker_cmd, get_pty=True)
            green_begin = "\033[92m"
            green_end = "\033[0m"

            while self.execute_ssh_thread:
                if stdout.channel.recv_ready():
                    # Read output line-by-line in a non-blocking manner
                    line = stdout.readline()
                    if line:
                        print(f"{green_begin}{line}{green_end}", end="", flush=True)
                time.sleep(0.25)

    def start_remote_server(self, docker_image: str, server_script: str, port: int) -> None:
        """
        Initiates the inference server execution in a separate thread.

        Args:
            docker_image (str): Name of the Docker image for execution.
            server_script (str): The script to execute inside the container.
            port (int): TCP port to listen on. Must match the port used by the client.
        """
        self.execute_ssh_thread = True
        self.ssh_thread = Thread(target=self.start_remote_server_thread, args=(docker_image, server_script, port,))
        self.ssh_thread.start()

    def close_ssh_thread(self) -> None:
        """
        Gracefully stops the SSH thread that is running the inference server.
        """
        self.execute_ssh_thread = False
        if self.ssh_thread is not None:
            self.ssh_thread.join()
        print("Stopped SSH thread")
