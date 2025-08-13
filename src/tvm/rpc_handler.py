"""
Copyright (c) 2025 Julien Posso
"""

import time
import subprocess
import warnings
from threading import Thread
from typing import Any, Optional

from paramiko import SSHClient, AutoAddPolicy


class RPCHandler:
    """
    Handles the setup of an RPC tracker on the host computer and maintains an SSH connection
    to a remote board to register the RPC tracker.

    Attributes:
        board (Any): An object representing the remote board. It must have the attributes:
            - ip (str): IP address of the board.
            - port (int or str): Port for the RPC tracker.
            - username (str): SSH username.
            - password (str): SSH password.
            - name (str): Name identifier for the board.
        print_ssh (bool): If True, prints the SSH output in green text.
        execute_ssh_thread (bool): Control flag for the SSH thread loop.
        host_ip (Optional[str]): The detected IP address of the host computer.
        ssh_thread (Optional[Thread]): Thread handling the remote SSH connection.
    """

    def __init__(self, board: Any, start_tracker: bool = True, start_ssh: bool = True, print_ssh: bool = False) -> None:
        """
        Initializes the RPCHandler by optionally starting the host RPC tracker,
        recovering the host IP, and starting an SSH thread to register the RPC tracker
        on the remote board.

        Args:
            board (Any): The board object with necessary attributes.
            start_tracker (bool, optional): Whether to start the RPC tracker on the host. Defaults to True.
            start_ssh (bool, optional): Whether to start the SSH thread to connect to the board. Defaults to True.
            print_ssh (bool, optional): If True, prints SSH command output. Defaults to False.
        """
        self.board = board
        self.print_ssh = print_ssh
        self.execute_ssh_thread = True
        self.ssh_thread: Optional[Thread] = None

        if start_tracker:
            self.start_host_tracker()

        self.host_ip = self.recover_host_ip()

        if start_ssh:
            self.ssh_thread = Thread(target=self.ssh_remote_thread)
            self.ssh_thread.start()

    def recover_host_ip(self) -> Optional[str]:
        """
        Automatically recovers the IP address of the host computer.

        It executes the hostname command to list IP addresses, then selects the IP that
        matches the board's IP range.

        Returns:
            Optional[str]: The detected host IP address or None if not found.
        """
        ip_string = subprocess.check_output(["hostname", "-I"], text=True)
        ip_list = ip_string.split()

        ip_address = []
        for ip in ip_list:
            # Maks Host RPC IP address. Works for mask 255.255.255.0
            ip_masked = '.'.join(ip.split('.')[:-1])
            # Looking for IPs in the same range as the board's IP address.
            if ip_masked in self.board.ip:
                ip_address.append(ip)

        if not ip_address:
            warnings.warn("IP not found. Please check the RPC tracker on the host, "
                          "unless you want to compile without auto-scheduling")
            return None
        elif len(ip_address) != 1:
            ip_address[0] = input(f"IP found multiple times ({ip_string}). \nPlease enter manually:")

        return ip_address[0]

    def get_host_ip(self) -> Optional[str]:
        """
        Retrieves the host IP address.

        Returns:
            Optional[str]: The host IP address.
        """
        return self.host_ip

    def start_host_tracker(self) -> None:
        """
        Starts the RPC tracker on the host computer using a subprocess.

        Raises:
            ValueError: If the subprocess returns a non-zero exit code.
        """
        cmd = f"python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port={self.board.port} &"
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise ValueError(f"Error while starting the RPC tracker on the computer. "
                             f"Return code = {proc.returncode}")

    def ssh_remote_thread(self) -> None:
        """
        Connects via SSH to the remote board and registers the RPC tracker running on the host.

        This method runs in its own thread and continuously checks for new output from the remote
        command. It exits gracefully when `self.execute_ssh_thread` is set to False.
        """
        cmd = (
            f"cd tvm/python; "
            f"python3 -m tvm.exec.rpc_server --tracker={self.host_ip}:{self.board.port} "
            f"--key={self.board.name.lower()}"
        )

        with SSHClient() as ssh:
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(
                self.board.ip,
                username=self.board.username,
                password=self.board.password,
                banner_timeout=20
            )

            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)

            # ANSI escape sequences for green text:
            begin = '\033[92m'
            end = '\033[0m'

            while self.execute_ssh_thread:
                if self.print_ssh and stdout.channel.recv_ready():
                    # Read line-by-line in a non-blocking way.
                    line = stdout.readline()
                    if line:
                        print(f"{begin}{line}{end}", end="")
                time.sleep(1)

    def close_ssh_thread(self) -> None:
        """
        Signals the SSH thread to stop and waits for it to terminate gracefully.
        """
        self.execute_ssh_thread = False
        if self.ssh_thread is not None:
            self.ssh_thread.join()
        print("Stopped SSH thread")
