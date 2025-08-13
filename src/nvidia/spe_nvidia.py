"""
Copyright (c) 2025 Julien Posso
"""

"""
Inference client running on the host.
- Robust I/O protocol using length-prefixed messages (4-byte big-endian header).
- Connect with retries, keepalive, and timeouts.
- Sends image size on startup, waits for READY signal from the server.
- No legacy tag-based fallbacks: only clean length-prefixed framing.
"""

import socket
import struct
import pickle
import time
import torch
from typing import Any, Tuple

# =====================
#  Framing utilities
# =====================
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from the socket (blocking until complete or closed)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server closed connection while reading.")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> bytes:
    """Read a length-prefixed message (4-byte size header, then payload)."""
    hdr = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", hdr)
    if length == 0:
        return b""
    return recv_exact(sock, length)


def send_msg(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed message (4-byte size header, then payload)."""
    sock.sendall(struct.pack("!I", len(payload)))
    if payload:
        sock.sendall(payload)


# =====================
#  Jetson client class
# =====================
class SPEJetson:
    """
    Manages an inference session with the Jetson over a persistent TCP socket.
    Assumes the inference server is already running on the Jetson (e.g. in a container with --network host).
    """

    def __init__(self, spe_utils, board: Any, inference_port: int = 50009,
                 img_size: Tuple[int, int, int, int] = (1, 3, 240, 384)):
        """
        Args:
            spe_utils: utility object for post-processing (last_activ, decode, modes...).
            board: object with at least `ip` (Jetson address).
            inference_port: server TCP port.
            img_size: input size (N, C, H, W) sent to the server.
        """
        self.spe_utils = spe_utils
        self.board = board
        self.inference_port = inference_port
        self.sock = self._connect_with_retry(sleep_s=1, max_attempts=60)  # 1 min window

        # Configure socket (timeouts + keepalive)
        self._configure_socket(self.sock)

        # Send image size and wait for READY
        send_msg(self.sock, pickle.dumps(img_size))
        ready = recv_msg(self.sock)
        if ready != b"<SERVER_READY>":
            raise RuntimeError(f"Server did not confirm READY. Received: {ready[:64]!r}")

    def _configure_socket(self, sock: socket.socket) -> None:
        """Apply a timeout and enable keepalive for robustness."""
        sock.settimeout(120)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass

    def _connect_with_retry(self, sleep_s: int, max_attempts: int) -> socket.socket:
        """Try to connect until success or max attempts exhausted."""
        last = None
        for i in range(1, max_attempts + 1):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.board.ip, self.inference_port))
                print("Connected to Jetson inference server!")
                return s
            except Exception as e:
                last = e
                print(f"Attempt {i}/{max_attempts}: connection failed ({e}). Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
        raise ConnectionError(f"Unable to connect to Jetson server after {max_attempts} attempts: {last}")

    def predict(self, image: torch.Tensor, num_predict: int = 1) -> Tuple[Any, float]:
        """
        Send an image tensor and number of inference repetitions.
        Returns (decoded pose, average inference time in ms).
        """
        # 1) Send request (pickle a dict)
        payload = {'image': image, 'num_predict': int(num_predict)}
        send_msg(self.sock, pickle.dumps(payload))

        # 2) Receive response (always length-prefixed)
        result_data = recv_msg(self.sock)
        msg = pickle.loads(result_data)

        # Server returns either (pred, avg_ms) or {"error": "..."}
        if isinstance(msg, dict) and "error" in msg:
            raise RuntimeError(f"Jetson server error: {msg['error']}")

        pred, avg_inference_time = msg  # avg_inference_time in ms

        # 3) Post-process according to modes
        if self.spe_utils.ori_mode == 'keypoints' and self.spe_utils.pos_mode == 'keypoints':
            pose = {'keypoints': pred.detach().cpu().numpy()}
        else:
            ori_key = 'ori_soft' if self.spe_utils.ori_mode == 'classification' else 'ori'
            pos_key = 'pos_soft' if self.spe_utils.pos_mode == 'classification' else 'pos'
            pose = {
                ori_key: pred[0].detach().cpu().numpy(),
                pos_key: pred[1].detach().cpu().numpy(),
            }
        pose = self.spe_utils.last_activ(pose)
        pose = self.spe_utils.decode(pose)

        return pose, float(avg_inference_time)

    def close(self) -> None:
        """Send termination command and close the socket cleanly."""
        try:
            send_msg(self.sock, b"TERMINATE")
            _ack = recv_msg(self.sock)  # expected: b"<TERMINATED>"
        except Exception:
            pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
