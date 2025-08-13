"""
Copyright (c) 2025 Julien Posso
"""

"""
Inference server running on the Jetson.
- Robust I/O protocol using length-prefixed messages (4-byte big-endian header).
- Socket is bound/listening BEFORE heavy imports to avoid "Connection refused" races.
- Torch/TensorRT imports, model load, and compilation happen AFTER accept() (lazy import) so the client can connect early.
- Network timeouts + keepalive + always sending a response (client never hangs waiting for data).
"""

import os
import argparse
import socket
import struct
import pickle
import time

# =====================
#  Framing utilities
# =====================
def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from the socket (blocking until complete or closed)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Peer closed connection while reading.")
        buf += chunk
    return bytes(buf)

def recv_msg(conn: socket.socket) -> bytes:
    """Read a length-prefixed message (4-byte size header, then payload)."""
    hdr = recv_exact(conn, 4)
    (length,) = struct.unpack("!I", hdr)
    if length == 0:
        return b""
    return recv_exact(conn, length)

def send_msg(conn: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed message (4-byte size header, then payload)."""
    conn.sendall(struct.pack("!I", len(payload)))
    if payload:
        conn.sendall(payload)

# =====================
#  Main server routine
# =====================
def main(bind_host: str, port: int) -> None:
    # 1) Bind/listen immediately (before heavy imports)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, port))
    srv.listen(1)
    print(f"[server] Listening on {bind_host or '0.0.0.0'}:{port}", flush=True)

    conn, addr = srv.accept()
    try:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except Exception:
        pass
    conn.settimeout(120)  # reasonable timeout to avoid deadlocks
    print(f"[server] Client connected: {addr}", flush=True)

    # 2) Receive image size BEFORE heavy imports (validates connection and protocol)
    try:
        img_size = pickle.loads(recv_msg(conn))  # expected: tuple (N, C, H, W)
        print(f"[server] Received image size: {img_size}", flush=True)
    except Exception as e:
        print(f"[server] Error reading image size: {e}", flush=True)
        try:
            send_msg(conn, pickle.dumps({"error": f"image_size: {e}"}))
        except Exception:
            pass
        conn.close()
        return

    # 3) Heavy imports AFTER accept()
    print("[server] Importing torch / torch_tensorrt ...", flush=True)
    import torch
    import torch_tensorrt

    # 4) Load + compile Torch-TensorRT model
    try:
        print("[server] Loading model...", flush=True)
        model = torch.jit.load("jit_model.pt").eval().to("cuda:0")

        print("[server] Compiling with Torch-TensorRT...", flush=True)
        compile_spec = {
            "inputs": [torch_tensorrt.Input(img_size)],
            "enabled_precisions": torch.int8,  # adjust if needed
            "ir": "ts",
        }
        model = torch_tensorrt.compile(model, **compile_spec)
        torch.jit.save(model, "model_int8_trt_jetson.ts")
        print("[server] Ready.", flush=True)

        # Send ready signal to client
        send_msg(conn, b"<SERVER_READY>")
    except Exception as e:
        print(f"[server] Model/compile error: {e}", flush=True)
        try:
            send_msg(conn, pickle.dumps({"error": f"compile: {e}"}))
        except Exception:
            pass
        conn.close()
        return

    # 5) Inference loop
    while True:
        try:
            payload = recv_msg(conn)

            # Termination command (control message, not pickled)
            if payload == b"TERMINATE":
                print("[server] TERMINATE received", flush=True)
                try:
                    send_msg(conn, b"<TERMINATED>")
                finally:
                    break

            # Inference request: pickled dict {"image": tensor, "num_predict": int}
            req = pickle.loads(payload)
            image = req["image"].to("cuda:0")
            num_predict = int(req.get("num_predict", 1))

            # Optional warm-up for throughput testing
            if num_predict > 100:
                for _ in range(60):
                    _ = model(image)

            total = 0.0
            out = None
            with torch.no_grad():
                for _ in range(num_predict):
                    t0 = time.time()
                    out = model(image)
                    total += (time.time() - t0)

            avg_ms = (total / max(num_predict, 1)) * 1000.0
            send_msg(conn, pickle.dumps((out, avg_ms)))

        except socket.timeout:
            # Never let the client hang — send an error frame
            send_msg(conn, pickle.dumps({"error": "timeout"}))
        except Exception as e:
            # Always send back an error frame on exception
            send_msg(conn, pickle.dumps({"error": str(e)}))

    try:
        conn.close()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="", help="Bind address (default: all interfaces).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("JETSON_PORT", "50009")),
                        help="TCP listen port.")
    args = parser.parse_args()
    main(args.host, args.port)

