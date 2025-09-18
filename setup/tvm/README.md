# TVM Installation and CPU Deployment

This README explains how to install and use **TVM** to build optimized CPU accelerators (`build_tvm.py`) and deploy (`deploy_tvm.py`) them on embedded boards (e.g. **Xilinx ZCU104**, **Ultra96**, **Raspberry Pi**).  
We provide instructions for both the **host machine (Docker)** and the **target CPU board (Linux/Pynq)**.

---

## 1. Installing PyTorch and TVM on the Host (Docker)

We recommend using the provided Dockerfile to build an image including PyTorch and TVM.  
This ensures reproducibility and simplifies cross-compilation for target boards.

### 1.1. Build the Docker image
Run this command at the root of the cloned repository:
```bash
docker build setup/tvm -t pose_tvm:latest
```

### 1.2. Run the Docker container
```bash
docker run --rm -it --gpus=all --ipc=host --net=host --hostname=tvm_pose \
  -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation \
  pose_tvm:latest
```

- Replace `$POSE_ESTIMATION_ROOT` with the path containing both the `Spacecraft-Pose-Estimation-Framework` project and the dataset.  
- The container opens a bash shell with all dependencies pre-installed.

---

## 2. Installing TVM on Target Boards

### 2.1. Installing the Operating System

#### 2.1.1. ZCU104 (Pynq 2.6.0)
Follow the same procedure as described in the section 2.1 of the FINN setup guide (setup/finn/README.md).  
Use the **Pynq 2.6.0** release for ZCU104 from the [Xilinx Pynq GitHub releases](https://github.com/Xilinx/PYNQ/releases).

#### 2.1.2. Ultra96 (Pynq 2.5)
Download **Pynq 2.5** for Ultra96 from the [Avnet Ultra96 Pynq releases](https://github.com/Avnet/Ultra96-PYNQ/releases).  
Follow the [Ultra96 getting started guide](https://ultra96-pynq.readthedocs.io/en/latest/getting_started.html).

#### 2.1.3. Raspberry Pi (Ubuntu Server 22.04)
Download **Ubuntu Server 22.04** for Raspberry Pi from [Ubuntu’s website](https://ubuntu.com/download/raspberry-pi).  
Flash the image to the MicroSD card, insert into the Raspberry Pi, and boot.  
Install the packages needed to build and run the TVM runtime (interned access needed):
```bash
sudo apt install build-essential
sudo apt install python3-pip
```

### 2.2. Installing the TVM Runtime (Common to All Boards)

Once the board is running its OS and you can access it via SSH, the TVM runtime installation steps are the same for all targets. We recommend transferring the sources using a mount point via **sshfs**, and checking out the same commit hash as used in the Dockerfile.

1. On your host computer, create a mount point and clone TVM into the board’s filesystem:
```bash
mkdir mount_point
sshfs USERNAME@BOARD_IP:/home/USERNAME mount_point 
cd mount_point
git clone --no-checkout --depth 1 https://github.com/apache/tvm.git tvm
cd tvm
git fetch --depth 1 origin TVM_COMMIT_HASH
git checkout TVM_COMMIT_HASH
git submodule update --init --recursive
cd ../..
sudo umount mount_point
rm -rf mount_point
```
- Replace `USERNAME` with the user of the board (`xilinx` for Pynq, `ubuntu` for Raspberry Pi).  
- Replace `BOARD_IP` with the IP address of the board.  
- Replace `TVM_COMMIT_HASH` with the same commit hash used in your Docker build (e.g. `e7f793d0ad5f141444fff41d308be17231ec6b86`).

2. Connect to the board via SSH:
```bash
ssh USERNAME@BOARD_IP
```

3. Build the TVM runtime on the board:
```bash
cd /home/USERNAME/tvm
mkdir build && cp cmake/config.cmake build
cd build
cmake ..
make runtime
```

4. Install dependencies on the board (optional):
```bash
pip install cloudpickle
```

---

## 3. Build and Deploy the Neural Network with TVM

### 3.1. Build the TVM Graph
Adjust the configuration files in the `src/config/build/tvm` directory as needed.  
By default, build outputs are stored in `experiments/build/tvm`.

```bash
python build_tvm.py
```

### 3.2. Deploy on the CPU Board
Before deployment, edit the `board_cfg.py` file in `src/boards/` to match your target configuration (ZCU104, Ultra96).  

```bash
python deploy_tvm.py
```

When prompted, select the output folder generated during the build step (located in `experiments/build/tvm`).

The deployment script automatically starts the TVM RPC server to handle board–host communication and model deployment.
