# FINN Installation and FPGA Deployment

This README explains how to install and use **FINN** to build FPGA accelerators (`build_finn.py`) and deploy them on FPGA boards (`deploy_finn.py`).  
We provide instructions for both the **host machine (Docker)** and the **target FPGA-SoC board (Xilinx ZCU104 with Pynq)**.

---

## 1. Installing FINN on the Host (Docker)

We recommend using the provided Dockerfile to build an image including FINN.  
This method is the easiest and ensures reproducibility. It is based on the official [FINN installation guide](https://finn.readthedocs.io/en/latest/getting_started.html).

> **Note:** In order to run FINN end-to-end and reproduce our results, you also need to install **[Vitis Core Development Kit - 2022.1 - Full Product Installation](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/archive-vitis.html)** (with the default installation options). Other versions may work but are not guaranteed to provide identical results.

### 1.1. Build the Docker image
Run this command at the root of the cloned repository:
```bash
docker build setup/finn -t pose_finn:latest
```

#### Build Arguments

| Argument        | Default Value                                                     | Description                                                                                              |
|-----------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `VITIS_VERSION` | `2022.1`                                                          | Version of Vitis to be used. Must match the version installed outside the container.                     |
| `XILINX_PATH`   | `/tools/Xilinx`                                                   | Directory path inside the container where Vitis is installed. Must match the host path.                  |
| `FINN_PATH`     | `/tools/finn`                                                     | Directory path inside the container where FINN will be installed.                                        |
| `PROJECT_ROOT`  | `/workspace/pose_estimation/Spacecraft-Pose-Estimation-Framework` | Directory path inside the container where the project will be mounted.                                   |
| `FINN_VERSION`  | `v0.9`                                                            | FINN release version from GitHub to be used in the project.                                              |

### 1.2. Run the Docker container
```bash
docker run --rm -it --net=host --hostname=finn_pose \
  -v /tools/Xilinx:/tools/Xilinx \
  -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation \
  pose_finn:latest
```

- The Xilinx install directory must be the same on the host and in the container (e.g. `/tools/Xilinx`).  
- Replace `$POSE_ESTIMATION_ROOT` with the path containing both the `Spacecraft-Pose-Estimation-Framework` project and the dataset.  
- The container starts at the root of the cloned repository and executes `finn_entrypoint.sh`.

### 1.3. Verify Installation
Inside the container:
```bash
pip install -e git+https://github.com/fbcotter/dataset_loading.git@0.0.4#egg=dataset_loading
bash /tools/finn/docker/quicktest.sh
```
Expected output: ~1285 passed, 256 skipped, 4 xfailed, 1 xpassed.

---

## 2. Installing FINN on the Target FPGA-SoC (Xilinx ZCU104)

### 2.1. Install Pynq on the Board
Follow the [Pynq getting started guide](https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html).  
To reproduce our results, use **Pynq 2.6.0** for the ZCU104.  
Download images from the [Pynq GitHub releases](https://github.com/Xilinx/PYNQ/releases).  

We recommend flashing the MicroSD card with [Balena Etcher](https://www.balena.io/etcher/).  
If the AppImage does not start, run with `--no-sandbox`:
```bash
./balenaEtcher.AppImage --no-sandbox
```

### 2.2. First-Time Setup

1. Insert the SD card into the ZCU104.  
2. Configure DIP switches as shown below.  
3. Follow the network setup instructions.  

![ZCU104 configuration](https://pynq.readthedocs.io/en/v2.7.0/_images/zcu104_setup.png)

#### 2.2.1. Network Configuration
- **Recommended:** connect the ZCU104 to the same switch/router as your PC → automatic internet access.  
- If it fails, assign a static IP on your PC (see [Ubuntu guide](https://linuxconfig.org/how-to-configure-static-ip-address-on-ubuntu-18-10-cosmic-cuttlefish-linux)).  
- If your organization uses MAC filtering, use the alternative setup: connect ZCU104 directly to your PC, assign a static IP address to your computer, and enable *“shared to other computers”* mode via `nm-connection-editor`. Reboot both your computer and the ZCU104. 

Default IP of the board: **192.168.2.99**  

Test connection:
```bash
ssh xilinx@192.168.2.99
ping 8.8.8.8
```

#### 2.2.2. Passwordless SSH
FINN requires passwordless SSH. The Dockerfile generates an RSA public/private key pair in the `/root/.ssh` folder of the container.  

Inside the Docker container:
```bash
ssh-copy-id -i /root/.ssh/id_rsa.pub xilinx@192.168.2.99
```

Test:
```bash
ssh xilinx@192.168.2.99
```

#### 2.2.3. Install Required Packages on ZCU104
On the board:
```bash
sudo pip3 install bitstring
```

---

## 3. Build and Deploy the Neural Network Accelerator

### 3.1. Build the Accelerator
- Adjust the configuration files in the `src/config/build/finn` directory as needed.  
- By default, build outputs and FPGA accelerator are stored in `experiments/build/finn`.
- FINN intermediate checkpoints are saved in the `finn_build` directory.

```bash
python build_finn.py
```

### 3.2. Deploy on the FPGA
Before deployment, edit the `ZCU104` class in `src/boards/board_cfg.py` to match your board configuration.

```bash
python deploy_finn.py
```
When prompted, select the output folder generated during the build step (located in `experiments/build/finn`).
