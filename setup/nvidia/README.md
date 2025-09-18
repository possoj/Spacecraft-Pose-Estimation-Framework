# Nvidia Deployment with PyTorch and TensorRT

This README explains how to install and use **PyTorch** and **TensorRT** for GPU deployment, both on the **host machine (Docker)** and the **Nvidia Jetson target (Orin Nano)**.

---

## 1. Installing PyTorch and TensorRT on the Host (Docker)

### 1.1. Build the Docker image
From the project root folder:
```bash
docker build setup/nvidia -t pose_nvidia:latest
```

### 1.2. Run the Docker container
```bash
docker run --rm -it --gpus=all --ipc=host --net=host \
  --entrypoint=bash --hostname=nvidia_pose \
  -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation \
  pose_nvidia:latest
```

- Replace `$POSE_ESTIMATION_ROOT` with the path containing both the `Spacecraft-Pose-Estimation-Framework` project and the dataset.  
- The container opens a bash shell with all dependencies pre-installed.

---

## 2. Installing PyTorch and TensorRT on the Nvidia Jetson Target

### 2.1. Install JetPack on the Target
Follow the [Jetson Orin Nano Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#intro) (**do not upgrade firmware to R36**).  
To reproduce our results, install **JetPack 5.1.3** from the [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) (L4T 35.5.0).  

The first boot requires a monitor, keyboard, and mouse.  

If you need a more advanced install (e.g. downgrading firmware), refer to:
1. [Download R35 Jetson Linux and File System](https://forums.developer.nvidia.com/t/downgrade-from-jetpack-6-to-jetpack-5/294256/3)
2. [Set the board in recovery mode](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_Nano/Jetpack_5.X/Cmd_Flash)
3. [Connect the Jetson to the host with USB-C](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html#usb-c-port-4)
4. Insert a micro-SD card (≥128 GB) and power on the board.
5. [Flash the Jetson Orin Nano (SD card)](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html#)

### 2.2. Network Configuration
Choose one of the following setups:

#### 2.2.1. Configuration 1: DHCP (Recommended)
Connect the Jetson to the same network (wired or wireless) as your computer and let DHCP assign the address automatically.

#### 2.2.2. Configuration 2: Direct Connection with Shared Internet
If your organization uses MAC filtering and the Jetson cannot access the internet:
- Connect the Jetson directly to your PC with an ethernet cable (or through a switch).  
- On the host PC, run:
  ```bash
  nm-connection-editor
  ```
  Double-click the ethernet interface for the Jetson. In the **IPv4** tab, set the method to **Shared to other computers**.  
- Assign a static IP address to your PC (e.g. `192.168.2.10`, mask `255.255.255.0`). Avoid using the Jetson’s IP (e.g. `192.168.2.185`).
- Reboot both the PC and the Jetson.  
- To discover devices on the local network, use:
  ```bash
  nmap -sn 192.168.2.0/24
  ```

Test connection:
```bash
ssh jetson@192.168.2.185
ping 8.8.8.8
```

#### 2.2.3. Configuration 3: Local Ethernet + Wi-Fi
- Connect both the Jetson and your PC via ethernet and configure Wi-Fi for internet access.  
- On the Jetson:
  ```bash
  nm-connection-editor
  ```
  Set the local ethernet interface to **Manual**, with a static IP (e.g. `192.168.2.185/24`). Disable IPv6.
- On the PC:
  ```bash
  nm-connection-editor
  ```
  Set the ethernet interface to **Manual**, with a static IP (e.g. `192.168.2.10/24`). Disable IPv6.
- Configure Wi-Fi as usual on both devices.

### 2.3. Install Updates on the Jetson
```bash
sudo apt update && sudo apt upgrade
```

### 2.4. Install PyTorch and Torch-TensorRT
The easiest method is to use [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master).  
Follow the [system setup guide](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md):
1. Clone the repository
2. Configure Docker default runtime
3. (Optional) Relocate Docker data root (if using SSD)
4. (Optional) Mount swap (if using SSD)
5. (Optional) Disable Desktop GUI (frees memory for throughput tests)
6. Add user to Docker group
7. Set power mode (default is sufficient)

Finally, launch the container once manually to pull the image:
```bash
jetson-containers run dustynv/torch_tensorrt:r35.3.1
```

---

## 3. Build and Deploy on Nvidia Jetson Orin Nano

### 3.1. Build
- Adjust the configuration files in the `src/config/build/nvidia` directory as needed.  
- By default, build outputs are stored in `experiments/build/nvidia`.

```bash
python build_nvidia.py
```

### 3.2. Deploy on Jetson
Before deployment, edit the `Jetson` class in `src/boards/board_cfg.py` to match your board configuration.

```bash
python deploy_nvidia.py
```
When prompted, choose the output folder created during the build step (typically under `experiments/build/nvidia`, e.g. `experiments/build/nvidia/exp_01`).
