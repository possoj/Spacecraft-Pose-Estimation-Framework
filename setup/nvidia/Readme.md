# 1. Installing Pytorch and Nvidia TensorRT on the host using Docker

## 1.1. Build the docker image

command to build the docker container (from the project root folder):
```shell
docker build setup/nvidia -t pose_nvidia:latest
```

## 1.2. Run the docker image

Command to run the docker image:
```shell
docker run --rm -it --gpus=all --ipc=host --net=host --entrypoint=bash --hostname=nvidia_pose -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_nvidia:latest
```
Replace $POSE_ESTIMATION_ROOT with your pose_estimation path that contains both the pose_estimation project and 
the dataset.

With podman rootless:
```shell
podman run --rm -it --net=host --entrypoint bash --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/-v /export/tmp/posso/recherche:/workspace/pose_estimation pose_nvidia:latest 
```

# 2. Installing Pytorch and Nvidia TensorRT on the Nvidia Jetson target

## 2.1. Install Jetpack on the target

Follow the [Jetson Orin Nano Developer Kit Getting Started Guide (do not upgrade firmware to R36)](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#intro). 
To reproduce our results, install the Jetpack 5.1.3 version on the [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) (L4T 35.5.0). 
The first boot requires a monitor, a keyboard and a mouse.

If you need a more advanced Jetpack install, like downgrading firmware, refer to these documentations:
1. [Download R35 Jetson Linux and File System](https://forums.developer.nvidia.com/t/downgrade-from-jetpack-6-to-jetpack-5/294256/3)
2. [Set the board in recovery mode](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_Nano/Jetpack_5.X/Cmd_Flash)
3. [Connect the Jetson to the host with usb-c cable](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html#usb-c-port-4)
4. Insert a micro-SD card in the Jetson (minimum 64 GB, but in my case it worked only with a 128 GB) and power on the board.
5. [Flash the Jetson Orin nano (SD card)](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html#)

## 2.2. Network configuration

Perform one of the three configuration

### 2.2.1. Configuration 1
If available, connect to the same network (wired or wireless) as your computer and let DHCP do the magic. 

### 2.2.2. Configuration 2
If your organization uses MAC filtering, and you are unable to access internet
from the Nvidia Jetson board, follow the next paragraph.

Directly connect the Jetson to your computer with an ethernet cable (eventually use an ethernet switch). 
Thus, you can access the Jetson through SSH but the Jetson will not have internet access. 
To handle that, on the host, open a terminal and nun `nm-connection-editor` and double click to the ethernet interface associated 
with the Jetson (should be the most recent one). In the IPv4 section, select "shared to other computers" as method, 
and add an IP address in the range 192.168.2.1/24 except the IP address of the board (e.g. IP=192.168.2.10, MASK=255.255.255.0). 
Reboot both your computer and the ZCU104. To know the IP address of other devices on the same local network, use the following command on the host `nmap -sn 192.168.2.0/24`.

Make sure the Jetson is powered on. Try to connect to the Jetson through ssh, e.g. `ssh jetson@192.168.2.185`. 
Once connected to the board, make sure you can access internet from the Jetson board: `ping 8.8.8.8`.

### 2.2.3. Configuration 3

The board and the computer are connected through ethernet on a local network. And both will access internet through Wi-Fi.

Run `nm-connection-editor` on the jetson board. Double-click on the local ethernet interface and go the IPv4 Settings tab.
Set the method to manual and add a static IP address (e.g. IP=192.168.2.185, MASK=24). Go to IPv6 and set the method to 
disabled. 

Run `nm-connection-editor` on the computer. Double-click on the local ethernet interface and go the IPv4 Settings tab.
Set the method to manual and add a static IP address (e.g. IP=192.168.2.10, MASK=24). Go to IPv6 and set the method to 
disabled.

Then, set up the Wi-Fi as usual on both the computer and the nvidia board.

## 2.3. Install updates on the Jetson

```shell
sudo apt update && sudo apt upgrade
```

## 2.4. Install PyTorch and Torch-TensorRT

The easiest way to use Pytorch and Torch-TensorRT on the Jetson board is to use [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master)
provided by Nvidia. First, follow the [system setup](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md).
1. Clone the Repo
2. Docker Default Runtime
3. Relocating Docker Data Root (only if SSD is used. Not my case)
4. Mounting Swap (only if SSD is used. Not my case)
5. Disabling the Desktop GUI (useful to save memory, especially for throughput tests)
6. Adding User to Docker Group
7. Setting the Power Mode (leaved default for now)

Then launch the container once manually to pull the image:
```shell
jetson-containers run dustynv/torch_tensorrt:r35.3.1
```
