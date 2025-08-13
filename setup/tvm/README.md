# 1. Installing Pytorch and TVM on the host using Docker

## 1.1. Build the docker image

command to build the docker container (from the project root folder):
```shell
docker build setup/tvm -t pose_tvm:latest
```

## 1.2. Run the docker image

Command to run the docker image:
```shell
docker run --rm -it --gpus=all --ipc=host --net=host --entrypoint=bash --hostname=tvm_pose -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_tvm:latest
```
Replace $POSE_ESTIMATION_ROOT with your pose_estimation path that contains both the pose_estimation project and 
the dataset.


# 2. TODO

[//]: # (The first step is to install Linux on the target board:)

[//]: # (- [Raspberry pi]&#40;https://gitlab.com/possoj/pose-tvm-cpu/-/wikis/Prepare-a-Raspberry-Pi-for-TVM&#41;)

[//]: # (- [Ultra96]&#40;https://gitlab.com/possoj/pose-tvm-cpu/-/wikis/Prepare-Ultra96-for-TVM&#41;)

[//]: # (- [Xilinx Pynq boards]&#40;https://gitlab.com/possoj/pose-tvm-cpu/-/wikis/Xilinx-Pynq-boards&#41;)

[//]: # ()
[//]: # (Once the board is prepared, you can follow [this tutorial]&#40;https://gitlab.com/possoj/pose-tvm-cpu/-/wikis/Install-TVM-on-Linux-target&#41; )

[//]: # (to install TVM on it.)

[//]: # ()
[//]: # (## Run the code)

[//]: # ()
[//]: # (The following configuration shows how to execute the code with the automatic configuration of RPC server and tracker &#40;default configuration&#41;.)

[//]: # (To see how to launch an RPC tracker and server manually, see [this wiki]&#40;https://gitlab.com/possoj/pose-tvm-cpu/-/wikis/Manually-start-RPC-tracker-and-server&#41;.)

[//]: # ()
[//]: # (1. Change the configuration in `src/config.py` file according to your environment.)

[//]: # ()
[//]: # (2. Connect the board to the computer &#40;host&#41; and turn the board &#40;wait for boot to complete&#41;.)

[//]: # ()
[//]: # (3. Execute the `main.py`. )
