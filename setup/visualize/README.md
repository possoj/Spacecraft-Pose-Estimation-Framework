# 1. Build the docker image

command to build the docker container (from the project root folder):
```shell
docker build setup/visualize -t pose_visu:latest
```

## 2. Run the docker image

Command to run the docker image:
```shell
docker run --rm -it --gpus=all --ipc=host --net=host --entrypoint=bash --hostname=visu_pose -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation pose_visu:latest
```
Replace $POSE_ESTIMATION_ROOT with your pose_estimation path that contains both the pose_estimation project and 
the dataset.
