# Visualization Environment Setup

This README explains how to set up the environment for **launching the GUI** to visualize spacecraft pose estimation results using Docker.

---

## 1. Build the Docker Image
From the project root folder:
```bash
docker build setup/visualize -t pose_visu:latest
```

---

## 2. Run the Docker Container
```bash
docker run --rm -it --gpus=all --ipc=host --net=host \
  --entrypoint=bash --hostname=visu_pose \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation \
  pose_visu:latest
```

- Replace `$POSE_ESTIMATION_ROOT` with the path containing both the `Spacecraft-Pose-Estimation-Framework` project and the dataset.  
- X11 forwarding (`-e DISPLAY` and `/tmp/.X11-unix`) enables graphical display on the host.  
- The container opens a bash shell with all dependencies pre-installed.

---

## 3. Launch the GUI

Once inside the container, run:
```bash
python gui.py
```
This will start the graphical user interface (GUI) for visualizing results interactively.
