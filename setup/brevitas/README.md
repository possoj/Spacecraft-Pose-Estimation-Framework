# Training Neural Networks (PyTorch / Brevitas QAT)

This guide explains how to set up the environment for training the spacecraft pose estimation networks, either in **standard PyTorch (Float32)** or with **Brevitas for Quantization-Aware Training (QAT)**.  

We recommend using **Docker** for maximum reproducibility.  
An **Anaconda/Pip** installation is also provided as an alternative.

---

## 1. Docker Setup (Recommended)

### 1.1. Build the Docker image
From the project root folder, run:
```bash
docker build setup/brevitas -t pose_brevitas:latest
```

### 1.2. Run the Docker container
```bash
docker run --rm -it --gpus=all --ipc=host --net=host \
  --entrypoint=bash --hostname=brevitas_pose \
  -v $POSE_ESTIMATION_ROOT:/workspace/pose_estimation \
  pose_brevitas:latest
```

- Replace `$POSE_ESTIMATION_ROOT` with the path containing both the `pose_estimation` project and the datasets.  
- The container will open a bash shell with all dependencies pre-installed.

---

## 2. Local Installation (Not recommended)

If you prefer not to use Docker, you can set up the environment manually.  
The following example uses **Conda** with Python 3.8.  

### 2.1. Create and activate environment
```bash
conda create --name pose_brevitas python=3.8.12
conda activate pose_brevitas
```

### 2.2. Install PyTorch
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### 2.3. Install additional dependencies
```bash
conda install -c conda-forge tqdm==4.62.3 matplotlib==3.4.3 opencv==4.5.5
pip install xlsxwriter==3.0.3 onnx==1.11.0 onnxruntime==1.11.1 pandas==1.2.3 tensorboard==2.6.0 protobuf==3.18.1
pip install brevitas==0.7.1 onnxoptimizer==0.2.7 yacs==0.1.8 scipy==1.6.3
```

---

## 3. Example Training Command

- Adjust the configuration files as needed in the `src/config/train` directory.  
- Logs and checkpoints will be saved in the `experiments/train/` directory by default.

Once the environment is ready, you can start training with:

```bash
python train.py
```

---

## Notes

- **Docker is recommended** for long-term reproducibility.  
- Local installation may require version adjustments depending on your CUDA driver.
