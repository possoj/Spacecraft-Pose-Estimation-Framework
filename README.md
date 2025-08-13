# Spacecraft Pose Estimation Framework (SPEF)

The **Spacecraft Pose Estimation Framework (SPEF)** is an open-source toolkit for **training, quantizing, and deploying deep neural networks** for spacecraft pose estimation from monocular images and video sequences.

SPEF has been developed in the context of a Ph.D. thesis and an upcoming IEEE TAES journal publication. It supports **real-time embedded implementations** on **CPU**, **GPU (NVIDIA Jetson)**, and **FPGA**, and includes utilities for **dataset generation**, **temporal evaluation**, and **visualization**.

---

## 🚀 Features

- **Pose Estimation Training**
- **Quantization (QAT)**:
  - **Brevitas + FINN** for FPGA
  - **PyTorch FX** for CPU
  - **pytorch-quantization-toolkit** for NVIDIA Jetson GPU
- **Embedded Deployment**:
  - CPU with TVM
  - GPU (Jetson) with TensorRT
  - FPGA (Xilinx UltraScale+) with FINN
- **Graphical User Interface (GUI)** for real-time pose visualization
- **Dataset Generation Tools** for D-SPEED dataset creation

---

## 📦 Installation (via Docker)

All installations are handled via Dockerfiles located in the `setup/` directory. Please refer to the individual README files in `setup/` for detailed build instructions and preparation steps for each target platform (**CPU**, **Jetson GPU**, and **FPGA**).

---

## 🛠 Usage

### 1. Training (Brevitas docker image)
Before launching training, go to `src/config/train/` and create your YAML configuration files:
- For **Brevitas models (QAT)** with mixed precision: create a folder `exp_X` (e.g., `exp_0`, `exp_1`) containing the YAML configuration file and JSON bit-width file.
- For **PyTorch models (Float32)**: create a single file `exp_X.yaml` (e.g., `exp_1.yaml`).

Training results will be stored in `experiments/train/`.
```bash
python train.py
```

### 2. Model Building (Use the corresponding Docker image for each target platform)
**GPU/CPU build pipeline**: 🎯 QAT → 🛠️ Compile → 📊 Evaluate → 💾 Store results:
- **GPU (NVIDIA)**: `build_nvidia.py` → configs in `src/build/nvidia/` → results in `experiments/build/nvidia/`
- **CPU (TVM)**: `build_tvm.py` → configs in `src/build/tvm/` → results in `experiments/build/tvm/`

**FPGA build pipeline** (QAT must be run manually with the Brevitas Docker image — see Step 1): 🛠️ Compile → 💾 Store results :
- **FPGA (FINN)**: `build_finn.py` → configs in `src/build/finn/` → results in `experiments/build/finn/`

### 3. Deployment (Use the corresponding Docker image for each target platform)
Deployment is automatic.  
Make sure to follow the steps in the `setup/` directory to configure the board (install OS, libraries, etc.).  
Adapt the **IP address**, **port**, **username**, and **password** in `src/boards/boards_cfg.py` according to your setup.

Then run the corresponding deployment script:
- **CPU (TVM)**: `deploy_cpu.py`  
- **GPU (NVIDIA)**: `deploy_nvidia.py`  
- **FPGA (FINN)**: `deploy_fpga.py`  

### 4. Additional Tools
- `temporal.py`: Evaluate on temporal D-SPEED data → results in `experiments/temporal/`
- `nn_stats.py`: Print per-layer network statistics (MACs, parameters)
- `soft_class_plot.py`: Visualize soft-classification encoding/decoding impact on pose accuracy

### 5. GUI
```bash
xhost +
python gui.py
```

---

## 📂 Repository Structure
```
Spacecraft-Pose-Estimation-Framework/
├── experiments/        # Stores all experiment results (training, build, temporal, etc.)
├── models/             # Manually copy models here after training for deployment or GUI visualization
├── setup/              # Dockerfiles and platform-specific README instructions
├── src/                # Source code
├── finn_build/         # Intermediate results/debug folder used by FINN
├── train.py            # Training entry point
├── eval.py             # Evaluation entry point
├── build_finn.py       # Build script for FPGA (FINN)
├── build_nvidia.py     # Build script for NVIDIA Jetson GPU (TensorRT)
├── build_tvm.py        # Build script for CPU (TVM)
├── deploy_finn.py      # Deployment script for FPGA
├── deploy_nvidia.py    # Deployment script for NVIDIA Jetson GPU
├── deploy_tvm.py       # Deployment script for CPU
├── gui.py              # Graphical User Interface for pose visualization
├── temporal.py         # Evaluation on temporal data from D-SPEED, outputs to experiments/temporal
├── nn_stats.py         # Per-layer network stats including MAC count and parameters
├── soft_class_plot.py  # Experiment to visualize impact of soft-classification encoding/decoding on pose accuracy
├── README.md           # This file
└── LICENSE             # MIT License
```

---

## 📊 Dataset

Supports multiple spacecraft pose estimation datasets, including:

**[D-SPEED: Dynamic-Spacecraft Pose Estimation Dataset](https://zenodo.org/records/15851302)**

[![Watch the video](https://img.youtube.com/vi/AbIYOj8LuNY/maxresdefault.jpg)](https://youtu.be/AbIYOj8LuNY?si=wQPHpHEwogIYLbT4)

---

## 📈 Results (Mobile-URSONet+ embedded)

| Platform             | Pose error | FPS  | Power   |
|----------------------|------------|------|---------|
| CPU (ARM)            | 0.2208     | 12.1 | 1.22 W  |
| Jetson Orin Nano     | 0.2088     | 560  | 4.28 W  |
| FPGA UltraScale+     | 0.3518     | 58.7 | 0.865 W |

Full details will be provided in the IEEE TAES article (coming soon, currently under review).

---

## 📜 License
[MIT License](LICENSE)

---

## 👤 Author
Julien Posso – Ph.D. Candidate, Polytechnique Montréal

---

## 📬 Contact

julien.posso@gmail.com / julien.posso@polymtl.ca
