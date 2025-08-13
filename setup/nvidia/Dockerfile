FROM nvcr.io/nvidia/pytorch:23.10-py3
# Pytorch 2.1.0 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10)

LABEL authors="Julien Posso"
LABEL description="Docker image for training (optionally quantized) DNN with Pytorch and compile it with Nvidia TensorRT"

# Avoid questions during build
ARG DEBIAN_FRONTEND="noninteractive"

ENV TZ="America/Montreal"

# Install python packages
RUN pip install xlsxwriter==3.0.3 paramiko==2.11.0 pandas==1.5.3 brevitas==0.7.1 yacs==0.1.8 torchlop==0.2 openpyxl==3.1.2
