FROM nvcr.io/nvidia/pytorch:23.10-py3
# Pytorch 2.1.0 (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10)

LABEL authors="Julien Posso"
LABEL description="Docker image for visualizing Spacecraft Pose Estimation"

# Avoid questions during build
ARG DEBIAN_FRONTEND="noninteractive"

ENV TZ="America/Montreal"

# Install packages for the GUI
RUN apt update \
    && apt install -y libtinfo-dev \
    && apt install -y libxml2-dev \
    && apt install -y xorg

# Install python packages
# For Nvidia with kivy version
RUN pip install xlsxwriter==3.2.0 brevitas==0.7.1 yacs==0.1.8 torchlop==0.2 openpyxl==3.1.4
RUN pip install paramiko==2.11.0
RUN pip install kivy==2.3.0
RUN pip install seaborn==0.13.2 plotly==5.22.0
