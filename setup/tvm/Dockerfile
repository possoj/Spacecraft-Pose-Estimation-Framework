FROM nvcr.io/nvidia/pytorch:22.05-py3
# Pytorch 1.12
# TVM 0.9.dev0

LABEL authors="Julien Posso"
LABEL description="Docker image for training DNN with Pytorch and compiling DNN with TVM"

# Avoid questions during build
ARG DEBIAN_FRONTEND="noninteractive"

# TVM version (https://github.com/apache/tvm/releases/tag/v0.9.0)
ARG TVM_COMMIT_HASH=e7f793d0ad5f141444fff41d308be17231ec6b86

ENV TZ="America/Montreal"
ENV TVM_ROOT="/opt/tvm"

# Install packages
RUN apt update \
#    && apt install -y libtinfo-dev=6.2-0ubuntu2 \
#    && apt install -y libxml2-dev=2.9.10+dfsg-5ubuntu0.20.04.3 \
    && apt install -y libtinfo-dev \
    && apt install -y libxml2-dev

# Install LLVM
RUN cd /opt && mkdir llvm && cd llvm \
    && wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz \
    && tar -xf clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz \
    && rm -f clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz

# Clone TVM repo
# RUN git clone --recursive https://github.com/apache/tvm.git $TVM_ROOT

RUN git clone --no-checkout --depth 1 https://github.com/apache/tvm.git $TVM_ROOT \
    && cd $TVM_ROOT \
    && git fetch --depth 1 origin $TVM_COMMIT_HASH \
    && git checkout $TVM_COMMIT_HASH \
    && git submodule update --init --recursive  # <-- This line initializes submodules

# Set build config
RUN cd $TVM_ROOT && mkdir build && cp cmake/config.cmake build \
    && echo 'set(USE_LLVM "/opt/llvm/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04/bin/llvm-config --link-static")' >> build/config.cmake \
    && echo 'set(HIDE_PRIVATE_SYMBOLS ON)' >> build/config.cmake

# Build TVM
RUN cd ${TVM_ROOT}/build && cmake .. && make -j8

# Set environment varibles
ENV TVM_HOME=$TVM_ROOT
ENV PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Open ports for Pynq
EXPOSE 8888 9091 9190

# Install python packages
# RUN conda install -c conda-forge xlsxwriter==3.0.3 paramiko==2.11.0
RUN pip install xlsxwriter==3.0.3 paramiko==2.11.0 pandas==1.5.3 brevitas==0.7.1 yacs==0.1.8 torchlop==0.2 openpyxl==3.1.2
