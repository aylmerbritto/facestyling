# FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04
#FROM nvidia/cuda-arm64:11.0-devel-ubuntu18.04
# Set environment variables
FROM ghcr.io/aylmerbritto/caricatureapp:base 
ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
#RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub
#RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub
# Add the NVIDIA GPG key
#RUN apt-get install -y wget
#RUN apt-get install -y gnupg && \
#    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub | apt-key add -
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN apt install -y curl

# RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/sbsa/7fa2af80.pub | apt-key add -

# Install system dependencies
# RUN apt-get update && \
#    apt-get install -y \
#        git \
#        python3-pip \
#        python3-dev \
#        python3-opencv \
#        libglib2.0-0 \
#	cmake
# Install any python packages you need
# COPY requirements.txt requirements.txt
# RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install -r requirements.txt

# Upgrade pip
#RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory
WORKDIR /app
COPY . .
# RUN pip3 install gabriel_server msrest dill scipy
# RUN apt-get install ninja-build
# RUN pip3 install opencv-python
#RUN pip3 cache purge
# RUN apt-get autoremove
# RUN apt-get autoclean
# RUN apt-get clean
# RUN rm -rf /var/lib/apt/lists/*
CMD ["python3", "app.py"]

