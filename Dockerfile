FROM nvidia/cuda:12.3.1-base-ubuntu22.04

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/lib/apt/lists/partial && \
    apt-get update -o Acquire::CompressionTypes::Order::=gz && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    wget \
    build-essential \
    libpng-dev \
    libjpeg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    TORCH_ALLOW_JIT_COMPILATION=1 \
    CUDA_LAUNCH_BLOCKING=1

VOLUME /work

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY requirements.txt /work/

WORKDIR /work

RUN pip3 install -r requirements.txt
RUN pip3 install fastapi uvicorn

EXPOSE 10288

ENTRYPOINT ["/entrypoint.sh"]