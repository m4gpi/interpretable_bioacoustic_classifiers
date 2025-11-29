FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/cargo/bin:$PATH"

RUN apt-get update && \
        apt-get install -y wget gnupg && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update

RUN apt-get install --no-install-recommends -y \
        cuda-11-8 \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        curl \
        vim \
        libncurses-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl \
        git \
        ffmpeg

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /cargo && \
        curl -LsSf https://astral.sh/uv/install.sh | CARGO_HOME=/cargo sh

# TODO: uncomment once not doing mounted
# ADD . /app
# WORKDIR /app
# RUN uv sync --locked
