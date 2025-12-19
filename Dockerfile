FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="SDXL Benchmark"
LABEL description="SDXL inference benchmark with power monitoring"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    sudo \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (but allow sudo for nvidia-smi -pl)
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o benchmark && \
    useradd -m -u $UID -g $GID -o -s /bin/bash benchmark && \
    echo "benchmark ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" >> /etc/sudoers

USER benchmark
WORKDIR /home/benchmark

# Set up Python environment
ENV PATH="/home/benchmark/.local/bin:$PATH"
ENV PIP_NO_CACHE_DIR=1

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (latest compatible versions for CUDA 12.8)
# Using cu128 index for CUDA 12.8 support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install diffusers and dependencies (latest compatible versions)
# These versions work well together and support SDXL with CLIPImageProcessor
RUN pip3 install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    invisible-watermark \
    pillow \
    numpy

# Install xformers (latest version compatible with PyTorch and CUDA 12.8)
RUN pip3 install xformers

# Copy benchmark script
COPY --chown=benchmark:benchmark benchmark.py /home/benchmark/benchmark.py

# Create results and cache directories with proper permissions
RUN mkdir -p /home/benchmark/results && \
    mkdir -p /home/benchmark/.cache/huggingface && \
    chown -R benchmark:benchmark /home/benchmark/.cache

# Pre-download model (optional - uncomment to bake model into image)
# RUN python3 -c "from diffusers import StableDiffusionXLPipeline; \
#     StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', \
#     torch_dtype='auto', variant='fp16')"

ENTRYPOINT ["python3", "/home/benchmark/benchmark.py"]
CMD ["--help"]
