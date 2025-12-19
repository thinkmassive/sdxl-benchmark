#!/bin/bash
#
# SDXL Benchmark Runner
#
# Usage:
#   ./run.sh                           # Show help
#   ./run.sh --power-limits auto       # Auto power range (needs privileged)
#   ./run.sh -p 200,250,300 -n 10      # Specific power limits (needs privileged)
#   ./run.sh --skip-power-control -n 5 # Single run at current power (no privileged needed)
#   ./run.sh --gpu 2 -n 5              # Run on GPU 2
#   ./run.sh --local                   # Run without Docker (uses local venv)
#
# For non-privileged container runs, set power on host first:
#   sudo nvidia-smi -i 0 -pl 250
#   ./run.sh --skip-power-control --gpu 0 --num-images 10
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments to extract flags we need for docker setup
USE_LOCAL=false
SKIP_POWER_CONTROL=false
GPU_INDEX=0
ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL=true
            shift
            ;;
        --skip-power-control)
            SKIP_POWER_CONTROL=true
            ARGS+=("$1")
            shift
            ;;
        --gpu)
            GPU_INDEX="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --gpu=*)
            GPU_INDEX="${1#*=}"
            ARGS+=("$1")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Function to run with Docker
run_docker() {
    echo -e "${GREEN}Running benchmark in Docker on GPU ${GPU_INDEX}...${NC}"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found${NC}"
        exit 1
    fi
    
    # Check for NVIDIA runtime
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime may not be configured${NC}"
    fi
    
    # Build image if needed
    if [[ "$(docker images -q sdxl-benchmark:latest 2>/dev/null)" == "" ]]; then
        echo -e "${YELLOW}Building Docker image (first run)...${NC}"
        docker build -t sdxl-benchmark:latest .
    fi
    
    # Create results directory and ensure it's writable
    # The container user (UID 1000) needs write access to this directory
    mkdir -p results
    chmod 777 results 2>/dev/null || true
    
    # Create HF cache directory on host and ensure it's writable
    # The container user (UID 1000) needs write access to this directory
    mkdir -p .cache/huggingface
    # Make directory writable by container user (world-writable for cache is acceptable)
    chmod 777 .cache/huggingface 2>/dev/null || true
    
    # Determine if we need privileged mode
    DOCKER_OPTS="--rm -it"
    if [[ "$SKIP_POWER_CONTROL" == true ]]; then
        echo -e "${GREEN}Running without --privileged (power control disabled)${NC}"
    else
        echo -e "${YELLOW}Running with --privileged (required for power control)${NC}"
        DOCKER_OPTS="$DOCKER_OPTS --privileged"
    fi
    
    # Use specific GPU - expose all GPUs but let the script select via torch.cuda.set_device
    # We need all GPUs visible so nvidia-smi can query the right one
    DOCKER_OPTS="$DOCKER_OPTS --gpus all"
    
    # Run benchmark
    docker run $DOCKER_OPTS \
        -v "$(pwd)/results:/home/benchmark/results" \
        -v "$(pwd)/.cache/huggingface:/home/benchmark/.cache/huggingface" \
        sdxl-benchmark:latest \
        "${ARGS[@]}"
}

# Function to run locally with venv
run_local() {
    echo -e "${GREEN}Running benchmark locally on GPU ${GPU_INDEX}...${NC}"
    
    VENV_DIR="$SCRIPT_DIR/.venv"
    
    # Create venv if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        
        echo -e "${YELLOW}Installing dependencies...${NC}"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip setuptools wheel
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install diffusers[torch] transformers accelerate safetensors xformers invisible-watermark
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    # Create results directory
    mkdir -p results
    
    # Run benchmark
    python3 benchmark.py "${ARGS[@]}"
}

# Main
if [[ "$USE_LOCAL" == true ]]; then
    run_local
else
    run_docker
fi
