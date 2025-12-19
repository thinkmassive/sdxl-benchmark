# SDXL Power/Performance Benchmark

Benchmarks Stable Diffusion XL inference performance across different GPU power limits to measure the performance-to-power curve.

## Features

- Measures throughput (images/sec), latency, and power consumption
- Tests multiple power limits automatically
- Calculates energy efficiency (Joules per image)
- Outputs results in CSV and JSON formats
- Supports both Docker (recommended) and local venv execution

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 3090, RTX 4090, A100)
- ~7GB VRAM minimum for SDXL at 1024x1024
- Docker with NVIDIA Container Toolkit (for Docker mode)
- OR Python 3.10+ with CUDA-capable PyTorch (for local mode)
- sudo/root access for changing power limits (`nvidia-smi -pl`)

## Quick Start

```bash
# Set power on host, run benchmark on GPU 0
sudo nvidia-smi -i 0 -pl 400
./run.sh --skip-power-control --gpu 0 --num-images 5
```

### Using Docker (Recommended)

```bash
# Clone or copy files to your machine
cd sdxl-benchmark

# Make run script executable
chmod +x run.sh

# Show help
./run.sh --help

# Run with automatic power limit range (50% to 100% of max TDP)
./run.sh --power-limits auto --num-images 5

# Run with specific power limits
./run.sh --power-limits 200,250,300,350 --num-images 10

# First run will:
# 1. Build the Docker image (~5 min)
# 2. Download SDXL model (~7GB, cached for future runs)
# 3. Run benchmarks
```

### Using Local Virtual Environment

```bash
# Run with --local flag to skip Docker
./run.sh --local --power-limits auto --num-images 5
```

### Manual Docker Commands

```bash
# Build image
docker build -t sdxl-benchmark .

# Run benchmark
docker run --rm -it \
    --gpus all \
    --privileged \
    -v $(pwd)/results:/home/benchmark/results \
    -v sdxl-benchmark-hf-cache:/home/benchmark/.cache/huggingface \
    sdxl-benchmark:latest \
    --power-limits auto --num-images 5
```

## Command Line Options

```
--power-limits, -p    Comma-separated power limits in watts, or 'auto'
                      Default: auto (50% to 100% of max TDP in 10% steps)

--num-images, -n      Number of images to generate per power level
                      Default: 5

--num-steps, -s       Number of inference steps per image
                      Default: 50 (standard for SDXL)

--resolution          Output resolution (WxH)
                      Default: 1024x1024

--prompt              Text prompt for generation
                      Default: "A photorealistic image of an astronaut..."

--output-dir, -o      Directory to save results
                      Default: ./results

--compile             Use torch.compile() for optimization (PyTorch 2.0+)

--num-warmup          Number of warmup iterations before benchmarking
                      Default: 3
```

## Output

Results are saved to `./results/` in two formats:

### CSV Format
```csv
timestamp,gpu_name,power_limit_watts,num_images,images_per_sec,avg_power_draw_watts,energy_per_image_joules,...
2024-01-15T10:30:00,NVIDIA GeForce RTX 4090,300,5,1.234,285.5,231.2,...
```

### JSON Format
```json
[
  {
    "timestamp": "2024-01-15T10:30:00",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "power_limit_watts": 300,
    "images_per_sec": 1.234,
    "avg_power_draw_watts": 285.5,
    "energy_per_image_joules": 231.2,
    ...
  }
]
```

### Console Summary
```
================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================
GPU: NVIDIA GeForce RTX 4090
Resolution: 1024x1024
Inference steps: 50
Images per test: 5
--------------------------------------------------------------------------------
 Power Limit    Img/sec    sec/Img    Avg Power   Energy/Img     Avg Temp
         (W)                                 (W)          (J)         (°C)
--------------------------------------------------------------------------------
         200      0.892      1.121        185.2        207.6         62.3
         250      1.056      0.947        231.5        219.2         67.8
         300      1.189      0.841        278.3        234.1         72.1
         350      1.234      0.811        325.6        263.9         76.4
================================================================================

Most efficient: 200W (207.6 J/image)
Fastest: 350W (1.234 img/sec)
```

## Interpreting Results

- **images_per_sec**: Raw throughput - higher is better
- **energy_per_image_joules**: Energy efficiency - lower is better
- **Efficiency sweet spot**: Usually 60-80% of max TDP offers best perf/watt
- **Diminishing returns**: Often see <10% perf gain for last 20% of power

## Troubleshooting

### "Could not set power limit"
- Ensure you have sudo access for `nvidia-smi -pl`
- In Docker, make sure `--privileged` flag is used

### Out of Memory
- Reduce resolution: `--resolution 768x768`
- Ensure no other GPU processes are running

### Slow First Run
- Model download (~7GB) is cached after first run
- Use named Docker volume to persist cache

### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

## Customization

### Testing Different Models
Edit `benchmark.py` and change the model in `load_pipeline()`:
```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "your-model-here",
    ...
)
```

### Adding Metrics
The `PowerMonitor` class can be extended to capture additional nvidia-smi metrics.

## License

MIT License - feel free to modify and distribute.
