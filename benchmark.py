#!/usr/bin/env python3
"""
SDXL Benchmark with Power Monitoring

Benchmarks Stable Diffusion XL inference performance at various GPU power limits,
measuring throughput, latency, and power consumption.

Usage:
    python benchmark.py --power-limits 150,200,250,300,350
    python benchmark.py --power-limits auto --num-images 10
"""

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class BenchmarkResult:
    timestamp: str
    gpu_name: str
    power_limit_watts: int
    num_images: int
    num_inference_steps: int
    resolution: str
    total_time_sec: float
    avg_time_per_image_sec: float
    images_per_sec: float
    avg_power_draw_watts: float
    max_power_draw_watts: float
    avg_gpu_temp_celsius: float
    max_gpu_temp_celsius: float
    avg_gpu_utilization_pct: float
    energy_per_image_joules: float
    pytorch_version: str
    cuda_version: str


class PowerMonitor:
    """Background thread to monitor GPU power, temperature, and utilization."""
    
    def __init__(self, gpu_index: int = 0, sample_interval_ms: int = 100):
        self.gpu_index = gpu_index
        self.sample_interval_ms = sample_interval_ms
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> list[dict]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples
    
    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i", str(self.gpu_index),
                        "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1.0
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        self.samples.append({
                            "timestamp": time.time(),
                            "power_watts": float(parts[0]),
                            "temp_celsius": float(parts[1]),
                            "utilization_pct": float(parts[2])
                        })
            except (subprocess.TimeoutExpired, ValueError, IndexError):
                pass
            
            time.sleep(self.sample_interval_ms / 1000.0)


def get_gpu_info(gpu_index: int = 0) -> tuple[str, int, int]:
    """Returns (gpu_name, current_power_limit, max_power_limit) in watts."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(gpu_index),
            "--query-gpu=name,power.limit,power.max_limit",
            "--format=csv,noheader,nounits"
        ],
        capture_output=True,
        text=True
    )
    parts = result.stdout.strip().split(", ")
    return parts[0], int(float(parts[1])), int(float(parts[2]))


def set_power_limit(watts: int, gpu_index: int = 0) -> bool:
    """Set GPU power limit. Requires root/sudo or appropriate permissions."""
    try:
        result = subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(gpu_index), "-pl", str(watts)],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def load_pipeline(use_compile: bool = False):
    """Load SDXL pipeline with optimizations."""
    from diffusers import StableDiffusionXLPipeline
    
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    # Enable memory-efficient attention (if xformers is available)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory-efficient attention enabled")
    except Exception as e:
        print(f"Warning: Could not enable xformers ({e}), continuing without it")
    
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    return pipe


def warmup(pipe, num_warmup: int = 3, num_steps: int = 20):
    """Warmup the pipeline to stabilize performance."""
    print(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        _ = pipe(
            prompt="warmup image",
            num_inference_steps=num_steps,
            output_type="latent"
        )
        torch.cuda.synchronize()
    print("Warmup complete.")


def run_benchmark(
    pipe,
    num_images: int,
    num_steps: int,
    prompt: str,
    power_limit: int,
    resolution: tuple[int, int] = (1024, 1024),
    skip_power_control: bool = False,
    gpu_index: int = 0
) -> BenchmarkResult:
    """Run benchmark at a specific power limit."""
    
    gpu_name, _, _ = get_gpu_info(gpu_index)
    
    if skip_power_control:
        print(f"\nRunning benchmark on GPU {gpu_index} (power control disabled, current limit: {power_limit}W)...")
    else:
        # Set power limit
        print(f"\nSetting GPU {gpu_index} power limit to {power_limit}W...")
        if not set_power_limit(power_limit, gpu_index):
            print(f"Warning: Could not set power limit to {power_limit}W (may need sudo)")
        
        # Allow GPU to stabilize at new power limit
        time.sleep(2.0)
    
    # Start power monitoring
    monitor = PowerMonitor(gpu_index=gpu_index, sample_interval_ms=100)
    monitor.start()
    
    # Run benchmark
    print(f"Generating {num_images} images at {power_limit}W...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(num_images):
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            height=resolution[1],
            width=resolution[0],
            output_type="latent"  # Skip decode for pure generation benchmark
        )
        torch.cuda.synchronize()
        print(f"  Image {i+1}/{num_images} complete")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Stop monitoring and collect samples
    samples = monitor.stop()
    
    # Calculate statistics
    if samples:
        power_values = [s["power_watts"] for s in samples]
        temp_values = [s["temp_celsius"] for s in samples]
        util_values = [s["utilization_pct"] for s in samples]
        
        avg_power = sum(power_values) / len(power_values)
        max_power = max(power_values)
        avg_temp = sum(temp_values) / len(temp_values)
        max_temp = max(temp_values)
        avg_util = sum(util_values) / len(util_values)
    else:
        avg_power = max_power = avg_temp = max_temp = avg_util = 0.0
    
    # Calculate derived metrics
    avg_time_per_image = total_time / num_images
    images_per_sec = num_images / total_time
    energy_per_image = (avg_power * total_time) / num_images  # Joules
    
    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        gpu_name=gpu_name,
        power_limit_watts=power_limit,
        num_images=num_images,
        num_inference_steps=num_steps,
        resolution=f"{resolution[0]}x{resolution[1]}",
        total_time_sec=round(total_time, 3),
        avg_time_per_image_sec=round(avg_time_per_image, 3),
        images_per_sec=round(images_per_sec, 4),
        avg_power_draw_watts=round(avg_power, 1),
        max_power_draw_watts=round(max_power, 1),
        avg_gpu_temp_celsius=round(avg_temp, 1),
        max_gpu_temp_celsius=round(max_temp, 1),
        avg_gpu_utilization_pct=round(avg_util, 1),
        energy_per_image_joules=round(energy_per_image, 1),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda or "unknown"
    )


def get_current_power_limit(gpu_index: int = 0) -> int:
    """Get the current GPU power limit in watts."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )
    return int(float(result.stdout.strip()))


def parse_power_limits(power_arg: str, max_power: int) -> list[int]:
    """Parse power limits argument."""
    if power_arg == "auto":
        # Generate range from 50% to 100% of max power in 10% increments
        step = max_power // 10
        return list(range(max_power // 2, max_power + 1, step))
    elif power_arg == "current":
        # Use whatever is currently set
        return [get_current_power_limit()]
    else:
        return [int(p.strip()) for p in power_arg.split(",")]


def save_results(results: list[BenchmarkResult], output_dir: Path):
    """Save results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    print(f"\nResults saved to {csv_path}")
    
    # Save as JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {json_path}")
    
    return csv_path, json_path


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"GPU: {results[0].gpu_name}")
    print(f"Resolution: {results[0].resolution}")
    print(f"Inference steps: {results[0].num_inference_steps}")
    print(f"Images per test: {results[0].num_images}")
    print("-" * 80)
    print(f"{'Power Limit':>12} {'Img/sec':>10} {'sec/Img':>10} {'Avg Power':>12} {'Energy/Img':>12} {'Avg Temp':>10}")
    print(f"{'(W)':>12} {'':>10} {'':>10} {'(W)':>12} {'(J)':>12} {'(°C)':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.power_limit_watts:>12} {r.images_per_sec:>10.3f} {r.avg_time_per_image_sec:>10.2f} "
              f"{r.avg_power_draw_watts:>12.1f} {r.energy_per_image_joules:>12.1f} {r.avg_gpu_temp_celsius:>10.1f}")
    
    print("=" * 80)
    
    # Find efficiency sweet spot
    best_efficiency = min(results, key=lambda r: r.energy_per_image_joules)
    print(f"\nMost efficient: {best_efficiency.power_limit_watts}W "
          f"({best_efficiency.energy_per_image_joules:.1f} J/image)")
    
    fastest = max(results, key=lambda r: r.images_per_sec)
    print(f"Fastest: {fastest.power_limit_watts}W "
          f"({fastest.images_per_sec:.3f} img/sec)")


def main():
    parser = argparse.ArgumentParser(
        description="SDXL Benchmark with Power Monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--power-limits", "-p",
        default="auto",
        help="Comma-separated power limits in watts, or 'auto' for automatic range"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=5,
        help="Number of images to generate per power level"
    )
    parser.add_argument(
        "--num-steps", "-s",
        type=int,
        default=50,
        help="Number of inference steps per image"
    )
    parser.add_argument(
        "--prompt",
        default="A photorealistic image of an astronaut riding a horse on Mars, detailed, 8k",
        help="Prompt for image generation"
    )
    parser.add_argument(
        "--resolution",
        default="1024x1024",
        help="Output resolution (WxH)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./results"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for additional optimization (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--skip-power-control",
        action="store_true",
        help="Don't change power limits - use current setting (set on host beforehand)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split("x"))
    resolution = (width, height)
    
    # Set CUDA device
    gpu_index = args.gpu
    torch.cuda.set_device(gpu_index)
    print(f"Using GPU {gpu_index}")
    
    # Get GPU info
    gpu_name, current_power, max_power = get_gpu_info(gpu_index)
    print(f"Detected GPU: {gpu_name}")
    print(f"Current power limit: {current_power}W")
    print(f"Maximum power limit: {max_power}W")
    
    # Parse power limits
    if args.skip_power_control:
        # Just use current power limit, don't try to change it
        power_limits = [current_power]
        print(f"Power control disabled - will benchmark at current {current_power}W")
    else:
        power_limits = parse_power_limits(args.power_limits, max_power)
        print(f"Will test power limits: {power_limits}W")
    
    # Validate power limits
    power_limits = [p for p in power_limits if p <= max_power]
    if not power_limits:
        print("Error: No valid power limits specified")
        return 1
    
    # Load model
    pipe = load_pipeline(use_compile=args.compile)
    
    # Warmup
    warmup(pipe, num_warmup=args.num_warmup, num_steps=args.num_steps)
    
    # Run benchmarks
    results = []
    for power_limit in power_limits:
        try:
            result = run_benchmark(
                pipe=pipe,
                num_images=args.num_images,
                num_steps=args.num_steps,
                prompt=args.prompt,
                power_limit=power_limit,
                resolution=resolution,
                skip_power_control=args.skip_power_control,
                gpu_index=gpu_index
            )
            results.append(result)
        except Exception as e:
            print(f"Error at {power_limit}W: {e}")
            continue
    
    if not results:
        print("No benchmark results collected")
        return 1
    
    # Restore original power limit (only if we changed it)
    if not args.skip_power_control:
        print(f"\nRestoring GPU {gpu_index} power limit to {current_power}W...")
        set_power_limit(current_power, gpu_index)
    
    # Save and display results
    save_results(results, args.output_dir)
    print_summary(results)
    
    return 0


if __name__ == "__main__":
    exit(main())
