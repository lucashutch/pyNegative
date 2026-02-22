#!/usr/bin/env python3
"""
Benchmark script for image processing pipeline operations.

Compares performance across three execution modes:
1. OpenCL GPU - Uses GPU via OpenCL
2. OpenCL CPU - Uses CPU via OpenCL
3. Native CPU - Pure CPU without OpenCL

Usage:
    uv run python scripts/benchmark_pipeline.py [options]

Options:
    --image PATH          Use test image (JPG, PNG, or auto-detected RAW)
    --raw PATH            Use RAW photo file (CR2, CR3, ARW, NEF, etc.) at multiple scales
    --resolutions SIZES   Comma-separated list of resolutions (default: 1024,2048,4096)
    --scales SCALES       Comma-separated list of scales for RAW: 0.25,0.5,1.0 (default: all)
    --iterations N        Number of timing iterations (default: 10)
    --warmup N            Number of warmup iterations (default: 3)
    --output PATH         Output JSON file for results
    --markdown PATH       Output Markdown report file

Examples:
    # Benchmark with synthetic image
    uv run python scripts/benchmark_pipeline.py

    # Benchmark with a JPG/PNG
    uv run python scripts/benchmark_pipeline.py --image photo.jpg

    # Benchmark with a RAW file (auto-detected)
    uv run python scripts/benchmark_pipeline.py --image photo.CR3

    # Benchmark RAW at multiple scales
    uv run python scripts/benchmark_pipeline.py --raw photo.CR3 --scales 0.25,0.5,1.0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from pynegative import core


def generate_test_image(width: int, height: int) -> np.ndarray:
    """Generate a synthetic test image with realistic patterns."""
    # Create base with gradient
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Add horizontal gradient
    for i in range(3):
        img[:, :, i] = np.linspace(0, 1, width)[None, :]

    # Add some noise
    img += np.random.normal(0, 0.05, img.shape)

    # Add some edges (simulating real image features)
    y, x = np.ogrid[:height, :width]
    mask = ((x - width // 3) ** 2 + (y - height // 2) ** 2) < (
        min(width, height) // 4
    ) ** 2
    img[mask] = img[mask] * 0.5 + 0.5

    return np.clip(img, 0, 1)


def load_raw_image(raw_path: str) -> tuple[np.ndarray, dict]:
    """Load a RAW image file using pynegative's RAW loader.

    Returns:
        Tuple of (full_resolution_image, metadata_dict)
    """
    print(f"Loading RAW image: {raw_path}")
    img = core.open_raw(raw_path)

    if img is None:
        raise ValueError(f"Failed to load RAW image: {raw_path}")

    h, w = img.shape[:2]
    print(f"  Loaded: {w}x{h} ({img.dtype})")

    return img, {"width": w, "height": h, "path": raw_path}


def create_scale_variants(
    img: np.ndarray, scales: list[float]
) -> dict[float, np.ndarray]:
    """Create scaled versions of an image.

    Args:
        img: Source image
        scales: List of scale factors (e.g., [0.25, 0.5, 1.0])

    Returns:
        Dictionary mapping scale factor to scaled image
    """
    variants = {}
    h, w = img.shape[:2]

    for scale in scales:
        if scale == 1.0:
            variants[scale] = img.copy()
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            variants[scale] = scaled
            print(f"  Created {scale * 100:.0f}% scale: {new_w}x{new_h}")

    return variants


def resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target max dimension."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def benchmark_operation(
    name: str,
    func,
    img: np.ndarray,
    iterations: int,
    warmup: int,
) -> dict:
    """Benchmark a single operation across all three modes."""
    results = {
        "operation": name,
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "modes": {},
    }

    # Test each mode
    modes = [
        ("opencl_gpu", True),
        ("opencl_cpu", False),
    ]

    for mode_name, use_opencl in modes:
        # Set OpenCL state
        if hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(use_opencl)

        # Warmup iterations
        for _ in range(warmup):
            try:
                func(img)
            except Exception as e:
                print(f"  Warmup error in {name}/{mode_name}: {e}")
                break

        # Timing iterations
        times = []

        for i in range(iterations):
            try:
                start = time.perf_counter()
                result = func(img)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                # Check actual backend used (if available from result)
                if isinstance(result, tuple) and len(result) > 1:
                    # Some functions return (result, metadata)
                    pass

            except Exception as e:
                print(f"  Error in {name}/{mode_name} iteration {i}: {e}")
                break

        if times:
            results["modes"][mode_name] = {
                "mean_ms": float(np.mean(times)),
                "median_ms": float(np.median(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "iterations": len(times),
            }
        else:
            results["modes"][mode_name] = {"error": "No successful iterations"}

    return results


def run_benchmarks(
    test_image: np.ndarray,
    resolutions: list[int],
    iterations: int,
    warmup: int,
) -> list[dict]:
    """Run all benchmarks."""
    all_results = []

    # Define operations to benchmark
    operations = [
        ("dehaze", lambda img: core.de_haze_image(img, 30)),
        ("denoise_nlmeans", lambda img: core.de_noise_image(img, 20, "NLMeans")),
        ("denoise_bilateral", lambda img: core.de_noise_image(img, 20, "High Quality")),
        ("sharpen", lambda img: core.sharpen_image(img, 0.5, 30)),
    ]

    # Check OpenCL availability
    has_opencl = hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL()
    print(f"OpenCL Available: {has_opencl}")
    if has_opencl:
        print(
            f"OpenCL Platform: {cv2.ocl.Device.getDefault().name() if cv2.ocl.Device.getDefault() else 'Unknown'}"
        )
    print()

    for resolution in resolutions:
        print(f"Testing resolution: {resolution}px max dimension")

        # Resize test image
        if resolution != max(test_image.shape[:2]):
            img = resize_image(test_image, resolution)
        else:
            img = test_image.copy()

        print(f"  Actual size: {img.shape[1]}x{img.shape[0]}")

        for op_name, op_func in operations:
            print(f"  Benchmarking {op_name}...", end=" ", flush=True)

            result = benchmark_operation(op_name, op_func, img, iterations, warmup)
            all_results.append(result)

            # Print quick summary
            if (
                "opencl_gpu" in result["modes"]
                and "mean_ms" in result["modes"]["opencl_gpu"]
            ):
                gpu_time = result["modes"]["opencl_gpu"]["mean_ms"]
                cpu_time = result["modes"]["opencl_cpu"].get("mean_ms", gpu_time)
                speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
                print(f"GPU: {gpu_time:.1f}ms, CPU: {cpu_time:.1f}ms ({speedup:.1f}x)")
            else:
                print("skipped")

        print()

    return all_results


def run_benchmarks_for_image(
    img: np.ndarray,
    iterations: int,
    warmup: int,
    label: str = None,
) -> list[dict]:
    """Run benchmarks on a single image (for RAW scale variants)."""
    all_results = []

    # Define operations to benchmark
    operations = [
        ("dehaze", lambda img: core.de_haze_image(img, 30)),
        ("denoise_nlmeans", lambda img: core.de_noise_image(img, 20, "NLMeans")),
        ("denoise_bilateral", lambda img: core.de_noise_image(img, 20, "High Quality")),
        ("sharpen", lambda img: core.sharpen_image(img, 0.5, 30)),
    ]

    # Check OpenCL availability
    has_opencl = hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL()
    if label:
        print(f"OpenCL Available: {has_opencl}")
        if has_opencl:
            print(
                f"OpenCL Platform: {cv2.ocl.Device.getDefault().name() if cv2.ocl.Device.getDefault() else 'Unknown'}"
            )
        print()

    print(f"  Actual size: {img.shape[1]}x{img.shape[0]}")

    for op_name, op_func in operations:
        print(f"  Benchmarking {op_name}...", end=" ", flush=True)

        result = benchmark_operation(op_name, op_func, img, iterations, warmup)
        all_results.append(result)

        # Print quick summary
        if (
            "opencl_gpu" in result["modes"]
            and "mean_ms" in result["modes"]["opencl_gpu"]
        ):
            gpu_time = result["modes"]["opencl_gpu"]["mean_ms"]
            cpu_time = result["modes"]["opencl_cpu"].get("mean_ms", gpu_time)
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            print(f"GPU: {gpu_time:.1f}ms, CPU: {cpu_time:.1f}ms ({speedup:.1f}x)")
        else:
            print("skipped")

    print()
    return all_results


def print_results_table(results: list[dict]):
    """Print formatted results table to console."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Group by resolution
    by_resolution = {}
    for r in results:
        res = r["image_size"]
        if res not in by_resolution:
            by_resolution[res] = []
        by_resolution[res].append(r)

    for resolution in sorted(by_resolution.keys(), key=lambda x: int(x.split("x")[0])):
        print(f"\nResolution: {resolution}")
        print("-" * 80)
        print(
            f"{'Operation':<20} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10} {'GPU ±':<10}"
        )
        print("-" * 80)

        for r in by_resolution[resolution]:
            op_name = r["operation"]
            modes = r["modes"]

            if "opencl_gpu" in modes and "mean_ms" in modes["opencl_gpu"]:
                gpu_mean = modes["opencl_gpu"]["mean_ms"]
                gpu_std = modes["opencl_gpu"]["std_ms"]
                cpu_mean = modes["opencl_cpu"].get("mean_ms", gpu_mean)
                speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 1.0

                print(
                    f"{op_name:<20} {gpu_mean:<12.2f} {cpu_mean:<12.2f} {speedup:<10.2f} {gpu_std:<10.2f}"
                )
            else:
                print(f"{op_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10}")

    print("\n" + "=" * 80)


def save_json_report(results: list[dict], output_path: str):
    """Save results to JSON file."""
    report = {
        "benchmark_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "opencv_version": cv2.__version__,
            "opencl_available": hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL(),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nJSON report saved to: {output_path}")


def save_markdown_report(results: list[dict], output_path: str):
    """Save results to Markdown file."""
    lines = [
        "# Image Processing Pipeline Benchmark Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**OpenCV Version:** {cv2.__version__}",
        f"**OpenCL Available:** {hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL()}",
        "",
        "## Summary",
        "",
        "Comparing OpenCL GPU vs OpenCL CPU performance for heavy image processing operations.",
        "",
    ]

    # Group by resolution
    by_resolution = {}
    for r in results:
        res = r["image_size"]
        if res not in by_resolution:
            by_resolution[res] = []
        by_resolution[res].append(r)

    for resolution in sorted(by_resolution.keys(), key=lambda x: int(x.split("x")[0])):
        lines.extend(
            [
                f"### Resolution: {resolution}",
                "",
                "| Operation | GPU (ms) | CPU (ms) | Speedup | GPU σ |",
                "|-----------|----------|----------|---------|-------|",
            ]
        )

        for r in by_resolution[resolution]:
            op_name = r["operation"]
            modes = r["modes"]

            if "opencl_gpu" in modes and "mean_ms" in modes["opencl_gpu"]:
                gpu_mean = modes["opencl_gpu"]["mean_ms"]
                gpu_std = modes["opencl_gpu"]["std_ms"]
                cpu_mean = modes["opencl_cpu"].get("mean_ms", gpu_mean)
                speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 1.0

                lines.append(
                    f"| {op_name} | {gpu_mean:.2f} | {cpu_mean:.2f} | {speedup:.2f}x | {gpu_std:.2f} |"
                )
            else:
                lines.append(f"| {op_name} | N/A | N/A | N/A | N/A |")

        lines.append("")

    lines.extend(
        [
            "## Methodology",
            "",
            "- **Warmup Iterations:** 3 (to allow OpenCL kernel compilation)",
            "- **Timing Iterations:** 10",
            "- **Measurements:** Mean, median, std, min, max execution times",
            "- **GPU Mode:** OpenCL with cv2.ocl.setUseOpenCL(True)",
            "- **CPU Mode:** OpenCL with cv2.ocl.setUseOpenCL(False)",
            "",
            "## Notes",
            "",
            "- Speedup > 1.0 indicates GPU is faster",
            "- Speedup < 1.0 indicates CPU is faster",
            "- Results may vary based on hardware, drivers, and image content",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark image processing pipeline operations"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (JPG/PNG/RAW auto-detected, default: generate synthetic)",
    )
    parser.add_argument(
        "--raw", type=str, help="Path to RAW photo file (CR2, ARW, NEF, etc.)"
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.25,0.5,1.0",
        help="Comma-separated list of scales for RAW: 0.25,0.5,1.0 (default: all)",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated list of max dimensions to test (default: 1024,2048,4096)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timing iterations per operation (default: 10)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--markdown", type=str, help="Output Markdown report file path")

    args = parser.parse_args()

    print("Image Processing Pipeline Benchmark")
    print("=" * 80)
    print()

    # Check for RAW image
    if args.raw:
        # Load RAW image
        raw_img, metadata = load_raw_image(args.raw)

        # Parse scales
        scales = [float(s.strip()) for s in args.scales.split(",")]
        print(f"Creating scale variants: {[f'{s * 100:.0f}%' for s in scales]}")

        # Create scale variants
        test_images = create_scale_variants(raw_img, scales)

        print(f"\nBenchmarking with RAW image at {len(scales)} scales")
        print(f"Iterations per operation: {args.iterations}")
        print(f"Warmup iterations: {args.warmup}")
        print()

        # Run benchmarks for each scale
        all_results = []
        for scale, img in test_images.items():
            print(
                f"\n--- Testing {scale * 100:.0f}% scale ({img.shape[1]}x{img.shape[0]}) ---"
            )
            # Use the scale as the "resolution" identifier
            results = run_benchmarks_for_image(
                img, args.iterations, args.warmup, label=f"{scale * 100:.0f}%"
            )
            all_results.extend(results)

        results = all_results

    else:
        # Original behavior: use resolutions
        resolutions = [int(r.strip()) for r in args.resolutions.split(",")]

        # Load or generate test image
        if args.image:
            # Check if it's a RAW file
            raw_extensions = {
                ".cr2",
                ".cr3",
                ".arw",
                ".nef",
                ".nrw",
                ".raf",
                ".orf",
                ".rw2",
                ".pef",
                ".dng",
            }
            if Path(args.image).suffix.lower() in raw_extensions:
                print(f"Detected RAW file, using RAW loader: {args.image}")
                raw_img, _ = load_raw_image(args.image)
                test_image = raw_img
            else:
                print(f"Loading test image: {args.image}")
                from PIL import Image

                img = Image.open(args.image).convert("RGB")
                test_image = np.array(img).astype(np.float32) / 255.0
        else:
            print("Generating synthetic test image (4096x2730)")
            test_image = generate_test_image(4096, 2730)

        print(f"Base image size: {test_image.shape[1]}x{test_image.shape[0]}")
        print(f"Resolutions to test: {resolutions}")
        print(f"Iterations per operation: {args.iterations}")
        print(f"Warmup iterations: {args.warmup}")
        print()

        # Run benchmarks
        results = run_benchmarks(test_image, resolutions, args.iterations, args.warmup)

    # Print results
    print_results_table(results)

    # Save reports
    if args.output:
        save_json_report(results, args.output)

    if args.markdown:
        save_markdown_report(results, args.markdown)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
