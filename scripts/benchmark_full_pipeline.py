#!/usr/bin/env python3
"""
Comprehensive benchmark script for all image processing pipeline operations.
Established as baseline before major architecture optimizations.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynegative import core
from pynegative.io import lens_resolver


class PipelineBenchmark:
    def __init__(self, iterations=10, warmup=3):
        self.iterations = iterations
        self.warmup = warmup
        self.results = []
        self.has_opencl = hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL()

    def log(self, msg):
        print(msg)

    def benchmark_func(self, name: str, func, img: np.ndarray, **kwargs) -> Dict:
        """Benchmark a single function call."""
        res_key = f"{img.shape[1]}x{img.shape[0]}"

        # Warmup
        for _ in range(self.warmup):
            try:
                func(img, **kwargs)
            except Exception as e:
                self.log(f"  Warmup error in {name}: {e}")
                break

        times = []
        for i in range(self.iterations):
            start = time.perf_counter()
            try:
                func(img, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                self.log(f"  Error in {name} iteration {i}: {e}")
                break

        if not times:
            return {
                "operation": name,
                "size": res_key,
                "error": "No successful iterations",
            }

        result = {
            "operation": name,
            "size": res_key,
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "iterations": len(times),
            "throughput_mpps": (img.shape[0] * img.shape[1])
            / (np.mean(times) / 1000.0)
            / 1e6,
        }
        return result

    def run_all(self, img_path: Path, resolutions: List[int]):
        self.log(f"Starting comprehensive benchmark with {img_path.name}")
        self.log(f"OpenCL Available: {self.has_opencl}")
        if self.has_opencl:
            self.log(f"OpenCL Device: {cv2.ocl.Device.getDefault().name()}")

        # 1. RAW Loading
        self.log("\n--- Stage 1: RAW Loading ---")
        if hasattr(core.open_raw, "cache_clear"):
            core.open_raw.cache_clear()

        load_times = []
        for _ in range(self.warmup):
            core.open_raw(img_path)
            if hasattr(core.open_raw, "cache_clear"):
                core.open_raw.cache_clear()

        for _ in range(self.iterations):
            if hasattr(core.open_raw, "cache_clear"):
                core.open_raw.cache_clear()
            t0 = time.perf_counter()
            core.open_raw(img_path)
            load_times.append((time.perf_counter() - t0) * 1000)

        raw_img = core.open_raw(img_path)
        self.results.append(
            {
                "operation": "raw_load_full",
                "size": f"{raw_img.shape[1]}x{raw_img.shape[0]}",
                "mean_ms": float(np.mean(load_times)),
                "std_ms": float(np.std(load_times)),
            }
        )
        self.log(f"RAW Load (Full): {np.mean(load_times):.2f}ms")

        # Get lens info for later
        source, lens_info = lens_resolver.resolve_lens_profile(img_path)
        if lens_info:
            self.log(f"Lens Profile: {lens_info.get('name')} (Source: {source})")
        else:
            self.log("No lens profile found, using fallback defaults for benchmarking.")
            lens_info = {}

        for res in resolutions:
            self.log(f"\n--- Testing Resolution: {res}px ---")
            scale = res / max(raw_img.shape[:2])
            img = cv2.resize(
                raw_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            self.log(f"Actual size: {img.shape[1]}x{img.shape[0]}")

            # 2. Lens Correction
            self.log("  Benchmarking Lens Correction...")
            # Ensure we have some parameters to force processing
            test_lens_info = lens_info.copy()
            if not test_lens_info.get("distortion"):
                test_lens_info["distortion"] = {"model": "poly3", "k1": 0.02}
            if not test_lens_info.get("vignetting"):
                test_lens_info["vignetting"] = {
                    "model": "pa",
                    "k1": -0.2,
                    "k2": 0.1,
                    "k3": 0.0,
                }

            settings_auto = {"lens_enabled": True, "lens_autocrop": True}
            self.results.append(
                self.benchmark_func(
                    "lens_correction_auto",
                    core.apply_lens_correction,
                    img,
                    settings=settings_auto,
                    lens_info=test_lens_info,
                    scale=scale,
                )
            )

            settings_manual = {
                "lens_enabled": True,
                "lens_distortion": 0.05,
                "lens_vignette": 0.5,
                "lens_autocrop": True,
            }
            self.results.append(
                self.benchmark_func(
                    "lens_correction_manual",
                    core.apply_lens_correction,
                    img,
                    settings=settings_manual,
                    scale=scale,
                )
            )

            # 3. Heavy Effects
            self.log("  Benchmarking Heavy Effects...")
            # Dehaze
            self.results.append(
                self.benchmark_func("dehaze_50", core.de_haze_image, img, strength=0.5)
            )
            # Denoise
            self.results.append(
                self.benchmark_func(
                    "denoise_nlmeans_fast",
                    core.de_noise_image,
                    img,
                    luma_strength=20,
                    chroma_strength=20,
                    method="NLMeans (Numba Fast+ YUV)",
                )
            )
            self.results.append(
                self.benchmark_func(
                    "denoise_nlmeans_hq",
                    core.de_noise_image,
                    img,
                    luma_strength=20,
                    chroma_strength=20,
                    method="NLMeans (Numba Hybrid YUV)",
                )
            )
            self.results.append(
                self.benchmark_func(
                    "denoise_bilateral",
                    core.de_noise_image,
                    img,
                    luma_strength=20,
                    chroma_strength=20,
                    method="High Quality",
                )
            )
            # Sharpen
            self.results.append(
                self.benchmark_func(
                    "sharpen_standard", core.sharpen_image, img, radius=1.0, percent=50
                )
            )

            # 4. Tone Mapping
            self.log("  Benchmarking Tone Mapping...")
            tm_settings = {
                "exposure": 0.5,
                "contrast": 1.2,
                "temperature": 0.1,
                "tint": 0.05,
                "blacks": 0.02,
                "whites": 0.98,
                "shadows": 5.0,
                "highlights": -5.0,
                "saturation": 1.1,
            }
            self.results.append(
                self.benchmark_func(
                    "tone_map_full", core.apply_tone_map, img, **tm_settings
                )
            )

            # 5. Defringe
            self.log("  Benchmarking Defringe...")
            df_settings = {
                "defringe_purple": 50,
                "defringe_green": 30,
                "defringe_radius": 2.0,
            }
            self.results.append(
                self.benchmark_func(
                    "defringe_standard", core.apply_defringe, img, settings=df_settings
                )
            )

            # 6. Geometry
            self.log("  Benchmarking Geometry...")
            self.results.append(
                self.benchmark_func(
                    "geometry_rotate_crop",
                    core.apply_geometry,
                    img,
                    rotate=5.0,
                    crop=(0.05, 0.05, 0.95, 0.95),
                )
            )

            # 7. Histogram
            self.log("  Benchmarking Histogram...")
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            def run_hist(image_u8):
                stride = max(
                    1, int(np.sqrt(image_u8.shape[0] * image_u8.shape[1] / 65536))
                )
                return core.numba_histogram_kernel(image_u8, stride)

            self.results.append(
                self.benchmark_func("histogram_standard", run_hist, img_uint8)
            )

    def save_report(self, output_path: str):
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "opencv_version": cv2.__version__,
            "opencl_available": self.has_opencl,
            "results": self.results,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"\nReport saved to {output_path}")

    def print_table(self):
        self.log("\n" + "=" * 100)
        self.log(
            f"{'Operation':<30} {'Resolution':<15} {'Mean (ms)':<12} {'MP/s':<10} {'Std (ms)':<10}"
        )
        self.log("-" * 100)
        for r in self.results:
            if "error" in r:
                continue
            self.log(
                f"{r['operation']:<30} {r['size']:<15} {r['mean_ms']:<12.2f} {r.get('throughput_mpps', 0):<10.2f} {r.get('std_ms', 0):<10.2f}"
            )
        self.log("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Benchmark")
    parser.add_argument("--image", type=str, required=True, help="Path to RAW image")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated resolutions",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=str, default="benchmark_results.json")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image {img_path} not found")
        sys.exit(1)

    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]

    bench = PipelineBenchmark(iterations=args.iterations, warmup=args.warmup)
    bench.run_all(img_path, resolutions)
    bench.print_table()
    bench.save_report(args.output)


if __name__ == "__main__":
    main()
