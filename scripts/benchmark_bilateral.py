#!/usr/bin/env python3
"""Quick benchmark of the bilateral kernel on a 256×256 image."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from pynegative.processing.effects import _apply_bilateral_path


def main():
    np.random.seed(42)
    size = (256, 256)

    # Create a noisy float32 RGB image
    img = np.random.uniform(0.1, 0.9, (*size, 3)).astype(np.float32)
    noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 1)

    l_str = 20.0
    c_str = 10.0

    print(f"Image size: {size[1]}×{size[0]}")
    print(f"Luma strength: {l_str}, Chroma strength: {c_str}")
    print()

    # Warmup (JIT compilation)
    print("Warming up (JIT compile)...")
    _ = _apply_bilateral_path(noisy.copy(), l_str, c_str)
    print("Warmup complete.\n")

    # Benchmark
    iterations = 10
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = _apply_bilateral_path(noisy.copy(), l_str, c_str)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i + 1}: {elapsed:.2f} ms")

    avg = np.mean(times)
    best = np.min(times)
    worst = np.max(times)
    print(f"\nResults ({iterations} iterations):")
    print(f"  Average: {avg:.2f} ms")
    print(f"  Best:    {best:.2f} ms")
    print(f"  Worst:   {worst:.2f} ms")


if __name__ == "__main__":
    main()
