#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pynegative.core import de_noise_image, open_raw


def benchmark_method(image, method, strength, iterations=3):
    print(f"Benchmarking {method}...")

    # Warmup
    _ = de_noise_image(
        image, luma_strength=strength, chroma_strength=strength, method=method
    )

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = de_noise_image(
            image, luma_strength=strength, chroma_strength=strength, method=method
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i + 1}: {elapsed:.2f} ms")

    avg = np.mean(times)
    return avg


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        print(f"Loading image from {image_path}...")
        # Load at half size for benchmarking efficiency if it's a huge RAW
        noisy_img = open_raw(image_path, half_size=True)
        size = noisy_img.shape[:2]
        print(f"Loaded image size: {size[1]}x{size[0]}")
    else:
        # Use a realistic image size for testing (e.g., 1024px width)
        size = (720, 1080)  # (H, W)
        print(f"Creating synthetic noisy image {size[1]}x{size[0]}...")

        # Create a gradient with some detail and noise
        y, x = np.mgrid[0 : size[0], 0 : size[1]]
        image = np.zeros((size[0], size[1], 3), dtype=np.float32)
        image[:, :, 0] = x / size[1]
        image[:, :, 1] = y / size[0]
        image[:, :, 2] = (x + y) / (size[0] + size[1])

        # Add gaussian noise
        noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
        noisy_img = np.clip(image + noise, 0, 1)

    strength = 20.0

    print(f"\nComparing Denoise Performance (Strength: {strength}):")
    print("-" * 50)

    # --- Benchmarking Suite ---
    methods = [
        ("High Quality", 3),
        ("NLMeans (Numba Hybrid YUV)", 3),
        ("NLMeans (Numba Fast+ YUV)", 5),
    ]

    results = {}
    for method, iters in methods:
        results[method] = benchmark_method(
            noisy_img, method, strength, iterations=iters
        )

    print("-" * 60)
    print(f"{'Method':<25} | {'Average Time':<15} | {'Speedup/Slowdown':<15}")
    print("-" * 60)

    base_time = results["High Quality"]
    for method, avg in results.items():
        ratio = avg / base_time
        print(f"{method:<25} | {avg:>10.2f} ms | {ratio:>10.2f}x slowest")

    print("-" * 60)
    print("Saving visual comparison to 'denoise_comparison.png'...")

    # We'll save the "Full" and "Fast" versions for comparison
    res_bilateral = de_noise_image(
        noisy_img,
        luma_strength=strength,
        chroma_strength=strength,
        method="High Quality",
    )
    res_hybrid = de_noise_image(
        noisy_img,
        luma_strength=strength,
        chroma_strength=strength,
        method="NLMeans (Numba Hybrid)",
    )
    res_fast_plus = de_noise_image(
        noisy_img,
        luma_strength=strength,
        chroma_strength=strength,
        method="NLMeans (Numba Fast+)",
    )

    # Convert to uint8 for saving
    def to_u8(img):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    h_stack = np.hstack(
        [
            to_u8(noisy_img),
            to_u8(res_bilateral),
            to_u8(res_hybrid),
            to_u8(res_fast_plus),
        ]
    )

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(h_stack, "Noisy", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(h_stack, "Bilateral", (size[1] + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(
        h_stack, "NLM Hybrid", (size[1] * 2 + 10, 30), font, 1, (255, 255, 255), 2
    )
    cv2.putText(
        h_stack, "NLM Fast+", (size[1] * 3 + 10, 30), font, 1, (255, 255, 255), 2
    )

    cv2.imwrite("denoise_comparison.png", cv2.cvtColor(h_stack, cv2.COLOR_RGB2BGR))
    print("Done!")


if __name__ == "__main__":
    main()
