#!/usr/bin/env python3
import time
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pynegative.utils.numba_kernels import nl_means_numba, nl_means_numba_multichannel


def benchmark(name, func, args, iters=3):
    print(f"Benchmarking {name}...")
    # Warmup
    _ = func(*args)

    times = []
    for i in range(iters):
        start = time.perf_counter()
        _ = func(*args)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Iter {i + 1}: {elapsed:.2f}ms")

    avg = np.mean(times)
    return avg


def main():
    # Use 720p for testing
    h, w = 720, 1080

    # Create a gradient with some detail and noise
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :, 0] = x_grid / w
    img[:, :, 1] = y_grid / h
    img[:, :, 2] = (x_grid + y_grid) / (w + h)

    # Add some gaussian noise
    noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 1)

    y = img[:, :, 0]
    u = img[:, :, 1]
    v = img[:, :, 2]
    uv_stack = np.ascontiguousarray(img[:, :, 1:])
    yuv_stack = np.ascontiguousarray(img)

    p_size, s_size = 3, 5  # Fast+ settings
    h_val = 1.0

    print(
        f"Testing on {w}x{h} image (Fast+ settings: patch={p_size}, search={s_size})\n"
    )

    # --- UV TEST ---
    def separate_uv(u, v, h_val, p, s):
        u_d = nl_means_numba(u, h_val, p, s)
        v_d = nl_means_numba(v, h_val, p, s)
        return u_d, v_d

    def combined_uv(uv, h_val, p, s):
        return nl_means_numba_multichannel(uv, (h_val, h_val), p, s)

    t_sep_uv = benchmark("Separate U and V", separate_uv, (u, v, h_val, p_size, s_size))
    t_com_uv = benchmark("Combined UV", combined_uv, (uv_stack, h_val, p_size, s_size))

    print("\nUV Results:")
    print(f"  Separate: {t_sep_uv:.2f}ms")
    print(f"  Combined: {t_com_uv:.2f}ms")
    print(f"  Improvement: {(t_sep_uv / t_com_uv - 1) * 100:.1f}%")

    # --- YUV TEST ---
    def separate_yuv(y, u, v, h_val, p, s):
        y_d = nl_means_numba(y, h_val, p, s)
        u_d = nl_means_numba(u, h_val, p, s)
        v_d = nl_means_numba(v, h_val, p, s)
        return y_d, u_d, v_d

    def combined_yuv(yuv, h_val, p, s):
        return nl_means_numba_multichannel(yuv, (h_val, h_val, h_val), p, s)

    t_sep_yuv = benchmark(
        "Separate Y, U, and V", separate_yuv, (y, u, v, h_val, p_size, s_size)
    )
    t_com_yuv = benchmark(
        "Combined YUV", combined_yuv, (yuv_stack, h_val, p_size, s_size)
    )

    print("\nYUV Results:")
    print(f"  Separate: {t_sep_yuv:.2f}ms")
    print(f"  Combined: {t_com_yuv:.2f}ms")
    print(f"  Improvement: {(t_sep_yuv / t_com_yuv - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
