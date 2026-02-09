import argparse
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2

# Import pynegative modules
from pynegative.core import open_raw
from pynegative.utils.numba_kernels import (
    tone_map_kernel,
    sharpen_kernel,
    bilateral_kernel_yuv,
    dehaze_recovery_kernel,
)


def apply_tone_map_numpy(
    img,
    exposure=0.0,
    contrast=1.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
):
    """Original NumPy implementation of apply_tone_map"""
    img = img.copy()

    # 0. White Balance
    t_scale = 0.4
    tint_scale = 0.2
    r_mult = np.exp(temperature * t_scale - tint * (tint_scale / 2))
    g_mult = np.exp(tint * tint_scale)
    b_mult = np.exp(-temperature * t_scale - tint * (tint_scale / 2))
    img[:, :, 0] *= r_mult
    img[:, :, 1] *= g_mult
    img[:, :, 2] *= b_mult

    # 1. Exposure
    if exposure != 0.0:
        img *= 2**exposure

    # 1.5 Contrast
    if contrast != 1.0:
        img -= 0.5
        img *= contrast
        img += 0.5

    # 2. Tone EQ
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    lum_3d = lum[:, :, np.newaxis]

    if blacks != 0.0:
        img -= blacks
    if whites != 1.0:
        denom = whites - blacks
        img /= denom if abs(denom) > 1e-6 else 1e-6

    if shadows != 0.0:
        s_mask = (1.0 - np.clip(lum_3d, 0, 1)) ** 2
        img *= 1.0 + shadows * s_mask

    if highlights != 0.0:
        if highlights < 0:
            h_mask = np.maximum(lum_3d, 0) ** 2
            img /= 1.0 + abs(highlights) * h_mask
        else:
            h_mask = np.clip(lum_3d, 0, 1) ** 2
            h_term = highlights * h_mask
            img = img * (1.0 - h_term) + h_term

    if saturation != 1.0:
        curr_lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        curr_lum_3d = np.clip(curr_lum, 0, 1)[:, :, np.newaxis]
        img -= curr_lum_3d
        img *= saturation
        img += curr_lum_3d

    np.clip(img, 0.0, 1.0, out=img)
    return img


def apply_tone_map_ocl(
    img,
    exposure=0.0,
    contrast=1.0,
    blacks=0.0,
    whites=1.0,
    shadows=0.0,
    highlights=0.0,
    saturation=1.0,
    temperature=0.0,
    tint=0.0,
):
    """OpenCL implementation of apply_tone_map using OpenCV Transparent API (UMat)"""
    if not cv2.ocl.haveOpenCL():
        return None

    cv2.ocl.setUseOpenCL(True)
    u_img = cv2.UMat(img)

    # 0. White Balance
    t_scale = 0.4
    tint_scale = 0.2
    r_mult = np.exp(temperature * t_scale - tint * (tint_scale / 2))
    g_mult = np.exp(tint * tint_scale)
    b_mult = np.exp(-temperature * t_scale - tint * (tint_scale / 2))

    channels = list(cv2.split(u_img))
    channels[0] = cv2.multiply(channels[0], float(r_mult))
    channels[1] = cv2.multiply(channels[1], float(g_mult))
    channels[2] = cv2.multiply(channels[2], float(b_mult))
    u_img = cv2.merge(channels)

    # 1. Exposure
    if exposure != 0.0:
        u_img = cv2.multiply(u_img, float(2**exposure))

    # 1.5 Contrast
    if contrast != 1.0:
        # img = (img - 0.5) * contrast + 0.5 = img * contrast + (0.5 - 0.5 * contrast)
        u_img = cv2.addWeighted(
            u_img, float(contrast), u_img, 0, 0.5 * (1.0 - float(contrast))
        )

    # 2. Tone EQ
    coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(1, 3)
    lum = cv2.transform(u_img, coeffs)

    if blacks != 0.0:
        u_img = cv2.subtract(u_img, float(blacks))
    if whites != 1.0:
        denom = whites - blacks
        u_img = cv2.multiply(u_img, 1.0 / (float(denom) if abs(denom) > 1e-6 else 1e-6))

    if shadows != 0.0:
        # Using threshold to clip
        _, clamped_lum = cv2.threshold(lum, 1.0, 1.0, cv2.THRESH_TRUNC)
        _, clamped_lum = cv2.threshold(clamped_lum, 0.0, 0.0, cv2.THRESH_TOZERO)

        s_mask = cv2.subtract(1.0, clamped_lum)
        s_mask = cv2.multiply(s_mask, s_mask)

        # img *= 1.0 + shadows * s_mask
        factor = cv2.addWeighted(s_mask, float(shadows), s_mask, 0, 1.0)
        u_img = cv2.multiply(u_img, cv2.merge([factor, factor, factor]))

    if highlights != 0.0:
        if highlights < 0:
            _, h_mask = cv2.threshold(lum, 0.0, 0.0, cv2.THRESH_TOZERO)
            h_mask = cv2.multiply(h_mask, h_mask)
            denom_val = cv2.addWeighted(h_mask, abs(float(highlights)), h_mask, 0, 1.0)
            u_img = cv2.divide(u_img, cv2.merge([denom_val, denom_val, denom_val]))
        else:
            _, clamped_lum = cv2.threshold(lum, 1.0, 1.0, cv2.THRESH_TRUNC)
            _, clamped_lum = cv2.threshold(clamped_lum, 0.0, 0.0, cv2.THRESH_TOZERO)
            h_mask = cv2.multiply(clamped_lum, clamped_lum)
            h_term = cv2.multiply(h_mask, float(highlights))
            one_minus_h_term = cv2.subtract(1.0, h_term)
            u_img = cv2.multiply(
                u_img, cv2.merge([one_minus_h_term, one_minus_h_term, one_minus_h_term])
            )
            u_img = cv2.add(u_img, cv2.merge([h_term, h_term, h_term]))

    if saturation != 1.0:
        curr_lum = cv2.transform(u_img, coeffs)
        _, clamped_curr_lum = cv2.threshold(curr_lum, 1.0, 1.0, cv2.THRESH_TRUNC)
        _, clamped_curr_lum = cv2.threshold(
            clamped_curr_lum, 0.0, 0.0, cv2.THRESH_TOZERO
        )
        u_img = cv2.addWeighted(
            u_img,
            float(saturation),
            cv2.merge([clamped_curr_lum] * 3),
            1.0 - float(saturation),
            0.0,
        )

    _, u_img = cv2.threshold(u_img, 1.0, 1.0, cv2.THRESH_TRUNC)
    _, u_img = cv2.threshold(u_img, 0.0, 0.0, cv2.THRESH_TOZERO)

    return u_img.get()


def create_synthetic_image(width=2000, height=1333):
    img = np.random.rand(height, width, 3).astype(np.float32)
    return img


def benchmark_tone_mapping(image, iterations=5):
    print(f"\n[Tone Mapping Benchmark] {image.shape}")

    # Parameters
    exposure = 0.5
    contrast = 1.1
    blacks = 0.02
    whites = 0.98
    shadows = 0.2
    highlights = -0.2
    saturation = 1.2
    temperature = 0.1
    tint = 0.05

    # 1. NumPy Reference
    times_numpy = []
    for _ in range(iterations):
        img_copy = image.copy()
        start = time.perf_counter()
        _ = apply_tone_map_numpy(
            img_copy,
            exposure,
            contrast,
            blacks,
            whites,
            shadows,
            highlights,
            saturation,
            temperature,
            tint,
        )
        times_numpy.append(time.perf_counter() - start)

    avg_numpy = np.mean(times_numpy) * 1000
    print(f"NumPy Reference:  {avg_numpy:.2f} ms")

    # 2. Numba JIT (Fused Kernel)
    t_scale = 0.4
    tint_scale = 0.2
    temp = temperature
    t = tint
    r_mult = np.exp(temp * t_scale - t * (tint_scale / 2))
    g_mult = np.exp(t * tint_scale)
    b_mult = np.exp(-temp * t_scale - t * (tint_scale / 2))
    exp_mult = 2**exposure

    # Warmup
    img_copy = image.copy()
    tone_map_kernel(
        img_copy,
        exp_mult,
        contrast,
        blacks,
        whites,
        shadows,
        highlights,
        saturation,
        r_mult,
        g_mult,
        b_mult,
    )

    times_numba = []
    for _ in range(iterations):
        img_copy = image.copy()
        start = time.perf_counter()
        tone_map_kernel(
            img_copy,
            exp_mult,
            contrast,
            blacks,
            whites,
            shadows,
            highlights,
            saturation,
            r_mult,
            g_mult,
            b_mult,
        )
        times_numba.append(time.perf_counter() - start)

    avg_numba = np.mean(times_numba) * 1000
    print(f"Numba JIT:        {avg_numba:.2f} ms")

    # 3. OpenCL (OpenCV UMat)
    times_ocl = []
    # Warmup
    _ = apply_tone_map_ocl(
        image,
        exposure,
        contrast,
        blacks,
        whites,
        shadows,
        highlights,
        saturation,
        temperature,
        tint,
    )

    for _ in range(iterations):
        start = time.perf_counter()
        _ = apply_tone_map_ocl(
            image,
            exposure,
            contrast,
            blacks,
            whites,
            shadows,
            highlights,
            saturation,
            temperature,
            tint,
        )
        times_ocl.append(time.perf_counter() - start)

    avg_ocl = np.mean(times_ocl) * 1000
    print(f"OpenCV OpenCL:    {avg_ocl:.2f} ms (Non-fused)")

    print("\nSpeedups:")
    print(f"Numba vs NumPy:      {avg_numpy / avg_numba:.2f}x")
    print(f"OpenCL vs NumPy:     {avg_numpy / avg_ocl:.2f}x")
    print(f"Numba vs OpenCL:     {avg_ocl / avg_numba:.2f}x")

    return {"numpy": avg_numpy, "numba": avg_numba, "ocl": avg_ocl}


def benchmark_sharpen(image, iterations=5):
    print(f"\n[Sharpening Benchmark] {image.shape}")
    radius = 2.0
    percent = 150.0

    # Baseline: GaussianBlur + NumPy arithmetic
    times_cv = []
    for _ in range(iterations):
        start = time.perf_counter()
        blur = cv2.GaussianBlur(image, (0, 0), radius)
        _ = image + (image - blur) * (percent / 100.0)
        times_cv.append(time.perf_counter() - start)
    avg_cv = np.mean(times_cv) * 1000
    print(f"OpenCV + NumPy:   {avg_cv:.2f} ms")

    # Numba
    # Warmup
    blur = cv2.GaussianBlur(image, (0, 0), radius)
    img_copy = image.copy()
    sharpen_kernel(img_copy, blur, percent)

    times_numba = []
    for _ in range(iterations):
        img_copy = image.copy()
        start = time.perf_counter()
        sharpen_kernel(img_copy, blur, percent)
        times_numba.append(time.perf_counter() - start)
    avg_numba = np.mean(times_numba) * 1000
    print(f"Numba JIT:        {avg_numba:.2f} ms")
    print(f"Speedup:          {avg_cv / avg_numba:.2f}x")


def benchmark_bilateral(image, iterations=3):
    print(f"\n[Bilateral Denoise Benchmark] {image.shape}")
    # Use a smaller size for Bilateral as it's very slow
    h, w = image.shape[:2]
    if w > 1024:
        scale = 1024.0 / w
        img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    else:
        img = image

    strength = 20.0
    s_scale = 1.0 / 255.0

    # Setup for baseline (OpenCV)
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    yuv_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV)

    times_cv = []
    for _ in range(iterations):
        start = time.perf_counter()
        y, u, v = cv2.split(yuv_uint8)
        u_denoised = cv2.bilateralFilter(
            u, 11, strength * 4.5 * s_scale, 2.0 + strength / 10.0
        )
        v_denoised = cv2.bilateralFilter(
            v, 11, strength * 4.5 * s_scale, 2.0 + strength / 10.0
        )
        y_denoised = cv2.bilateralFilter(
            y, 3, strength * 0.4 * s_scale, 0.5 + strength / 100.0
        )
        _ = cv2.cvtColor(
            cv2.merge([y_denoised, u_denoised, v_denoised]), cv2.COLOR_YUV2RGB
        )
        times_cv.append(time.perf_counter() - start)
    avg_cv = np.mean(times_cv) * 1000
    print(f"OpenCV (Bilateral): {avg_cv:.2f} ms")

    # Numba
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    sigma_color_uv = float(strength) * 4.5 * s_scale
    sigma_space_uv = 2.0 + (float(strength) / 10.0)
    sigma_color_y = float(strength) * 0.4 * s_scale
    sigma_space_y = 0.5 + (float(strength) / 100.0)

    # Warmup
    _ = bilateral_kernel_yuv(
        img_yuv, strength, sigma_color_y, sigma_space_y, sigma_color_uv, sigma_space_uv
    )

    times_numba = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = bilateral_kernel_yuv(
            img_yuv,
            strength,
            sigma_color_y,
            sigma_space_y,
            sigma_color_uv,
            sigma_space_uv,
        )
        times_numba.append(time.perf_counter() - start)
    avg_numba = np.mean(times_numba) * 1000
    print(f"Numba JIT:          {avg_numba:.2f} ms")
    print(f"Speedup:            {avg_cv / avg_numba:.2f}x")


def benchmark_dehaze(image, iterations=5):
    print(f"\n[Dehaze Recovery Benchmark] {image.shape}")
    rows, cols = image.shape[:2]
    transmission = np.random.rand(rows, cols).astype(np.float32)
    atmospheric_light = np.array([0.8, 0.8, 0.8], dtype=np.float32)

    # Baseline: NumPy
    times_np = []
    for _ in range(iterations):
        start = time.perf_counter()
        t_3d = transmission[:, :, np.newaxis]
        _ = (image - atmospheric_light) / np.maximum(t_3d, 0.1) + atmospheric_light
        times_np.append(time.perf_counter() - start)
    avg_np = np.mean(times_np) * 1000
    print(f"NumPy:            {avg_np:.2f} ms")

    # Numba
    # Warmup
    _ = dehaze_recovery_kernel(image, transmission, atmospheric_light)

    times_numba = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = dehaze_recovery_kernel(image, transmission, atmospheric_light)
        times_numba.append(time.perf_counter() - start)
    avg_numba = np.mean(times_numba) * 1000
    print(f"Numba JIT:        {avg_numba:.2f} ms")
    print(f"Speedup:          {avg_np / avg_numba:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Image size (square)")
    parser.add_argument("--raw", help="Path to real RAW file")
    args = parser.parse_args()

    if args.raw and os.path.exists(args.raw):
        print(f"Loading {args.raw}...")
        image = open_raw(args.raw)
    else:
        print(f"Generated synthetic image {args.size}x{args.size}")
        image = create_synthetic_image(args.size, args.size)

    # Benchmark 1: Tone Mapping
    benchmark_tone_mapping(image)

    # Benchmark 2: Sharpening
    benchmark_sharpen(image)

    # Benchmark 3: Dehaze
    benchmark_dehaze(image)

    # Benchmark 4: Bilateral
    benchmark_bilateral(image)


if __name__ == "__main__":
    main()
