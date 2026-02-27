"""
Comprehensive denoise performance benchmarks.

Tests bilateral denoise optimization on 24MP synthetic images at various render scales.
Compares luma-only, chroma-only, and combined denoising performance.
"""

import time
import numpy as np
import pytest

from pynegative.processing import effects


class BenchmarkDenoiseConfig:
    """Configuration for denoise benchmarks."""

    # 24MP image dimensions: 6144 x 4080 (Canon EOS R5)
    IMAGE_WIDTH = 6144
    IMAGE_HEIGHT = 4080

    SCALES = [
        ("1:1", 1.0, (IMAGE_HEIGHT, IMAGE_WIDTH)),
        ("1:2", 0.5, (IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2)),
        ("1:4", 0.25, (IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4)),
    ]

    # Strong realistic denoise settings
    DENOISE_PROFILES = [
        ("low", {"luma": 2.0, "chroma": 2.0}),
        ("medium", {"luma": 7.0, "chroma": 7.0}),
        ("high", {"luma": 15.0, "chroma": 15.0}),
    ]


def create_synthetic_image(height, width, noise_type="gaussian"):
    """Create a synthetic RGB test image with realistic noise.

    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    noise_type : str
        Type of noise to add: 'gaussian', 'poisson', 'mixed'

    Returns
    -------
    np.ndarray
        Float32 RGB image with noise, normalized to [0, 1]
    """
    # Create base image with gradients and patterns
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    xx, yy = np.meshgrid(x, y)

    # Composite pattern with edges
    base = 0.5 + 0.25 * np.sin(xx) + 0.2 * np.cos(yy) + 0.15 * np.sin(xx + yy)
    base = np.clip(base, 0, 1)

    # Create RGB from single channel with different scaling
    img_rgb = np.stack(
        [
            base,
            np.clip(base + 0.1, 0, 1),
            np.clip(base - 0.05, 0, 1),
        ],
        axis=2,
    ).astype(np.float32)

    # Add realistic noise
    if noise_type in ("gaussian", "mixed"):
        gaussian_noise = np.random.normal(0, 0.02, img_rgb.shape).astype(np.float32)
        img_rgb = np.clip(img_rgb + gaussian_noise, 0, 1)

    if noise_type in ("poisson", "mixed"):
        # Poisson-like noise (shot noise) - scale dependent
        scaling = 100
        img_scaled = img_rgb * scaling
        img_poisson = np.random.poisson(img_scaled).astype(np.float32) / scaling
        if noise_type == "poisson":
            img_rgb = np.clip(img_poisson, 0, 1)
        else:  # mixed
            img_rgb = np.clip((img_rgb + img_poisson) / 2, 0, 1)

    return img_rgb.astype(np.float32)


def benchmark_denoise_kernel(img, luma_str, chroma_str, iterations=3):
    """Benchmark a single denoise configuration.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image (float32, [0, 1])
    luma_str : float
        Luma denoise strength (0-50)
    chroma_str : float
        Chroma denoise strength (0-50)
    iterations : int
        Number of iterations to average

    Returns
    -------
    float
        Mean elapsed time in milliseconds
    """
    times = []
    for _ in range(iterations):
        # Ensure image is contiguous for optimal Numba performance
        img_contig = np.ascontiguousarray(img)

        start = time.perf_counter()
        _ = effects.de_noise_image(
            img_contig,
            luma_strength=luma_str,
            chroma_strength=chroma_str,
            method="Bilateral",
        )
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return np.mean(times)


@pytest.mark.parametrize(
    "scale_name, scale_factor, dims", BenchmarkDenoiseConfig.SCALES
)
@pytest.mark.parametrize(
    "profile_name, profile_strengths", BenchmarkDenoiseConfig.DENOISE_PROFILES
)
def test_bilateral_denoise_luma_only(
    scale_name, scale_factor, dims, profile_name, profile_strengths, benchmark
):
    """Benchmark bilateral denoise with luma-only processing."""
    h, w = dims
    img = create_synthetic_image(h, w, noise_type="gaussian")

    # Run benchmark
    elapsed = benchmark_denoise_kernel(
        img,
        luma_str=profile_strengths["luma"],
        chroma_str=0.0,  # Luma only
        iterations=3,
    )

    # Suppress warning about no assertions
    assert elapsed >= 0


@pytest.mark.parametrize(
    "scale_name, scale_factor, dims", BenchmarkDenoiseConfig.SCALES
)
@pytest.mark.parametrize(
    "profile_name, profile_strengths", BenchmarkDenoiseConfig.DENOISE_PROFILES
)
def test_bilateral_denoise_chroma_only(
    scale_name, scale_factor, dims, profile_name, profile_strengths, benchmark
):
    """Benchmark bilateral denoise with chroma-only processing."""
    h, w = dims
    img = create_synthetic_image(h, w, noise_type="gaussian")

    # Run benchmark
    elapsed = benchmark_denoise_kernel(
        img,
        luma_str=0.0,  # Chroma only
        chroma_str=profile_strengths["chroma"],
        iterations=3,
    )

    # Suppress warning about no assertions
    assert elapsed >= 0


@pytest.mark.parametrize(
    "scale_name, scale_factor, dims", BenchmarkDenoiseConfig.SCALES
)
@pytest.mark.parametrize(
    "profile_name, profile_strengths", BenchmarkDenoiseConfig.DENOISE_PROFILES
)
def test_bilateral_denoise_combined(
    scale_name, scale_factor, dims, profile_name, profile_strengths, benchmark
):
    """Benchmark bilateral denoise with both luma and chroma processing."""
    h, w = dims
    img = create_synthetic_image(h, w, noise_type="gaussian")

    # Run benchmark
    elapsed = benchmark_denoise_kernel(
        img,
        luma_str=profile_strengths["luma"],
        chroma_str=profile_strengths["chroma"],
        iterations=3,
    )

    # Suppress warning about no assertions
    assert elapsed >= 0


def test_denoise_correctness_luma_only():
    """Verify luma-only denoising produces valid output."""
    img = create_synthetic_image(256, 256)
    result = effects.de_noise_image(img, luma_strength=7.0, chroma_strength=0.0)

    # Output should be float32, same shape, and in valid range
    assert result.dtype == np.float32
    assert result.shape == img.shape
    assert np.all(result >= 0.0) and np.all(result <= 1.0)

    # Result should be different from input (denoising happened)
    assert not np.allclose(result, img)


def test_denoise_correctness_chroma_only():
    """Verify chroma-only denoising produces valid output."""
    img = create_synthetic_image(256, 256)
    result = effects.de_noise_image(img, luma_strength=0.0, chroma_strength=7.0)

    # Output should be float32, same shape, and in valid range
    assert result.dtype == np.float32
    assert result.shape == img.shape
    assert np.all(result >= 0.0) and np.all(result <= 1.0)

    # Result should be different from input (denoising happened)
    assert not np.allclose(result, img)


def test_denoise_correctness_combined():
    """Verify combined luma+chroma denoising produces valid output."""
    img = create_synthetic_image(256, 256)
    result = effects.de_noise_image(img, luma_strength=7.0, chroma_strength=7.0)

    # Output should be float32, same shape, and in valid range
    assert result.dtype == np.float32
    assert result.shape == img.shape
    assert np.all(result >= 0.0) and np.all(result <= 1.0)

    # Result should be different from input (denoising happened)
    assert not np.allclose(result, img)


def test_denoise_zero_strength():
    """Verify zero strength returns original image."""
    img = create_synthetic_image(256, 256)
    result = effects.de_noise_image(img, luma_strength=0.0, chroma_strength=0.0)

    # Should return the same image
    assert np.allclose(result, img)


def test_denoise_edge_case_small_image():
    """Verify denoise works on small images."""
    img = create_synthetic_image(32, 32)
    result = effects.de_noise_image(img, luma_strength=5.0, chroma_strength=5.0)

    assert result.shape == img.shape
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_denoise_edge_case_large_strength():
    """Verify denoise with very high strength doesn't crash."""
    img = create_synthetic_image(128, 128)
    result = effects.de_noise_image(img, luma_strength=50.0, chroma_strength=50.0)

    assert result.shape == img.shape
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


if __name__ == "__main__":
    # Manual benchmark run for verification
    print("Running manual denoise benchmarks...")

    for scale_name, scale_factor, dims in BenchmarkDenoiseConfig.SCALES:
        h, w = dims
        print(f"\n--- Scale: {scale_name} ({w}x{h}) ---")

        img = create_synthetic_image(h, w)

        for profile_name, strengths in BenchmarkDenoiseConfig.DENOISE_PROFILES:
            print(
                f"Profile: {profile_name} (luma={strengths['luma']}, chroma={strengths['chroma']})"
            )

            # Luma only
            t_luma = benchmark_denoise_kernel(img, strengths["luma"], 0.0, iterations=5)
            print(f"  Luma only:    {t_luma:6.2f} ms")

            # Chroma only
            t_chroma = benchmark_denoise_kernel(
                img, 0.0, strengths["chroma"], iterations=5
            )
            print(f"  Chroma only:  {t_chroma:6.2f} ms")

            # Combined (fused)
            t_combined = benchmark_denoise_kernel(
                img, strengths["luma"], strengths["chroma"], iterations=5
            )
            print(f"  Combined:     {t_combined:6.2f} ms")

            # Check for expected improvement
            expected_time = t_luma + t_chroma
            speedup = expected_time / t_combined if t_combined > 0 else 0
            print(f"  Speedup:      {speedup:.2f}x (combined vs sequential)")
