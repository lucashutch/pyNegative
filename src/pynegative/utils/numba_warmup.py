"""
Pre-compile all Numba JIT kernels so the user never experiences a
compilation hitch during interactive editing.

Call ``warmup_kernels()`` once at startup (behind the splash screen).
If the Numba on-disk cache is already warm the calls return almost
instantly (<100 ms); on a cold cache the full LLVM compilation runs
(~15-30 s on first launch).
"""

import logging
import time

import numpy as np

logger = logging.getLogger("pynegative.core")


def warmup_kernels() -> tuple[bool, float]:
    """Trigger JIT compilation of every Numba kernel used by the app.

    Returns
    -------
    is_first_run : bool
        ``True`` when the compilation took long enough that it was
        likely a cold-cache (first-launch) run.
    elapsed_ms : float
        Wall-clock time spent warming up, in milliseconds.
    """
    from .numba_color import preprocess_kernel, tone_map_kernel
    from .numba_dehaze import (
        dark_channel_kernel,
        dehaze_recovery_kernel,
        transmission_dark_channel_kernel,
    )
    from .numba_denoise import (
        bilateral_kernel_yuv,
        nl_means_numba,
        nl_means_numba_multichannel,
    )
    from .numba_detail import sharpen_kernel

    start = time.perf_counter()

    img3 = np.zeros((4, 4, 3), dtype=np.float32)
    img2d = np.zeros((30, 30), dtype=np.float32)
    img3_mc = np.zeros((30, 30, 3), dtype=np.float32)
    transmission = np.full((4, 4), 0.5, dtype=np.float32)
    atmospheric = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    preprocess_kernel(img3.copy(), 1.0, 1.0, 1.0, 0.0)

    tone_map_kernel(img3.copy(), 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, True)

    # 2. sharpen_kernel(img, blurred, percent)
    sharpen_kernel(img3.copy(), img3.copy(), 50.0)

    # 3. bilateral_kernel_yuv(img_yuv, strength, σ_color_y, σ_space_y,
    #                         σ_color_uv, σ_space_uv)
    bilateral_kernel_yuv(img3.copy(), 1.0, 0.1, 1.0, 0.1, 1.0)

    # 4. nl_means_numba  (2D single-channel)
    #    Minimum image size for defaults (patch=7, search=21):
    #    2*(10+3)+1 = 27  → use 30×30
    nl_means_numba(img2d, h=10.0, patch_size=7, search_size=21)

    # Also warm the specialised 3×3 patch path
    nl_means_numba(img2d, h=10.0, patch_size=3, search_size=5)

    # 5. nl_means_numba_multichannel  (3-channel)
    nl_means_numba_multichannel(
        img3_mc, h=(10.0, 10.0, 10.0), patch_size=7, search_size=21
    )
    nl_means_numba_multichannel(
        img3_mc, h=(10.0, 10.0, 10.0), patch_size=3, search_size=5
    )

    # 6. dark_channel_kernel
    dark_channel_kernel(img3.copy())

    # 7. transmission_dark_channel_kernel (fused normalization + dark channel)
    transmission_dark_channel_kernel(img3.copy(), atmospheric)

    # 8. dehaze_recovery_kernel
    dehaze_recovery_kernel(img3.copy(), transmission, atmospheric)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Heuristic: if it took more than 2 s it was very likely a cold cache
    is_first_run = elapsed_ms > 2000.0

    if is_first_run:
        logger.info(
            "First-launch Numba kernel compilation completed in %.0f ms",
            elapsed_ms,
        )
    else:
        logger.debug("Numba kernel cache warm (%.0f ms)", elapsed_ms)

    return is_first_run, elapsed_ms
