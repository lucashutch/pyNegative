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
    from .numba_color import float32_to_uint8, preprocess_kernel, tone_map_kernel
    from .numba_dehaze import (
        dark_channel_kernel,
        dehaze_recovery_kernel,
        transmission_dark_channel_kernel,
    )
    from .numba_denoise import (
        bilateral_kernel_yuv,
        bilateral_kernel_luma,
        bilateral_kernel_chroma,
    )
    from .numba_defringe import defringe_kernel
    from ..processing.lens import (
        vignette_kernel,
        generate_ptlens_map,
        generate_poly3_map,
        generate_tca_maps,
    )

    start = time.perf_counter()

    img3 = np.zeros((4, 4, 3), dtype=np.float32)
    transmission = np.full((4, 4), 0.5, dtype=np.float32)
    atmospheric = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    preprocess_kernel(img3.copy(), 1.0, 1.0, 1.0, 0.0)

    tone_map_kernel(img3.copy(), 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, True)

    # 3. bilateral kernels for denoising
    #    bilateral_kernel_yuv(img_yuv, strength, σ_color_y, σ_space_y,
    #                         σ_color_uv, σ_space_uv)
    bilateral_kernel_yuv(img3.copy(), 1.0, 0.1, 1.0, 0.1, 1.0)

    # Warmup the optimized luma and chroma variants
    bilateral_kernel_luma(img3.copy(), 1.0, 0.1, 1.0)
    plane_y = img3[:, :, 0].copy()
    bilateral_kernel_chroma(img3.copy(), plane_y, 1.0, 0.1, 1.0)

    # 4. dark_channel_kernel
    dark_channel_kernel(img3.copy())

    # 5. transmission_dark_channel_kernel (fused normalization + dark channel)
    transmission_dark_channel_kernel(img3.copy(), atmospheric)

    # 6. dehaze_recovery_kernel
    dehaze_recovery_kernel(img3.copy(), transmission, atmospheric)

    # 7. defringe_kernel(img, out, purple_thresh, green_thresh, edge_thresh, radius)
    defringe_kernel(img3.copy(), np.empty_like(img3), 0.5, 0.5, 0.05, 1.0)

    # 8. Lens kernels
    vignette_kernel(img3.copy(), 0.1, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0)
    generate_ptlens_map(4, 4, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 1.0)
    generate_poly3_map(4, 4, 0.0, 2.0, 2.0, 4.0, 4.0, 1.0)
    generate_tca_maps(
        4,
        4,
        0,
        np.zeros(4, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2.0,
        2.0,
        4.0,
        4.0,
        1.0,
    )

    # 9. float32_to_uint8 conversion kernel
    float32_to_uint8(img3.copy())

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
