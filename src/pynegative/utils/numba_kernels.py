"""
Shim for backward compatibility and to group Numba kernels.
"""

from .numba_color import float32_to_uint8, preprocess_kernel, tone_map_kernel
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
from .numba_histogram import numba_histogram_kernel

__all__ = [
    "tone_map_kernel",
    "preprocess_kernel",
    "float32_to_uint8",
    "sharpen_kernel",
    "bilateral_kernel_yuv",
    "nl_means_numba",
    "nl_means_numba_multichannel",
    "dark_channel_kernel",
    "dehaze_recovery_kernel",
    "transmission_dark_channel_kernel",
    "numba_histogram_kernel",
]
