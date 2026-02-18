"""
Shim for backward compatibility and to group Numba kernels.
"""

from .numba_color import tone_map_kernel, preprocess_kernel
from .numba_detail import sharpen_kernel
from .numba_denoise import (
    bilateral_kernel_yuv,
    nl_means_numba,
    nl_means_numba_multichannel,
)
from .numba_dehaze import dark_channel_kernel, dehaze_recovery_kernel
from .numba_histogram import numba_histogram_kernel

__all__ = [
    "tone_map_kernel",
    "preprocess_kernel",
    "sharpen_kernel",
    "bilateral_kernel_yuv",
    "nl_means_numba",
    "nl_means_numba_multichannel",
    "dark_channel_kernel",
    "dehaze_recovery_kernel",
    "numba_histogram_kernel",
]
