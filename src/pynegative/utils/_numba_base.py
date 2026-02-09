import logging

# Use 'pynegative.core' logger so messages appear alongside other core processing logs
_logger = logging.getLogger("pynegative.core")

try:
    from numba import njit, prange
except ImportError:
    _logger.warning("Numba not found, using pure python fallback decorators")

    def njit(*args, **kwargs):
        def decorator(f):
            return f

        return decorator

    prange = range
