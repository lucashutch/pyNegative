"""Tests for the Numba kernel warmup module."""

from pynegative.utils.numba_warmup import warmup_kernels


def test_warmup_returns_tuple():
    """warmup_kernels() should return (bool, float) and not raise."""
    result = warmup_kernels()
    assert isinstance(result, tuple)
    assert len(result) == 2

    is_first_run, elapsed_ms = result
    assert isinstance(is_first_run, bool)
    assert isinstance(elapsed_ms, float)
    assert elapsed_ms >= 0


def test_warmup_warm_cache():
    """A second call should always be fast (cache warm)."""
    # First call ensures compilation
    warmup_kernels()
    # Second call should be cache-warm
    is_first_run, elapsed_ms = warmup_kernels()
    assert not is_first_run, "Second call should be cache-warm"
    # Should complete well under 2s on warm cache
    assert elapsed_ms < 2000.0
