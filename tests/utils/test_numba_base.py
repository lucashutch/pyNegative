import sys
from unittest.mock import patch
import importlib


def test_numba_base_fallback():
    # We must ensure the module is reloaded after setting sys.modules['numba'] to None
    with patch.dict(sys.modules, {"numba": None}):
        import pynegative.utils._numba_base as nb

        # Reloading will trigger the 'except ImportError' block
        importlib.reload(nb)

        @nb.njit()
        def f(x):
            return x

        assert f(10) == 10
        assert list(nb.prange(3)) == [0, 1, 2]


def test_numba_base_actual():
    # Just ensure it's there
    import pynegative.utils._numba_base

    importlib.reload(pynegative.utils._numba_base)
    from pynegative.utils._numba_base import njit

    assert njit is not None
