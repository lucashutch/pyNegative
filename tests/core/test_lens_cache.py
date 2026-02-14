import time
from pynegative.processing.lens import _LENS_MAP_CACHE


def test_lens_map_cache_basic():
    _LENS_MAP_CACHE.clear()

    w, h = 100, 100
    model = "poly3"
    params = {"k1": 0.1}
    cx, cy = 50.0, 50.0
    fw, fh = 100, 100
    zoom = 1.0

    # First call: generate
    t0 = time.perf_counter()
    mx1, my1 = _LENS_MAP_CACHE.get_maps(w, h, model, params, cx, cy, fw, fh, zoom)
    t1 = time.perf_counter()
    gen_time = t1 - t0

    assert mx1 is not None
    assert my1 is not None

    # Second call: cache hit
    t0 = time.perf_counter()
    mx2, my2 = _LENS_MAP_CACHE.get_maps(w, h, model, params, cx, cy, fw, fh, zoom)
    t1 = time.perf_counter()
    cache_time = t1 - t0

    assert mx2 is mx1  # Should be the same object
    assert my2 is my1
    # Cache should be significantly faster (though for 100x100 it might be small,
    # but still measurable)
    print(f"Gen time: {gen_time:.6f}, Cache time: {cache_time:.6f}")
    assert cache_time < gen_time


def test_lens_map_cache_eviction():
    _LENS_MAP_CACHE.clear()
    # Use a small max entries for testing if it was configurable,
    # but it's 8 by default.

    w, h = 10, 10
    model = "poly3"
    cx, cy = 5.0, 5.0
    fw, fh = 10, 10
    zoom = 1.0

    # Fill cache
    maps = []
    for i in range(10):
        m = _LENS_MAP_CACHE.get_maps(w, h, model, {"k1": i * 0.1}, cx, cy, fw, fh, zoom)
        maps.append(m)

    # Check if first ones are evicted (max entries is 8)
    # Entry 0 should be gone
    params_0 = tuple(sorted({"k1": 0.0}.items()))
    key_0 = (w, h, model, params_0, cx, cy, fw, fh, zoom)

    with _LENS_MAP_CACHE._lock:
        assert key_0 not in _LENS_MAP_CACHE._cache

    # Entry 9 should be there
    params_9 = tuple(sorted({"k1": 0.9}.items()))
    key_9 = (w, h, model, params_9, cx, cy, fw, fh, zoom)
    with _LENS_MAP_CACHE._lock:
        assert key_9 in _LENS_MAP_CACHE._cache
