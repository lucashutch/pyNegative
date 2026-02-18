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

    t0 = time.perf_counter()
    mx1, my1 = _LENS_MAP_CACHE.get_maps(w, h, model, params, cx, cy, fw, fh, zoom)
    t1 = time.perf_counter()
    gen_time = t1 - t0

    assert mx1 is not None
    assert my1 is not None

    t0 = time.perf_counter()
    mx2, my2 = _LENS_MAP_CACHE.get_maps(w, h, model, params, cx, cy, fw, fh, zoom)
    t1 = time.perf_counter()
    cache_time = t1 - t0

    assert mx2 is mx1
    assert my2 is my1
    print(f"Gen time: {gen_time:.6f}, Cache time: {cache_time:.6f}")
    assert cache_time < gen_time


def test_lens_map_cache_lru_eviction():
    _LENS_MAP_CACHE.clear()

    w, h = 10, 10
    model = "poly3"
    cx, cy = 5.0, 5.0
    fw, fh = 10, 10
    zoom = 1.0

    for i in range(8):
        _LENS_MAP_CACHE.get_maps(w, h, model, {"k1": i * 0.1}, cx, cy, fw, fh, zoom)

    _LENS_MAP_CACHE.get_maps(w, h, model, {"k1": 0.0}, cx, cy, fw, fh, zoom)

    for i in range(8, 10):
        _LENS_MAP_CACHE.get_maps(w, h, model, {"k1": i * 0.1}, cx, cy, fw, fh, zoom)

    params_1 = tuple(sorted({"k1": 0.1}.items()))
    key_1 = (w, h, model, params_1, cx, cy, fw, fh, zoom)

    params_0 = tuple(sorted({"k1": 0.0}.items()))
    key_0 = (w, h, model, params_0, cx, cy, fw, fh, zoom)

    with _LENS_MAP_CACHE._lock:
        assert key_1 not in _LENS_MAP_CACHE._cache
        assert key_0 in _LENS_MAP_CACHE._cache


def test_lens_map_cache_fifo_eviction():
    _LENS_MAP_CACHE.clear()

    w, h = 10, 10
    model = "poly3"
    cx, cy = 5.0, 5.0
    fw, fh = 10, 10
    zoom = 1.0

    maps = []
    for i in range(10):
        m = _LENS_MAP_CACHE.get_maps(w, h, model, {"k1": i * 0.1}, cx, cy, fw, fh, zoom)
        maps.append(m)

    params_0 = tuple(sorted({"k1": 0.0}.items()))
    key_0 = (w, h, model, params_0, cx, cy, fw, fh, zoom)

    with _LENS_MAP_CACHE._lock:
        assert key_0 not in _LENS_MAP_CACHE._cache

    params_9 = tuple(sorted({"k1": 0.9}.items()))
    key_9 = (w, h, model, params_9, cx, cy, fw, fh, zoom)
    with _LENS_MAP_CACHE._lock:
        assert key_9 in _LENS_MAP_CACHE._cache
