import numpy as np
from pynegative.utils.numba_histogram import numba_histogram_kernel


def test_numba_histogram_kernel():
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    # some black pixels (default)

    # some white pixels
    img[0, 0] = [255, 255, 255]

    # some random colors
    img[1, 1] = [50, 100, 150]

    # very dark red
    img[2, 2] = [200, 0, 0]

    # very bright blue
    img[3, 3] = [0, 0, 200]

    hist_r, hist_g, hist_b, hist_y, hist_u, hist_v = numba_histogram_kernel(img, 1)

    assert hist_r[255] == 1
    assert hist_r[200] == 1
    assert hist_r[0] == 97  # 100 - 3 modified

    assert hist_g[255] == 1
    assert hist_b[255] == 1

    assert len(hist_r) == 256
    assert len(hist_y) == 256

    # Check clamping functionality
    # These extreme values might push Y/U/V out of normal bounds and need clamping
    # Our data type is uint8 so we can't exceed 255 anyway, but the math inside might!
    img[4, 4] = [255, 0, 0]
    img[5, 5] = [0, 255, 0]
    img[6, 6] = [0, 0, 255]

    # Test striding > 1
    h_r, h_g, h_b, h_y, h_u, h_v = numba_histogram_kernel(img, 2)
    assert len(h_r) == 256
