import numpy as np
from pynegative.utils.numba_defringe import defringe_kernel


def test_defringe_logic():
    # Create a test image (10x10)
    # Background: black
    # White square in middle (5x5) to create contrasty edges
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[2:8, 2:8, :] = 1.0

    # Add a magenta fringe at the right edge of the white square
    # White is at x=7, so x=8 is black.
    img[5, 8, 0] = 1.0  # Red
    img[5, 8, 1] = 0.2  # Green (low)
    img[5, 8, 2] = 1.0  # Blue
    # Result at (5,8) is magenta

    out = np.zeros_like(img)

    # Purple thresh 1.0 (aggressive), edge 0.05, radius 1
    defringe_kernel(img, out, 1.0, 0.0, 0.05, 1.0)

    # Check that the magenta pixel at (5,8) was desaturated
    # Luma of (1, 0.2, 1) is roughly 0.299*1 + 0.587*0.2 + 0.114*1 = 0.53
    # Channels should move towards ~0.53
    assert out[5, 8, 0] < 0.7
    assert out[5, 8, 1] > 0.4
    assert out[5, 8, 2] < 0.7

    # Check that white square middle (5,5) is untouched (low contrast neighborhood)
    # Wait, my kernel checks neighborhood. (5,5) neighborhood is all white, so contrast is 0.
    assert out[5, 5, 0] == 1.0
    assert out[5, 5, 1] == 1.0
    assert out[5, 5, 2] == 1.0


def test_defringe_no_contrast():
    # Magenta pixel in a sea of magenta (no contrast)
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[:, :, 0] = 1.0
    img[:, :, 1] = 0.2
    img[:, :, 2] = 1.0

    out = np.zeros_like(img)
    defringe_kernel(img, out, 1.0, 0.0, 0.1, 1.0)

    # Should be untouched because contrast is 0
    assert np.all(img == out)
