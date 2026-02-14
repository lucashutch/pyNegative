import numpy as np
from unittest.mock import MagicMock, patch
from pynegative.io.raw import open_raw


def test_open_raw_linear_params():
    """Verify that open_raw calls rawpy.postprocess with gamma=(1,1)."""
    path = "test.cr2"

    with patch("rawpy.imread") as mock_imread:
        mock_raw = MagicMock()
        mock_imread.return_value.__enter__.return_value = mock_raw

        # Mock postprocess to return a dummy array
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint16)
        mock_raw.postprocess.return_value = dummy_rgb

        open_raw(path, output_bps=16)

        # Check that postprocess was called with gamma=(1, 1)
        args, kwargs = mock_raw.postprocess.call_args
        assert kwargs.get("gamma") == (1, 1)
        assert kwargs.get("no_auto_bright") is True


def test_tone_map_gamma():
    """Verify that tone_map_kernel applies gamma (implicitly by checking output values)."""
    from pynegative.processing.tonemap import apply_tone_map

    # Create a linear ramp
    img = np.linspace(0, 1, 100).reshape(1, 100, 1)
    img = np.repeat(img, 3, axis=2).astype(np.float32)

    # Apply tone map with neutral settings (no exposure, no contrast change, etc.)
    # Note: contrast center is 0.18 now, so 1.0 contrast means no change.
    # We need to set contrast to 1.0 and blacks=0, whites=1.
    out, _ = apply_tone_map(
        img, exposure=0.0, contrast=1.0, blacks=0.0, whites=1.0, calculate_stats=False
    )

    # In linear to sRGB, 0.18 should become ~0.46
    # (0.18 is roughly the 18% gray)
    mid_idx = 18
    linear_val = img[0, mid_idx, 0]
    gamma_val = out[0, mid_idx, 0]

    assert gamma_val > linear_val
    # 0.18 sRGB is approx 0.46
    assert 0.45 < out[0, 18, 0] < 0.48


def test_standard_image_linearization():
    """Verify that standard images are linearized on load."""
    from PIL import Image
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test.jpg"
        # Create a middle gray image (128 is ~0.5 sRGB)
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        img.save(tmp_path)

        # Load it
        rgb = open_raw(tmp_path)

        # 128/255 = 0.502
        # Linearized 0.502 should be approx 0.214
        assert 0.20 < rgb[0, 0, 0] < 0.23
