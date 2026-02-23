import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pynegative.io.raw import open_raw, extract_thumbnail, save_image


def test_open_raw_std():
    mock_img = MagicMock()
    mock_img.mode = "RGB"
    mock_img.width = 100
    mock_img.height = 100
    mock_img.__array__ = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

    with patch("pynegative.io.raw.Image.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_img
        open_raw("test.jpg", half_size=True)
        open_raw("test.png", half_size=False)


def test_open_raw_raw():
    with patch("pynegative.io.raw.rawpy.imread") as mock_imread:
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = np.zeros((10, 10, 3), dtype=np.uint16)
        mock_imread.return_value.__enter__.return_value = mock_raw

        open_raw("test.cr2", output_bps=16)
        open_raw("test.cr3", output_bps=8)


def test_extract_thumbnail_std():
    with patch("pynegative.io.raw.Image.open"):
        extract_thumbnail("test.jpg")


def test_extract_thumbnail_raw():
    with patch("pynegative.io.raw.rawpy.imread") as mock_imread:
        mock_raw = MagicMock()
        thumb = MagicMock()
        import rawpy

        thumb.format = rawpy.ThumbFormat.JPEG
        thumb.data = b"fake_jpeg_data"
        mock_raw.extract_thumb.return_value = thumb
        mock_imread.return_value.__enter__.return_value = mock_raw

        with patch("pynegative.io.raw.Image.open"):
            extract_thumbnail("test.cr2")

        # fallback
        mock_raw.extract_thumb.side_effect = Exception("No thumb")
        extract_thumbnail("test.dng")


def test_save_image():
    mock_img = MagicMock()
    save_image(mock_img, "out.jpg")
    try:
        save_image(mock_img, "out.heic")
    except RuntimeError:
        pass
    with pytest.raises(ValueError):
        save_image(mock_img, "out.xyz")
