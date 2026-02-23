import pytest
from unittest.mock import MagicMock, patch
from PySide6 import QtGui
from pynegative.ui.loaders import ThumbnailLoader, RawLoader


@pytest.fixture
def mock_pixmap():
    with patch("PySide6.QtGui.QPixmap.fromImage") as mock:
        mock.return_value = MagicMock(spec=QtGui.QPixmap)
        yield mock


def test_thumbnail_loader_cache(mock_pixmap, tmp_path):
    path = tmp_path / "test.jpg"
    path.touch()

    # First run to populate cache
    loader = ThumbnailLoader(str(path))
    with (
        patch(
            "pynegative.core.load_cached_thumbnail",
            return_value=(MagicMock(), {"meta": 1}),
        ),
        patch("PIL.ImageQt.ImageQt"),
    ):
        loader.run()

    # Second run should hit cache
    loader2 = ThumbnailLoader(str(path))
    loader2.signals.finished = MagicMock()
    loader2.run()

    loader2.signals.finished.emit.assert_called()
    args = loader2.signals.finished.emit.call_args[0]
    assert args[0] == str(path)
    assert args[2] == {"meta": 1}


def test_thumbnail_loader_generate(mock_pixmap, tmp_path):
    path = tmp_path / "gen.jpg"
    path.touch()

    loader = ThumbnailLoader(str(path))
    mock_pil = MagicMock()
    mock_pil.width = 100
    mock_pil.height = 100

    with (
        patch("pynegative.core.load_cached_thumbnail", return_value=(None, None)),
        patch("pynegative.core.extract_thumbnail", return_value=mock_pil),
        patch("pynegative.core.save_cached_thumbnail"),
        patch("PIL.ImageQt.ImageQt"),
    ):
        loader.run()

    assert loader.signals.finished is not None


def test_thumbnail_loader_not_found(tmp_path):
    loader = ThumbnailLoader(str(tmp_path / "missing.jpg"))
    loader.signals.finished = MagicMock()
    loader.run()
    loader.signals.finished.emit.assert_not_called()


def test_raw_loader_basic(tmp_path):
    path = tmp_path / "test.ARW"
    path.touch()

    loader = RawLoader(str(path))
    loader.signals.finished = MagicMock()

    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)

    with (
        patch("pynegative.core.open_raw", return_value=mock_img),
        patch("pynegative.core.load_sidecar", return_value={"exp": 1.0}),
    ):
        loader.run()

    loader.signals.finished.emit.assert_called_with(str(path), mock_img, {"exp": 1.0})


def test_raw_loader_fallback(tmp_path):
    path = tmp_path / "test2.ARW"
    path.touch()

    loader = RawLoader(str(path))
    loader.signals.finished = MagicMock()

    mock_img = MagicMock()
    mock_img.shape = (100, 100, 3)

    with (
        patch("pynegative.core.open_raw", return_value=mock_img),
        patch("pynegative.core.load_sidecar", return_value=None),
        patch("pynegative.core.calculate_auto_exposure", return_value={"auto": True}),
    ):
        loader.run()

    loader.signals.finished.emit.assert_called_with(str(path), mock_img, {"auto": True})


def test_raw_loader_error(tmp_path):
    path = tmp_path / "error.ARW"
    path.touch()

    loader = RawLoader(str(path))
    loader.signals.finished = MagicMock()

    with patch("pynegative.core.open_raw", side_effect=Exception("fail")):
        loader.run()

    loader.signals.finished.emit.assert_called_with(str(path), None, None)
