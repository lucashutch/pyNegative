import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PySide6 import QtCore
from pynegative.ui.exportprocessor import ExportProcessor, ExportJob


@pytest.fixture
def thread_pool():
    return QtCore.QThreadPool()


def test_export_processor_validate_settings():
    assert not ExportProcessor.validate_export_settings({"format": "JPEG"})
    assert (
        "Unsupported format"
        in ExportProcessor.validate_export_settings({"format": "TIFF"})[0]
    )
    assert (
        "Max width must be positive"
        in ExportProcessor.validate_export_settings(
            {"format": "JPEG", "max_width": -1}
        )[0]
    )
    assert (
        "Max width must be a number"
        in ExportProcessor.validate_export_settings(
            {"format": "JPEG", "max_width": "abc"}
        )[0]
    )


def test_export_processor_run_jpeg(tmp_path):
    signals = MagicMock()
    signals.fileProcessed = MagicMock()
    signals.batchCompleted = MagicMock()
    signals.progress = MagicMock()

    file = tmp_path / "img1.ARW"
    file.touch()

    settings = {
        "format": "JPEG",
        "jpeg_quality": 90,
        "max_width": 100,
        "max_height": 100,
    }
    processor = ExportProcessor(signals, [file], settings, tmp_path / "out")

    mock_img = np.zeros((10, 10, 3), dtype=np.float32)

    with patch(
        "pynegative.ui.exportprocessor.pynegative.open_raw", return_value=mock_img
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.load_sidecar",
        return_value={"de_haze": 10, "sharpen_value": 5},
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_preprocess",
        return_value=mock_img,
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.de_haze_image",
        return_value=(mock_img, None),
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.sharpen_image", return_value=mock_img
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_tone_map",
        return_value=(mock_img, None),
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_defringe", return_value=mock_img
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_geometry", return_value=mock_img
    ), patch(
        "pynegative.io.lens_resolver.resolve_lens_profile", return_value=(None, None)
    ), patch("PIL.Image.Image.save"):
        processor.run()

    signals.fileProcessed.emit.assert_called()
    signals.batchCompleted.emit.assert_called_with(1, 0, 1)


def test_export_processor_run_heif(tmp_path):
    signals = MagicMock()
    signals.fileProcessed = MagicMock()
    signals.batchCompleted = MagicMock()

    file = tmp_path / "img1.ARW"
    file.touch()

    settings = {"format": "HEIF", "heif_quality": 95, "heif_bit_depth": "10-bit"}
    processor = ExportProcessor(signals, [file], settings, tmp_path / "out")

    mock_img = np.zeros((10, 10, 3), dtype=np.float32)

    with patch(
        "pynegative.ui.exportprocessor.pynegative.open_raw", return_value=mock_img
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.load_sidecar", return_value={}
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_preprocess",
        return_value=mock_img,
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_tone_map",
        return_value=(mock_img, None),
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_defringe", return_value=mock_img
    ), patch(
        "pynegative.ui.exportprocessor.pynegative.apply_geometry", return_value=mock_img
    ), patch(
        "pynegative.io.lens_resolver.resolve_lens_profile", return_value=(None, None)
    ), patch("PIL.Image.Image.save"):
        processor.run()

    signals.batchCompleted.emit.assert_called_with(1, 0, 1)


def test_export_processor_cancel(tmp_path):
    signals = MagicMock()
    files = [tmp_path / f"img{i}.ARW" for i in range(5)]
    for f in files:
        f.touch()

    processor = ExportProcessor(signals, files, {"format": "JPEG"}, tmp_path / "out")
    processor.cancel()
    processor.run()

    signals.fileProcessed.emit.assert_not_called()


def test_export_job_start(thread_pool):
    job = ExportJob(thread_pool)
    files = ["test.ARW"]
    settings = {"format": "JPEG"}

    with patch.object(thread_pool, "start") as mock_start:
        assert job.start_export(files, settings, "/tmp/out") is True
        assert job.is_exporting()
        mock_start.assert_called()
