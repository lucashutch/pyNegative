import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pynegative.ui.pipeline.worker import (
    TierGeneratorWorker,
    ImageProcessorWorker,
    ImageProcessorSignals,
)
from pynegative.ui.pipeline.stages import (
    process_denoise_stage,
    process_heavy_stage,
    resolve_vignette_params,
    get_fused_geometry,
    apply_fused_remap,
    calculate_histograms,
)


@pytest.fixture
def signals():
    return MagicMock(spec=ImageProcessorSignals)


def test_tier_generator_worker(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    worker = TierGeneratorWorker(signals, img, 123)

    with patch("PySide6.QtGui.QImage"), patch("PySide6.QtGui.QPixmap.fromImage"):
        worker.run()

    signals.uneditedPixmapGenerated.emit.assert_called()
    assert signals.tierGenerated.emit.call_count >= 1


def test_image_processor_worker_update_preview_viewport(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    worker = ImageProcessorWorker(
        signals, img, {0.125: img, 0.5: img, 1.0: img}, {"exposure": 1.0}, 1
    )
    worker.target_on_screen_width = 100
    worker.calculate_lowres = True
    worker.calculate_histogram = True
    worker.zoom_scale = 1.0
    worker.visible_scene_rect = (10, 10, 20, 20)

    with (
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_preprocess",
            return_value=img,
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_tone_map",
            return_value=(img, None),
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_defringe", return_value=img
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.float32_to_uint8",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        ),
        patch("PySide6.QtGui.QImage"),
        patch("PySide6.QtGui.QPixmap.fromImage"),
        patch("pynegative.ui.pipeline.worker.calculate_histograms", return_value={}),
        patch("pynegative.ui.pipeline.worker.process_denoise_stage", return_value=img),
        patch("pynegative.ui.pipeline.worker.process_heavy_stage", return_value=img),
        patch("pynegative.ui.pipeline.worker.apply_fused_remap", return_value=img),
    ):
        result = worker._update_preview()
        assert len(result) == 6


def test_image_processor_worker_run(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    worker = ImageProcessorWorker(signals, img, {0.5: img}, {"exposure": 1.0}, 1)

    with patch.object(
        worker, "_update_preview", return_value=(MagicMock(), 100, 100, 0, None, None)
    ):
        worker.run()

    signals.finished.emit.assert_called()


def test_image_processor_worker_denoise(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)

    with patch("pynegative.core.de_noise_image", return_value=img) as mock_denoise:
        heavy_params = {
            "denoise_luma": 5.0,
            "denoise_chroma": 5.0,
            "denoise_method": "High Quality",
        }
        result = process_denoise_stage(img, "tier_1.0", heavy_params, 1.0)
        mock_denoise.assert_called()
        assert result.shape == (100, 100, 3)


def test_image_processor_worker_heavy(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)

    heavy_params = {
        "de_haze": 50,
        "sharpen_value": 50,
        "sharpen_radius": 1.0,
        "sharpen_percent": 1.0,
    }

    with (
        patch("pynegative.core.de_haze_image", return_value=(img, 0.5)) as mock_dehaze,
        patch("pynegative.core.sharpen_image", return_value=img) as mock_sharpen,
    ):
        process_heavy_stage(img, "tier_1.0", heavy_params, 1.0, "de_haze", None)
        mock_dehaze.assert_called()
        mock_sharpen.assert_called()


def test_resolve_vignette_params(signals):
    worker = ImageProcessorWorker(
        signals, None, {}, {"lens_enabled": True, "lens_vignette": 0.5}, 1
    )
    worker.lens_info = {
        "vignetting": {"model": "pa", "k1": 0.1, "k2": 0.05, "k3": 0.02}
    }

    vig_k1, vig_k2, vig_k3, cx, cy, fw, fh = resolve_vignette_params(
        worker.settings, worker.lens_info, roi_offset=(0, 0), full_size=(1000, 800)
    )
    assert vig_k1 == pytest.approx(0.6)
    assert vig_k2 == 0.05
    assert cx == 500
    assert cy == 400


def test_get_fused_geometry(signals):
    settings = {"lens_enabled": False}
    # Testing affine only path
    fused_maps, out_w, out_h, zoom = get_fused_geometry(
        settings, None, 100, 100, 90, None, False, False
    )
    assert len(fused_maps) == 1
    assert out_w == 100
    assert out_h == 100


def test_apply_fused_remap_affine(signals):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    M = np.eye(2, 3, dtype=np.float32)

    result = apply_fused_remap(img, [M], 100, 100)
    assert result.shape == (100, 100, 3)


def test_histogram_emitted_without_0625_tier(signals):
    """Histogram must be computed even when 0.0625 tier is absent."""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    tiers = {0.125: np.zeros((12, 12, 3), dtype=np.float32), 0.5: img}
    worker = ImageProcessorWorker(signals, img, tiers, {"exposure": 1.0}, 1)
    worker.target_on_screen_width = 100
    worker.calculate_lowres = True
    worker.calculate_histogram = True
    worker.zoom_scale = 1.0
    worker.visible_scene_rect = (10, 10, 20, 20)

    with (
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_preprocess",
            return_value=img,
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_tone_map",
            return_value=(img, None),
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.apply_defringe", return_value=img
        ),
        patch(
            "pynegative.ui.pipeline.worker.pynegative.float32_to_uint8",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        ),
        patch("PySide6.QtGui.QImage"),
        patch("PySide6.QtGui.QPixmap.fromImage"),
        patch(
            "pynegative.ui.pipeline.worker.calculate_histograms",
            return_value={"R": [1]},
        ) as mock_hist,
        patch("pynegative.ui.pipeline.worker.process_denoise_stage", return_value=img),
        patch("pynegative.ui.pipeline.worker.process_heavy_stage", return_value=img),
        patch("pynegative.ui.pipeline.worker.apply_fused_remap", return_value=img),
    ):
        worker._update_preview()
        mock_hist.assert_called()
        signals.histogramUpdated.emit.assert_called()


def test_calculate_histograms(signals):
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "pynegative.ui.pipeline.stages.pynegative.numba_histogram_kernel",
        return_value=(np.zeros(256),) * 6,
    ):
        hist = calculate_histograms(img)
        expected = {
            "R",
            "G",
            "B",
            "Y",
            "U",
            "V",
            "waveform_R",
            "waveform_G",
            "waveform_B",
            "waveform_Y",
        }
        assert set(hist.keys()) == expected
        assert len(hist["R"]) == 256
        assert hist["waveform_R"].shape == (256, 100)  # 100-col image
        assert hist["waveform_Y"].shape == (256, 100)


def test_calculate_histograms_waveform_wide():
    """Waveform should downsample to 256 columns for wide images."""
    img = np.random.randint(0, 256, (50, 500, 3), dtype=np.uint8)
    hist = calculate_histograms(img)
    assert hist["waveform_R"].shape == (256, 256)
    assert hist["waveform_Y"].shape == (256, 256)
