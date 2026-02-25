import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PySide6 import QtCore, QtGui
from pynegative.ui.imageprocessing import ImageProcessingPipeline


@pytest.fixture
def thread_pool():
    pool = QtCore.QThreadPool()
    yield pool
    pool.waitForDone(1000)


@pytest.fixture
def pipeline(qtbot, thread_pool):
    p = ImageProcessingPipeline(thread_pool)
    p.tiers = {}
    return p


def test_pipeline_set_image(pipeline):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    with patch("pynegative.ui.imageprocessing.TierGeneratorWorker") as mock_worker:
        pipeline.set_image(img)
        assert pipeline.base_img_full is img
        mock_worker.assert_called()


def test_pipeline_set_image_none(pipeline):
    mock_slot = MagicMock()
    pipeline.previewUpdated.connect(mock_slot)
    pipeline.set_image(None)
    assert pipeline.base_img_full is None
    mock_slot.assert_called()


def test_get_unedited_pixmap(pipeline):
    img = np.zeros((100, 100, 3), dtype=np.float32)
    pipeline.base_img_full = img
    pipeline._unedited_img_full = img

    # We need to patch the one imported in imageprocessing.py
    with patch("pynegative.ui.imageprocessing.QtGui.QPixmap.fromImage") as mock_pixmap:
        pipeline.get_unedited_pixmap()
        mock_pixmap.assert_called()


def test_set_processing_params(pipeline):
    with patch.object(pipeline, "request_update") as mock_req:
        pipeline.set_processing_params(exposure=1.5, de_haze=10)
        assert pipeline._processing_params["exposure"] == 1.5
        assert pipeline._last_heavy_adjusted == "de_haze"
        mock_req.assert_called()


def test_process_pending_update(pipeline):
    pipeline.base_img_full = np.zeros((100, 100, 3), dtype=np.float32)
    view = MagicMock()
    view.transform.return_value.m11.return_value = 1.0
    view.viewport.return_value.size.return_value = QtCore.QSize(800, 600)
    view.viewport.return_value.rect.return_value = QtCore.QRect(0, 0, 800, 600)
    view._crop_item.isVisible.return_value = False
    view.mapToScene.return_value.boundingRect.return_value = QtCore.QRectF(
        0, 0, 800, 600
    )
    view.sceneRect.return_value = QtCore.QRectF(0, 0, 1000, 800)

    pipeline.set_view_reference(view)
    pipeline._render_pending = True

    with patch("pynegative.ui.imageprocessing.ImageProcessorWorker") as mock_worker:
        pipeline._process_pending_update()
        assert not pipeline._render_pending
        mock_worker.assert_called()


def test_process_pending_update_recovers_orphan_pending_tile(pipeline):
    pipeline.base_img_full = np.zeros((100, 100, 3), dtype=np.float32)
    pipeline._processing_params = {}
    pipeline._last_settings = {}
    pipeline._last_requested_zoom = 1.0
    pipeline._last_is_fitting = False
    pipeline._last_is_cropping = False
    view = MagicMock()
    view._is_fitting = False
    view.transform.return_value.m11.return_value = 1.0
    view.viewport.return_value.size.return_value = QtCore.QSize(200, 200)
    view.viewport.return_value.rect.return_value = QtCore.QRect(0, 0, 200, 200)
    view._crop_item.isVisible.return_value = False
    view.mapToScene.return_value.boundingRect.return_value = QtCore.QRectF(
        0, 0, 200, 200
    )
    view.sceneRect.return_value = QtCore.QRectF(0, 0, 200, 200)

    pipeline.set_view_reference(view)
    pipeline._render_pending = True
    pipeline._current_render_state_id = 1
    tile_key = (0, 0, 256)
    pipeline._tile_cache[tile_key] = "pending_1"
    pipeline._active_workers = 0

    with patch("pynegative.ui.imageprocessing.ImageProcessorWorker") as mock_worker:
        pipeline._process_pending_update()
        mock_worker.assert_called_once()
    assert pipeline._tile_cache[tile_key] == "pending_1"


def test_process_pending_update_keeps_pending_tile_when_workers_active(pipeline):
    pipeline.base_img_full = np.zeros((100, 100, 3), dtype=np.float32)
    pipeline._processing_params = {}
    pipeline._last_settings = {}
    pipeline._last_requested_zoom = 1.0
    pipeline._last_is_fitting = False
    pipeline._last_is_cropping = False
    view = MagicMock()
    view._is_fitting = False
    view.transform.return_value.m11.return_value = 1.0
    view.viewport.return_value.size.return_value = QtCore.QSize(200, 200)
    view.viewport.return_value.rect.return_value = QtCore.QRect(0, 0, 200, 200)
    view._crop_item.isVisible.return_value = False
    view.mapToScene.return_value.boundingRect.return_value = QtCore.QRectF(
        0, 0, 200, 200
    )
    view.sceneRect.return_value = QtCore.QRectF(0, 0, 200, 200)

    pipeline.set_view_reference(view)
    pipeline._render_pending = True
    pipeline._current_render_state_id = 1
    tile_key = (0, 0, 256)
    pipeline._tile_cache[tile_key] = "pending_1"
    pipeline._active_workers = 1

    with patch("pynegative.ui.imageprocessing.ImageProcessorWorker") as mock_worker:
        pipeline._process_pending_update()
        mock_worker.assert_not_called()
    assert pipeline._tile_cache[tile_key] == "pending_1"


def test_on_worker_finished(pipeline):
    mock_slot = MagicMock()
    pipeline.previewUpdated.connect(mock_slot)
    pix = MagicMock(spec=QtGui.QPixmap)

    pipeline._on_worker_finished(pix, 100, 100, 0, None, None, 1, (0, 0, 256), 1, 1)

    mock_slot.assert_called()
    assert pipeline._tile_cache[(0, 0, 256)] == "done_1"


def test_shutdown(pipeline):
    pipeline.render_timer.start(100)
    pipeline.shutdown()
    assert pipeline._shutting_down
    assert not pipeline.render_timer.isActive()


def test_set_histogram_enabled_triggers_update(pipeline):
    pipeline.base_img_full = MagicMock()
    pipeline._view_ref = MagicMock()
    pipeline._current_render_state_id = 10
    pipeline._tile_cache = {"some_key": "done_10"}

    pipeline.request_update = MagicMock()

    pipeline.set_histogram_enabled(True)

    assert pipeline._current_render_state_id > 10
    assert len(pipeline._tile_cache) == 0
    pipeline.request_update.assert_called_once()


def test_on_histogram_updated_accepts_recent_request(pipeline):
    mock_slot = MagicMock()
    pipeline.histogramUpdated.connect(mock_slot)
    pipeline._current_request_id = 12
    pipeline._last_processed_id = 10

    payload = {"R": [1], "G": [1], "B": [1], "Y": [1], "U": [1], "V": [1]}
    pipeline._on_histogram_updated(payload, 11)

    mock_slot.assert_called_once_with(payload)


def test_on_histogram_updated_drops_old_request(pipeline):
    mock_slot = MagicMock()
    pipeline.histogramUpdated.connect(mock_slot)
    pipeline._last_processed_id = 10

    payload = {"R": [1], "G": [1], "B": [1], "Y": [1], "U": [1], "V": [1]}
    pipeline._on_histogram_updated(payload, 9)

    mock_slot.assert_not_called()
