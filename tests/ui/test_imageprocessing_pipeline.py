import pytest
import numpy as np
from PySide6 import QtCore
from pynegative.ui.imageprocessing import ImageProcessingPipeline
from unittest.mock import MagicMock, patch


@pytest.fixture
def pipeline(qtbot):
    mock_thread_pool = MagicMock()
    pipeline = ImageProcessingPipeline(mock_thread_pool)

    # Mock base image so request_update doesn't return early
    pipeline.base_img_full = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.tiers = {0.5: np.zeros((50, 50, 3)), 0.25: np.zeros((25, 25, 3))}

    # Mock view_ref
    mock_view = MagicMock()
    mock_view.viewport().rect.return_value = QtCore.QRect(0, 0, 100, 100)
    mock_view.mapToScene.return_value.boundingRect.return_value = QtCore.QRectF(
        0, 0, 100, 100
    )
    pipeline._view_ref = mock_view

    return pipeline


def test_cache_invalidation_batching(pipeline, qtbot):
    with patch("pynegative.ui.imageprocessing.logger") as mock_logger:
        # Multiple geometry changes
        pipeline.set_processing_params(rotation=90)
        pipeline.set_processing_params(flip_h=True)
        pipeline.set_processing_params(crop=(0, 0, 0.5, 0.5))

        # Verify no log yet
        invalid_logs = [
            call
            for call in mock_logger.debug.call_args_list
            if "Invalidating pipeline cache" in call[0][0]
        ]
        assert len(invalid_logs) == 0

        # Trigger update (normally called after set_processing_params)
        pipeline.request_update()

        # Wait for render_timer (20ms)
        qtbot.wait(50)

        # Verify log happened exactly once
        invalid_logs = [
            call
            for call in mock_logger.debug.call_args_list
            if "Invalidating pipeline cache" in call[0][0]
        ]
        assert len(invalid_logs) == 1
        assert "geom_changed=True" in invalid_logs[0][0][0]


def test_cache_invalidation_lens_and_geom(pipeline, qtbot):
    with patch("pynegative.ui.imageprocessing.logger") as mock_logger:
        # Mixed changes
        pipeline.set_processing_params(rotation=90)
        pipeline.set_lens_info({"model": "test"})

        pipeline.request_update()
        qtbot.wait(50)

        # Verify log happened exactly once with both flags
        invalid_logs = [
            call
            for call in mock_logger.debug.call_args_list
            if "Invalidating pipeline cache" in call[0][0]
        ]
        assert len(invalid_logs) == 1
        assert "lens_changed=True" in invalid_logs[0][0][0]
        assert "geom_changed=True" in invalid_logs[0][0][0]
