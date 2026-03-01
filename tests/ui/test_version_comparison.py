"""Tests for Phase 4: Version comparison via the comparison overlay."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6 import QtGui, QtWidgets

from pynegative.ui.editor_managers.comparison_manager import (
    ComparisonManager,
    ComparisonSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(qtbot):
    """Create a ComparisonManager wired to a lightweight parent widget."""
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    parent.image_processor = MagicMock()
    parent.image_processor.get_unedited_pixmap.return_value = QtGui.QPixmap(10, 10)
    parent.image_processor.render_snapshot_pixmap.return_value = QtGui.QPixmap(10, 10)
    parent.show_toast = MagicMock()

    mgr = ComparisonManager(parent)
    mgr.editor = parent

    # Minimal UI mocks (bypass setup_ui)
    mgr.canvas_frame = MagicMock()
    mgr.canvas_frame.height.return_value = 600
    mgr.canvas_frame.width.return_value = 800

    view = MagicMock()
    bg_item = MagicMock()
    bg_item.pixmap.return_value = QtGui.QPixmap(10, 10)
    view._bg_item = bg_item
    mgr.view = view

    mgr.comparison_btn = MagicMock()
    mgr.comparison_btn.isChecked.return_value = True
    mgr.comparison_handle = MagicMock()
    mgr.comparison_overlay = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# ComparisonSource enum
# ---------------------------------------------------------------------------


class TestComparisonSource:
    def test_values(self):
        assert ComparisonSource.UNEDITED.value == "unedited"
        assert ComparisonSource.CURRENT.value == "current"
        assert ComparisonSource.SNAPSHOT.value == "snapshot"


# ---------------------------------------------------------------------------
# ComparisonManager - default sources
# ---------------------------------------------------------------------------


class TestDefaultSources:
    def test_initial_sources(self, manager):
        assert manager._left_source == ComparisonSource.UNEDITED
        assert manager._right_source == ComparisonSource.CURRENT

    def test_toggle_resets_sources(self, manager):
        manager._left_source = ComparisonSource.SNAPSHOT
        manager._right_source = ComparisonSource.SNAPSHOT
        manager.toggle_comparison()
        assert manager._left_source == ComparisonSource.UNEDITED
        assert manager._right_source == ComparisonSource.CURRENT

    def test_toggle_emits_signal(self, manager):
        handler = MagicMock()
        manager.comparisonToggled.connect(handler)
        manager.toggle_comparison()
        handler.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# Snapshot comparison
# ---------------------------------------------------------------------------


class TestSnapshotComparison:
    def test_set_left_snapshot(self, manager):
        manager.enabled = True
        settings = {"exposure": 1.5}
        manager.set_left_snapshot(settings)

        assert manager._left_source == ComparisonSource.SNAPSHOT
        assert manager._left_snapshot_settings == settings
        manager.editor.image_processor.render_snapshot_pixmap.assert_called_once_with(
            settings
        )
        manager.comparison_overlay.setUneditedPixmap.assert_called_once()

    def test_set_right_snapshot(self, manager):
        manager.enabled = True
        settings = {"contrast": 0.3}
        manager.set_right_snapshot(settings)

        assert manager._right_source == ComparisonSource.SNAPSHOT
        assert manager._right_snapshot_settings == settings
        manager.editor.image_processor.render_snapshot_pixmap.assert_called_once_with(
            settings
        )
        manager.comparison_overlay.setEditedPixmap.assert_called_once()

    def test_set_left_snapshot_null_pixmap_no_update(self, manager):
        manager.editor.image_processor.render_snapshot_pixmap.return_value = (
            QtGui.QPixmap()
        )
        manager.enabled = True
        manager.set_left_snapshot({"exposure": 0.0})
        manager.comparison_overlay.setUneditedPixmap.assert_not_called()

    def test_reset_sources(self, manager):
        manager.enabled = True
        manager._left_source = ComparisonSource.SNAPSHOT
        manager._right_source = ComparisonSource.SNAPSHOT
        manager.reset_sources()
        assert manager._left_source == ComparisonSource.UNEDITED
        assert manager._right_source == ComparisonSource.CURRENT
        assert manager._left_snapshot_settings is None
        assert manager._right_snapshot_settings is None


# ---------------------------------------------------------------------------
# update_pixmaps respects source tracking
# ---------------------------------------------------------------------------


class TestUpdatePixmapsSourceAware:
    def test_update_unedited_when_source_is_unedited(self, manager):
        manager.enabled = True
        manager._left_source = ComparisonSource.UNEDITED
        pix = QtGui.QPixmap(5, 5)
        manager.update_pixmaps(unedited=pix)
        manager.editor.image_processor.render_snapshot_pixmap.assert_called_with({})
        manager.comparison_overlay.setUneditedPixmap.assert_called_once()

    def test_skip_unedited_when_source_is_snapshot(self, manager):
        manager.enabled = True
        manager._left_source = ComparisonSource.SNAPSHOT
        pix = QtGui.QPixmap(5, 5)
        manager.update_pixmaps(unedited=pix)
        manager.comparison_overlay.setUneditedPixmap.assert_not_called()

    def test_update_edited_when_source_is_current(self, manager):
        manager.enabled = True
        manager._right_source = ComparisonSource.CURRENT
        pix = QtGui.QPixmap(5, 5)
        manager.update_pixmaps(edited=pix)
        manager.comparison_overlay.setEditedPixmap.assert_called_once_with(pix)

    def test_skip_edited_when_source_is_snapshot(self, manager):
        manager.enabled = True
        manager._right_source = ComparisonSource.SNAPSHOT
        pix = QtGui.QPixmap(5, 5)
        manager.update_pixmaps(edited=pix)
        manager.comparison_overlay.setEditedPixmap.assert_not_called()


# ---------------------------------------------------------------------------
# render_snapshot_pixmap (ImageProcessingPipeline)
# ---------------------------------------------------------------------------


class TestRenderSnapshotPixmap:
    def test_returns_null_pixmap_when_no_image(self, qtbot):
        from pynegative.ui.imageprocessing import ImageProcessingPipeline

        thread_pool = MagicMock()
        pipeline = ImageProcessingPipeline(thread_pool)
        result = pipeline.render_snapshot_pixmap({"exposure": 0.0})
        assert result.isNull()

    @patch("pynegative.ui.imageprocessing.pynegative")
    @patch("pynegative.ui.imageprocessing.process_heavy_stage")
    @patch("pynegative.ui.imageprocessing.apply_fused_remap")
    @patch("pynegative.ui.imageprocessing.get_fused_geometry")
    @patch("pynegative.ui.imageprocessing.process_denoise_stage")
    @patch("pynegative.ui.imageprocessing.resolve_vignette_params")
    def test_renders_with_tier(
        self,
        mock_vig,
        mock_denoise,
        mock_geometry,
        mock_remap,
        mock_heavy,
        mock_core,
        qtbot,
    ):
        import numpy as np

        mock_vig.return_value = (0, 0, 0, 0.5, 0.5, 100, 100)

        img_3ch = np.zeros((100, 200, 3), dtype=np.float32)
        mock_core.apply_preprocess.return_value = img_3ch
        mock_denoise.return_value = img_3ch
        mock_geometry.return_value = (MagicMock(), 200, 100, None)
        mock_remap.return_value = img_3ch
        mock_heavy.return_value = img_3ch
        mock_core.apply_tone_map.return_value = (img_3ch, None)
        mock_core.apply_defringe.return_value = img_3ch
        mock_core.float32_to_uint8.return_value = np.zeros(
            (100, 200, 3), dtype=np.uint8
        )

        from pynegative.ui.imageprocessing import ImageProcessingPipeline

        thread_pool = MagicMock()
        pipeline = ImageProcessingPipeline(thread_pool)
        pipeline.base_img_full = np.zeros((400, 800, 3), dtype=np.float32)
        pipeline.tiers = {0.25: np.zeros((100, 200, 3), dtype=np.float32)}

        result = pipeline.render_snapshot_pixmap({"exposure": 1.0}, max_width=100)
        assert not result.isNull()
        mock_core.apply_preprocess.assert_called_once()
        mock_denoise.assert_called_once()
        mock_geometry.assert_called_once()

    @patch("pynegative.ui.imageprocessing.pynegative")
    @patch("pynegative.ui.imageprocessing.process_heavy_stage")
    @patch("pynegative.ui.imageprocessing.apply_fused_remap")
    @patch("pynegative.ui.imageprocessing.get_fused_geometry")
    @patch("pynegative.ui.imageprocessing.process_denoise_stage")
    @patch("pynegative.ui.imageprocessing.resolve_vignette_params")
    def test_exception_returns_null(
        self,
        mock_vig,
        mock_denoise,
        mock_geometry,
        mock_remap,
        mock_heavy,
        mock_core,
        qtbot,
    ):
        import numpy as np

        mock_vig.side_effect = RuntimeError("boom")

        from pynegative.ui.imageprocessing import ImageProcessingPipeline

        thread_pool = MagicMock()
        pipeline = ImageProcessingPipeline(thread_pool)
        pipeline.base_img_full = np.zeros((100, 200, 3), dtype=np.float32)
        pipeline.tiers = {}

        result = pipeline.render_snapshot_pixmap({"exposure": 0.0})
        assert result.isNull()


# ---------------------------------------------------------------------------
# Editor wiring (setLeftComparison / setRightComparison)
# ---------------------------------------------------------------------------


class TestEditorComparisonWiring:
    def test_on_set_left_comparison_calls_manager(self):
        from pynegative.ui.editor import EditorWidget

        with patch.object(EditorWidget, "__init__", lambda self, *a, **kw: None):
            editor = EditorWidget.__new__(EditorWidget)
            editor.history_panel = MagicMock()
            editor.comparison_manager = MagicMock()
            editor.comparison_manager.enabled = True

            snap = {"id": "abc", "settings": {"exposure": 2.0}}
            editor.history_panel.get_snapshot_by_id.return_value = snap

            editor._on_set_left_comparison("abc")
            editor.comparison_manager.set_left_snapshot.assert_called_once_with(
                {"exposure": 2.0}
            )

    def test_on_set_right_comparison_calls_manager(self):
        from pynegative.ui.editor import EditorWidget

        with patch.object(EditorWidget, "__init__", lambda self, *a, **kw: None):
            editor = EditorWidget.__new__(EditorWidget)
            editor.history_panel = MagicMock()
            editor.comparison_manager = MagicMock()
            editor.comparison_manager.enabled = True

            snap = {"id": "def", "settings": {"contrast": 0.5}}
            editor.history_panel.get_snapshot_by_id.return_value = snap

            editor._on_set_right_comparison("def")
            editor.comparison_manager.set_right_snapshot.assert_called_once_with(
                {"contrast": 0.5}
            )

    def test_comparison_skipped_when_disabled(self):
        from pynegative.ui.editor import EditorWidget

        with patch.object(EditorWidget, "__init__", lambda self, *a, **kw: None):
            editor = EditorWidget.__new__(EditorWidget)
            editor.history_panel = MagicMock()
            editor.comparison_manager = MagicMock()
            editor.comparison_manager.enabled = False
            editor.history_panel.get_snapshot_by_id.return_value = {
                "id": "abc",
                "settings": {"exposure": 0.2},
            }

            editor._on_set_left_comparison("abc")
            editor.comparison_manager.comparison_btn.setChecked.assert_called_once_with(
                True
            )
            editor.comparison_manager.toggle_comparison.assert_called_once()
            editor.comparison_manager.set_left_snapshot.assert_called_once_with(
                {"exposure": 0.2}
            )

    def test_comparison_skipped_when_snapshot_missing(self):
        from pynegative.ui.editor import EditorWidget

        with patch.object(EditorWidget, "__init__", lambda self, *a, **kw: None):
            editor = EditorWidget.__new__(EditorWidget)
            editor.history_panel = MagicMock()
            editor.comparison_manager = MagicMock()
            editor.comparison_manager.enabled = True
            editor.history_panel.get_snapshot_by_id.return_value = None

            editor._on_set_left_comparison("missing")
            editor.comparison_manager.set_left_snapshot.assert_not_called()
