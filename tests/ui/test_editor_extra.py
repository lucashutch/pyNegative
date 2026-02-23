import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt, QPointF
from pynegative.ui.editor import EditorWidget

class MockSignal:
    def connect(self, slot): pass
    def emit(self, *args): pass
    def disconnect(self, slot=None): pass

class MockQWidget(QtWidgets.QWidget):
    def __getattr__(self, name):
        # Cache mocks to return the same one
        if name.endswith("Changed") or name.endswith("Clicked") or name.endswith("Requested") or name.endswith("Applied") or name == "doubleClicked":
            m = MockSignal()
        else:
            m = MagicMock()
        setattr(self, name, m)
        return m

@pytest.fixture
def editor(qtbot):
    mock_pool = MagicMock()

    # Mocking ComparisonManager too
    mock_comparison_manager = MagicMock()
    mock_comparison_manager.comparison_btn = MagicMock()

    mock_carousel_manager = MagicMock()
    mock_carousel_manager.get_widget.return_value = MockQWidget()

    mock_floating_ui_manager = MagicMock()

    with patch("pynegative.ui.editor.SettingsManager", return_value=MagicMock()), \
         patch("pynegative.ui.editor.EditingControls", return_value=MockQWidget()), \
         patch("pynegative.ui.editor.ImageProcessingPipeline", return_value=MagicMock()), \
         patch("pynegative.ui.editor.CarouselManager", return_value=mock_carousel_manager), \
         patch("pynegative.ui.editor.MetadataPanel", return_value=MockQWidget()), \
         patch("pynegative.ui.editor.ZoomControls", return_value=MockQWidget()), \
         patch("pynegative.ui.editor.ZoomableGraphicsView", return_value=MockQWidget()), \
         patch("pynegative.ui.editor.ComparisonManager", return_value=mock_comparison_manager), \
         patch("pynegative.ui.editor.FloatingUIManager", return_value=mock_floating_ui_manager):

        # CarouselManager needs get_widget to return a QWidget
        widget = EditorWidget(mock_pool)
        qtbot.addWidget(widget)
        return widget

def test_editor_load_image(editor, tmp_path):
    path = tmp_path / "test.ARW"
    path.write_text("fake")

    with patch("pynegative.core.load_cached_thumbnail") as mock_thumb:
        mock_thumb.return_value = (None, None)
        with patch("pynegative.ui.editor.RawLoader") as mock_loader:
            editor.load_image(path)
            assert editor.raw_path == path
            mock_loader.assert_called()

def test_editor_setting_changed(editor):
    editor._on_setting_changed("val_exposure", 1.5)
    editor.image_processor.set_processing_params.assert_called()

def test_editor_auto_save(editor, tmp_path):
    editor.raw_path = tmp_path / "test.ARW"
    editor.editing_controls.get_all_settings.return_value = {"exposure": 0.0}
    editor.editing_controls.star_rating_widget.rating.return_value = 4

    editor.view.sceneRect.return_value = MagicMock()
    editor.view.sceneRect.return_value.width.return_value = 0

    editor._auto_save_sidecar()
    editor.settings_manager.auto_save_sidecar.assert_called()

def test_editor_undo_redo(editor):
    state = {"settings": {"exposure": 1.0}, "rating": 5}
    editor.settings_manager.undo.return_value = state

    editor._undo()
    editor.settings_manager.undo.assert_called()

    editor.settings_manager.redo.return_value = state
    editor._redo()
    editor.settings_manager.redo.assert_called()
