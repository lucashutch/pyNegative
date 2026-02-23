import pytest
import math
from unittest.mock import MagicMock, patch
from PySide6 import QtCore
from PySide6.QtCore import QRectF, QObject
from pynegative.ui.editor_managers.crop_manager import CropManager

class MockEditor(QObject):
    def __init__(self):
        super().__init__()
        self.view = MagicMock()
        self.image_processor = MagicMock()
        self.editing_controls = MagicMock()
        self.settings_manager = MagicMock()
        self.save_timer = MagicMock()

    def _request_update_from_view(self): pass
    def show_toast(self, msg): pass
    def _auto_save_sidecar(self): pass

def test_crop_manager_toggle_crop():
    editor = MockEditor()
    manager = CropManager(editor)

    # Mock settings
    editor.image_processor.get_current_settings.return_value = {"rotation": 10.0}
    editor.image_processor.base_img_full = MagicMock()
    editor.image_processor.base_img_full.shape = (1000, 1500, 3)

    # Mock view methods to return real QRectF
    editor.view.get_crop_rect.return_value = QRectF(100, 100, 800, 600)
    editor.view.sceneRect.return_value = QRectF(0, 0, 1500, 1000)

    # Enable crop
    manager.toggle_crop(True)
    editor.view.set_crop_mode.assert_called_with(True)
    editor.view.set_rotation.assert_called_with(10.0)

    # Disable crop
    manager.toggle_crop(False)
    editor.view.set_crop_mode.assert_called_with(False)
    editor.image_processor.set_processing_params.assert_called()

def test_crop_manager_rotation():
    editor = MockEditor()
    manager = CropManager(editor)

    manager.handle_rotation_changed(15.0)
    assert manager._pending_rotation_from_handle == 15.0
    editor.editing_controls.set_slider_value.assert_called_with("rotation", 15.0, silent=True)

    manager._apply_pending_rotation()
    editor.image_processor.set_processing_params.assert_called_with(rotation=15.0)
    assert manager._pending_rotation_from_handle is None

def test_text_to_ratio():
    manager = CropManager(MockEditor())
    assert manager._text_to_ratio("1:1") == 1.0
    assert manager._text_to_ratio("16:9") == 16.0 / 9.0
    assert manager._text_to_ratio("Free") is None
