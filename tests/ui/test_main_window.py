import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PySide6 import QtWidgets, QtGui
from pynegative.ui.main_window import MainWindow


class MockSignal:
    def connect(self, slot):
        pass

    def emit(self, *args):
        pass

    def disconnect(self, slot=None):
        pass


class MockQWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Some explicit properties to avoid common issues
        self.raw_path = None
        self.current_folder = None
        self._metadata_panel_visible = False
        self._is_large_preview = False
        self._gallery_metadata_visible = False

    def __getattr__(self, name):
        if (
            name.endswith("Changed")
            or name == "imageSelected"
            or name == "folderLoaded"
        ):
            m = MockSignal()
        else:
            m = MagicMock()
        setattr(self, name, m)
        return m


@pytest.fixture
def main_window(qtbot):
    # Mock all major sub-widgets to avoid real logic
    with patch(
        "pynegative.ui.main_window.GalleryWidget", return_value=MockQWidget()
    ), patch(
        "pynegative.ui.main_window.EditorWidget", return_value=MockQWidget()
    ), patch(
        "pynegative.ui.main_window.ExportWidget", return_value=MockQWidget()
    ), patch("pynegative.ui.main_window.StarRatingWidget", return_value=MockQWidget()):
        win = MainWindow()
        qtbot.addWidget(win)
        return win


def test_main_window_init(main_window):
    assert main_window.windowTitle() == "pyNegative"
    assert main_window.stack.count() == 3


def test_main_window_switches(main_window):
    main_window.switch_to_edit()
    assert main_window.btn_edit.isChecked()

    main_window.switch_to_export()
    assert main_window.btn_export.isChecked()

    main_window.switch_to_gallery()
    assert main_window.btn_gallery.isChecked()


def test_main_window_metadata_toggle(main_window):
    main_window.switch_to_edit()
    # Mock isChecked since we manually set it above
    with patch.object(main_window.metadata_btn, "isChecked", return_value=True):
        main_window._on_metadata_toggle()

    main_window.editor.metadata_panel.setVisible.assert_called_with(True)


def test_on_gallery_list_changed(main_window):
    # Setup editor with a fake path
    main_window.editor.raw_path = Path("test.ARW")

    image_list = ["test.ARW", "other.ARW"]
    main_window._on_gallery_list_changed(image_list)
    main_window.editor.set_carousel_images.assert_called()


def test_close_event(main_window):
    event = QtGui.QCloseEvent()
    main_window.closeEvent(event)
    main_window.editor.image_processor.shutdown.assert_called()
    assert event.isAccepted()
