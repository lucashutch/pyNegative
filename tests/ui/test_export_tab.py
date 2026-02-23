import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PySide6 import QtWidgets, QtCore
from pynegative.ui.export_tab import ExportWidget


@pytest.fixture
def thread_pool():
    return QtCore.QThreadPool()


@pytest.fixture
def export_widget(qtbot, thread_pool):
    # Patch all managers and dialogs before instantiation
    with patch("pynegative.ui.export_tab.ExportSettingsManager"), patch(
        "pynegative.ui.export_tab.ExportGalleryManager"
    ) as mock_gallery_class, patch(
        "pynegative.ui.export_tab.RenameSettingsManager"
    ) as mock_rename_settings_class, patch("pynegative.ui.export_tab.ExportJob"), patch(
        "pynegative.ui.export_tab.RenamePreviewDialog"
    ) as mock_dialog_class:
        # Ensure gallery_manager.get_widget() returns a QWidget
        mock_gallery_class.return_value.get_widget.return_value = QtWidgets.QWidget()
        # Ensure dialog mock works
        mock_dialog_class.return_value.exec.return_value = (
            QtWidgets.QDialog.DialogCode.Accepted
        )

        widget = ExportWidget(thread_pool)
        # Set default mock behaviors to avoid UI hangs
        widget.settings_manager.get_current_settings.return_value = {"format": "HEIF", "jpeg_quality": 95, "heif_quality": 95}
        widget.rename_settings_manager.get_current_settings.return_value = {"enabled": False}

        qtbot.addWidget(widget)
        widget.show()
        return widget


def test_export_widget_init(export_widget):
    assert "0 items selected" in export_widget.selection_label.text()


def test_on_selection_count_changed(export_widget):
    export_widget._on_selection_count_changed(5)
    assert "5 items selected" in export_widget.selection_label.text()
    assert export_widget.export_button.isEnabled() is True


def test_on_format_changed(export_widget):
    export_widget.settings_manager.get_current_settings.return_value = {
        "format": "JPEG"
    }
    export_widget.format_combo.addItem("JPEG")
    export_widget.format_combo.setCurrentText("JPEG")

    export_widget._on_format_changed(0)
    assert export_widget.jpeg_settings.isVisible()
    assert not export_widget.heif_settings.isVisible()


def test_start_export_flow(export_widget):
    export_widget.gallery_manager.get_selected_paths.return_value = ["img1.ARW"]
    export_widget.settings_manager.get_current_settings.return_value = {
        "format": "JPEG"
    }
    export_widget._export_destination = Path(QtCore.QDir.tempPath())

    with patch.object(
        export_widget.export_job, "start_export", return_value=True
    ) as mock_start:
        export_widget.start_export()
        mock_start.assert_called()
        assert export_widget.progress_bar.isVisible()


def test_export_completed(export_widget):
    export_widget.progress_bar.setVisible(True)
    export_widget._on_export_completed(10, 0, 10)
    assert not export_widget.progress_bar.isVisible()


def test_rename_preview_dialog_trigger(export_widget):
    export_widget.rename_settings_manager.get_current_settings.return_value = {
        "enabled": True
    }
    export_widget.gallery_manager.get_selected_paths.return_value = ["img1.ARW"]
    export_widget.format_combo.addItem("JPEG")

    # Use the mock dialog created in the fixture
    mock_dialog = export_widget._rename_preview_dialog
    mock_dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
    mock_dialog.get_rename_mapping.return_value = {"img1.ARW": "new1.jpg"}

    result = export_widget._show_rename_preview()
    assert result is True
    mock_dialog.set_preview_data.assert_called()


def test_load_folder(export_widget, tmp_path):
    # Mock main window (parent of parent...) or use a real one
    main_win = MagicMock()
    main_win.filter_combo.currentText.return_value = "Match"
    main_win.filter_rating_widget.rating.return_value = 0

    with patch.object(export_widget, "window", return_value=main_win):
        export_widget.load_folder(str(tmp_path))
        assert export_widget.current_folder == tmp_path
        export_widget.gallery_manager.load_folder.assert_called()


def test_on_preset_applied(export_widget):
    export_widget.settings_manager.get_current_settings.return_value = {
        "format": "HEIF",
        "jpeg_quality": 90,
        "heif_quality": 85,
        "heif_bit_depth": "10-bit",
        "max_width": "2000",
        "max_height": "1000",
    }
    export_widget.format_combo.addItems(["JPEG", "HEIF"])
    export_widget.heif_bit_depth.addItems(["8-bit", "10-bit", "12-bit"])

    export_widget._on_preset_applied("Test")

    assert export_widget.format_combo.currentText() == "HEIF"
    assert export_widget.heif_quality.value() == 85
    assert export_widget.max_width.text() == "2000"


def test_choose_export_destination(export_widget):
    with patch(
        "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
        return_value="/tmp/test_export",
    ):
        export_widget._choose_export_destination()
        assert str(export_widget._export_destination) == "/tmp/test_export"
        assert "test_export" in export_widget.dest_path_label.text()
