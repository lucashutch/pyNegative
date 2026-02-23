from pathlib import Path
from unittest.mock import MagicMock, patch
from PySide6 import QtCore

from pynegative.ui.editor import EditorWidget


@patch("PySide6.QtCore.QThreadPool.start")
def test_editor_widget_coverage(mock_start, qapp):
    thread_pool = QtCore.QThreadPool()
    editor = EditorWidget(thread_pool)

    editor.floating_ui_manager.show_toast = MagicMock()

    # general methods
    editor._load_carousel_height()
    editor._on_carousel_splitter_moved(100, 1)
    editor._save_ui_settings()
    editor.clear()

    # load image
    with patch("pynegative.ui.editor.RawLoader"):
        with patch(
            "pynegative.ui.editor.pynegative.load_cached_thumbnail"
        ) as mock_thumb:
            mock_thumb.return_value = (None, None)
            editor.load_image("/tmp/test.cr2")

    # rating
    editor.raw_path = Path("/tmp/test.cr2")
    editor.update_rating_for_path("/tmp/test.cr2", 4)
    editor._on_rating_changed(5)
    editor._on_preview_rating_changed(3)

    # carousel
    editor.load_carousel_folder("/tmp/")
    editor.set_carousel_images(["/tmp/img1.cr2"], "/tmp/img1.cr2")

    # open
    editor.open("/tmp/img1.cr2")
    editor.open("/tmp/img1.cr2", image_list=["/tmp/img1.cr2"])

    # preview mode
    editor.set_preview_mode(True)
    editor.set_preview_mode(False)

    # settings changed
    editor.image_processor = MagicMock()
    editor.image_processor.get_current_settings.return_value = {
        "crop": (0, 0, 1, 1),
        "rotation": 0,
    }
    editor._on_setting_changed("exposure", 1.0)
    editor._on_setting_changed("crop", None)
    editor._on_setting_changed("flip_h", True)
    editor._on_setting_changed("flip_v", True)

    editor.image_processor.base_img_full = MagicMock()
    editor.image_processor.base_img_full.shape = (100, 100)
    editor._on_setting_changed("rotation", 45.0)

    editor._on_setting_changed("lens_name_override", "Test")

    # Auto WB
    editor.image_processor.base_img_preview = MagicMock()
    with patch("pynegative.ui.editor.pynegative.calculate_auto_wb") as mock_auto_wb:
        mock_auto_wb.return_value = {"temperature": 5000, "tint": 0}
        editor._on_auto_wb_requested()

    # Preset
    editor._on_preset_applied("test")

    # Denoise mode
    editor._cycle_denoise_mode()

    # Raw loaded
    import numpy as np

    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    editor.raw_path = Path("/tmp/test.cr2")

    # Mocking out lens resolver
    with (
        patch("pynegative.io.lens_resolver.resolve_lens_profile") as mock_resolve,
        patch("PySide6.QtWidgets.QMessageBox.critical"),
    ):
        mock_resolve.return_value = (None, None)
        editor._on_raw_loaded("/tmp/test.cr2", dummy_img, {"exposure": 0})
        editor._on_raw_loaded("/tmp/test.cr2", None, {})  # should hit QMessageBox

    # Update from view
    editor._request_update_from_view()

    # auto save sidecar
    editor._auto_save_sidecar()

    # Carousel events
    editor._on_carousel_image_selected("/tmp/other.cr2")
    editor._on_carousel_selection_changed(["/tmp/test.cr2"])
    editor._on_carousel_keyboard_navigation(["/tmp/test.cr2"])

    # Undo / Redo
    with patch.object(editor.settings_manager, "undo") as mock_undo:
        mock_undo.return_value = {"settings": {}, "rating": 0}
        editor.shortcut_manager.undo()

    with patch.object(editor.settings_manager, "redo") as mock_redo:
        mock_redo.return_value = {"settings": {}, "rating": 0}
        editor.shortcut_manager.redo()

    # Perf
    editor._on_performance_measured(10.5)
    editor.shortcut_manager.toggle_performance_overlay()

    # Context menus
    with patch("PySide6.QtWidgets.QMenu.exec_"):
        editor.context_menu_manager.show_main_photo_context_menu(QtCore.QPoint(0, 0))

        carousel_mock = MagicMock()
        carousel_mock.get_selected_paths.return_value = ["/tmp/test.cr2"]
        editor.context_menu_manager.handle_carousel_context_menu(
            "carousel", (QtCore.QPoint(0, 0), "/tmp/test.cr2", carousel_mock)
        )

    # Shortcuts
    editor.carousel_manager.get_selected_paths = MagicMock(
        return_value=["/tmp/test.cr2"]
    )
    editor.shortcut_manager.handle_copy_shortcut()
    editor.shortcut_manager.handle_paste_shortcut()
