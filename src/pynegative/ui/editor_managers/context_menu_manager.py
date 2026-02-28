from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ... import core as pynegative


def _position_menu_on_screen(menu, global_pos):
    """Position menu to stay within screen bounds."""
    screen = QtWidgets.QApplication.screenAt(global_pos)
    if not screen:
        return global_pos

    screen_geo = screen.availableGeometry()
    menu_size = menu.sizeHint()

    x = global_pos.x()
    y = global_pos.y()

    if y + menu_size.height() > screen_geo.bottom():
        y = screen_geo.bottom() - menu_size.height()

    if x + menu_size.width() > screen_geo.right():
        x = screen_geo.right() - menu_size.width()

    return QtCore.QPoint(x, y)


class ContextMenuManager:
    """Manages Context Menus within the EditorWidget context."""

    def __init__(self, editor):
        self.editor = editor

    def show_main_photo_context_menu(self, pos):
        if not self.editor.raw_path:
            return
        menu = QtWidgets.QMenu(self.editor)
        copy_action = menu.addAction("Copy Settings")
        copy_action.triggered.connect(
            lambda: self.editor.settings_manager.copy_settings_from_current(
                self.editor.image_processor.get_current_settings()
            )
        )
        copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)

        copy_selective_action = menu.addAction("Copy Settings Selective...")
        copy_selective_action.triggered.connect(self.editor.show_selective_copy_dialog)
        copy_selective_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))

        paste_action = menu.addAction("Paste Settings")
        paste_action.triggered.connect(
            lambda: self.editor.settings_manager.paste_settings_to_current(
                self.editor.editing_controls.apply_settings
            )
        )
        paste_action.setEnabled(self.editor.settings_manager.has_clipboard_content())
        paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)

        paste_selective_action = menu.addAction("Paste Settings Selective...")
        paste_selective_action.triggered.connect(
            self.editor.show_selective_paste_dialog
        )
        paste_selective_action.setEnabled(
            self.editor.settings_manager.has_clipboard_content()
        )
        paste_selective_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))

        adjusted_pos = _position_menu_on_screen(menu, self.editor.view.mapToGlobal(pos))
        menu.exec_(adjusted_pos)

    def handle_carousel_context_menu(self, context_type, data):
        if context_type == "carousel":
            pos, item_path, carousel_widget = data
            menu = QtWidgets.QMenu(self.editor)
            selected_paths = carousel_widget.get_selected_paths()
            if item_path in selected_paths:
                copy_action = menu.addAction("Copy Settings from Selected")
                copy_action.triggered.connect(
                    lambda: self.editor.settings_manager.copy_settings_from_path(
                        Path(selected_paths[0]) if selected_paths else Path(item_path)
                    )
                )
                copy_action.setShortcut(QtGui.QKeySequence.StandardKey.Copy)

                copy_selective_action = menu.addAction("Copy Settings Selective...")
                copy_selective_action.triggered.connect(
                    self.editor.show_selective_copy_dialog
                )
                copy_selective_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))
            else:
                copy_action = menu.addAction(
                    f"Copy Settings from {Path(item_path).name}"
                )
                copy_action.triggered.connect(
                    lambda: self.editor.settings_manager.copy_settings_from_path(
                        item_path
                    )
                )

                copy_selective_action = menu.addAction("Copy Settings Selective...")
                copy_selective_action.triggered.connect(
                    self.editor.show_selective_copy_dialog
                )
                copy_selective_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))

            paste_action = menu.addAction("Paste Settings to Selected")
            paste_action.triggered.connect(
                lambda: self.editor.settings_manager.paste_settings_to_selected(
                    selected_paths, self.editor.editing_controls.apply_settings
                )
            )
            paste_action.setEnabled(
                self.editor.settings_manager.has_clipboard_content()
                and len(selected_paths) > 0
            )
            paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)

            paste_selective_action = menu.addAction("Paste Settings Selective...")
            paste_selective_action.triggered.connect(
                self.editor.show_selective_paste_dialog
            )
            paste_selective_action.setEnabled(
                self.editor.settings_manager.has_clipboard_content()
                and len(selected_paths) > 0
            )
            paste_selective_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))

            menu.addSeparator()
            select_all_action = menu.addAction("Select All")
            select_all_action.triggered.connect(carousel_widget.select_all_items)
            select_all_action.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)

            menu.addSeparator()
            left_action = menu.addAction("Set as Left Comparison Image")
            left_action.triggered.connect(
                lambda checked=False, p=item_path: self._set_carousel_comparison(
                    p, "left"
                )
            )
            right_action = menu.addAction("Set as Right Comparison Image")
            right_action.triggered.connect(
                lambda checked=False, p=item_path: self._set_carousel_comparison(
                    p, "right"
                )
            )

            adjusted_pos = _position_menu_on_screen(
                menu, carousel_widget.mapToGlobal(pos)
            )
            menu.exec_(adjusted_pos)

    def _set_carousel_comparison(self, image_path: str, side: str):
        """Load sidecar settings from *image_path* and set as comparison."""
        if not self.editor.comparison_manager.enabled:
            self.editor.comparison_manager.comparison_btn.setChecked(True)
            self.editor.comparison_manager.toggle_comparison()

        settings = pynegative.load_sidecar(image_path)
        if settings is None:
            settings = {}
        if side == "left":
            self.editor.comparison_manager.set_left_snapshot(settings)
        else:
            self.editor.comparison_manager.set_right_snapshot(settings)
