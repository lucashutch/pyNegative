from pathlib import Path
from PySide6 import QtGui, QtWidgets


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
        paste_action = menu.addAction("Paste Settings")
        paste_action.triggered.connect(
            lambda: self.editor.settings_manager.paste_settings_to_current(
                self.editor.editing_controls.apply_settings
            )
        )
        paste_action.setEnabled(self.editor.settings_manager.has_clipboard_content())
        paste_action.setShortcut(QtGui.QKeySequence.StandardKey.Paste)
        menu.exec_(self.editor.view.mapToGlobal(pos))

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
            else:
                copy_action = menu.addAction(
                    f"Copy Settings from {Path(item_path).name}"
                )
                copy_action.triggered.connect(
                    lambda: self.editor.settings_manager.copy_settings_from_path(
                        item_path
                    )
                )
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
            menu.addSeparator()
            select_all_action = menu.addAction("Select All")
            select_all_action.triggered.connect(carousel_widget.select_all_items)
            select_all_action.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)
            menu.exec_(carousel_widget.mapToGlobal(pos))
