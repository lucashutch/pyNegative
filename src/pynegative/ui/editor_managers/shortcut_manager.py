from functools import partial
from PySide6 import QtGui
from PySide6.QtCore import Qt


class ShortcutManager:
    """Manages keyboard shortcuts for the EditorWidget."""

    def __init__(self, editor):
        self.editor = editor

    def setup_shortcuts(self):
        editor = self.editor

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, editor, self.undo)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, editor, self.redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), editor, self.redo)
        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Copy, editor, self.handle_copy_shortcut
        )
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+C"),
            editor,
            editor.show_selective_copy_dialog,
        )
        QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Paste, editor, self.handle_paste_shortcut
        )
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+V"),
            editor,
            editor.show_selective_paste_dialog,
        )
        QtGui.QShortcut(
            QtGui.QKeySequence("F12"), editor, self.toggle_performance_overlay
        )
        QtGui.QShortcut(QtGui.QKeySequence("F10"), editor, editor._cycle_denoise_mode)
        QtGui.QShortcut(
            QtGui.QKeySequence("U"),
            editor,
            self.toggle_comparison,
        )

        for key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_0]:
            rating = 0 if key == Qt.Key_0 else key.value - Qt.Key_0.value
            QtGui.QShortcut(key, editor, partial(self.set_rating_by_number, rating))

        nav_left = QtGui.QShortcut(Qt.Key_Left, editor, self.navigate_previous)
        nav_left.setContext(Qt.ApplicationShortcut)
        nav_right = QtGui.QShortcut(Qt.Key_Right, editor, self.navigate_next)
        nav_right.setContext(Qt.ApplicationShortcut)

        # Zoom shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+="), editor, editor.view.zoom_in)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl++"), editor, editor.view.zoom_in)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+-"), editor, editor.view.zoom_out)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), editor, editor.view.reset_zoom)

    def handle_copy_shortcut(self):
        from pathlib import Path

        editor = self.editor
        selected_paths = editor.carousel_manager.get_selected_paths()
        if len(selected_paths) > 0:
            if editor.raw_path and str(editor.raw_path) in selected_paths:
                editor.settings_manager.copy_settings_from_current(
                    editor.image_processor.get_current_settings()
                )
            else:
                editor.settings_manager.copy_settings_from_path(Path(selected_paths[0]))
        else:
            editor.settings_manager.copy_settings_from_current(
                editor.image_processor.get_current_settings()
            )

    def handle_paste_shortcut(self):
        editor = self.editor
        selected_paths = editor.carousel_manager.get_selected_paths()
        if len(selected_paths) > 0:
            editor.settings_manager.paste_settings_to_selected(
                selected_paths, editor.editing_controls.apply_settings
            )
        else:
            editor.settings_manager.paste_settings_to_current(
                editor.editing_controls.apply_settings
            )

    def undo(self):
        state = self.editor.settings_manager.undo()
        if state:
            self._restore_state(state)

    def redo(self):
        state = self.editor.settings_manager.redo()
        if state:
            self._restore_state(state)

    def _restore_state(self, state):
        editor = self.editor
        settings = state["settings"]
        rating = state["rating"]
        editor.editing_controls.apply_settings(settings)
        editor.image_processor.set_processing_params(**settings)
        editor._request_update_from_view()
        editor.editing_controls.set_rating(rating)
        editor.settings_manager.set_current_settings(settings, rating)

    def toggle_performance_overlay(self):
        editor = self.editor
        is_visible = editor.floating_ui_manager.toggle_perf_visibility()
        editor.show_toast(f"Performance Overlay {'On' if is_visible else 'Off'}")

    def toggle_comparison(self):
        editor = self.editor
        if hasattr(editor.comparison_manager, "comparison_btn"):
            editor.comparison_manager.comparison_btn.animateClick()

    def set_rating_by_number(self, rating):
        editor = self.editor
        editor.preview_rating_widget.set_rating(rating)
        editor.editing_controls.set_rating(rating)
        editor._on_rating_changed(rating)

    def navigate_previous(self):
        if self.editor.isVisible():
            self.editor.carousel_manager.select_previous()

    def navigate_next(self):
        if self.editor.isVisible():
            self.editor.carousel_manager.select_next()
