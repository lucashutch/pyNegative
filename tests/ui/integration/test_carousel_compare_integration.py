from unittest.mock import MagicMock, patch

from PySide6 import QtCore

from pynegative.ui.editor_managers.context_menu_manager import ContextMenuManager


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self):
        for callback in self._callbacks:
            callback()


class _FakeAction:
    def __init__(self, text):
        self.text = text
        self.triggered = _FakeSignal()

    def setShortcut(self, *_args, **_kwargs):
        return None

    def setEnabled(self, *_args, **_kwargs):
        return None


class _FakeMenu:
    def __init__(self, *_args, **_kwargs):
        self.actions = []

    def addAction(self, text):
        action = _FakeAction(text)
        self.actions.append(action)
        return action

    def addSeparator(self):
        return None

    def sizeHint(self):
        return QtCore.QSize(220, 180)

    def exec_(self, _pos):
        return None


def _build_manager():
    editor = MagicMock()
    editor.raw_path = "/img/current.dng"
    editor.settings_manager = MagicMock()
    editor.settings_manager.has_clipboard_content.return_value = False
    editor.editing_controls = MagicMock()
    editor.comparison_manager = MagicMock()
    editor.comparison_manager.enabled = False
    editor.comparison_manager.comparison_btn = MagicMock()
    return ContextMenuManager(editor)


def _build_carousel_widget():
    carousel_widget = MagicMock()
    carousel_widget.get_selected_paths.return_value = []
    carousel_widget.select_all_items = MagicMock()
    return carousel_widget


def _find_action(menu, text):
    for action in menu.actions:
        if action.text == text:
            return action
    raise AssertionError(f"Action '{text}' was not found in menu")


class TestCarouselCompareIntegration:
    def test_cross_photo_left_action_uses_clicked_image_pixmap(self):
        mgr = _build_manager()
        carousel_widget = _build_carousel_widget()
        menu = _FakeMenu()

        mock_pixmap = MagicMock()
        mock_pixmap.isNull.return_value = False

        with (
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.QtWidgets.QMenu",
                return_value=menu,
            ),
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.get_menu_exec_position",
                return_value=QtCore.QPoint(0, 0),
            ),
            patch.object(
                ContextMenuManager,
                "_load_comparison_pixmap",
                return_value=mock_pixmap,
            ),
            patch("pynegative.ui.editor_managers.context_menu_manager.pynegative"),
        ):
            mgr.handle_carousel_context_menu(
                "carousel",
                (QtCore.QPoint(5, 5), "/img/other_left.dng", carousel_widget),
            )

            _find_action(menu, "Set as Left Comparison Image").triggered.emit()

        mgr.editor.comparison_manager.comparison_btn.setChecked.assert_called_once_with(
            True
        )
        mgr.editor.comparison_manager.toggle_comparison.assert_called_once()
        mgr.editor.comparison_manager.set_left_pixmap.assert_called_once_with(
            mock_pixmap,
            "other_left.dng",
        )
        mgr.editor.comparison_manager.set_left_snapshot.assert_not_called()

    def test_cross_photo_right_action_uses_clicked_image_pixmap(self):
        mgr = _build_manager()
        carousel_widget = _build_carousel_widget()
        menu = _FakeMenu()

        mock_pixmap = MagicMock()
        mock_pixmap.isNull.return_value = False

        with (
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.QtWidgets.QMenu",
                return_value=menu,
            ),
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.get_menu_exec_position",
                return_value=QtCore.QPoint(0, 0),
            ),
            patch.object(
                ContextMenuManager,
                "_load_comparison_pixmap",
                return_value=mock_pixmap,
            ),
            patch("pynegative.ui.editor_managers.context_menu_manager.pynegative"),
        ):
            mgr.handle_carousel_context_menu(
                "carousel",
                (QtCore.QPoint(5, 5), "/img/other_right.dng", carousel_widget),
            )

            _find_action(menu, "Set as Right Comparison Image").triggered.emit()

        mgr.editor.comparison_manager.comparison_btn.setChecked.assert_called_once_with(
            True
        )
        mgr.editor.comparison_manager.toggle_comparison.assert_called_once()
        mgr.editor.comparison_manager.set_right_pixmap.assert_called_once_with(
            mock_pixmap,
            "other_right.dng",
        )
        mgr.editor.comparison_manager.set_right_snapshot.assert_not_called()

    def test_same_photo_left_action_falls_back_to_snapshot_settings(self):
        mgr = _build_manager()
        carousel_widget = _build_carousel_widget()
        menu = _FakeMenu()
        mgr.editor.comparison_manager.enabled = True

        with (
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.QtWidgets.QMenu",
                return_value=menu,
            ),
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.get_menu_exec_position",
                return_value=QtCore.QPoint(0, 0),
            ),
            patch.object(
                ContextMenuManager,
                "_load_comparison_pixmap",
            ) as mock_load_pixmap,
            patch(
                "pynegative.ui.editor_managers.context_menu_manager.pynegative.load_sidecar",
                return_value={"exposure": 0.6},
            ) as mock_load_sidecar,
        ):
            mgr.handle_carousel_context_menu(
                "carousel",
                (QtCore.QPoint(5, 5), "/img/current.dng", carousel_widget),
            )

            _find_action(menu, "Set as Left Comparison Image").triggered.emit()

        mock_load_pixmap.assert_not_called()
        mock_load_sidecar.assert_called_once_with("/img/current.dng")
        mgr.editor.comparison_manager.set_left_snapshot.assert_called_once_with(
            {"exposure": 0.6}
        )
