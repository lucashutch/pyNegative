from unittest.mock import MagicMock, patch

from PySide6 import QtCore
from PySide6 import QtWidgets

from pynegative.ui.context_menu_positioning import get_menu_exec_position


class _ViewportWidget(QtWidgets.QWidget):
    def __init__(self, point):
        super().__init__()
        self._point = point
        self.calls = 0

    def mapToGlobal(self, pos):
        self.calls += 1
        return self._point


def test_uses_viewport_map_to_global_when_available(qapp):
    menu = MagicMock()
    menu.sizeHint.return_value = QtCore.QSize(200, 120)

    viewport = _ViewportWidget(QtCore.QPoint(780, 590))

    source_widget = MagicMock()
    source_widget.viewport.return_value = viewport
    source_widget.mapToGlobal.return_value = QtCore.QPoint(10, 10)

    screen = MagicMock()
    screen.availableGeometry.return_value = QtCore.QRect(0, 0, 800, 600)

    with patch(
        "pynegative.ui.context_menu_positioning.QtWidgets.QApplication.screenAt",
        return_value=screen,
    ):
        result = get_menu_exec_position(menu, source_widget, QtCore.QPoint(5, 5))

    assert viewport.calls == 1
    source_widget.mapToGlobal.assert_not_called()
    assert result == QtCore.QPoint(580, 470)


def test_falls_back_to_source_widget_when_viewport_not_qwidget():
    menu = MagicMock()
    menu.sizeHint.return_value = QtCore.QSize(100, 100)

    source_widget = MagicMock()
    source_widget.viewport.return_value = MagicMock()
    source_widget.mapToGlobal.return_value = QtCore.QPoint(20, 30)

    screen = MagicMock()
    screen.availableGeometry.return_value = QtCore.QRect(0, 0, 800, 600)

    with patch(
        "pynegative.ui.context_menu_positioning.QtWidgets.QApplication.screenAt",
        return_value=screen,
    ):
        result = get_menu_exec_position(menu, source_widget, QtCore.QPoint(1, 2))

    source_widget.mapToGlobal.assert_called_once_with(QtCore.QPoint(1, 2))
    assert result == QtCore.QPoint(20, 30)


def test_uses_primary_screen_when_screen_at_not_found():
    menu = MagicMock()
    menu.sizeHint.return_value = QtCore.QSize(300, 300)

    source_widget = MagicMock()
    source_widget.mapToGlobal.return_value = QtCore.QPoint(-100, -100)

    primary_screen = MagicMock()
    primary_screen.availableGeometry.return_value = QtCore.QRect(0, 0, 1920, 1080)

    with (
        patch(
            "pynegative.ui.context_menu_positioning.QtWidgets.QApplication.screenAt",
            return_value=None,
        ),
        patch(
            "pynegative.ui.context_menu_positioning.QtGui.QGuiApplication.primaryScreen",
            return_value=primary_screen,
        ),
    ):
        result = get_menu_exec_position(menu, source_widget, QtCore.QPoint(0, 0))

    assert result == QtCore.QPoint(0, 0)


def test_prefers_opening_upward_when_bottom_space_insufficient():
    menu = MagicMock()
    menu.sizeHint.return_value = QtCore.QSize(220, 180)

    source_widget = MagicMock()
    source_widget.mapToGlobal.return_value = QtCore.QPoint(300, 560)

    screen = MagicMock()
    screen.availableGeometry.return_value = QtCore.QRect(0, 0, 800, 600)

    with patch(
        "pynegative.ui.context_menu_positioning.QtWidgets.QApplication.screenAt",
        return_value=screen,
    ):
        result = get_menu_exec_position(menu, source_widget, QtCore.QPoint(0, 0))

    assert result == QtCore.QPoint(300, 380)


def test_prefers_opening_left_when_right_space_insufficient():
    menu = MagicMock()
    menu.sizeHint.return_value = QtCore.QSize(260, 120)

    source_widget = MagicMock()
    source_widget.mapToGlobal.return_value = QtCore.QPoint(760, 200)

    screen = MagicMock()
    screen.availableGeometry.return_value = QtCore.QRect(0, 0, 800, 600)

    with patch(
        "pynegative.ui.context_menu_positioning.QtWidgets.QApplication.screenAt",
        return_value=screen,
    ):
        result = get_menu_exec_position(menu, source_widget, QtCore.QPoint(0, 0))

    assert result == QtCore.QPoint(500, 200)
