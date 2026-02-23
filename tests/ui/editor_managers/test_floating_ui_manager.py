import pytest
from unittest.mock import MagicMock
from PySide6 import QtWidgets, QtCore
from pynegative.ui.editor_managers.floating_ui_manager import FloatingUIManager


class MockEditor(QtCore.QObject):
    pass


@pytest.fixture
def manager(qtbot):
    editor = MockEditor()
    return FloatingUIManager(editor)


def test_setup_ui(manager):
    frame = QtWidgets.QFrame()
    manager.setup_ui(frame)
    assert isinstance(manager.perf_label, QtWidgets.QLabel)
    assert manager.perf_label.parent() == frame
    assert manager.toast is not None


def test_reposition(manager):
    frame = QtWidgets.QFrame()
    frame.resize(800, 600)
    manager.setup_ui(frame)
    frame.show()

    view = QtWidgets.QWidget(frame)
    view.resize(600, 400)
    view.move(10, 10)

    zoom_ctrl = QtWidgets.QWidget(frame)
    zoom_ctrl.resize(50, 50)

    rating = QtWidgets.QWidget(frame)
    rating.resize(100, 20)

    comp_mgr = MagicMock()
    comp_mgr.comparison_btn = QtWidgets.QPushButton(frame)
    comp_mgr.comparison_btn.resize(30, 30)

    manager.reposition(view, zoom_ctrl, rating, comp_mgr)

    # Check zoom ctrl position
    assert zoom_ctrl.x() > 0
    assert zoom_ctrl.y() > 0

    # Check comparison btn
    assert comp_mgr.comparison_btn.isVisible()


def test_perf_visibility(manager):
    frame = QtWidgets.QFrame()
    manager.setup_ui(frame)
    frame.show()

    manager.perf_label.hide()
    assert manager.toggle_perf_visibility() is True
    assert manager.perf_label.isVisible()
    assert manager.toggle_perf_visibility() is False
    assert not manager.perf_label.isVisible()


def test_show_toast(manager):
    frame = QtWidgets.QFrame()
    manager.setup_ui(frame)
    manager.toast = MagicMock()
    manager.show_toast("Hello")
    manager.toast.show_message.assert_called_with("Hello")
