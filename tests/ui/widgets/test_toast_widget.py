import pytest
from PySide6 import QtWidgets, QtCore
from pynegative.ui.widgets.toast import ToastWidget


@pytest.fixture
def toast_widget(qtbot):
    """Provides a ToastWidget instance for testing."""
    # Use a longer duration so fade-in can complete before fade-out starts
    widget = ToastWidget(duration=500)
    qtbot.addWidget(widget)
    return widget


def test_initialization(toast_widget):
    """Test that the widget initializes with correct properties."""
    assert toast_widget.windowFlags() & QtCore.Qt.FramelessWindowHint
    assert toast_widget.windowFlags() & QtCore.Qt.WindowStaysOnTopHint
    assert toast_widget.testAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
    assert toast_widget.testAttribute(QtCore.Qt.WA_TranslucentBackground)
    assert toast_widget._label.text() == ""
    assert toast_widget._opacity == 0.0


def test_show_message(toast_widget, qtbot):
    """Test showing a message."""
    message = "Test Toast Message"
    toast_widget.show_message(message)

    assert toast_widget._label.text() == message
    assert toast_widget.isVisible()


def test_positioning(qtbot):
    """Test positioning when a parent is provided."""
    parent = QtWidgets.QWidget()
    parent.resize(800, 600)
    parent.show()
    qtbot.addWidget(parent)

    toast = ToastWidget(parent)
    toast.show_message("Position test")

    # Check positioning (centered horizontally at bottom)
    parent_rect = parent.rect()
    expected_x = (parent_rect.width() - toast.width()) // 2
    expected_y = parent_rect.height() - toast.height() - 40

    assert toast.x() == expected_x
    assert toast.y() == expected_y


def test_fade_animation(toast_widget, qtbot):
    """Test that the fade animation works."""
    toast_widget.show_message("Fading...")

    # Wait for fade in to complete (internal opacity)
    qtbot.waitUntil(lambda: toast_widget._opacity >= 1.0, timeout=1000)
    assert toast_widget._opacity == 1.0

    # Wait for hide timer to trigger fade out
    qtbot.waitUntil(lambda: toast_widget._opacity < 1.0, timeout=1000)

    # Wait for fade out to complete
    qtbot.waitUntil(lambda: not toast_widget.isVisible(), timeout=1000)
    assert toast_widget._opacity == 0.0


def test_multiple_messages(toast_widget, qtbot):
    """Test that showing a second message resets the timers."""
    toast_widget.show_message("Message 1")
    qtbot.wait(50)

    toast_widget.show_message("Message 2")
    assert toast_widget._label.text() == "Message 2"
    assert toast_widget._opacity == 0.0  # Resets opacity to 0 on show
