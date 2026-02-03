import pytest
from PySide6 import QtWidgets, QtCore, QtGui
# QTest is not used for direct event injection, but keep for qtbot fixture

from pynegative.ui.widgets.starrating import StarRatingWidget


@pytest.fixture
def star_rating_widget(qtbot):
    """Provides a StarRatingWidget instance for testing."""
    widget = StarRatingWidget()
    widget.resize(widget.sizeHint())
    widget.show()  # Ensure the widget is visible and in the event loop
    qtbot.addWidget(widget)  # Manages widget lifecycle
    return widget


def test_initialization(star_rating_widget):
    """Test that the widget initializes with a default rating of 0."""
    assert star_rating_widget.rating() == 0
    assert star_rating_widget._hover_rating == -1


def test_set_rating_valid(star_rating_widget, qtbot):
    """Test setting a valid rating and signal emission."""
    with qtbot.waitSignal(star_rating_widget.ratingChanged) as blocker:
        star_rating_widget.set_rating(3)
    assert star_rating_widget.rating() == 3
    assert blocker.args == [3]

    with qtbot.waitSignal(star_rating_widget.ratingChanged) as blocker:
        star_rating_widget.set_rating(5)
    assert star_rating_widget.rating() == 5
    assert blocker.args == [5]


def test_set_rating_to_zero(star_rating_widget, qtbot):
    """Test setting a rating to 0."""
    star_rating_widget.set_rating(3)  # Set an initial rating
    with qtbot.waitSignal(star_rating_widget.ratingChanged) as blocker:
        star_rating_widget.set_rating(0)
    assert star_rating_widget.rating() == 0
    assert blocker.args == [0]


def test_set_rating_no_change(star_rating_widget, qtbot):
    """Test that signal is not emitted if rating does not change."""
    star_rating_widget.set_rating(3)

    # Attempt to set to the same value
    with qtbot.assertNotEmitted(star_rating_widget.ratingChanged):
        star_rating_widget.set_rating(3)
    assert star_rating_widget.rating() == 3


def test_mouse_move_event_hover(star_rating_widget, qtbot):
    """Test mouse movement for hover effect using direct event injection."""
    star_width = star_rating_widget.star_empty_pixmap.width() + 4

    # Simulate mouse moving over the 3rd star (index 2)
    x_pos = (2 * star_width) + (star_width // 2)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(x_pos, 5),
        QtCore.QPoint(x_pos, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event)
    qtbot.wait(10)  # Give event loop a moment
    assert star_rating_widget._hover_rating == 3

    # Simulate mouse moving over the 5th star (index 4)
    x_pos = (4 * star_width) + (star_width // 2)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(x_pos, 5),
        QtCore.QPoint(x_pos, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event)
    qtbot.wait(10)
    assert star_rating_widget._hover_rating == 5

    # Simulate mouse moving over the first star (x=0 is valid for the first star)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(0, 5),
        QtCore.QPoint(0, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event)
    qtbot.wait(10)
    assert star_rating_widget._hover_rating == 1  # Should be 1, not -1


def test_mouse_leave_event(star_rating_widget, qtbot):
    """Test mouse leaving the widget area using direct event injection."""
    star_width = star_rating_widget.star_empty_pixmap.width() + 4

    # First, move mouse over to set a hover rating
    event_move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(star_width * 2, 5),
        QtCore.QPoint(star_width * 2, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event_move)
    qtbot.wait(10)
    assert star_rating_widget._hover_rating == 3

    # Simulate mouse leaving
    event_leave = QtGui.QMouseEvent(
        QtCore.QEvent.Leave,
        QtCore.QPoint(-1, -1),  # Coordinates don't matter much for Leave
        QtCore.QPoint(-1, -1),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event_leave)
    qtbot.wait(10)
    assert star_rating_widget._hover_rating == -1


def test_mouse_press_event_set_rating(star_rating_widget, qtbot):
    """Test mouse click to set a rating using direct event injection (Press and Release)."""
    star_width = star_rating_widget.star_empty_pixmap.width() + 4
    x_pos = (2 * star_width) + (star_width // 2)  # Middle of 3rd star

    # First, simulate mouse move to set _hover_rating
    event_move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(x_pos, 5),
        QtCore.QPoint(x_pos, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event_move)
    qtbot.wait(10)  # Give event loop a moment for hover to register
    assert star_rating_widget._hover_rating == 3  # Ensure hover is set

    with qtbot.waitSignal(star_rating_widget.ratingChanged) as blocker:
        event_press = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            QtCore.QPoint(x_pos, 5),
            QtCore.QPoint(x_pos, 5),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(star_rating_widget, event_press)
        qtbot.wait(10)  # Give event loop a moment

        event_release = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease,
            QtCore.QPoint(x_pos, 5),
            QtCore.QPoint(x_pos, 5),
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(star_rating_widget, event_release)
        qtbot.wait(10)  # Give event loop a moment
    assert star_rating_widget.rating() == 3
    assert blocker.args == [3]


def test_mouse_press_event_clear_rating(star_rating_widget, qtbot):
    """Test mouse click on an already set rating to clear it using direct event injection (Press and Release)."""
    star_rating_widget.set_rating(3)  # Set initial rating
    star_width = star_rating_widget.star_empty_pixmap.width() + 4
    x_pos = (2 * star_width) + (star_width // 2)  # Middle of 3rd star

    # First, simulate mouse move to set _hover_rating
    event_move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(x_pos, 5),
        QtCore.QPoint(x_pos, 5),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(star_rating_widget, event_move)
    qtbot.wait(10)  # Give event loop a moment for hover to register
    assert star_rating_widget._hover_rating == 3  # Ensure hover is set

    with qtbot.waitSignal(star_rating_widget.ratingChanged) as blocker:
        event_press = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            QtCore.QPoint(x_pos, 5),
            QtCore.QPoint(x_pos, 5),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(star_rating_widget, event_press)
        qtbot.wait(10)  # Give event loop a moment

        event_release = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease,
            QtCore.QPoint(x_pos, 5),
            QtCore.QPoint(x_pos, 5),
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(star_rating_widget, event_release)
        qtbot.wait(10)  # Give event loop a moment
    assert star_rating_widget.rating() == 0
    assert blocker.args == [0]


def test_size_hint(star_rating_widget):
    """Test that sizeHint returns a reasonable size."""
    size = star_rating_widget.sizeHint()
    expected_width = (star_rating_widget.star_empty_pixmap.width() + 4) * 5
    expected_height = star_rating_widget.star_empty_pixmap.height()
    assert size.width() >= expected_width - 5  # Allow for slight variations
    assert size.height() == expected_height
