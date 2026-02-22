import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from pynegative.ui.widgets.combobox import ComboBox


@pytest.fixture
def combo_box(qtbot):
    """Provides a ComboBox instance."""
    widget = ComboBox()
    widget.resize(150, 30)
    widget.show()
    qtbot.addWidget(widget)

    # Add some test items
    for i in range(5):
        widget.addItem(f"Item {i}")

    qtbot.addWidget(widget)
    return widget


def test_initialization(combo_box):
    """Test that the combo box initializes correctly."""
    assert combo_box.count() == 5


def test_item_count(combo_box):
    """Test that items can be added and counted."""
    initial_count = combo_box.count()
    combo_box.addItem("New Item")
    assert combo_box.count() == initial_count + 1


def test_wheel_event_does_nothing(combo_box, qtbot):
    """Test that wheel event is disabled (does nothing)."""
    initial_index = combo_box.currentIndex()

    event = QtGui.QWheelEvent(
        QtCore.QPoint(75, 15),
        QtCore.QPoint(75, 15),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(combo_box, event)

    assert combo_box.currentIndex() == initial_index


def test_current_index(combo_box):
    """Test getting and setting current index."""
    combo_box.setCurrentIndex(2)
    assert combo_box.currentIndex() == 2


def test_current_text(combo_box):
    """Test getting current text."""
    combo_box.setCurrentIndex(1)
    assert combo_box.currentText() == "Item 1"


def test_find_text(combo_box):
    """Test finding item by text."""
    index = combo_box.findText("Item 3")
    assert index == 3


def test_find_text_not_found(combo_box):
    """Test finding non-existent text."""
    index = combo_box.findText("NonExistent")
    assert index == -1


def test_insert_item(combo_box):
    """Test inserting item at specific index."""
    combo_box.insertItem(0, "Inserted Item")
    assert combo_box.itemText(0) == "Inserted Item"


def test_remove_item(combo_box):
    """Test removing item at index."""
    combo_box.removeItem(0)
    assert combo_box.count() == 4


def test_clear(combo_box):
    """Test clearing all items."""
    combo_box.clear()
    assert combo_box.count() == 0


def test_set_item_text(combo_box):
    """Test setting item text at index."""
    combo_box.setItemText(0, "Modified Item")
    assert combo_box.itemText(0) == "Modified Item"


def test_item_data(combo_box):
    """Test getting and setting item data."""
    combo_box.setItemData(0, "custom_data")
    assert combo_box.itemData(0) == "custom_data"


def test_multiple_items(combo_box):
    """Test adding multiple items."""
    combo_box.addItem("A")
    combo_box.addItem("B")
    combo_box.addItem("C")
    assert combo_box.count() == 8
