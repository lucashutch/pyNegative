import pytest
from PySide6 import QtCore, QtWidgets

from pynegative.ui.widgets.horizontallist import HorizontalListWidget


@pytest.fixture
def list_widget(qtbot):
    """Provides a HorizontalListWidget with some test items."""
    widget = HorizontalListWidget()
    widget.resize(800, 100)
    widget.show()
    qtbot.addWidget(widget)

    for i in range(5):
        item = QtWidgets.QListWidgetItem(f"Item {i}")
        item.setData(QtCore.Qt.UserRole, f"/path/to/item_{i}.jpg")
        widget.addItem(item)

    return widget


def test_initialization(list_widget):
    """Test that the widget initializes with correct properties."""
    assert list_widget.selectionMode() == QtWidgets.QListWidget.ExtendedSelection
    assert list_widget.hasMouseTracking()
    assert len(list_widget.selected_paths) == 0


def test_select_all(list_widget):
    """Test select_all_items method."""
    list_widget.select_all_items()

    assert len(list_widget.selected_paths) == 5
    assert "/path/to/item_0.jpg" in list_widget.selected_paths
    assert "/path/to/item_4.jpg" in list_widget.selected_paths


def test_clear_selection(list_widget):
    """Test clear_selection method."""
    list_widget.select_all_items()
    assert len(list_widget.selected_paths) == 5

    list_widget.clear_selection()
    assert len(list_widget.selected_paths) == 0


def test_get_selected_paths(list_widget):
    """Test get_selected_paths returns correct list."""
    list_widget.toggle_selection("/path/to/item_0.jpg")
    list_widget.toggle_selection("/path/to/item_2.jpg")

    paths = list_widget.get_selected_paths()
    assert len(paths) == 2
    assert "/path/to/item_0.jpg" in paths
    assert "/path/to/item_2.jpg" in paths


def test_should_show_circles(list_widget):
    """Test should_show_circles returns True for multiple items."""
    assert list_widget.should_show_circles() is True


def test_should_show_circles_single(list_widget):
    """Test should_show_circles returns False for single item."""
    list_widget.clear()
    item = QtWidgets.QListWidgetItem("Single")
    item.setData(QtCore.Qt.UserRole, "/path/to/single.jpg")
    list_widget.addItem(item)

    assert list_widget.should_show_circles() is False


def test_toggle_selection(list_widget):
    """Test toggle_selection method."""
    path = "/path/to/item_0.jpg"

    assert path not in list_widget.selected_paths

    list_widget.toggle_selection(path)
    assert path in list_widget.selected_paths

    list_widget.toggle_selection(path)
    assert path not in list_widget.selected_paths


def test_is_multi_select_active(list_widget):
    """Test is_multi_select_active method."""
    assert list_widget.is_multi_select_active() is False

    list_widget.toggle_selection("/path/to/item_0.jpg")
    assert list_widget.is_multi_select_active() is True

    list_widget.clear_selection()
    assert list_widget.is_multi_select_active() is False


def test_clear_method_clears_selections(list_widget):
    """Test clear method clears selections too."""
    list_widget.select_all_items()
    assert len(list_widget.selected_paths) == 5

    list_widget.clear()
    assert len(list_widget.selected_paths) == 0
    assert list_widget.count() == 0


def test_selected_paths_type(list_widget):
    """Test that selected_paths is a set."""
    assert isinstance(list_widget.selected_paths, set)


def test_selection_change_signal(list_widget, qtbot):
    """Test that selectionChanged signal is emitted."""
    with qtbot.waitSignal(list_widget.selectionChanged):
        list_widget.toggle_selection("/path/to/item_0.jpg")


def test_double_selection_no_duplicate(list_widget):
    """Test that selecting same item twice doesn't duplicate."""
    list_widget.toggle_selection("/path/to/item_0.jpg")
    list_widget.toggle_selection("/path/to/item_0.jpg")
    assert len(list_widget.selected_paths) == 0


def test_item_data_access(list_widget):
    """Test that items store and return correct data."""
    item = list_widget.item(0)
    assert item.data(QtCore.Qt.UserRole) == "/path/to/item_0.jpg"
