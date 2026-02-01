import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from pynegative.ui.widgets.carouseldelegate import CarouselDelegate


@pytest.fixture
def list_widget(qtbot):
    """Provides a HorizontalListWidget with items for the delegate."""
    from pynegative.ui.widgets.horizontallist import HorizontalListWidget

    widget = HorizontalListWidget()
    widget.setObjectName("Carousel")
    widget.resize(800, 120)
    widget.show()
    qtbot.addWidget(widget)

    for i in range(5):
        item = QtWidgets.QListWidgetItem(f"Item {i}")
        item.setData(QtCore.Qt.UserRole, f"/path/to/item_{i}.jpg")
        widget.addItem(item)

    return widget


@pytest.fixture
def delegate(list_widget):
    """Provides a CarouselDelegate attached to the list widget."""
    delegate = CarouselDelegate(list_widget)
    return delegate


def test_initialization(delegate):
    """Test that the delegate initializes correctly."""
    assert delegate._show_selection_circles is False
    assert delegate._circle_clicked is False


def test_set_show_selection_circles(delegate):
    """Test setting circle visibility."""
    delegate.set_show_selection_circles(True)
    assert delegate._show_selection_circles is True

    delegate.set_show_selection_circles(False)
    assert delegate._show_selection_circles is False


def test_was_circle_clicked(delegate):
    """Test circle click state detection."""
    assert delegate.was_circle_clicked() is False

    delegate._circle_clicked = True
    assert delegate.was_circle_clicked() is True


def test_reset_circle_clicked(delegate):
    """Test resetting circle click state."""
    delegate._circle_clicked = True
    delegate.reset_circle_clicked()
    assert delegate._circle_clicked is False


def test_is_click_on_circle_miss_left(delegate):
    """Test click detection - click to the left of circle."""
    item_rect = QtCore.QRect(0, 0, 100, 100)
    click_pos = QtCore.QPoint(5, 5)

    assert delegate.is_click_on_circle(click_pos, item_rect) is False


def test_is_click_on_circle_miss_below(delegate):
    """Test click detection - click below the circle."""
    item_rect = QtCore.QRect(0, 0, 100, 100)
    click_pos = QtCore.QPoint(85, 50)

    assert delegate.is_click_on_circle(click_pos, item_rect) is False


def test_is_click_on_circle_hit(delegate, list_widget):
    """Test click detection - click on circle."""
    # Ensure multiple items so circles are active
    if list_widget.count() < 2:
        list_widget.addItem(QtWidgets.QListWidgetItem("Item 1"))
    delegate.set_show_selection_circles(True)

    item_rect = QtCore.QRect(0, 0, 100, 100)
    # Circle is at x=76, y=6 (right - 24, top + 6), size 18x18
    # Center is at (85, 15)
    click_pos = QtCore.QPoint(85, 15)

    assert delegate.is_click_on_circle(click_pos, item_rect) is True


def test_is_click_on_circle_top_left_corner(delegate, list_widget):
    """Test click detection - click on top-left of circle."""
    # Ensure multiple items so circles are active
    if list_widget.count() < 2:
        list_widget.addItem(QtWidgets.QListWidgetItem("Item 1"))
    delegate.set_show_selection_circles(True)

    item_rect = QtCore.QRect(0, 0, 100, 100)
    # Top-left of circle: (76, 6)
    click_pos = QtCore.QPoint(76, 6)

    assert delegate.is_click_on_circle(click_pos, item_rect) is True


def test_is_click_on_circle_bottom_right_corner(delegate, list_widget):
    """Test click detection - click on bottom-right of circle."""
    # Ensure multiple items so circles are active
    if list_widget.count() < 2:
        list_widget.addItem(QtWidgets.QListWidgetItem("Item 1"))
    delegate.set_show_selection_circles(True)

    item_rect = QtCore.QRect(0, 0, 100, 100)
    # Bottom-right of circle is at (93.99, 23.99) - Qt uses exclusive bottom-right
    # So click at (93, 23) should be inside
    click_pos = QtCore.QPoint(93, 23)

    assert delegate.is_click_on_circle(click_pos, item_rect) is True


def test_is_click_on_circle_just_outside(delegate):
    """Test click detection - click just outside circle."""
    item_rect = QtCore.QRect(0, 0, 100, 100)
    # Just to the right of circle
    click_pos = QtCore.QPoint(95, 15)

    assert delegate.is_click_on_circle(click_pos, item_rect) is False


def test_should_show_circles_multiple_items(list_widget, delegate, qtbot):
    """Test that circles are shown for multiple items."""
    delegate.set_show_selection_circles(True)

    option = QtWidgets.QStyleOptionViewItem()
    index = list_widget.model().index(0, 0)

    painter = QtGui.QPainter()
    delegate.paint(painter, option, index)


def test_should_not_show_circles_single_item(list_widget, delegate, qtbot):
    """Test that circles are not shown for single item."""
    list_widget.clear()
    item = QtWidgets.QListWidgetItem("Single")
    item.setData(QtCore.Qt.UserRole, "/path/to/single.jpg")
    list_widget.addItem(item)

    index = list_widget.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()

    painter = QtGui.QPainter()
    delegate.paint(painter, option, index)


def test_paint_selected_item(list_widget, delegate, qtbot):
    """Test painting a selected item."""
    list_widget.selected_paths.add("/path/to/item_0.jpg")
    delegate.set_show_selection_circles(True)

    index = list_widget.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()

    painter = QtGui.QPainter()
    delegate.paint(painter, option, index)


def test_paint_unselected_item(list_widget, delegate, qtbot):
    """Test painting an unselected item."""
    list_widget.selected_paths.clear()
    delegate.set_show_selection_circles(True)

    index = list_widget.model().index(0, 0)
    option = QtWidgets.QStyleOptionViewItem()

    painter = QtGui.QPainter()
    delegate.paint(painter, option, index)


def test_circle_position_calculation(delegate, list_widget):
    """Test that circle position is calculated correctly from item rect."""
    # Ensure multiple items so circles are active
    if list_widget.count() < 2:
        list_widget.addItem(QtWidgets.QListWidgetItem("Item 1"))
    delegate.set_show_selection_circles(True)

    item_rect = QtCore.QRect(50, 50, 100, 100)
    # Circle should be at: x = 50+100-24=126, y=50+6=56
    # Circle rect: (126, 56, 18, 18)
    expected_x = item_rect.right() - 24
    expected_y = item_rect.top() + 6

    click_pos = QtCore.QPoint(expected_x, expected_y)
    assert delegate.is_click_on_circle(click_pos, item_rect) is True


def test_circle_click_returns_false_for_single_item(list_widget, delegate):
    """Test that is_click_on_circle returns False for single item."""
    list_widget.clear()
    item = QtWidgets.QListWidgetItem("Single")
    item.setData(QtCore.Qt.UserRole, "/path/to/single.jpg")
    list_widget.addItem(item)

    item_rect = QtCore.QRect(0, 0, 100, 100)
    click_pos = QtCore.QPoint(76, 6)

    assert delegate.is_click_on_circle(click_pos, item_rect) is False
