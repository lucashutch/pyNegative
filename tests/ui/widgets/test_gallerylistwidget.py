import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from pynegative.ui.widgets.gallerylistwidget import GalleryListWidget


@pytest.fixture
def gallery_list(qtbot):
    """Provides a GalleryListWidget with some test items."""
    widget = GalleryListWidget()
    widget.resize(300, 400)
    widget.show()
    qtbot.addWidget(widget)

    for i in range(5):
        item = QtWidgets.QListWidgetItem(f"Image {i}")
        item.setData(QtCore.Qt.UserRole, f"/path/to/image_{i}.jpg")
        widget.addItem(item)

    return widget


@pytest.fixture
def empty_gallery_list(qtbot):
    """Provides an empty GalleryListWidget."""
    widget = GalleryListWidget()
    widget.resize(300, 400)
    widget.show()
    qtbot.addWidget(widget)
    return widget


class TestGalleryListWidget:
    def test_initialization(self, gallery_list):
        """Test that the widget initializes with correct properties."""
        assert gallery_list.hasMouseTracking()
        assert gallery_list._hovered_item is None
        # The widget uses ExtendedSelection mode
        assert gallery_list.selectionMode() == QtWidgets.QListWidget.ExtendedSelection

    def test_mouse_move_event(self, gallery_list, qtbot):
        """Test mouse move event updates hovered item."""
        # Move mouse over first item
        item = gallery_list.item(0)
        item_rect = gallery_list.visualItemRect(item)

        # Simulate mouse move by creating event directly
        mouse_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(item_rect.center()),
            QtCore.QPointF(item_rect.center()),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        gallery_list.mouseMoveEvent(mouse_event)

        assert gallery_list._hovered_item == item
        assert gallery_list.get_hovered_item() == item

    def test_mouse_move_different_items(self, gallery_list, qtbot):
        """Test mouse move between different items."""
        item0 = gallery_list.item(0)
        item1 = gallery_list.item(1)

        item0_rect = gallery_list.visualItemRect(item0)
        item1_rect = gallery_list.visualItemRect(item1)

        # Move to first item
        mouse_event0 = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(item0_rect.center()),
            QtCore.QPointF(item0_rect.center()),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        gallery_list.mouseMoveEvent(mouse_event0)
        assert gallery_list._hovered_item == item0

        # Move to second item
        mouse_event1 = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(item1_rect.center()),
            QtCore.QPointF(item1_rect.center()),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        gallery_list.mouseMoveEvent(mouse_event1)
        assert gallery_list._hovered_item == item1

    def test_leave_event(self, gallery_list, qtbot):
        """Test leave event clears hovered item."""
        item = gallery_list.item(0)
        item_rect = gallery_list.visualItemRect(item)

        # Move into widget
        mouse_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(item_rect.center()),
            QtCore.QPointF(item_rect.center()),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        gallery_list.mouseMoveEvent(mouse_event)
        assert gallery_list._hovered_item == item

        # Leave widget
        leave_event = QtCore.QEvent(QtCore.QEvent.Type.Leave)
        gallery_list.leaveEvent(leave_event)
        assert gallery_list._hovered_item is None

    def test_get_hovered_item_initial(self, gallery_list):
        """Test get_hovered_item returns None initially."""
        assert gallery_list.get_hovered_item() is None

    def test_selection_changed_signal(self, gallery_list, qtbot):
        """Test that selectionChanged signal is emitted."""
        with qtbot.waitSignal(gallery_list.selectionChanged):
            # Trigger selection by calling selectionChange method directly
            # This is more reliable than trying to simulate mouse clicks in tests
            selected = QtCore.QItemSelection()
            deselected = QtCore.QItemSelection()

            # Simulate selecting the first item
            index = gallery_list.model().index(0, 0)
            selected.select(index, index)

            gallery_list.selectionChange(selected, deselected)

    def test_selection_change_method(self, gallery_list, qtbot):
        """Test that selectionChange method emits signal."""
        with qtbot.waitSignal(gallery_list.selectionChanged):
            # Test with a different item
            selected = QtCore.QItemSelection()
            deselected = QtCore.QItemSelection()

            # Simulate selecting the second item
            index = gallery_list.model().index(1, 0)
            selected.select(index, index)

            gallery_list.selectionChange(selected, deselected)

    def test_empty_widget(self, empty_gallery_list):
        """Test that empty widget works correctly."""
        assert empty_gallery_list.count() == 0
        assert empty_gallery_list.get_hovered_item() is None

    def test_item_count(self, gallery_list):
        """Test that items are added correctly."""
        assert gallery_list.count() == 5

    def test_item_data(self, gallery_list):
        """Test that items store correct data."""
        for i in range(5):
            item = gallery_list.item(i)
            assert item.data(QtCore.Qt.UserRole) == f"/path/to/image_{i}.jpg"

    def test_visual_item_rect(self, gallery_list):
        """Test that visualItemRect returns valid rectangles."""
        for i in range(5):
            item = gallery_list.item(i)
            rect = gallery_list.visualItemRect(item)
            assert rect.isValid()
            assert rect.width() > 0
            assert rect.height() > 0

    def test_selection_mode(self, gallery_list):
        """Test that selection mode is correct."""
        # The widget uses ExtendedSelection mode
        assert gallery_list.selectionMode() == QtWidgets.QListWidget.ExtendedSelection

    def test_mouse_tracking(self, gallery_list):
        """Test that mouse tracking is enabled."""
        assert gallery_list.hasMouseTracking()
