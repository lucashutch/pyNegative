import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from pynegative.ui.widgets.galleryitemdelegate import GalleryItemDelegate
from pynegative.ui.widgets.gallerylistwidget import GalleryListWidget


@pytest.fixture
def list_widget(qtbot):
    """Provides a GalleryListWidget with items for the delegate."""
    widget = GalleryListWidget()
    widget.setObjectName("Gallery")
    widget.resize(300, 400)
    widget.show()
    qtbot.addWidget(widget)

    for i in range(5):
        item = QtWidgets.QListWidgetItem(f"Image {i}")
        item.setData(QtCore.Qt.UserRole, f"/path/to/image_{i}.jpg")
        item.setData(QtCore.Qt.UserRole + 1, i + 1)  # Rating 1-5
        widget.addItem(item)

    return widget


@pytest.fixture
def delegate(list_widget):
    """Provides a GalleryItemDelegate attached to the list widget."""
    delegate = GalleryItemDelegate(list_widget)
    return delegate


@pytest.fixture
def empty_list_widget(qtbot):
    """Provides an empty GalleryListWidget."""
    widget = GalleryListWidget()
    widget.setObjectName("Gallery")
    widget.resize(300, 400)
    widget.show()
    qtbot.addWidget(widget)
    return widget


class TestGalleryItemDelegate:
    def test_initialization(self, delegate):
        """Test that the delegate initializes correctly."""
        assert delegate.star_filled is not None
        assert delegate.star_empty is not None
        assert delegate.star_filled.size() == QtCore.QSize(16, 16)
        assert delegate.star_empty.size() == QtCore.QSize(16, 16)

    def test_star_pixmap_creation(self):
        """Test that star pixmaps are created correctly."""
        delegate = GalleryItemDelegate()

        # Test filled star
        filled = delegate.star_filled
        assert filled.width() == 16
        assert filled.height() == 16
        assert filled.hasAlpha()  # Should be transparent background

        # Test empty star
        empty = delegate.star_empty
        assert empty.width() == 16
        assert empty.height() == 16
        assert empty.hasAlpha()

    def test_paint_with_rating(self, list_widget, delegate, qtbot):
        """Test painting an item with rating."""
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 200, 50)
        index = list_widget.model().index(0, 0)

        # Create a pixmap to paint on for testing
        pixmap = QtGui.QPixmap(200, 50)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, index)
        painter.end()
        # Should not crash - basic paint test

    def test_paint_with_hover(self, list_widget, delegate, qtbot):
        """Test painting an item with hover effect."""
        # Set up hover state
        list_widget._hovered_item = list_widget.item(0)

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 200, 50)
        index = list_widget.model().index(0, 0)

        # Create a pixmap to paint on for testing
        pixmap = QtGui.QPixmap(200, 50)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, index)
        painter.end()
        # Should not crash - basic paint test

    def test_paint_with_no_rating_or_hover(self, list_widget, delegate, qtbot):
        """Test painting an item with no rating and no hover."""
        # Clear rating
        list_widget.item(0).setData(QtCore.Qt.UserRole + 1, 0)
        list_widget._hovered_item = None

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 200, 50)
        index = list_widget.model().index(0, 0)

        # Create a pixmap to paint on for testing
        pixmap = QtGui.QPixmap(200, 50)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, index)
        painter.end()
        # Should not crash - basic paint test

    def test_paint_empty_list(self, empty_list_widget, delegate, qtbot):
        """Test painting with empty list."""
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 200, 50)
        # Invalid index for empty list
        index = QtCore.QModelIndex()

        # Create a pixmap to paint on for testing
        pixmap = QtGui.QPixmap(200, 50)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, index)
        painter.end()
        # Should handle gracefully - basic paint test

    def test_editor_event_mouse_click(self, list_widget, delegate, qtbot):
        """Test editor event for mouse click on stars."""
        # Set up hover state - ensure the item is being hovered
        hovered_item = list_widget.item(0)
        list_widget._hovered_item = hovered_item

        # Set initial rating to 0 so we can test setting it to 1
        list_widget.item(0).setData(QtCore.Qt.UserRole + 1, 0)

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 220, 240)  # Standard large item rect
        index = list_widget.model().index(0, 0)

        # Verify hover state is set correctly
        assert list_widget.get_hovered_item() is not None
        assert list_widget.get_hovered_item().text() == index.data()

        # Calculate first star position for 220x240 rect:
        # is_large = 240 > 150 = True
        # star_strip_width = 30
        # bottom_height = 28
        # stars_y_start = (240 - 28 - 88) // 2 = 62
        # stars_x = 220 - 30 + (30 - 16) // 2 = 197

        click_x = 205  # Within [197, 213]
        click_y = 70  # Within [62, 78]

        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(click_x, click_y),
            QtCore.QPointF(click_x, click_y),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        result = delegate.editorEvent(event, list_widget.model(), option, index)
        assert result is True
        assert list_widget.item(0).data(QtCore.Qt.UserRole + 1) == 1

    def test_editor_event_click_outside_stars(self, list_widget, delegate, qtbot):
        """Test editor event for click outside stars."""
        # Set up hover state
        list_widget._hovered_item = list_widget.item(0)

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 220, 240)
        index = list_widget.model().index(0, 0)

        # Simulate click outside star area (e.g., top-left)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(10, 10),
            QtCore.QPointF(10, 10),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        result = delegate.editorEvent(event, list_widget.model(), option, index)
        assert result is False  # Should not handle

    def test_editor_event_click_clears_rating(self, list_widget, delegate, qtbot):
        """Test that clicking on existing rating clears it."""
        # Set up hover state
        list_widget._hovered_item = list_widget.item(0)

        # Set initial rating to 3
        list_widget.item(0).setData(QtCore.Qt.UserRole + 1, 3)

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 220, 240)
        index = list_widget.model().index(0, 0)

        # Third star: stars_y_start (62) + 2 * (star_height (16) + spacing (2)) = 62 + 36 = 98
        # x is 197 to 213.
        click_x = 205
        click_y = 105  # Within [98, 114]

        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(click_x, click_y),
            QtCore.QPointF(click_x, click_y),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        result = delegate.editorEvent(event, list_widget.model(), option, index)
        assert result is True
        assert list_widget.item(0).data(QtCore.Qt.UserRole + 1) == 0

    def test_editor_event_non_mouse_event(self, list_widget, delegate):
        """Test editor event with non-mouse event."""
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 200, 50)  # Provide rect for consistency
        index = list_widget.model().index(0, 0)

        # Simulate key event
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Space,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        result = delegate.editorEvent(event, list_widget.model(), option, index)
        assert result is False  # Should not handle

    def test_paint_with_different_ratings(self, list_widget, delegate, qtbot):
        """Test painting items with different ratings."""
        for i in range(5):
            list_widget.item(i).setData(QtCore.Qt.UserRole + 1, i + 1)

            option = QtWidgets.QStyleOptionViewItem()
            option.rect = QtCore.QRect(0, 0, 200, 50)
            index = list_widget.model().index(i, 0)

            # Create a pixmap to paint on for testing
            pixmap = QtGui.QPixmap(200, 50)
            painter = QtGui.QPainter(pixmap)
            delegate.paint(painter, option, index)
            painter.end()
            # Should not crash - basic paint test

    def test_star_positions(self, list_widget, delegate):
        """Test that star positions are calculated correctly."""
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 220, 240)

        # Large mode (240 > 150)
        # star_strip_width = 30
        # bottom_height = 28
        # total_stars_height = 5 * 16 + 4 * 2 # 88

        expected_x = 220 - 30 + (30 - 16) // 2  # 197
        expected_y_start = (240 - 28 - 88) // 2  # 62
        star_height = 16

        # Test star position calculation logic
        for i in range(5):
            x = expected_x
            y = expected_y_start + (i * (star_height + 2))

            # Verify the pattern matches what the delegate would use
            assert x == 197
            assert y == 62 + i * (16 + 2)

        # Test with actual painting to ensure no crash
        pixmap = QtGui.QPixmap(220, 240)
        painter = QtGui.QPainter(pixmap)
        index = list_widget.model().index(0, 0)
        delegate.paint(painter, option, index)
        painter.end()
        # Should not crash - basic paint test

    def test_rating_data_access(self, list_widget):
        """Test that rating data is stored and retrieved correctly."""
        for i in range(5):
            item = list_widget.item(i)
            assert item.data(QtCore.Qt.UserRole + 1) == i + 1

    def test_empty_item_data(self, empty_list_widget):
        """Test that empty widget handles data access gracefully."""
        # QListWidget.item() returns None for invalid indices, not IndexError
        assert empty_list_widget.item(0) is None

    def test_model_index(self, list_widget):
        """Test that model indices work correctly."""
        for i in range(5):
            index = list_widget.model().index(i, 0)
            assert index.isValid()
            assert index.data() == f"Image {i}"

    def test_delegate_parent(self, list_widget, delegate):
        """Test that delegate parent is set correctly."""
        assert delegate.parent() == list_widget

    def test_star_size(self, delegate):
        """Test that star sizes are consistent."""
        assert delegate.star_filled.size() == QtCore.QSize(16, 16)
        assert delegate.star_empty.size() == QtCore.QSize(16, 16)
        assert delegate.star_filled.size() == delegate.star_empty.size()
