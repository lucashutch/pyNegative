from PySide6 import QtCore, QtWidgets, QtGui
from .galleryitemdelegate import GalleryItemDelegate


class CarouselDelegate(GalleryItemDelegate):
    """Custom delegate for carousel items with purple selection circles and stars."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        """Paint item with selection highlight and circles."""
        # We need to manually handle 'selected' state if HorizontalListWidget
        # uses its own 'selected_paths' set instead of standard QItemSelection
        list_widget = self.parent()
        is_selected = False

        if list_widget and hasattr(list_widget, "selected_paths"):
            item_path = index.data(QtCore.Qt.UserRole)
            is_selected = item_path in list_widget.selected_paths
        else:
            is_selected = option.state & QtWidgets.QStyle.State_Selected

        # Temporarily modify option state to reflect our custom selection if needed
        original_state = option.state
        if is_selected:
            option.state |= QtWidgets.QStyle.State_Selected
        else:
            option.state &= ~QtWidgets.QStyle.State_Selected

        super().paint(painter, option, index)

        # Restore state
        option.state = original_state

    def is_click_on_circle(self, pos, item_rect):
        """Check if a position is on the selection circle."""
        # Always allow circle clicks when there are multiple items in the carousel
        list_widget = self.parent()
        if not list_widget or list_widget.count() <= 1:
            return False

        return super().is_click_on_circle(pos, item_rect)
