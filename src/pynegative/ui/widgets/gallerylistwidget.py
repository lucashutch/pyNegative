from PySide6 import QtCore, QtWidgets


class GalleryListWidget(QtWidgets.QListWidget):
    """ListWidget with selection changed signal for gallery."""

    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.setMouseTracking(True)
        self._hovered_item = None

        # Connect builtin signal to custom signal
        self.itemSelectionChanged.connect(self.selectionChanged.emit)

    def _get_event_pos(self, event):
        """Compatibility helper for Qt6 mouse events."""
        if hasattr(event, "position"):
            return event.position()
        return event.pos()

    def mousePressEvent(self, event):
        """Handle mouse press with multi-selection support."""
        pos = self._get_event_pos(event)
        item = self.itemAt(pos.toPoint())

        if item:
            item_rect = self.visualItemRect(item)

            # Check if click is on the selection circle
            delegate = self.itemDelegate()
            if hasattr(delegate, "is_click_on_circle"):
                is_circle_click = delegate.is_click_on_circle(pos.toPoint(), item_rect)

                if is_circle_click:
                    # Toggle selection via circle click
                    item.setSelected(not item.isSelected())
                    self.update()
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        pos = self._get_event_pos(event)
        item = self.itemAt(pos.toPoint())
        if item != self._hovered_item:
            if self._hovered_item:
                # Trigger update on old item to remove hover effect
                self.update(self.visualItemRect(self._hovered_item))
            self._hovered_item = item
            if self._hovered_item:
                # Trigger update on new item to show hover effect
                self.update(self.visualItemRect(self._hovered_item))

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._hovered_item:
            self.update(self.visualItemRect(self._hovered_item))
            self._hovered_item = None

    def get_hovered_item(self):
        return self._hovered_item

    def selectionChange(self, selected, deselected):
        """Deprecated: use itemSelectionChanged or selectionChanged signal instead."""
        # Kept for compatibility with existing tests
        self.selectionChanged.emit()
