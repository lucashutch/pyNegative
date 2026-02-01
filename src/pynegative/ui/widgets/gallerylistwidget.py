from PySide6 import QtWidgets, QtCore


class GalleryListWidget(QtWidgets.QListWidget):
    """ListWidget with selection changed signal for gallery."""

    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.setMouseTracking(True)
        self._hovered_item = None

    def mousePressEvent(self, event):
        """Handle mouse press with multi-selection support."""
        item = self.itemAt(event.position().toPoint())

        if item:
            item_rect = self.visualItemRect(item)

            # Check if click is on the selection circle
            delegate = self.itemDelegate()
            if hasattr(delegate, "is_click_on_circle"):
                is_circle_click = delegate.is_click_on_circle(
                    event.position().toPoint(), item_rect
                )

                if is_circle_click:
                    # Toggle selection via circle click
                    item.setSelected(not item.isSelected())
                    self.selectionChanged.emit()
                    self.update()
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        item = self.itemAt(event.position().toPoint())
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
        """Override selectionChange to emit custom signal."""
        # Call the base class implementation - but QListWidget doesn't override this
        # so we skip the super call and just emit our signal
        self.selectionChanged.emit()
