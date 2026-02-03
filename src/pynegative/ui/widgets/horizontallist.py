from PySide6 import QtWidgets, QtCore


class HorizontalListWidget(QtWidgets.QListWidget):
    """A ListWidget that scrolls horizontally with the mouse wheel and supports multi-selection."""

    selectionChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.setMouseTracking(True)

        # Track selected paths (for delegate rendering)
        self.selected_paths = set()
        self._last_clicked_item = None

        # Sync selection state and emit custom signal
        self.itemSelectionChanged.connect(self._sync_selection)
        self.itemSelectionChanged.connect(self.selectionChanged.emit)

    def _sync_selection(self):
        """Sync internal selected_paths with actual QListWidget selection."""
        self.selected_paths.clear()
        for item in self.selectedItems():
            path = item.data(QtCore.Qt.UserRole)
            if path:
                self.selected_paths.add(path)
        self.update()

    def wheelEvent(self, event):
        if event.angleDelta().y():
            # Scroll horizontally instead of vertically
            delta = event.angleDelta().y()
            # Most mice return 120 per notch. We apply a small multiplier for speed.
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta
            )
            event.accept()
        else:
            super().wheelEvent(event)

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
                    self._last_clicked_item = item
                    # Accept and return to prevent base class from processing the click
                    event.accept()
                    return

            # Skip selection logic for right-click (context menu)
            if event.button() == QtCore.Qt.RightButton:
                super().mousePressEvent(event)
                return

            # Normal selection logic - let base class handle it but track last clicked
            self._last_clicked_item = item

        super().mousePressEvent(event)

    def toggle_selection(self, item_path):
        """Toggle selection of an item by path."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(QtCore.Qt.UserRole) == item_path:
                item.setSelected(not item.isSelected())
                break

    def clear_selection(self):
        """Clear all selections."""
        self.clearSelection()

    def select_all_items(self):
        """Select all items in the carousel."""
        self.selectAll()

    def get_selected_paths(self):
        """Get list of selected item paths."""
        return list(self.selected_paths)

    def should_show_circles(self):
        """Check if circles should be shown (carousel has multiple items)."""
        return self.count() > 1

    def is_multi_select_active(self):
        """Check if multi-selection is active (more than 0 items selected)."""
        return len(self.selected_paths) > 0

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if (
            event.key() == QtCore.Qt.Key_A
            and event.modifiers() & QtCore.Qt.ControlModifier
        ):
            self.selectAll()
            event.accept()
            return
        super().keyPressEvent(event)
