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
        item = self.itemAt(event.pos())

        if item:
            item_path = item.data(QtCore.Qt.UserRole)
            item_rect = self.visualItemRect(item)

            # Check if click is on the selection circle
            delegate = self.itemDelegate()
            if hasattr(delegate, "is_click_on_circle"):
                is_circle_click = delegate.is_click_on_circle(event.pos(), item_rect)

                if is_circle_click:
                    # Toggle selection via circle click
                    self.toggle_selection(item_path)
                    self._last_clicked_item = item
                    self.selectionChanged.emit()
                    self.update()
                    return

            # Skip selection logic for right-click (context menu)
            if event.button() == QtCore.Qt.RightButton:
                super().mousePressEvent(event)
                return

            # Normal selection logic
            if event.modifiers() & QtCore.Qt.ControlModifier:
                # Ctrl+Click: Toggle selection
                self.toggle_selection(item_path)
                self._last_clicked_item = item
                return
            elif (
                event.modifiers() & QtCore.Qt.ShiftModifier and self._last_clicked_item
            ):
                # Shift+Click: Range selection
                self._select_range(self._last_clicked_item, item)
                self._last_clicked_item = item
                return
            else:
                # Normal click: Select this item only and clear others
                self.clear_selection()
                self.selected_paths.add(item_path)
                self._last_clicked_item = item
                self.selectionChanged.emit()
                self.update()

        super().mousePressEvent(event)

    def toggle_selection(self, item_path):
        """Toggle selection of an item."""
        if item_path in self.selected_paths:
            self.selected_paths.remove(item_path)
        else:
            self.selected_paths.add(item_path)
        self.selectionChanged.emit()
        self.update()

    def clear_selection(self):
        """Clear all selections."""
        self.selected_paths.clear()
        self.selectionChanged.emit()
        self.update()

    def clear(self):
        """Override clear to also update circle visibility."""
        super().clear()
        self.selected_paths.clear()
        self.selectionChanged.emit()
        self.update()

    def select_all_items(self):
        """Select all items in the carousel."""
        self.selected_paths.clear()
        for i in range(self.count()):
            item = self.item(i)
            item_path = item.data(QtCore.Qt.UserRole)
            self.selected_paths.add(item_path)
        self.selectionChanged.emit()
        self.update()

    def _select_range(self, from_item, to_item):
        """Select all items between from_item and to_item."""
        from_idx = self.row(from_item)
        to_idx = self.row(to_item)

        start = min(from_idx, to_idx)
        end = max(from_idx, to_idx)

        self.selected_paths.clear()
        for i in range(start, end + 1):
            item = self.item(i)
            item_path = item.data(QtCore.Qt.UserRole)
            self.selected_paths.add(item_path)

        self.selectionChanged.emit()
        self.update()

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
            self.select_all_items()
            event.accept()
            return
        super().keyPressEvent(event)
