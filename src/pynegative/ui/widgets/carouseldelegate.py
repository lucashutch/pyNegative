from PySide6 import QtWidgets, QtGui, QtCore


class CarouselDelegate(QtWidgets.QStyledItemDelegate):
    """Custom delegate for carousel items with purple selection circles."""

    # Accent colors from styles.qss
    FILLED_CIRCLE_COLOR = QtGui.QColor("#6366f1")  # Primary accent
    EMPTY_CIRCLE_COLOR = QtGui.QColor("#8b5cf6")  # Secondary accent
    CIRCLE_SIZE = 18
    CIRCLE_MARGIN = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._show_selection_circles = False
        self._circle_clicked = False

    def set_show_selection_circles(self, show):
        """Show or hide selection circles."""
        self._show_selection_circles = show

    def paint(self, painter, option, index):
        """Paint item with selection circle overlay."""
        # Paint standard item (icon + text)
        super().paint(painter, option, index)

        # Always show circles when there are multiple items
        list_widget = self.parent()
        if not list_widget or list_widget.count() <= 1:
            return

        # Get item's path to check selection
        item_path = index.data(QtCore.Qt.UserRole)
        is_selected = item_path in list_widget.selected_paths

        # Calculate circle position (top-right corner)
        rect = option.rect
        circle_x = rect.right() - self.CIRCLE_SIZE - self.CIRCLE_MARGIN
        circle_y = rect.top() + self.CIRCLE_MARGIN

        # Draw circle
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        circle_rect = QtCore.QRectF(
            circle_x, circle_y, self.CIRCLE_SIZE, self.CIRCLE_SIZE
        )

        if is_selected:
            # Draw filled purple circle
            painter.setBrush(QtGui.QBrush(self.FILLED_CIRCLE_COLOR))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(circle_rect)

            # Draw white checkmark inside
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            check_padding = 5
            check_start_x = circle_x + check_padding
            check_start_y = circle_y + self.CIRCLE_SIZE / 2
            check_mid_x = circle_x + self.CIRCLE_SIZE / 2 - 1
            check_mid_y = circle_y + self.CIRCLE_SIZE - check_padding - 1
            check_end_x = circle_x + self.CIRCLE_SIZE - check_padding + 1
            check_end_y = circle_y + check_padding + 2

            painter.drawLine(
                int(check_start_x),
                int(check_start_y),
                int(check_mid_x),
                int(check_mid_y),
            )
            painter.drawLine(
                int(check_mid_x), int(check_mid_y), int(check_end_x), int(check_end_y)
            )
        else:
            # Draw empty purple circle outline
            pen = QtGui.QPen(self.EMPTY_CIRCLE_COLOR, 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(circle_rect)

    def was_circle_clicked(self):
        """Check if the circle was clicked in the last event."""
        return self._circle_clicked

    def reset_circle_clicked(self):
        """Reset the circle clicked flag."""
        self._circle_clicked = False

    def is_click_on_circle(self, pos, item_rect):
        """Check if a position is on the selection circle."""
        # Always allow circle clicks when there are multiple items
        list_widget = self.parent()
        if not list_widget or list_widget.count() <= 1:
            return False

        circle_x = item_rect.right() - self.CIRCLE_SIZE - self.CIRCLE_MARGIN
        circle_y = item_rect.top() + self.CIRCLE_MARGIN

        circle_rect = QtCore.QRectF(
            circle_x, circle_y, self.CIRCLE_SIZE, self.CIRCLE_SIZE
        )

        return circle_rect.contains(pos)
