from PySide6 import QtWidgets, QtGui, QtCore


class GalleryItemDelegate(QtWidgets.QStyledItemDelegate):
    # Accent colors from styles.qss
    FILLED_CIRCLE_COLOR = QtGui.QColor("#6366f1")  # Primary accent
    EMPTY_CIRCLE_COLOR = QtGui.QColor("#4a4a4a")  # Subtle border for unselected
    SELECTION_HIGHLIGHT = QtGui.QColor(99, 102, 241, 40)  # 15% opacity primary accent
    CIRCLE_SIZE = 18
    CIRCLE_MARGIN = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.star_filled = self._create_star_pixmap(True)
        self.star_empty = self._create_star_pixmap(False)
        self._show_selection_circles = False
        self._circle_clicked = False

    def was_circle_clicked(self):
        """Check if the circle was clicked in the last event."""
        return self._circle_clicked

    def reset_circle_clicked(self):
        """Reset the circle clicked flag."""
        self._circle_clicked = False

    def set_show_selection_circles(self, show):
        """Show or hide selection circles."""
        self._show_selection_circles = show

    def _create_star_pixmap(self, filled):
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#909090"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")
        painter.end()
        return pixmap

    def paint(self, painter, option, index):
        # Paint standard item (icon + text)
        super().paint(painter, option, index)

        list_widget = self.parent()
        if not list_widget:
            return

        is_hovered = False
        if hasattr(list_widget, "get_hovered_item"):
            hovered_item = list_widget.get_hovered_item()
            is_hovered = (
                hovered_item is not None and hovered_item.text() == index.data()
            )

        # Handle selection highlight and circles
        is_selected = option.state & QtWidgets.QStyle.State_Selected

        # Draw highlight overlay for selected items (in addition to stylesheet border)
        if is_selected:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(self.SELECTION_HIGHLIGHT))
            painter.setPen(QtCore.Qt.NoPen)
            # Use item rect with a small padding
            highlight_rect = option.rect.adjusted(2, 2, -2, -2)
            painter.drawRoundedRect(highlight_rect, 6, 6)
            painter.restore()

        # Draw rating stars
        rating = index.data(QtCore.Qt.UserRole + 1)
        if rating is None:
            rating = 0

        if rating > 0 or is_hovered:
            y = option.rect.y() + 5
            for i in range(5):
                star_icon = self.star_empty
                if i < rating:
                    star_icon = self.star_filled

                x = option.rect.x() + 5 + (i * (self.star_empty.width() + 2))
                painter.drawPixmap(x, y, star_icon)

        # Draw selection circle if enabled
        if self._show_selection_circles:
            # Calculate circle position (top-right corner)
            rect = option.rect
            circle_x = rect.right() - self.CIRCLE_SIZE - self.CIRCLE_MARGIN
            circle_y = rect.top() + self.CIRCLE_MARGIN
            circle_rect = QtCore.QRectF(
                circle_x, circle_y, self.CIRCLE_SIZE, self.CIRCLE_SIZE
            )

            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

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
                    int(check_mid_x),
                    int(check_mid_y),
                    int(check_end_x),
                    int(check_end_y),
                )
            else:
                # Draw empty subtle circle outline for non-selected items
                pen = QtGui.QPen(self.EMPTY_CIRCLE_COLOR, 2)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(circle_rect)

            painter.restore()

    def is_click_on_circle(self, pos, item_rect):
        """Check if a position is on the selection circle."""
        if not self._show_selection_circles:
            return False

        circle_x = item_rect.right() - self.CIRCLE_SIZE - self.CIRCLE_MARGIN
        circle_y = item_rect.top() + self.CIRCLE_MARGIN

        circle_rect = QtCore.QRectF(
            circle_x, circle_y, self.CIRCLE_SIZE, self.CIRCLE_SIZE
        )

        return circle_rect.contains(pos)

    def editorEvent(self, event, model, option, index):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            # If clicking on the selection circle, let the ListWidget handle it
            if self.is_click_on_circle(event.position().toPoint(), option.rect):
                self._circle_clicked = True
                return False

            list_widget = self.parent()
            is_hovered = False
            if list_widget and hasattr(list_widget, "get_hovered_item"):
                hovered_item = list_widget.get_hovered_item()
                is_hovered = (
                    hovered_item is not None and hovered_item.text() == index.data()
                )

            if is_hovered:
                # Check if click is on the stars
                y = option.rect.y() + 5
                x_start = option.rect.x() + 5
                star_width = self.star_empty.width()
                star_height = self.star_empty.height()

                if (
                    event.position().y() >= y
                    and event.position().y() <= y + star_height
                ):
                    for i in range(5):
                        x = x_start + (i * (star_width + 2))
                        if (
                            event.position().x() >= x
                            and event.position().x() <= x + star_width
                        ):
                            new_rating = i + 1
                            current_rating = index.data(QtCore.Qt.UserRole + 1)
                            if current_rating == new_rating:
                                new_rating = 0  # Allow clearing
                            model.setData(index, new_rating, QtCore.Qt.UserRole + 1)
                            return True

        return super().editorEvent(event, model, option, index)
