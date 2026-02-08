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
        # Initialize style option
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        widget = opt.widget
        style = widget.style() if widget else QtWidgets.QApplication.style()

        # Draw Background (CSS styling via PE_PanelItemViewItem)
        style.drawPrimitive(QtWidgets.QStyle.PE_PanelItemViewItem, opt, painter, widget)

        # Determine layout based on item size
        is_large = opt.rect.height() > 150

        # New Layout: Vertical star strip on the right, Filename at the bottom
        star_strip_width = 30 if is_large else 24
        bottom_height = 28 if is_large else 22

        # Symmetric padding for perfect centering in the item box boundaries.
        # We use a bit more padding as requested, but keeping it equal for centering.
        h_padding = 15 if is_large else 10
        v_padding = 10 if is_large else 6

        # 1. Draw Icon (Centered in the thumbnail area)
        if opt.features & QtWidgets.QStyleOptionViewItem.HasDecoration:
            icon = opt.icon

            # Define the target rect as the full area above the filename
            # This ensures the visual center of the thumbnail aligns with the box center.
            image_area_rect = QtCore.QRect(
                opt.rect.left(),
                opt.rect.top(),
                opt.rect.width(),
                opt.rect.height() - bottom_height,
            )

            # Apply symmetric padding
            target_rect = image_area_rect.adjusted(
                h_padding, v_padding, -h_padding, -v_padding
            )

            # Get and scale pixmap
            pixmap = icon.pixmap(target_rect.size())
            scaled_pixmap = pixmap.scaled(
                target_rect.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

            # Calculate centered position within target_rect
            draw_rect = QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight,
                QtCore.Qt.AlignCenter,
                scaled_pixmap.size(),
                target_rect,
            )

            painter.drawPixmap(draw_rect, scaled_pixmap)

        # 2. Draw Text (Centered at the bottom)
        if opt.features & QtWidgets.QStyleOptionViewItem.HasDisplay:
            text = opt.text

            # Use full box width (minus small margins) for centering logic
            text_rect = QtCore.QRect(
                opt.rect.left() + 10,
                opt.rect.bottom() - bottom_height,
                opt.rect.width() - 20,
                bottom_height - 4,
            )

            text_color = opt.palette.text().color()
            if opt.state & QtWidgets.QStyle.State_Selected:
                text_color = opt.palette.highlightedText().color()

            painter.setPen(text_color)
            painter.save()
            font = painter.font()
            if not is_large:
                if font.pointSize() > 1:
                    font.setPointSize(font.pointSize() - 1)
                elif font.pixelSize() > 1:
                    font.setPixelSize(font.pixelSize() - 1)
            painter.setFont(font)

            font_metrics = painter.fontMetrics()
            elided_text = font_metrics.elidedText(
                text, QtCore.Qt.ElideMiddle, text_rect.width()
            )

            painter.drawText(
                text_rect,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                elided_text,
            )
            painter.restore()

        # Custom overlays (Stars, Selection Highlight, Circles)
        list_widget = self.parent()
        if not list_widget:
            return

        is_hovered = False
        if hasattr(list_widget, "get_hovered_item"):
            hovered_item = list_widget.get_hovered_item()
            is_hovered = (
                hovered_item is not None and hovered_item.text() == index.data()
            )

        # Handle selection highlight (Custom Overlay)
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
            # Position stars vertically on the right side
            star_height = self.star_empty.height()
            total_stars_height = 5 * star_height + 4 * 2  # 5 stars + 4 spacers

            stars_y_start = (
                opt.rect.top()
                + (opt.rect.height() - bottom_height - total_stars_height) // 2
            )
            stars_x = (
                opt.rect.right()
                - star_strip_width
                + (star_strip_width - self.star_empty.width()) // 2
            )

            for i in range(5):
                star_icon = self.star_empty
                if i < rating:
                    star_icon = self.star_filled

                y = int(stars_y_start + (i * (star_height + 2)))
                painter.drawPixmap(int(stars_x), y, star_icon)

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
                is_large = option.rect.height() > 150
                star_strip_width = 30 if is_large else 24
                bottom_height = 28 if is_large else 22

                star_width = self.star_empty.width()
                star_height = self.star_empty.height()
                total_stars_height = 5 * star_height + 4 * 2

                stars_x = (
                    option.rect.right()
                    - star_strip_width
                    + (star_strip_width - star_width) // 2
                )
                stars_y_start = (
                    option.rect.top()
                    + (option.rect.height() - bottom_height - total_stars_height) // 2
                )

                if (
                    event.position().x() >= stars_x
                    and event.position().x() <= stars_x + star_width
                ):
                    for i in range(5):
                        y = stars_y_start + (i * (star_height + 2))
                        if (
                            event.position().y() >= y
                            and event.position().y() <= y + star_height
                        ):
                            new_rating = i + 1
                            current_rating = index.data(QtCore.Qt.UserRole + 1)
                            if current_rating == new_rating:
                                new_rating = 0  # Allow clearing
                            model.setData(index, new_rating, QtCore.Qt.UserRole + 1)
                            return True

        return super().editorEvent(event, model, option, index)
