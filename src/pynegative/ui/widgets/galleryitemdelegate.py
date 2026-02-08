from PySide6 import QtWidgets, QtGui, QtCore


class GalleryItemDelegate(QtWidgets.QStyledItemDelegate):
    # Accent colors from styles.qss
    FILLED_CIRCLE_COLOR = QtGui.QColor("#6366f1")  # Primary accent
    EMPTY_CIRCLE_COLOR = QtGui.QColor("#4a4a4a")  # Subtle border for unselected
    SELECTION_HIGHLIGHT = QtGui.QColor(99, 102, 241, 40)  # 15% opacity primary accent
    CIRCLE_SIZE = 18
    CIRCLE_MARGIN = 6

    # Layout Constants (will be used as defaults/bases)
    BOTTOM_HEIGHT = 28
    STAR_STRIP_WIDTH = 26
    STAR_SIZE = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._item_size = 200
        self.star_filled = self._create_star_pixmap(True)
        self.star_empty = self._create_star_pixmap(False)
        self._show_selection_circles = False
        self._circle_clicked = False

        self._resizing = False
        self._resize_cooldown = QtCore.QTimer()
        self._resize_cooldown.setSingleShot(True)
        self._resize_cooldown.setInterval(150)
        self._resize_cooldown.timeout.connect(self._stop_resizing)

    def set_item_size(self, size):
        """Update the item size."""
        if self._item_size != size:
            self._item_size = size
            self._resizing = True
            self._resize_cooldown.start()

    def _stop_resizing(self):
        """Called when resize interaction stops."""
        self._resizing = False
        # Request a repaint of all items to get smooth quality back
        parent = self.parent()
        if parent:
            parent.update()

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
        pixmap = QtGui.QPixmap(self.STAR_SIZE, self.STAR_SIZE)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        if filled:
            painter.setPen(QtGui.QColor("#f0c419"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
        else:
            painter.setPen(QtGui.QColor("#909090"))
            painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")
        painter.end()
        return pixmap

    def sizeHint(self, option, index):
        return QtCore.QSize(self._item_size, self._item_size)

    def paint(self, painter, option, index):
        # Initialize style option
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        widget = opt.widget
        style = widget.style() if widget else QtWidgets.QApplication.style()

        # Draw Background (CSS styling via PE_PanelItemViewItem)
        # Ensure we use the full rect
        style.drawPrimitive(QtWidgets.QStyle.PE_PanelItemViewItem, opt, painter, widget)

        # 1. Draw Icon (Centered in the area above the filename)
        if opt.features & QtWidgets.QStyleOptionViewItem.HasDecoration:
            icon = opt.icon

            # The area available for the image is the full width minus some padding,
            # and the height minus the bottom filename area.
            # 2px border + 6px padding = 8px offset.
            # We use -5 for the height subtraction from common area to keep it close to text
            # but still having some breathing room.
            image_area_rect = QtCore.QRect(
                opt.rect.left() + 8,
                opt.rect.top() + 8,
                opt.rect.width() - 16,
                opt.rect.height() - self.BOTTOM_HEIGHT - 12,
            )

            # Get the pixmap at a size that fits the area
            # ThumbnailLoader already provides a 200px max size pixmap.
            pixmap = icon.pixmap(image_area_rect.size())
            if not pixmap.isNull():
                # Use FastTransformation during active resize for much better performance
                transform = (
                    QtCore.Qt.FastTransformation
                    if self._resizing
                    else QtCore.Qt.SmoothTransformation
                )

                scaled_pixmap = pixmap.scaled(
                    image_area_rect.size(),
                    QtCore.Qt.KeepAspectRatio,
                    transform,
                )

                # Calculate centered position within image_area_rect
                draw_rect = QtWidgets.QStyle.alignedRect(
                    QtCore.Qt.LeftToRight,
                    QtCore.Qt.AlignCenter,
                    scaled_pixmap.size(),
                    image_area_rect,
                )

                painter.drawPixmap(draw_rect, scaled_pixmap)

        # 2. Draw Text (Centered at the bottom)
        if opt.features & QtWidgets.QStyleOptionViewItem.HasDisplay:
            text = opt.text
            text_rect = QtCore.QRect(
                opt.rect.left() + 8,
                opt.rect.bottom() - self.BOTTOM_HEIGHT,
                opt.rect.width() - 16,
                self.BOTTOM_HEIGHT - 4,
            )

            text_color = opt.palette.text().color()
            if opt.state & QtWidgets.QStyle.State_Selected:
                text_color = opt.palette.highlightedText().color()

            painter.setPen(text_color)
            painter.save()
            font = painter.font()

            # Scale font size slightly for small items
            font_size = 10
            if self._item_size < 150:
                font_size = 8
            elif self._item_size > 300:
                font_size = 12

            font.setPointSize(font_size)
            painter.setFont(font)

            # Skip expensive elision during active resize for 60fps
            if self._resizing:
                elided_text = text if len(text) < 15 else text[:12] + "..."
            else:
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

        # 3. Custom Overlays (Stars, Selection Highlight, Circles)
        list_widget = self.parent()
        if not list_widget:
            return

        is_hovered = False
        if hasattr(list_widget, "get_hovered_item"):
            hovered_item = list_widget.get_hovered_item()
            # If it's a QListWidget, hovered_item is QListWidgetItem.
            # If it's a QListView, it might be different. Let's compare index if possible.
            if hovered_item:
                if isinstance(hovered_item, QtWidgets.QListWidgetItem):
                    is_hovered = hovered_item.text() == index.data()
                else:
                    # Fallback for QListView where generic implementation might return index
                    is_hovered = hovered_item == index

        is_selected = option.state & QtWidgets.QStyle.State_Selected

        # Draw highlight overlay for selected items (in addition to stylesheet border)
        if is_selected:
            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(self.SELECTION_HIGHLIGHT))
            painter.setPen(QtCore.Qt.NoPen)
            highlight_rect = option.rect.adjusted(2, 2, -2, -2)
            painter.drawRoundedRect(highlight_rect, 6, 6)
            painter.restore()

        # 4. Draw Rating Stars (Vertical on the right)
        rating = index.data(QtCore.Qt.UserRole + 1)
        if rating is None:
            rating = 0

        if rating > 0 or is_hovered:
            total_stars_height = 5 * self.STAR_SIZE + 4 * 4  # 5 stars + 4 spacers
            stars_y_start = (
                opt.rect.top()
                + (opt.rect.height() - self.BOTTOM_HEIGHT - total_stars_height) // 2
            )
            stars_x = (
                opt.rect.right()
                - self.STAR_STRIP_WIDTH
                + (self.STAR_STRIP_WIDTH - self.STAR_SIZE) // 2
            )

            for i in range(5):
                star_icon = self.star_empty
                if i < rating:
                    star_icon = self.star_filled
                elif not is_hovered and i >= rating:
                    continue  # Don't draw empty stars if not hovered

                y = int(stars_y_start + (i * (self.STAR_SIZE + 4)))
                painter.drawPixmap(int(stars_x), y, star_icon)

        # 5. Draw Selection Circle if enabled
        if self._show_selection_circles:
            rect = option.rect
            circle_x = rect.right() - self.CIRCLE_SIZE - self.CIRCLE_MARGIN
            circle_y = rect.top() + self.CIRCLE_MARGIN
            circle_rect = QtCore.QRectF(
                circle_x, circle_y, self.CIRCLE_SIZE, self.CIRCLE_SIZE
            )

            painter.save()
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            if is_selected:
                painter.setBrush(QtGui.QBrush(self.FILLED_CIRCLE_COLOR))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(circle_rect)

                # Draw white checkmark
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
            # If clicking on the selection circle
            if self.is_click_on_circle(event.position().toPoint(), option.rect):
                self._circle_clicked = True
                return False

            list_widget = self.parent()
            is_hovered = False
            if list_widget and hasattr(list_widget, "get_hovered_item"):
                hovered_item = list_widget.get_hovered_item()
                if hovered_item:
                    if isinstance(hovered_item, QtWidgets.QListWidgetItem):
                        is_hovered = hovered_item.text() == index.data()
                    else:
                        is_hovered = hovered_item == index

            if is_hovered:
                # Check if click is on the vertical star strip
                total_stars_height = 5 * self.STAR_SIZE + 4 * 4
                stars_x_start = option.rect.right() - self.STAR_STRIP_WIDTH
                stars_y_start = (
                    option.rect.top()
                    + (option.rect.height() - self.BOTTOM_HEIGHT - total_stars_height)
                    // 2
                )

                if event.position().x() >= stars_x_start:
                    # Determine which star was clicked
                    rel_y = event.position().y() - stars_y_start
                    for i in range(5):
                        star_y = i * (self.STAR_SIZE + 4)
                        if star_y <= rel_y <= star_y + self.STAR_SIZE:
                            new_rating = i + 1
                            current_rating = index.data(QtCore.Qt.UserRole + 1)
                            if current_rating == new_rating:
                                new_rating = 0  # Allow clearing
                            model.setData(index, new_rating, QtCore.Qt.UserRole + 1)
                            return True

        return super().editorEvent(event, model, option, index)
