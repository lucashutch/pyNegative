from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt


class CropRectItem(QtWidgets.QGraphicsObject):
    """
    Interactive Crop Rectangle with resize handles, rotation handles, and Rule of Thirds grid.
    """

    cropChanged = QtCore.Signal(QtCore.QRectF)
    rotationChanged = QtCore.Signal(float)  # Emit rotation angle in degrees

    def __init__(self, rect=QtCore.QRectF(0, 0, 100, 100), parent=None):
        super().__init__(parent)
        self._rect = rect
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self._handle_size = 15
        self._active_handle = None
        self._mouse_press_pos = None
        self._mouse_press_screen_pos = None
        self._mouse_press_rect = None
        self._aspect_ratio = 0.0  # 0.0 means free crop
        self._safe_bounds = None  # QRectF of valid image area

        # Rotation handle state
        self._rotation_angle = 0.0  # Current rotation in degrees
        self._active_rotation_handle = None  # Which rotation handle is being dragged
        self._rotation_start_angle = 0.0  # Starting angle at mouse press
        self._rotation_start_rotation = 0.0  # Starting rotation value at mouse press
        self._rotation_handle_distance = 50  # Distance from edge in screen pixels
        self._last_rotation_emit_time = 0  # Track last signal emission for throttling
        self._rotation_throttle_ms = 33  # ~30fps (1000ms / 30 = 33.3ms)

        # Colors
        self._pen = QtGui.QPen(QtCore.Qt.white, 1, QtCore.Qt.SolidLine)
        self._pen.setCosmetic(True)  # Keep line width constant regardless of zoom

        self._grid_pen = QtGui.QPen(
            QtGui.QColor(255, 255, 255, 100), 1, QtCore.Qt.DashLine
        )
        self._grid_pen.setCosmetic(True)

    def boundingRect(self):
        # Add padding for handles
        # Use a very generous padding to ensure rotation handles are hit-testable
        # even when zoomed out on high-resolution images.
        pad = 5000
        return self._rect.adjusted(-pad, -pad, pad, pad)

    def set_rect(self, rect):
        self._rect = rect
        self.prepareGeometryChange()
        self.update()

    def get_rect(self):
        return self._rect

    def set_aspect_ratio(self, ratio):
        """Set aspect ratio (0.0 for free) and update current rect."""
        self._aspect_ratio = ratio
        if ratio > 0:
            # Adjust current rect to match ratio
            r = QtCore.QRectF(self._rect)
            center = r.center()

            # Current dimensions
            w = r.width()
            h = r.height()

            # Try to keep width, adjust height
            new_h = w / ratio
            if new_h <= h:
                h = new_h
            else:
                # If new height is larger than current, keep height and adjust width
                w = h * ratio

            r.setSize(QtCore.QSizeF(w, h))
            r.moveCenter(center)

            # Clamp to bounds
            bounds = self._safe_bounds
            if not bounds:
                scene = self.scene()
                if scene:
                    bounds = scene.sceneRect()

            if bounds:
                if r.left() < bounds.left():
                    r.moveLeft(bounds.left())
                if r.top() < bounds.top():
                    r.moveTop(bounds.top())
                if r.right() > bounds.right():
                    r.moveRight(bounds.right())
                if r.bottom() > bounds.bottom():
                    r.moveBottom(bounds.bottom())

            self._rect = r.normalized()
            self.prepareGeometryChange()
            self.update()
            self.cropChanged.emit(self._rect)

    def set_safe_bounds(self, rect):
        """Set the bounds within which the crop rectangle must stay."""
        self._safe_bounds = rect
        if rect:
            # Force current rect inside new safe bounds
            r = QtCore.QRectF(self._rect)
            if r.left() < rect.left():
                r.setLeft(rect.left())
            if r.top() < rect.top():
                r.setTop(rect.top())
            if r.right() > rect.right():
                r.setRight(rect.right())
            if r.bottom() > rect.bottom():
                r.setBottom(rect.bottom())

            # If it became invalid or we want to maintain ratio,
            # we should ideally re-apply ratio logic.
            # For now, just ensure it's normalized and within.
            self._rect = r.normalized()
            if self._aspect_ratio > 0:
                # Re-trigger ratio adjustment if needed
                self.set_aspect_ratio(self._aspect_ratio)
            else:
                self.prepareGeometryChange()
                self.update()
                self.cropChanged.emit(self._rect)

    def paint(self, painter, option, widget):
        # 1. Dim the area OUTSIDE the crop
        scene = self.scene()
        if scene:
            bounds = scene.sceneRect()
            # Draw overlay
            overlay_color = QtGui.QColor(0, 0, 0, 160)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(overlay_color)

            # Create a path for the whole area and subtract the crop rect
            path = QtGui.QPainterPath()
            path.addRect(bounds)
            path.addRect(self._rect)
            painter.drawPath(
                path
            )  # Odd-even fill rules means the rect will be "cut out"

        # 2. Rect Border
        painter.setPen(self._pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(self._rect)

        # 2. Rule of Thirds Grid
        left = self._rect.left()
        top = self._rect.top()
        w = self._rect.width()
        h = self._rect.height()

        painter.setPen(self._grid_pen)

        # Verticals
        painter.drawLine(QtCore.QLineF(left + w / 3, top, left + w / 3, top + h))
        painter.drawLine(
            QtCore.QLineF(left + 2 * w / 3, top, left + 2 * w / 3, top + h)
        )

        # Horizontals
        painter.drawLine(QtCore.QLineF(left, top + h / 3, left + w, top + h / 3))
        painter.drawLine(
            QtCore.QLineF(left, top + 2 * h / 3, left + w, top + 2 * h / 3)
        )

        # 3. Handles (Corners and Sides)
        painter.setBrush(QtCore.Qt.white)
        painter.setPen(QtCore.Qt.NoPen)
        # Use simple rectangles for handles
        handles = self._get_handles()

        # Use simple fixed size for cosmetic pen (it's in screen pixels)
        # No need to manually scale radius for drawing if we use setCosmetic(True)
        # But we might want access to scale for other logic?
        # For drawing dots, cosmetic pen does the work.

        # Cosmetic drawing logic:
        # Use semi-transparent white color
        color = QtGui.QColor(255, 255, 255, 180)
        handle_pen = QtGui.QPen(color, self._handle_size)
        handle_pen.setCosmetic(True)  # Screen space size
        handle_pen.setCapStyle(QtCore.Qt.RoundCap)  # Round dots
        painter.setPen(handle_pen)

        for k, pt in handles.items():
            painter.drawPoint(pt)

        # 4. Rotation Handles (Top Center and Middle Right)
        rotation_handles = self._get_rotation_handles()
        handle_color = QtGui.QColor(74, 144, 226, 200)  # Blue #4A90E2
        rotation_handle_pen = QtGui.QPen(handle_color, self._handle_size)
        rotation_handle_pen.setCosmetic(True)
        rotation_handle_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(rotation_handle_pen)

        for k, pt in rotation_handles.items():
            painter.drawPoint(pt)
            # Draw rotation icon (double curved arrows)
            self._draw_rotation_icon(painter, pt)

    def _hit_test(self, pos):
        # Check handles first
        handles = self._get_handles()

        # Hit radius should be constant in SCREEN pixels, not Scene pixels
        # Default roughly 20px screen distance
        # We need access to view... tricky from just Item logic without explicit ref
        # Fallback: Assume cosmetic pen size which is ~15px screen
        radius = 50  # Scene units fallback

        # Try to deduce scale from scene/view? No easy way.
        # IF we assume the user is clicking handles, we can use a generous approach

        views = self.scene().views()
        if views:
            view = views[0]
            scale = view.transform().m11()
            if scale > 0:
                radius = 20 / scale  # 20px screen radius

        for k, pt in handles.items():
            if QtCore.QLineF(pos, pt).length() < radius:
                return k
        return None

    def _get_handles(self):
        r = self._rect
        return {
            "tl": r.topLeft(),
            "tr": r.topRight(),
            "bl": r.bottomLeft(),
            "br": r.bottomRight(),
            "t": QtCore.QPointF(r.center().x(), r.top()),
            "b": QtCore.QPointF(r.center().x(), r.bottom()),
            "l": QtCore.QPointF(r.left(), r.center().y()),
            "r": QtCore.QPointF(r.right(), r.center().y()),
        }

    def hoverMoveEvent(self, event):
        pos = event.position()

        # Check rotation zone first (handle + far outside)
        rotation_handle = self._hit_test_rotation(pos)

        views = self.scene().views() if self.scene() else []
        scale = 1.0
        if views:
            scale = views[0].transform().m11()

        buffer = 30 / scale
        is_outside = not self._rect.adjusted(-buffer, -buffer, buffer, buffer).contains(
            pos
        )

        if rotation_handle or is_outside:
            self.setCursor(Qt.PointingHandCursor)
            super().hoverMoveEvent(event)
            return

        handle = self._hit_test(pos)

        cursor = Qt.ArrowCursor
        if handle == "tl" or handle == "br":
            cursor = Qt.SizeFDiagCursor
        elif handle == "tr" or handle == "bl":
            cursor = Qt.SizeBDiagCursor
        elif handle == "l" or handle == "r":
            cursor = Qt.SizeHorCursor
        elif handle == "t" or handle == "b":
            cursor = Qt.SizeVerCursor
        elif self._rect.contains(pos):
            cursor = Qt.SizeAllCursor

        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()

            # 1. Check specific rotation handle (top middle)
            handle = self._hit_test_rotation(pos)
            if handle:
                self._active_rotation_handle = handle
                self._mouse_press_pos = pos
                self._rotation_start_angle = self._calculate_angle_from_center(pos)
                self._rotation_start_rotation = self._rotation_angle
                event.accept()
                return

            # 2. Check resize handles
            handle = self._hit_test(pos)
            if handle:
                self._active_handle = handle
                self._mouse_press_pos = pos
                self._mouse_press_screen_pos = event.screenPos()
                self._mouse_press_rect = QtCore.QRectF(self._rect)
                event.accept()
                return

            # 3. Check if inside crop rect (move)
            if self._rect.contains(pos):
                self._active_handle = "move"
                self._mouse_press_pos = pos
                self._mouse_press_screen_pos = event.screenPos()
                self._mouse_press_rect = QtCore.QRectF(self._rect)
                event.accept()
                return

            # 4. Outside interaction: Buffer check
            views = self.scene().views() if self.scene() else []
            scale = 1.0
            if views:
                scale = views[0].transform().m11()

            # 30px screen-space buffer to avoid accidental rotations near the crop area
            buffer = 30 / scale

            if not self._rect.adjusted(-buffer, -buffer, buffer, buffer).contains(pos):
                # Far enough outside to rotate
                self._active_rotation_handle = "general"
                self._mouse_press_pos = pos
                self._rotation_start_angle = self._calculate_angle_from_center(pos)
                self._rotation_start_rotation = self._rotation_angle
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._active_rotation_handle:
            # Calculate rotation angle
            current_angle = self._calculate_angle_from_center(event.position())
            delta_angle = current_angle - self._rotation_start_angle

            # Handle angle wrapping (crossing the 180/-180 boundary)
            while delta_angle > 180:
                delta_angle -= 360
            while delta_angle < -180:
                delta_angle += 360

            new_rotation = self._rotation_start_rotation - delta_angle

            # Shift-key snapping
            modifiers = QtGui.QGuiApplication.queryKeyboardModifiers()
            if modifiers & Qt.ShiftModifier:
                new_rotation = self._snap_angle(new_rotation)

            # Clamp to ±45°
            new_rotation = max(-45.0, min(45.0, new_rotation))
            self._rotation_angle = new_rotation

            # Throttle signal emission (30fps)
            import time

            current_time = time.time() * 1000  # milliseconds
            if (
                current_time - self._last_rotation_emit_time
                >= self._rotation_throttle_ms
            ):
                self.rotationChanged.emit(self._rotation_angle)
                self._last_rotation_emit_time = current_time

            # Always update (this is cheap)
            self.update()
            event.accept()
            return

        if self._active_handle and self._mouse_press_rect:
            # Calculate delta in screen pixels
            delta_px = event.screenPos() - self._mouse_press_screen_pos

            # Convert to scene units
            scale = 1.0
            views = self.scene().views()
            if views:
                scale = views[0].transform().m11()

            diff = delta_px / scale
            self._update_geometry(self._active_handle, diff)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._active_rotation_handle:
            # Emit final rotation value on release (ensures exact final value is set)
            self.rotationChanged.emit(self._rotation_angle)
            self._active_rotation_handle = None
            self.update()

        self._active_handle = None
        self._mouse_press_pos = None
        self._mouse_press_rect = None
        super().mouseReleaseEvent(event)

    def _update_geometry(self, handle, diff):
        r = QtCore.QRectF(self._mouse_press_rect)
        scene = self.scene()

        # Use safe bounds if available, else scene rect
        bounds = self._safe_bounds
        if not bounds:
            bounds = scene.sceneRect() if scene else QtCore.QRectF(0, 0, 10000, 10000)

        if handle == "move":
            r.translate(diff)
            # Clamp to bounds
            if r.left() < bounds.left():
                r.moveLeft(bounds.left())
            if r.top() < bounds.top():
                r.moveTop(bounds.top())
            if r.right() > bounds.right():
                r.moveRight(bounds.right())
            if r.bottom() > bounds.bottom():
                r.moveBottom(bounds.bottom())
        else:
            # Resizing logic
            if self._aspect_ratio <= 0:
                # Free Resize
                if "l" in handle:
                    r.setLeft(r.left() + diff.x())
                if "r" in handle:
                    r.setRight(r.right() + diff.x())
                if "t" in handle:
                    r.setTop(r.top() + diff.y())
                if "b" in handle:
                    r.setBottom(r.bottom() + diff.y())

                # Clamp edges
                if r.left() < bounds.left():
                    r.setLeft(bounds.left())
                if r.top() < bounds.top():
                    r.setTop(bounds.top())
                if r.right() > bounds.right():
                    r.setRight(bounds.right())
                if r.bottom() > bounds.bottom():
                    r.setBottom(bounds.bottom())
            else:
                # Aspect Ratio Locked Resize
                # 1. Determine anchor point (opposite of handle)
                anchor = ""
                if "t" in handle:
                    anchor += "b"
                elif "b" in handle:
                    anchor += "t"
                if "l" in handle:
                    anchor += "r"
                elif "r" in handle:
                    anchor += "l"

                # If side handle, we need to pick a default anchor side
                if handle == "t":
                    anchor = "b"
                elif handle == "b":
                    anchor = "t"
                elif handle == "l":
                    anchor = "r"
                elif handle == "r":
                    anchor = "l"

                # 2. Apply initial diff
                if "l" in handle:
                    r.setLeft(r.left() + diff.x())
                if "r" in handle:
                    r.setRight(r.right() + diff.x())
                if "t" in handle:
                    r.setTop(r.top() + diff.y())
                if "b" in handle:
                    r.setBottom(r.bottom() + diff.y())

                # 3. Enforce ratio from anchor
                # Corner handles
                if len(handle) == 2:
                    # Find fixed point
                    if anchor == "br":
                        fixed = self._mouse_press_rect.bottomRight()
                    elif anchor == "bl":
                        fixed = self._mouse_press_rect.bottomLeft()
                    elif anchor == "tr":
                        fixed = self._mouse_press_rect.topRight()
                    elif anchor == "tl":
                        fixed = self._mouse_press_rect.topLeft()

                    new_w = abs(r.right() - r.left())
                    new_h = abs(r.bottom() - r.top())

                    # Use the larger dimension change to determine the other
                    if new_w / self._aspect_ratio > new_h:
                        new_h = new_w / self._aspect_ratio
                    else:
                        new_w = new_h * self._aspect_ratio

                    # Re-apply relative to fixed point
                    if "l" in handle:
                        r.setLeft(fixed.x() - new_w)
                    else:
                        r.setRight(fixed.x() + new_w)

                    if "t" in handle:
                        r.setTop(fixed.y() - new_h)
                    else:
                        r.setBottom(fixed.y() + new_h)

                # Side handles
                else:
                    if handle in ["l", "r"]:
                        new_w = r.width()
                        new_h = new_w / self._aspect_ratio
                        center_y = self._mouse_press_rect.center().y()
                        r.setTop(center_y - new_h / 2)
                        r.setBottom(center_y + new_h / 2)
                    else:  # t or b
                        new_h = r.height()
                        new_w = new_h * self._aspect_ratio
                        center_x = self._mouse_press_rect.center().x()
                        r.setLeft(center_x - new_w / 2)
                        r.setRight(center_x + new_w / 2)

                # 4. Final Clamp (may break ratio slightly if at edge, but safer for bounds)
                # To perfectly maintain ratio, we should shrink BOTH dimensions if one hits a boundary.
                if (
                    r.left() < bounds.left()
                    or r.right() > bounds.right()
                    or r.top() < bounds.top()
                    or r.bottom() > bounds.bottom()
                ):
                    # Shrink to fit bounds
                    if r.left() < bounds.left():
                        r.moveLeft(bounds.left())
                        if r.right() > bounds.right():
                            r.setRight(bounds.right())
                    if r.right() > bounds.right():
                        r.moveRight(bounds.right())
                        if r.left() < bounds.left():
                            r.setLeft(bounds.left())
                    if r.top() < bounds.top():
                        r.moveTop(bounds.top())
                        if r.bottom() > bounds.bottom():
                            r.setBottom(bounds.bottom())
                    if r.bottom() > bounds.bottom():
                        r.moveBottom(bounds.bottom())
                        if r.top() < bounds.top():
                            r.setTop(bounds.top())

                    # Re-normalize and re-force ratio if clamped
                    r = r.normalized()
                    if r.width() / self._aspect_ratio > r.height():
                        new_w = r.height() * self._aspect_ratio
                        # Keep center
                        cx = r.center().x()
                        r.setLeft(cx - new_w / 2)
                        r.setRight(cx + new_w / 2)
                    else:
                        new_h = r.width() / self._aspect_ratio
                        cy = r.center().y()
                        r.setTop(cy - new_h / 2)
                        r.setBottom(cy + new_h / 2)

        # Use normalized rect to prevent negative size
        self._rect = r.normalized()
        self.prepareGeometryChange()
        self.update()
        self.cropChanged.emit(self._rect)

    def set_rotation(self, angle: float) -> None:
        """Set rotation angle externally (from slider)."""
        self._rotation_angle = max(-45.0, min(45.0, angle))
        self.update()

    def get_rotation(self) -> float:
        """Get current rotation angle."""
        return self._rotation_angle

    def _get_rotation_handles(self):
        """Return positions of rotation handles (top center)."""
        r = self._rect

        # Calculate screen-space offset (50px outside edge)
        views = self.scene().views() if self.scene() else []
        scale = 1.0
        if views:
            scale = views[0].transform().m11()

        offset_distance = (
            self._rotation_handle_distance / scale
        )  # Convert to scene units

        return {
            "rot_top": QtCore.QPointF(r.center().x(), r.top() - offset_distance),
        }

    def _hit_test_rotation(self, pos):
        """Check if position hits a rotation handle."""
        handles = self._get_rotation_handles()

        # Hit radius in screen pixels
        views = self.scene().views() if self.scene() else []
        scale = 1.0
        if views:
            scale = views[0].transform().m11()

        # Use a generous 30px screen radius for easier hitting
        radius = 30 / scale

        for k, pt in handles.items():
            if QtCore.QLineF(pos, pt).length() < radius:
                return k
        return None

    def _calculate_angle_from_center(self, pos: QtCore.QPointF) -> float:
        """Calculate angle in degrees from rect center to position."""
        import math

        center = self._rect.center()
        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def _snap_angle(self, angle: float) -> float:
        """Snap angle to cardinal directions (0°, ±90°, ±180°) if within threshold."""
        snap_threshold = 5.0  # degrees
        snap_angles = [0.0, 90.0, -90.0, 180.0, -180.0]

        for snap in snap_angles:
            if abs(angle - snap) < snap_threshold:
                return snap

        return angle

    def _draw_rotation_icon(self, painter, center_pos):
        """Draw double curved arrow rotation icon at handle position."""
        import math

        painter.save()

        # Get scale for cosmetic drawing
        views = self.scene().views() if self.scene() else []
        scale = 1.0
        if views:
            scale = views[0].transform().m11()

        icon_radius = 8 / scale  # 8px in screen space

        # Draw circular arrows (rotation symbol)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.5 / scale)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)

        # Draw two arc arrows forming a rotation symbol
        rect = QtCore.QRectF(
            center_pos.x() - icon_radius,
            center_pos.y() - icon_radius,
            icon_radius * 2,
            icon_radius * 2,
        )

        # First arc (top-right to bottom-right, clockwise)
        start_angle = -45 * 16  # Qt uses 1/16th degree units
        span_angle = 200 * 16
        painter.drawArc(rect, start_angle, span_angle)

        # Arrow head for first arc
        arrow_angle = math.radians(-45 + 200)
        arrow_x = center_pos.x() + icon_radius * math.cos(arrow_angle)
        arrow_y = center_pos.y() - icon_radius * math.sin(arrow_angle)
        arrow_size = 3 / scale

        # Draw small arrow head
        arrow_head = QtGui.QPolygonF(
            [
                QtCore.QPointF(arrow_x, arrow_y),
                QtCore.QPointF(
                    arrow_x + arrow_size * math.cos(arrow_angle + math.radians(150)),
                    arrow_y - arrow_size * math.sin(arrow_angle + math.radians(150)),
                ),
                QtCore.QPointF(
                    arrow_x + arrow_size * math.cos(arrow_angle - math.radians(150)),
                    arrow_y - arrow_size * math.sin(arrow_angle - math.radians(150)),
                ),
            ]
        )
        painter.setBrush(QtGui.QColor(255, 255, 255, 220))
        painter.drawPolygon(arrow_head)

        # Second arc (bottom-left to top-left, clockwise) - mirror of first
        start_angle2 = 135 * 16
        span_angle2 = 200 * 16
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawArc(rect, start_angle2, span_angle2)

        # Arrow head for second arc
        arrow_angle2 = math.radians(135 + 200)
        arrow_x2 = center_pos.x() + icon_radius * math.cos(arrow_angle2)
        arrow_y2 = center_pos.y() - icon_radius * math.sin(arrow_angle2)

        arrow_head2 = QtGui.QPolygonF(
            [
                QtCore.QPointF(arrow_x2, arrow_y2),
                QtCore.QPointF(
                    arrow_x2 + arrow_size * math.cos(arrow_angle2 + math.radians(150)),
                    arrow_y2 - arrow_size * math.sin(arrow_angle2 + math.radians(150)),
                ),
                QtCore.QPointF(
                    arrow_x2 + arrow_size * math.cos(arrow_angle2 - math.radians(150)),
                    arrow_y2 - arrow_size * math.sin(arrow_angle2 - math.radians(150)),
                ),
            ]
        )
        painter.setBrush(QtGui.QColor(255, 255, 255, 220))
        painter.drawPolygon(arrow_head2)

        painter.restore()
