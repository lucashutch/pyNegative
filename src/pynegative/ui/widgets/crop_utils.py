import math
import PySide6.QtCore as QtCore
from PySide6 import QtGui


def update_crop_geometry(
    handle: str,
    diff: QtCore.QPointF,
    start_rect: QtCore.QRectF,
    aspect_ratio: float,
    bounds: QtCore.QRectF,
) -> QtCore.QRectF:
    """Calculate the new crop rectangle based on a handle drag."""
    r = QtCore.QRectF(start_rect)

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
        if aspect_ratio <= 0:
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
                    fixed = start_rect.bottomRight()
                elif anchor == "bl":
                    fixed = start_rect.bottomLeft()
                elif anchor == "tr":
                    fixed = start_rect.topRight()
                elif anchor == "tl":
                    fixed = start_rect.topLeft()

                new_w = abs(r.right() - r.left())
                new_h = abs(r.bottom() - r.top())

                # Use the larger dimension change to determine the other
                if new_w / aspect_ratio > new_h:
                    new_h = new_w / aspect_ratio
                else:
                    new_w = new_h * aspect_ratio

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
                    new_h = new_w / aspect_ratio
                    center_y = start_rect.center().y()
                    r.setTop(center_y - new_h / 2)
                    r.setBottom(center_y + new_h / 2)
                else:  # t or b
                    new_h = r.height()
                    new_w = new_h * aspect_ratio
                    center_x = start_rect.center().x()
                    r.setLeft(center_x - new_w / 2)
                    r.setRight(center_x + new_w / 2)

            # 4. Final Clamp (may break ratio slightly if at edge, but safer for bounds)
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
                if r.width() / aspect_ratio > r.height():
                    new_w = r.height() * aspect_ratio
                    # Keep center
                    cx = r.center().x()
                    r.setLeft(cx - new_w / 2)
                    r.setRight(cx + new_w / 2)
                else:
                    new_h = r.width() / aspect_ratio
                    cy = r.center().y()
                    r.setTop(cy - new_h / 2)
                    r.setBottom(cy + new_h / 2)

    return r.normalized()


def calculate_angle_from_center(pos: QtCore.QPointF, center: QtCore.QPointF) -> float:
    """Calculate angle in degrees from center point to position."""
    dx = pos.x() - center.x()
    dy = pos.y() - center.y()
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def snap_angle(angle: float, snap_threshold: float = 5.0) -> float:
    """Snap angle to cardinal directions if within threshold."""
    snap_angles = [0.0, 90.0, -90.0, 180.0, -180.0]
    for snap in snap_angles:
        if abs(angle - snap) < snap_threshold:
            return snap
    return angle


def draw_rotation_icon(painter, center_pos, scale=1.0):
    """Draw double curved arrow rotation icon at handle position."""
    painter.save()

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
