"""Qt version compatibility helpers and shared widget utilities."""

from PySide6 import QtCore, QtGui


def get_event_pos(event):
    """Return the local position from a mouse event (Qt5/Qt6 compatible)."""
    if hasattr(event, "position"):
        return event.position()
    return event.pos()


STAR_COLOR_FILLED = QtGui.QColor("#f0c419")
STAR_COLOR_EMPTY = QtGui.QColor("#808080")


def create_star_pixmap(
    filled: bool,
    size: int = 24,
    font_size: int = 20,
    *,
    empty_color: QtGui.QColor | None = None,
) -> QtGui.QPixmap:
    """Create a star rating pixmap (filled ★ or empty ☆)."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    font = painter.font()
    try:
        font.setPointSize(font_size)
    except Exception:
        pass
    painter.setFont(font)
    if filled:
        painter.setPen(STAR_COLOR_FILLED)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "★")
    else:
        painter.setPen(empty_color or STAR_COLOR_EMPTY)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignCenter, "☆")
    painter.end()
    return pixmap
