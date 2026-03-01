"""Qt version compatibility helpers and shared widget utilities."""

from PySide6 import QtGui

from ..icons import get_heroicon


def get_event_pos(event):
    """Return the local position from a mouse event (Qt5/Qt6 compatible)."""
    if hasattr(event, "position"):
        return event.position()
    return event.pos()


STAR_COLOR_FILLED = "#f0c419"
STAR_COLOR_EMPTY = "#808080"


def create_star_pixmap(
    filled: bool,
    size: int = 24,
    font_size: int = 20,
    *,
    empty_color: QtGui.QColor | None = None,
) -> QtGui.QPixmap:
    """Create a star rating pixmap using Heroicons."""
    if filled:
        icon = get_heroicon("star", size=size, color=STAR_COLOR_FILLED, variant="solid")
    else:
        color = empty_color.name() if empty_color else STAR_COLOR_EMPTY
        icon = get_heroicon("star", size=size, color=color, variant="outline")
    return icon.pixmap(size, size)
