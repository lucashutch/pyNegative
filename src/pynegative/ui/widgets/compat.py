"""Qt version compatibility helpers."""


def get_event_pos(event):
    """Return the local position from a mouse event (Qt5/Qt6 compatible)."""
    if hasattr(event, "position"):
        return event.position()
    return event.pos()
