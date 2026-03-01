from PySide6 import QtCore, QtGui, QtWidgets


def _get_anchor_widget(source_widget):
    viewport_getter = getattr(source_widget, "viewport", None)
    if callable(viewport_getter):
        viewport = viewport_getter()
        if isinstance(viewport, QtWidgets.QWidget):
            return viewport
    return source_widget


def get_menu_exec_position(menu, source_widget, pos):
    """Return a screen-safe global position for context menu execution."""
    anchor_widget = _get_anchor_widget(source_widget)
    global_pos = anchor_widget.mapToGlobal(pos)

    screen = QtWidgets.QApplication.screenAt(global_pos)
    if screen is None and isinstance(anchor_widget, QtWidgets.QWidget):
        screen = anchor_widget.screen()
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return global_pos

    screen_geo = screen.availableGeometry()
    menu_size = menu.sizeHint()

    x = global_pos.x()
    y = global_pos.y()

    space_right = screen_geo.right() - x + 1
    space_left = x - screen_geo.left()
    if menu_size.width() > space_right and space_left >= menu_size.width():
        x = x - menu_size.width()

    space_below = screen_geo.bottom() - y + 1
    space_above = y - screen_geo.top()
    if menu_size.height() > space_below and space_above >= menu_size.height():
        y = y - menu_size.height()

    min_x = screen_geo.left()
    min_y = screen_geo.top()
    max_x = screen_geo.right() - menu_size.width() + 1
    max_y = screen_geo.bottom() - menu_size.height() + 1

    if max_x < min_x:
        max_x = min_x
    if max_y < min_y:
        max_y = min_y

    clamped_x = min(max(x, min_x), max_x)
    clamped_y = min(max(y, min_y), max_y)

    return QtCore.QPoint(clamped_x, clamped_y)
