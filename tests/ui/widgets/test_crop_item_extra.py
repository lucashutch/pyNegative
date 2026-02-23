from PySide6.QtCore import QRectF, QPointF
from PySide6.QtGui import QPainter, QImage
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView
from pynegative.ui.widgets.crop_item import CropRectItem


def test_crop_item_full_coverage(qtbot):
    scene = QGraphicsScene(0, 0, 1000, 1000)
    view = QGraphicsView(scene)
    item = CropRectItem(QRectF(100, 100, 200, 200))
    scene.addItem(item)

    view.show()
    qtbot.addWidget(view)

    item.set_rect(QRectF(150, 150, 100, 100))
    assert item.get_rect() == QRectF(150, 150, 100, 100)

    item.set_safe_bounds(QRectF(0, 0, 800, 800))
    item.set_aspect_ratio(1.5)

    assert item._hit_test(QPointF(150, 150)) == "tl"

    # Need to correctly hit test rot_top, since offset changes based on scale
    handles = item._get_rotation_handles()
    rot_top = handles["rot_top"]
    assert item._hit_test_rotation(rot_top) == "rot_top"

    item.set_rotation(15.0)
    assert item.get_rotation() == 15.0

    img = QImage(100, 100, QImage.Format_ARGB32)
    painter = QPainter(img)
    item.paint(painter, None, None)
    painter.end()

    # _update_geometry manual triggers
    item._mouse_press_rect = QRectF(150, 150, 100, 100)
    item._active_handle = "move"
    item._update_geometry("move", QPointF(10, 10))

    item._aspect_ratio = 0.0
    item._active_handle = "br"
    item._update_geometry("br", QPointF(10, 10))

    item.set_aspect_ratio(1.0)
    item._mouse_press_rect = item._rect
    item._update_geometry("br", QPointF(20, 20))
    item._mouse_press_rect = item._rect
    item._update_geometry("l", QPointF(-10, 0))
    item._mouse_press_rect = item._rect
    item._update_geometry("t", QPointF(0, -10))

    # Try snapping boundaries
    item.set_safe_bounds(QRectF(160, 160, 50, 50))

    # angle snaps
    assert item._snap_angle(4.0) == 0.0
    assert item._snap_angle(92.0) == 90.0
    assert item._snap_angle(18.0) == 18.0
