import PySide6.QtCore as QtCore
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QWheelEvent
from pynegative.ui.widgets.combobox import ComboBox


def test_combobox_extra_coverage(qtbot):
    box = ComboBox()
    qtbot.addWidget(box)

    box.addItem("Test 1")
    box.addItem("Test 2")

    # Wheel event
    event = QWheelEvent(
        QPointF(0, 0),
        QPointF(0, 0),
        QtCore.QPoint(0, 120),
        QtCore.QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollBegin,
        False,
    )
    box.wheelEvent(event)

    from PySide6.QtGui import QPaintEvent

    pe = QPaintEvent(box.rect())
    box.paintEvent(pe)

    qtbot.mousePress(box, Qt.LeftButton)
