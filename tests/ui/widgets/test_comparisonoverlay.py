from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent
from pynegative.ui.widgets.comparisonoverlay import ComparisonOverlay, ComparisonHandle


def test_comparison_handle(qtbot):
    handle = ComparisonHandle()
    qtbot.addWidget(handle)

    handle.repaint()

    called = []
    handle.dragged.connect(lambda x: called.append(x))

    qtbot.mousePress(handle, Qt.LeftButton)
    assert handle._dragging

    import PySide6.QtCore as QtCore

    event = QMouseEvent(
        QtCore.QEvent.MouseMove,
        QPointF(5, 5),
        QPointF(100, 100),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    handle.mouseMoveEvent(event)
    assert len(called) > 0

    qtbot.mouseRelease(handle, Qt.LeftButton)
    assert not handle._dragging


def test_comparison_overlay(qtbot):
    overlay = ComparisonOverlay()
    qtbot.addWidget(overlay)

    overlay.setSplitPosition(0.7)
    assert overlay._split_position == 0.7

    overlay.setComparisonActive(True)
    assert overlay._comparison_active

    from PySide6.QtGui import QPixmap

    pm1 = QPixmap(100, 100)
    pm2 = QPixmap(100, 100)

    overlay.setUneditedPixmap(pm1)
    assert overlay._unedited_pixmap is not None

    overlay.setEditedPixmap(pm2)
    assert overlay._edited_pixmap is not None

    overlay.updateEditedPixmap(pm2)

    overlay.repaint()

    # Boundary tests
    overlay.setSplitPosition(1.5)
    assert overlay._split_position == 1.0

    overlay.setSplitPosition(-0.5)
    assert overlay._split_position == 0.0

    class DummyScrollBar:
        def __init__(self):
            class DummySignal:
                def connect(self, x):
                    pass

            self.valueChanged = DummySignal()

    class DummyView:
        def __init__(self):
            class DummySignal:
                def connect(self, x):
                    pass

            self.zoomChanged = DummySignal()
            self._bg_item = None

        def horizontalScrollBar(self):
            return DummyScrollBar()

        def verticalScrollBar(self):
            return DummyScrollBar()

    overlay.setView(DummyView())
