from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QMouseEvent, QPixmap
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


def test_calculate_scaled_rect_preserves_aspect_ratio():
    """Test that _calculate_scaled_rect preserves aspect ratio when scaling."""
    overlay = ComparisonOverlay()

    # Test landscape image in square target
    landscape_pixmap = QPixmap(200, 100)
    square_target = QRectF(0, 0, 100, 100)
    dest_rect, source_rect = overlay._calculate_scaled_rect(
        landscape_pixmap, square_target
    )

    # Should scale to fit width, maintaining aspect ratio
    assert dest_rect.width() == 100
    assert dest_rect.height() == 50  # Half width = half height to preserve ratio
    assert dest_rect.x() == 0  # Centered horizontally
    assert dest_rect.y() == 25  # Centered vertically (100 - 50) / 2
    assert source_rect.width() == 200
    assert source_rect.height() == 100


def test_calculate_scaled_rect_portrait_in_landscape():
    """Test portrait image scaled to fit in landscape target."""
    overlay = ComparisonOverlay()

    # Test portrait image in landscape target
    portrait_pixmap = QPixmap(100, 200)
    landscape_target = QRectF(0, 0, 200, 100)
    dest_rect, source_rect = overlay._calculate_scaled_rect(
        portrait_pixmap, landscape_target
    )

    # Should scale to fit height, maintaining aspect ratio
    assert dest_rect.height() == 100
    assert dest_rect.width() == 50  # Half height = half width to preserve ratio
    assert dest_rect.x() == 75  # Centered horizontally (200 - 50) / 2
    assert dest_rect.y() == 0  # Centered vertically
    assert source_rect.width() == 100
    assert source_rect.height() == 200


def test_calculate_scaled_rect_same_aspect_ratio():
    """Test scaling when image and target have same aspect ratio."""
    overlay = ComparisonOverlay()

    pixmap = QPixmap(200, 100)
    target = QRectF(0, 0, 400, 200)
    dest_rect, source_rect = overlay._calculate_scaled_rect(pixmap, target)

    # Should fill the entire target
    assert dest_rect.width() == 400
    assert dest_rect.height() == 200
    assert dest_rect.x() == 0
    assert dest_rect.y() == 0


def test_calculate_scaled_rect_null_pixmap():
    """Test handling of null pixmap."""
    overlay = ComparisonOverlay()

    null_pixmap = QPixmap()
    target = QRectF(0, 0, 100, 100)
    dest_rect, source_rect = overlay._calculate_scaled_rect(null_pixmap, target)

    assert dest_rect.isEmpty()
    assert source_rect.isEmpty()


def test_comparison_with_different_orientations(qtbot):
    """Test comparison with portrait vs landscape images doesn't stretch."""
    overlay = ComparisonOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(400, 400)

    # Create a portrait image (600x900) and landscape image (900x600)
    portrait_pixmap = QPixmap(600, 900)
    landscape_pixmap = QPixmap(900, 600)

    overlay.setUneditedPixmap(landscape_pixmap)
    overlay.setEditedPixmap(portrait_pixmap)
    overlay.setComparisonActive(True)

    # Set up a mock view with a square viewport
    class DummyBgItem:
        def sceneBoundingRect(self):
            from PySide6.QtCore import QRectF

            return QRectF(0, 0, 400, 400)

    class DummyScrollBar:
        class DummySignal:
            def connect(self, x):
                pass

        valueChanged = DummySignal()

    class DummyView:
        class DummySignal:
            def connect(self, x):
                pass

        zoomChanged = DummySignal()
        _bg_item = DummyBgItem()

        def mapFromScene(self, rect):
            from PySide6.QtGui import QPolygon

            return QPolygon([rect.topLeft().toPoint()])

        def horizontalScrollBar(self):
            return DummyScrollBar()

        def verticalScrollBar(self):
            return DummyScrollBar()

    overlay.setView(DummyView())

    # Should paint without errors
    overlay.repaint()

    # Both pixmaps should be set
    assert overlay._unedited_pixmap is not None
    assert overlay._edited_pixmap is not None
    assert overlay._unedited_pixmap.width() == 900
    assert overlay._unedited_pixmap.height() == 600
    assert overlay._edited_pixmap.width() == 600
    assert overlay._edited_pixmap.height() == 900


def test_comparison_landscape_vs_landscape_different_ratios(qtbot):
    """Test comparison between two landscape images with different aspect ratios."""
    overlay = ComparisonOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(800, 400)

    # Create two landscape images with different aspect ratios
    # 16:9 ratio
    wide_landscape = QPixmap(1600, 900)
    # 4:3 ratio
    standard_landscape = QPixmap(1200, 900)

    overlay.setUneditedPixmap(wide_landscape)
    overlay.setEditedPixmap(standard_landscape)
    overlay.setComparisonActive(True)
    overlay.setSplitPosition(0.5)

    # Set up a mock view
    class DummyBgItem:
        def sceneBoundingRect(self):
            from PySide6.QtCore import QRectF

            return QRectF(0, 0, 800, 400)

    class DummyScrollBar:
        class DummySignal:
            def connect(self, x):
                pass

        valueChanged = DummySignal()

    class DummyView:
        class DummySignal:
            def connect(self, x):
                pass

        zoomChanged = DummySignal()
        _bg_item = DummyBgItem()

        def mapFromScene(self, rect):
            from PySide6.QtGui import QPolygon

            return QPolygon([rect.topLeft().toPoint()])

        def horizontalScrollBar(self):
            return DummyScrollBar()

        def verticalScrollBar(self):
            return DummyScrollBar()

    overlay.setView(DummyView())

    # Should paint without errors and maintain aspect ratios
    overlay.repaint()

    # Verify original dimensions are preserved (not stretched)
    assert overlay._unedited_pixmap.width() == 1600
    assert overlay._unedited_pixmap.height() == 900
    assert overlay._edited_pixmap.width() == 1200
    assert overlay._edited_pixmap.height() == 900
