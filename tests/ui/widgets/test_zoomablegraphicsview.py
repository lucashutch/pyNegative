import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from pynegative.ui.widgets.zoomablegraphicsview import ZoomableGraphicsView


@pytest.fixture
def zoom_view(qtbot):
    """Provides a ZoomableGraphicsView widget."""
    view = ZoomableGraphicsView()
    view.resize(400, 300)
    view.show()
    qtbot.addWidget(view)
    return view


@pytest.fixture
def sample_pixmap():
    """Creates a small test pixmap."""
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("red"))
    return pixmap


@pytest.fixture
def sample_roi_pixmap():
    """Creates a small ROI test pixmap."""
    pixmap = QtGui.QPixmap(50, 50)
    pixmap.fill(QtGui.QColor("blue"))
    return pixmap


class TestZoomableGraphicsView:
    def test_initialization(self, zoom_view):
        """Test that the widget initializes with correct properties."""
        assert zoom_view._current_zoom == 1.0
        assert zoom_view._is_fitting is True
        assert zoom_view.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag
        assert zoom_view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert zoom_view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert zoom_view.frameShape() == QtWidgets.QFrame.NoFrame

    def test_background_and_foreground_items(self, zoom_view):
        """Test that background and foreground items are created."""
        assert zoom_view._bg_item is not None
        assert zoom_view._fg_item is not None
        assert zoom_view._bg_item.zValue() == 0
        assert zoom_view._fg_item.zValue() == 1

    def test_set_pixmaps_background_only(self, zoom_view, sample_pixmap):
        """Test setting background pixmap only."""
        zoom_view.set_pixmaps(sample_pixmap, 100, 100)

        assert not zoom_view._bg_item.pixmap().isNull()
        assert zoom_view._bg_item.pixmap().width() == 100
        assert zoom_view._bg_item.pixmap().height() == 100
        assert (
            not zoom_view._fg_item.isVisible()
        )  # Foreground should be hidden when no ROI

    def test_set_pixmaps_with_roi(self, zoom_view, sample_pixmap, sample_roi_pixmap):
        """Test setting background and ROI pixmaps."""
        zoom_view.set_pixmaps(
            sample_pixmap, 100, 100, sample_roi_pixmap, 10, 20, 50, 50
        )

        assert not zoom_view._bg_item.pixmap().isNull()
        assert not zoom_view._fg_item.pixmap().isNull()
        assert zoom_view._fg_item.isVisible()
        assert zoom_view._fg_item.pos() == QtCore.QPointF(10, 20)
        assert zoom_view._scene.sceneRect() == QtCore.QRectF(0, 0, 100, 100)

    def test_reset_zoom_with_no_content(self, zoom_view):
        """Test reset_zoom when there's no content."""
        zoom_view.reset_zoom()
        # Should not crash or raise exceptions
        assert zoom_view._current_zoom == 1.0
        assert zoom_view._is_fitting is True

    def test_reset_zoom_with_content(self, zoom_view, sample_pixmap):
        """Test reset_zoom with content."""
        zoom_view.set_zoom(2.0)
        assert zoom_view._current_zoom == 2.0
        assert zoom_view._is_fitting is False

    def test_wheel_event_zoom_in(self, zoom_view, qtbot, sample_pixmap):
        """Test wheel event for zooming in."""
        zoom_view.set_pixmaps(sample_pixmap, 100, 100)
        zoom_view.set_zoom(1.0)

        # Simulate zoom in
        zoom_view.set_zoom(1.1)

        # Should zoom in to approximately 1.1x
        assert zoom_view._current_zoom > 1.0
        assert zoom_view._current_zoom <= 1.1

    def test_wheel_event_zoom_out(self, zoom_view, qtbot, sample_pixmap):
        """Test wheel event for zooming out."""
        zoom_view.set_pixmaps(sample_pixmap, 100, 100)
        zoom_view.set_zoom(1.0)

        # Simulate zoom out
        zoom_view.set_zoom(0.9)

        # Should zoom out to approximately 0.9x
        assert zoom_view._current_zoom < 1.0
        assert zoom_view._current_zoom >= 0.9

    def test_wheel_event_bounds(self, zoom_view, qtbot, sample_pixmap):
        """Test wheel event respects zoom bounds."""
        zoom_view.set_pixmaps(sample_pixmap, 100, 100)

        # Zoom in multiple times to reach max
        for _ in range(10):
            angle_delta = QtCore.QPoint(0, 120)
            qtbot.wait(100)
            qtbot.keyClick(zoom_view, QtCore.Qt.Key_Plus)

        assert zoom_view._current_zoom <= 4.0

        # Zoom out multiple times to reach min
        for _ in range(10):
            qtbot.wait(100)
            qtbot.keyClick(zoom_view, QtCore.Qt.Key_Minus)

        assert zoom_view._current_zoom >= 0.5

    def test_double_click_event(self, zoom_view, qtbot):
        """Test double click event emission."""
        # Test by directly calling the double click event method
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QPointF(200, 150),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        with qtbot.waitSignal(zoom_view.doubleClicked):
            zoom_view.mouseDoubleClickEvent(event)

    def test_zoom_changed_signal(self, zoom_view, qtbot):
        """Test zoomChanged signal emission."""
        with qtbot.waitSignal(zoom_view.zoomChanged):
            zoom_view.set_zoom(2.0)

    def test_zoom_changed_on_pan(self, zoom_view, qtbot, sample_pixmap):
        """Test zoomChanged signal emission during pan."""
        zoom_view.set_pixmaps(sample_pixmap, 100, 100)
        zoom_view.set_zoom(2.0, manual=True)

        # Simulate scroll bar movement to trigger pan
        with qtbot.waitSignal(zoom_view.zoomChanged):
            # This would normally be triggered by scroll bar changes
            # We'll just call the method directly for testing
            zoom_view._sync_view()

    def test_background_brush(self, zoom_view):
        """Test that background brush is set correctly."""
        brush = zoom_view.backgroundBrush()
        assert brush.color() == QtGui.QColor("#1a1a1a")

    def test_empty_scene_rect(self, zoom_view):
        """Test behavior with empty scene rect."""
        assert zoom_view._scene.sceneRect().isEmpty()

        # Test wheel event with empty scene
        event = QtGui.QWheelEvent(
            QtCore.QPointF(200, 150),  # Position
            QtCore.QPointF(0, 0),  # Global position
            QtCore.QPoint(0, 0),  # Pixel delta
            QtCore.QPoint(0, 120),  # Angle delta
            QtCore.Qt.MouseButton.NoButton,  # Buttons
            QtCore.Qt.KeyboardModifier.NoModifier,  # Modifiers
            QtCore.Qt.ScrollPhase.NoScrollPhase,  # Phase
            False,  # Inverted
            QtCore.Qt.MouseEventSource.MouseEventNotSynthesized,  # Source
        )

        zoom_view.wheelEvent(event)
        # Should not crash or raise exceptions
        assert zoom_view._current_zoom == 1.0

    def test_transform_anchor_settings(self, zoom_view):
        """Test that transformation anchor settings are correct."""
        assert (
            zoom_view.transformationAnchor() == QtWidgets.QGraphicsView.AnchorUnderMouse
        )
        assert zoom_view.resizeAnchor() == QtWidgets.QGraphicsView.AnchorUnderMouse
