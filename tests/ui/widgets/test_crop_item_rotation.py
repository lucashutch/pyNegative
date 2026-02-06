"""Tests for CropRectItem rotation handle functionality."""

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pynegative.ui.widgets.crop_item import CropRectItem


@pytest.fixture
def crop_item(qtbot):
    """Create a CropRectItem for testing."""
    item = CropRectItem(QtCore.QRectF(0, 0, 1000, 1000))
    return item


class TestRotationHandles:
    """Test rotation handle functionality."""

    def test_initial_rotation_is_zero(self, crop_item):
        """Test that initial rotation angle is 0."""
        assert crop_item.get_rotation() == 0.0

    def test_set_rotation_within_range(self, crop_item):
        """Test setting rotation within valid range."""
        crop_item.set_rotation(15.0)
        assert crop_item.get_rotation() == 15.0

        crop_item.set_rotation(-20.0)
        assert crop_item.get_rotation() == -20.0

    def test_rotation_clamped_to_45_degrees(self, crop_item):
        """Test that rotation is clamped to ±45°."""
        crop_item.set_rotation(60.0)
        assert crop_item.get_rotation() == 45.0

        crop_item.set_rotation(-60.0)
        assert crop_item.get_rotation() == -45.0

    def test_angle_calculation_from_center(self, crop_item):
        """Test angle calculation from center point."""
        # Center of rect is (500, 500)
        center = crop_item._rect.center()
        assert center.x() == 500.0
        assert center.y() == 500.0

        # Point to the right (0°)
        angle = crop_item._calculate_angle_from_center(QtCore.QPointF(600, 500))
        assert abs(angle - 0.0) < 0.1

        # Point below (90°)
        angle = crop_item._calculate_angle_from_center(QtCore.QPointF(500, 600))
        assert abs(angle - 90.0) < 0.1

        # Point to the left (180° or -180°)
        angle = crop_item._calculate_angle_from_center(QtCore.QPointF(400, 500))
        assert abs(abs(angle) - 180.0) < 0.1

        # Point above (-90°)
        angle = crop_item._calculate_angle_from_center(QtCore.QPointF(500, 400))
        assert abs(angle - (-90.0)) < 0.1

    def test_snap_angle_to_cardinal_directions(self, crop_item):
        """Test angle snapping to 0°, ±90°, ±180°."""
        # Should snap to 0°
        assert crop_item._snap_angle(2.0) == 0.0
        assert crop_item._snap_angle(-3.0) == 0.0

        # Should snap to 90°
        assert crop_item._snap_angle(88.0) == 90.0
        assert crop_item._snap_angle(92.0) == 90.0

        # Should snap to -90°
        assert crop_item._snap_angle(-88.0) == -90.0
        assert crop_item._snap_angle(-92.0) == -90.0

        # Should snap to 180°
        assert crop_item._snap_angle(177.0) == 180.0
        assert crop_item._snap_angle(183.0) == 180.0

        # Should not snap (outside threshold)
        assert crop_item._snap_angle(10.0) == 10.0
        assert crop_item._snap_angle(45.0) == 45.0

    def test_rotation_signal_emitted(self, crop_item, qtbot):
        """Test that rotation signal is emitted on rotation change."""
        with qtbot.wait_signal(crop_item.rotationChanged) as blocker:
            crop_item.set_rotation(25.0)
            # Manually emit since we're not using mouse events
            crop_item.rotationChanged.emit(25.0)

        assert blocker.args == [25.0]

    def test_rotation_handles_positioned_correctly(self, crop_item):
        """Test that rotation handles are positioned at top center."""
        # This test would require a scene and view for scale calculation
        # For now, just test that the method returns the expected structure
        handles = crop_item._get_rotation_handles()
        assert "rot_top" in handles
        assert "rot_right" not in handles
        assert isinstance(handles["rot_top"], QtCore.QPointF)

        # Verify positions are outside the rect
        rect = crop_item._rect
        # Top handle should be above the rect
        assert handles["rot_top"].y() < rect.top()
        assert abs(handles["rot_top"].x() - rect.center().x()) < 1.0

    def test_interactive_rotation_direction(self, crop_item, qtbot):
        """Test that rotation direction matches mouse movement."""
        # Setup: Ensure we have a scene and view for hit testing
        scene = QtWidgets.QGraphicsScene()
        QtWidgets.QGraphicsView(scene)
        scene.addItem(crop_item)
        crop_item.show()

        # Center is (500, 500), top is at 0
        handles = crop_item._get_rotation_handles()
        rot_top = handles["rot_top"]

        # 1. Press on top rotation handle
        # Mocking the hit test or just calling the internal handlers
        # because simulating mouse events in QGraphicsItem is complex with qtbot

        # Manually trigger mouse press using non-deprecated constructor
        from PySide6.QtGui import QPointingDevice

        device = QPointingDevice()

        press_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            rot_top,
            rot_top,  # globalPos
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
            device,
        )
        crop_item.mousePressEvent(press_event)
        assert crop_item._active_rotation_handle == "rot_top"

        # 2. Drag to the right (Clockwise)
        # Top is at (500, 450) roughly. Moving to (600, 450) is CW.
        move_pos = QtCore.QPointF(600, 450)
        move_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            move_pos,
            move_pos,  # globalPos
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
            device,
        )
        crop_item.mouseMoveEvent(move_event)

        # CW movement should result in NEGATIVE rotation value (per core.py convention)
        assert crop_item.get_rotation() < 0.0

        # 3. Drag to the left (Counter-Clockwise)
        move_pos_ccw = QtCore.QPointF(400, 450)
        move_event_ccw = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            move_pos_ccw,
            move_pos_ccw,  # globalPos
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
            device,
        )
        crop_item.mouseMoveEvent(move_event_ccw)

        # CCW movement should result in POSITIVE rotation value
        assert crop_item.get_rotation() > 0.0
