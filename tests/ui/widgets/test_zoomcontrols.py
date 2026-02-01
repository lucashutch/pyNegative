import pytest
from pynegative.ui.widgets.zoomcontrols import ZoomControls


@pytest.fixture
def zoom_controls(qtbot):
    """Provides a ZoomControls instance."""
    widget = ZoomControls()
    widget.resize(250, 40)
    widget.show()
    qtbot.addWidget(widget)
    return widget


def test_initialization(zoom_controls):
    """Test that the controls initialize with default values."""
    assert zoom_controls.slider.value() == 100
    assert zoom_controls.spin.value() == 100


def test_zoom_changed_signal(zoom_controls, qtbot):
    """Test that zoomChanged signal is emitted on valid change."""
    with qtbot.waitSignal(zoom_controls.zoomChanged) as blocker:
        zoom_controls.slider.setValue(150)

    assert len(blocker.args) == 1
    assert abs(blocker.args[0] - 1.5) < 0.01


def test_spinbox_to_slider_sync(zoom_controls):
    """Test that changing spinbox updates slider."""
    zoom_controls.spin.setValue(200)
    assert zoom_controls.slider.value() == 200


def test_slider_to_spinbox_sync(zoom_controls):
    """Test that changing slider updates spinbox."""
    zoom_controls.slider.setValue(250)
    assert zoom_controls.spin.value() == 250


def test_update_zoom_no_signal(zoom_controls, qtbot):
    """Test that update_zoom doesn't emit signal."""
    with qtbot.assertNotEmitted(zoom_controls.zoomChanged):
        zoom_controls.update_zoom(2.0)

    assert zoom_controls.spin.value() == 200
    assert zoom_controls.slider.value() == 200


def test_zoom_bounds_min(zoom_controls):
    """Test that zoom is clamped to minimum."""
    zoom_controls.slider.setValue(10)
    # It should be clamped to 10
    assert zoom_controls.slider.value() >= 10


def test_zoom_bounds_max(zoom_controls):
    """Test that zoom is clamped to maximum."""
    zoom_controls.slider.setValue(500)
    # It should be clamped to 400
    assert zoom_controls.slider.value() <= 400


def test_normalized_value_calculation(zoom_controls, qtbot):
    """Test normalized value calculation."""
    with qtbot.waitSignal(zoom_controls.zoomChanged) as blocker:
        zoom_controls.slider.setValue(200)

    expected = 2.0
    assert abs(blocker.args[0] - expected) < 0.01


def test_display_value_percentage(zoom_controls):
    """Test that spinbox displays percentage."""
    zoom_controls.spin.setValue(150)
    assert zoom_controls.spin.value() == 150
    assert zoom_controls.spin.suffix() == "%"


def test_update_zoom_external(zoom_controls):
    """Test update_zoom with various values."""
    test_values = [0.5, 1.0, 2.0, 4.0]

    for val in test_values:
        zoom_controls.update_zoom(val)
        display_value = int(val * 100)
        assert zoom_controls.spin.value() == display_value


def test_slider_range(zoom_controls):
    """Test that slider has correct range."""
    assert zoom_controls.slider.minimum() == 10
    assert zoom_controls.slider.maximum() == 400


def test_spinbox_range(zoom_controls):
    """Test that spinbox has correct range."""
    assert zoom_controls.spin.minimum() == 10
    assert zoom_controls.spin.maximum() == 400


def test_signal_emits_normalized_value(zoom_controls, qtbot):
    """Test that both slider and spin emit normalized values."""
    # Test slider - set to 200 to ensure change from default
    with qtbot.waitSignal(zoom_controls.zoomChanged) as blocker:
        zoom_controls.slider.setValue(200)
    assert abs(blocker.args[0] - 2.0) < 0.01

    # Reset to 100
    zoom_controls.slider.setValue(100)

    # Test spinbox
    with qtbot.waitSignal(zoom_controls.zoomChanged) as blocker:
        zoom_controls.spin.setValue(300)
    assert abs(blocker.args[0] - 3.0) < 0.01


def test_update_zoom_clamps_values(zoom_controls):
    """Test that update_zoom handles out-of-range values."""
    zoom_controls.update_zoom(0.1)  # Below minimum
    assert zoom_controls.spin.value() == 10

    zoom_controls.update_zoom(10.0)  # Above maximum
    assert zoom_controls.spin.value() == 400
