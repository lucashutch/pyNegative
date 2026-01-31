import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from pynegative.ui.widgets.resetableslider import ResetableSlider


@pytest.fixture
def slider(qtbot):
    """Provides a ResetableSlider instance."""
    widget = ResetableSlider(QtCore.Qt.Horizontal)
    widget.setRange(0, 100)
    widget.setValue(50)
    widget.resize(200, 30)
    widget.show()
    qtbot.addWidget(widget)
    return widget


def test_initialization(slider):
    """Test that the slider initializes with correct properties."""
    assert slider.orientation() == QtCore.Qt.Horizontal
    assert slider.default_slider_value == 0


def test_set_value(slider):
    """Test setting a value."""
    slider.setValue(75)
    assert slider.value() == 75


def test_reset_mimic(slider):
    """Test that resetting sets value to default_slider_value (0)."""
    slider.setValue(75)
    slider.setValue(slider.default_slider_value)
    assert slider.value() == 0


def test_value_changed_signal(slider, qtbot):
    """Test that valueChanged signal is emitted on reset."""
    slider.setValue(50)

    with qtbot.waitSignal(slider.valueChanged) as blocker:
        slider.setValue(0)

    assert slider.value() == 0
    assert blocker.args == [0]


def test_no_signal_if_already_zero(qtbot):
    """Test that no signal is emitted if value is already zero."""
    slider = ResetableSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(0)
    slider.resize(200, 30)
    slider.show()
    qtbot.addWidget(slider)

    assert slider.value() == 0

    with qtbot.assertNotEmitted(slider.valueChanged):
        slider.setValue(0)


def test_vertical_orientation():
    """Test vertical orientation slider."""
    slider = ResetableSlider(QtCore.Qt.Vertical)
    assert slider.orientation() == QtCore.Qt.Vertical
    slider.deleteLater()


def test_range_values():
    """Test setting values within a typical range."""
    slider = ResetableSlider(QtCore.Qt.Horizontal)
    slider.setRange(-100, 100)
    slider.setValue(50)
    assert slider.value() == 50
    slider.deleteLater()


def test_reset_from_different_values():
    """Test reset from various values."""
    slider = ResetableSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 100)
    slider.resize(200, 30)
    slider.show()

    test_values = [10, 25, 50, 75, 100]

    for val in test_values:
        slider.setValue(val)
        assert slider.value() == val
        slider.setValue(0)
        assert slider.value() == 0


def test_default_slider_value_constant():
    """Test that default_slider_value is 0."""
    slider = ResetableSlider(QtCore.Qt.Horizontal)
    assert slider.default_slider_value == 0
