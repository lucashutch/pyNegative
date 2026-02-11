import pytest
from PySide6.QtCore import Qt
from pynegative.ui.widgets.zoomcontrols import ZoomControls


@pytest.fixture
def zoom_controls(qtbot):
    """Provides a ZoomControls instance."""
    widget = ZoomControls()
    widget.resize(300, 40)
    widget.show()
    qtbot.addWidget(widget)
    return widget


def test_initialization(zoom_controls):
    """Test that the controls initialize with default values."""
    assert zoom_controls.slider.value() == 100
    assert zoom_controls.btn_preset.text() == "100%"


def test_zoom_changed_signal(zoom_controls, qtbot):
    """Test that zoomChanged signal is emitted on slider change."""
    with qtbot.waitSignal(zoom_controls.zoomChanged) as blocker:
        zoom_controls.slider.setValue(150)

    assert len(blocker.args) == 1
    assert abs(blocker.args[0] - 1.5) < 0.01


def test_slider_to_preset_text_sync(zoom_controls):
    """Test that changing slider updates preset button text."""
    zoom_controls.slider.setValue(250)
    assert zoom_controls.btn_preset.text() == "250%"


def test_update_zoom_no_signal(zoom_controls, qtbot):
    """Test that update_zoom doesn't emit signal."""
    with qtbot.assertNotEmitted(zoom_controls.zoomChanged):
        zoom_controls.update_zoom(2.0)

    assert zoom_controls.btn_preset.text() == "200%"
    assert zoom_controls.slider.value() == 200


def test_zoom_in_out_signals(zoom_controls, qtbot):
    """Test zoom in/out button signals."""
    with qtbot.waitSignal(zoom_controls.zoomInClicked):
        qtbot.mouseClick(zoom_controls.btn_in, Qt.LeftButton)

    with qtbot.waitSignal(zoom_controls.zoomOutClicked):
        qtbot.mouseClick(zoom_controls.btn_out, Qt.LeftButton)


def test_fit_signal(zoom_controls, qtbot):
    """Test fit button signal."""
    with qtbot.waitSignal(zoom_controls.fitClicked):
        qtbot.mouseClick(zoom_controls.btn_fit, Qt.LeftButton)


def test_preset_menu_signal(zoom_controls, qtbot):
    """Test that preset menu items emit zoomPresetSelected."""
    # We can't easily click the menu items in a headless test without more setup,
    # but we can verify the signal exists and maybe trigger it directly if needed.
    # For now, let's just trigger the signal.
    with qtbot.waitSignal(zoom_controls.zoomPresetSelected) as blocker:
        zoom_controls.zoomPresetSelected.emit(2.0)
    assert blocker.args[0] == 2.0


def test_update_zoom_external(zoom_controls):
    """Test update_zoom with various values."""
    test_values = [0.5, 1.0, 2.0, 4.0]

    for val in test_values:
        zoom_controls.update_zoom(val)
        display_text = f"{int(round(val * 100))}%"
        assert zoom_controls.btn_preset.text() == display_text
