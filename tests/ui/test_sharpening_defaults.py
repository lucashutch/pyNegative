from pynegative.ui.editingcontrols import EditingControls


def test_sharpening_default_value(qtbot):
    """Test that the sharpening slider defaults to 0."""
    controls = EditingControls()
    qtbot.addWidget(controls)

    # Check initial value
    assert controls.val_sharpen_value == 0.0

    # Check slider value (slider is scaled by 1000)
    assert controls.val_sharpen_value_slider.value() == 0


def test_sharpening_reset_value(qtbot):
    """Test that resetting the DETAILS section sets sharpening to 0, not 20."""
    controls = EditingControls()
    qtbot.addWidget(controls)

    # Manually set to something else
    controls.set_slider_value("val_sharpen_value", 30.0)
    assert controls.val_sharpen_value == 30.0

    # Reset the section
    # Trigger the reset signal which calls _reset_section("details")
    controls.details_section.resetClicked.emit()

    # This is where we expect it to be 0.0
    assert controls.val_sharpen_value == 0.0
