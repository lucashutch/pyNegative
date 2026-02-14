import pytest
from unittest.mock import patch
from PySide6 import QtWidgets, QtCore
from pynegative.ui.editingcontrols import EditingControls


# Mock control classes that inherit from QWidget
class MockControl(QtWidgets.QWidget):
    settingChanged = QtCore.Signal(str, object)
    autoWbRequested = QtCore.Signal()
    presetApplied = QtCore.Signal(str)
    cropToggled = QtCore.Signal(bool)
    aspectRatioChanged = QtCore.Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliders = {}
        self.labels = {}
        self.btn_flip_h = QtWidgets.QPushButton()
        self.btn_flip_v = QtWidgets.QPushButton()
        self.crop_btn = QtWidgets.QPushButton()
        self.crop_btn.setCheckable(True)
        self.val_flip_h = False
        self.val_flip_v = False
        self.val_sharpen_radius = 0.5
        self.val_sharpen_percent = 50.0

    def get_value(self, name):
        return 0.0

    def set_slider_value(self, name, value, silent=False):
        pass

    def reset_section(self):
        pass


class MockLensControls(MockControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(["Auto", "Canon EOS R50"])
        self.lens_combo = QtWidgets.QComboBox()
        self.lens_combo.addItems(["Auto", "Canon RF 50mm"])


@pytest.fixture
def mock_controls(qtbot):
    with (
        patch("pynegative.ui.editingcontrols.ToneControls", side_effect=MockControl),
        patch("pynegative.ui.editingcontrols.ColorControls", side_effect=MockControl),
        patch("pynegative.ui.editingcontrols.DetailControls", side_effect=MockControl),
        patch(
            "pynegative.ui.editingcontrols.GeometryControls", side_effect=MockControl
        ),
        patch(
            "pynegative.ui.editingcontrols.LensControls", side_effect=MockLensControls
        ),
    ):
        yield


def test_apply_lens_overrides(qtbot, mock_controls):
    controls = EditingControls()
    qtbot.addWidget(controls)

    # Access the lens controls instance
    lens_ctrl = controls.lens_controls

    # Test 1: Apply override
    settings = {
        "lens_camera_override": "Canon EOS R50",
        "lens_name_override": "Canon RF 50mm",
    }

    controls.apply_settings(settings)

    assert lens_ctrl.camera_combo.currentText() == "Canon EOS R50"
    assert lens_ctrl.lens_combo.currentText() == "Canon RF 50mm"

    # Test 2: Apply Auto (None or "Auto")
    settings_auto = {"lens_camera_override": "Auto", "lens_name_override": None}
    controls.apply_settings(settings_auto)

    assert lens_ctrl.camera_combo.currentIndex() == 0  # Auto index
    assert lens_ctrl.lens_combo.currentIndex() == 0  # Auto index
