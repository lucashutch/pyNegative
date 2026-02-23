from pynegative.ui.widgets.metadata_panel import MetadataPanel


def test_metadata_panel_extra_coverage(qtbot):
    panel = MetadataPanel()
    qtbot.addWidget(panel)

    # Test property
    assert panel.content_layout is not None

    # Test show empty
    panel.show_empty()
    assert panel._current_path is None

    # Test clear
    panel.clear()

    # Test internal populate formatting logic
    data = {
        "file_name": "test.raw",
        "file_location": "/tmp",
        "iso": 100,
        "shutter_speed": "1/200",
        "aperture": "9/5",
        "focal_length": "50/1",
        "focal_length_35mm": 75,
        "camera_make": "Sony",
        "camera_model": "Sony A7",
        "lens_model": "Sony FE 50mm",
        "lens_source": "MANUAL_OVERRIDE",
        "lens_model_resolved": "FE 50mm FE 50mm",
        "max_aperture": "2.8",
        "date_taken": "2023:01:01 12:00:00",
        "exposure_compensation": "-1.0",
        "metering_mode": "Pattern",
        "exposure_program": "Manual",
        "exposure_mode": "Auto",
        "white_balance": "Auto",
        "color_space": "sRGB",
        "flash": "Off",
        "scene_capture_type": "Standard",
        "contrast": "Normal",
        "saturation": "High",
        "sharpness": "Hard",
        "digital_zoom": "1.0",
        "gps_latitude": "45.0N",
        "gps_longitude": "10.0E",
        "width": 6000,
        "height": 4000,
        "file_size": 1048576 * 10,
        "orientation": "Horizontal",
        "artist": "John Doe",
        "copyright": "2023",
        "image_description": "A test image",
        "software": "pyNegative",
    }
    panel._populate(data)

    panel.clear()

    class FakeRatio:
        def __init__(self, n, d):
            self.num = n
            self.den = d

    data2 = {
        "shutter_speed": FakeRatio(1, 400),
        "aperture": FakeRatio(14, 5),
        "focal_length": FakeRatio(50, 1),
        "max_aperture": FakeRatio(8, 2),
        "file_size": None,
        "date_taken": None,
        "lens_source": "NONE",
        "lens_model": "Sig Sigma",
    }
    panel._populate(data2)

    data3 = {
        "shutter_speed": FakeRatio(30, 1),
        "aperture": FakeRatio(1, 0),
        "focal_length": FakeRatio(1, 0),
    }
    panel._populate(data3)

    data4 = {
        "shutter_speed": "0",
        "aperture": "0",
        "focal_length": "0",
        "max_aperture": FakeRatio(1, 0),
    }
    panel._populate(data4)
