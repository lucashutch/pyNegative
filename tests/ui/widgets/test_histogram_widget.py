import numpy as np
from pynegative.ui.widgets.histogram import HistogramWidget


def test_histogram_widget_init(qtbot):
    widget = HistogramWidget()
    qtbot.addWidget(widget)
    assert widget.mode == "Auto"
    assert widget.data is None


def test_histogram_widget_set_data(qtbot):
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    data = {
        "R": np.random.rand(256),
        "G": np.random.rand(256),
        "B": np.random.rand(256),
        "Y": np.random.rand(256),
        "U": np.random.rand(256),
        "V": np.random.rand(256),
    }

    widget.set_data(data)
    # Check that data is stored (numpy arrays might need special comparison but here we just check if it's the same dict)
    assert widget.data == data


def test_histogram_widget_grayscale_detection(qtbot):
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    # Grayscale data
    hist = np.random.rand(256).astype(np.float32)
    data = {"R": hist, "G": hist, "B": hist}

    widget.set_data(data)
    assert widget._is_grayscale

    # Non-grayscale data
    data["R"] = np.random.rand(256).astype(np.float32)
    widget.set_data(data)
    # Since it's random, it's very unlikely to be identical
    assert not widget._is_grayscale


def test_histogram_widget_mode_change(qtbot):
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    widget.set_mode("RGB")
    assert widget.mode == "RGB"

    widget.set_mode("YUV")
    assert widget.mode == "YUV"
