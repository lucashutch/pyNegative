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


def test_set_mode_rebuilds_cached_paths(qtbot):
    """Changing mode must rebuild _cached_paths so paintEvent uses the new mode."""
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

    widget.set_mode("RGB")
    assert set(widget._cached_paths.keys()) == {"R", "G", "B"}

    widget.set_mode("YUV")
    assert set(widget._cached_paths.keys()) == {"Y", "U", "V"}

    widget.set_mode("Luminance")
    assert "Y" in widget._cached_paths


def test_waveform_mode_renders(qtbot):
    """Waveform (RGB) mode should prepare a _waveform_pixmap from per-channel data."""
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    wf = np.random.rand(256, 256).astype(np.float32)
    data = {
        "R": np.random.rand(256),
        "G": np.random.rand(256),
        "B": np.random.rand(256),
        "Y": np.random.rand(256),
        "U": np.random.rand(256),
        "V": np.random.rand(256),
        "waveform_R": wf,
        "waveform_G": wf,
        "waveform_B": wf,
        "waveform_Y": wf,
    }
    widget.set_mode("Waveform (RGB)")
    widget.set_data(data)

    assert widget._waveform_pixmap is not None
    assert not widget._waveform_pixmap.isNull()


def test_waveform_luma_mode_renders(qtbot):
    """Waveform (Luma) mode should prepare a white/grey waveform pixmap."""
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    wf = np.random.rand(256, 256).astype(np.float32)
    data = {
        "R": np.random.rand(256),
        "G": np.random.rand(256),
        "B": np.random.rand(256),
        "Y": np.random.rand(256),
        "U": np.random.rand(256),
        "V": np.random.rand(256),
        "waveform_R": wf,
        "waveform_G": wf,
        "waveform_B": wf,
        "waveform_Y": wf,
    }
    widget.set_data(data)
    widget.set_mode("Waveform (Luma)")

    assert widget._waveform_pixmap is not None
    assert not widget._waveform_pixmap.isNull()


def test_waveform_set_mode_after_data(qtbot):
    """set_mode('Waveform (RGB)') after set_data should also prepare the pixmap."""
    widget = HistogramWidget()
    qtbot.addWidget(widget)

    wf = np.random.rand(256, 256).astype(np.float32)
    data = {
        "R": np.random.rand(256),
        "G": np.random.rand(256),
        "B": np.random.rand(256),
        "Y": np.random.rand(256),
        "U": np.random.rand(256),
        "V": np.random.rand(256),
        "waveform_R": wf,
        "waveform_G": wf,
        "waveform_B": wf,
        "waveform_Y": wf,
    }
    widget.set_data(data)
    assert widget._waveform_pixmap is None  # mode is Auto, not Waveform

    widget.set_mode("Waveform (RGB)")
    assert widget._waveform_pixmap is not None
    assert not widget._waveform_pixmap.isNull()
