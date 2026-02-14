import pytest
from PySide6 import QtWidgets
from pynegative.ui.widgets.metadata_panel import MetadataPanel


@pytest.fixture
def panel(qtbot):
    widget = MetadataPanel()
    qtbot.addWidget(widget)
    return widget


def get_form_layout_rows(layout):
    """Helper to extract key-value pairs from QFormLayout."""
    rows = {}
    for i in range(layout.rowCount()):
        item = layout.itemAt(i, QtWidgets.QFormLayout.LabelRole)
        field = layout.itemAt(i, QtWidgets.QFormLayout.FieldRole)
        if item and field:
            label = item.widget().text()
            # Remove HTML bold tags if present
            label = label.replace("<b>", "").replace("</b>", "").replace(":", "")
            value = field.widget().text()
            rows[label] = value
    return rows


def test_lens_info_display_full(panel):
    """Test display when all lens info is present."""
    data = {
        "camera_make": "Canon",
        "camera_model": "EOS R5",
        "lens_model": "RF 50mm F1.2",
        "lens_source": "LENSFUN_DB",
        "lens_model_resolved": "Canon RF 50mm F1.2L USM",
    }

    panel._populate(data)
    rows = get_form_layout_rows(panel.content_layout)

    assert rows["Camera"] == "Canon EOS R5"
    assert rows["Lens"] == "RF 50mm F1.2"
    assert "Canon RF 50mm F1.2L USM" in rows["Lens Profile"]
    assert "Lensfun Match" in rows["Lens Profile"]


def test_lens_info_missing_model_but_camera_present(panel):
    """Test fallback when lens model is missing but camera is known."""
    data = {
        "camera_make": "Canon",
        "camera_model": "EOS R5",
        "lens_model": "",
        "lens_source": "NONE",
        "lens_model_resolved": None,
    }

    panel._populate(data)
    rows = get_form_layout_rows(panel.content_layout)

    assert rows["Camera"] == "Canon EOS R5"
    assert rows["Lens"] == "Unknown / Not detected"
    assert "No Profile Found (Manual Mode)" in rows["Lens Profile"]


def test_lens_profile_none(panel):
    """Test display when lens profile search failed (Manual Mode)."""
    data = {
        "camera_make": "Canon",
        "camera_model": "EOS R5",
        "lens_model": "Unknown Lens",
        "lens_source": "NONE",
        "lens_model_resolved": None,
    }

    panel._populate(data)
    rows = get_form_layout_rows(panel.content_layout)

    assert rows["Lens"] == "Unknown Lens"
    assert "No Profile Found (Manual Mode)" in rows["Lens Profile"]
