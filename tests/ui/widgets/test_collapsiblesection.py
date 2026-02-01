import pytest
from PySide6 import QtWidgets
from pynegative.ui.widgets.collapsiblesection import CollapsibleSection


@pytest.fixture
def section(qtbot):
    """Provides a CollapsibleSection instance."""
    widget = CollapsibleSection("Test Section")
    widget.resize(300, 200)
    widget.show()
    qtbot.addWidget(widget)
    return widget


def test_initialization(section):
    """Test that the section initializes with correct properties."""
    assert section.header.text() == "Test Section"


def test_initial_expanded_state(section):
    """Test that section is expanded by default."""
    assert section.header.isChecked() is True
    assert section.content.isVisible()


def test_toggle_expands(section):
    """Test that toggle method hides content when header is unchecked."""
    section.header.setChecked(False)
    section.toggle()
    assert section.header.isChecked() is False
    assert not section.content.isVisible()


def test_toggle_collapses(section):
    """Test that toggle method shows content when header is checked."""
    section.header.setChecked(False)
    section.header.setChecked(True)
    section.toggle()
    assert section.header.isChecked() is True
    assert section.content.isVisible()


def test_add_widget(section):
    """Test adding widgets to the section."""
    test_widget = QtWidgets.QPushButton("Test Button")
    section.add_widget(test_widget)

    assert section.content_layout.count() > 0


def test_content_widget_initially_visible(section):
    """Test content widget is visible when expanded."""
    assert section.content.isVisible()


def test_collapsed_initialization(qtbot):
    """Test section initialized as collapsed."""
    widget = CollapsibleSection("Collapsed Section", expanded=False)
    widget.show()
    qtbot.addWidget(widget)

    assert widget.header.isChecked() is False
    assert not widget.content.isVisible()


def test_toggle_via_click(qtbot):
    """Test that clicking the header toggles the section."""
    widget = CollapsibleSection("Toggle Test")
    widget.show()
    qtbot.addWidget(widget)

    widget.header.click()
    assert widget.header.isChecked() is False


def test_content_layout_exists(section):
    """Test that content layout exists."""
    assert section.content_layout is not None


def test_header_exists(section):
    """Test that header exists."""
    assert section.header is not None


def test_checkbox_is_checked(section):
    """Test that checkbox (header) is checked by default."""
    assert section.header.isChecked()


def test_multiple_toggles(section):
    """Test multiple toggle operations."""
    for i in range(5):
        section.header.setChecked(i % 2 == 0)
        section.toggle()
        expected_visible = i % 2 == 0
        assert section.content.isVisible() == expected_visible


def test_add_multiple_widgets(section):
    """Test adding multiple widgets."""
    for i in range(3):
        btn = QtWidgets.QPushButton(f"Button {i}")
        section.add_widget(btn)

    assert section.content_layout.count() == 3


def test_section_title(section):
    """Test that title is correctly set."""
    assert "Test Section" in section.header.text()
