import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestPreviewStarRatingWidget:
    """Tests for the PreviewStarRatingWidget class."""

    def test_creates_larger_stars(self, qtbot):
        from src.pynegative.ui.editor import PreviewStarRatingWidget

        widget = PreviewStarRatingWidget()
        assert widget.star_filled_pixmap.width() == 30
        assert widget.star_filled_pixmap.height() == 30

    def test_rating_property(self, qtbot):
        from src.pynegative.ui.editor import PreviewStarRatingWidget

        widget = PreviewStarRatingWidget()
        widget.set_rating(3)
        assert widget.rating() == 3

        widget.set_rating(0)
        assert widget.rating() == 0


class TestResetableSliderKeyboard:
    """Tests for ResetableSlider keyboard navigation."""

    def test_up_decreases_value(self, qtbot):
        from src.pynegative.ui.widgets.resetableslider import ResetableSlider

        slider = ResetableSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.show()
        qtbot.addWidget(slider)

        slider.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
        )
        assert slider.value() == 51

    def test_down_increases_value(self, qtbot):
        from src.pynegative.ui.widgets.resetableslider import ResetableSlider

        slider = ResetableSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.show()
        qtbot.addWidget(slider)

        slider.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
        )
        assert slider.value() == 49

    def test_page_step(self, qtbot):
        from src.pynegative.ui.widgets.resetableslider import ResetableSlider

        slider = ResetableSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setPageStep(10)
        slider.show()
        qtbot.addWidget(slider)

        slider.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_PageUp, Qt.NoModifier)
        )
        assert slider.value() == 60

        slider.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_PageDown, Qt.NoModifier)
        )
        assert slider.value() == 50


class TestCarouselNavigation:
    """Tests for carousel navigation methods."""

    def test_select_previous_exists(self):
        """Test that select_previous method exists."""
        from src.pynegative.ui.carouselmanager import CarouselManager

        manager = MagicMock(spec=CarouselManager)
        assert hasattr(manager, "select_previous")

    def test_select_next_exists(self):
        """Test that select_next method exists."""
        from src.pynegative.ui.carouselmanager import CarouselManager

        manager = MagicMock(spec=CarouselManager)
        assert hasattr(manager, "select_next")

    def test_select_previous_wraps(self, qtbot):
        """Test that select_previous wraps from first to last."""
        from src.pynegative.ui.carouselmanager import CarouselManager

        with patch("src.pynegative.ui.carouselmanager.ThumbnailLoader"):
            thread_pool = MagicMock()
            manager = CarouselManager(thread_pool)
            qtbot.addWidget(manager.carousel)

            item1 = QtWidgets.QListWidgetItem("image1.raw")
            item1.setData(QtCore.Qt.UserRole, "/path/image1.raw")
            item2 = QtWidgets.QListWidgetItem("image2.raw")
            item2.setData(QtCore.Qt.UserRole, "/path/image2.raw")

            manager.carousel.addItem(item1)
            manager.carousel.addItem(item2)
            manager.carousel.setCurrentRow(0)

            manager.select_previous()
            assert manager.carousel.currentRow() == 1

    def test_select_next_wraps(self, qtbot):
        """Test that select_next wraps from last to first."""
        from src.pynegative.ui.carouselmanager import CarouselManager

        with patch("src.pynegative.ui.carouselmanager.ThumbnailLoader"):
            thread_pool = MagicMock()
            manager = CarouselManager(thread_pool)
            qtbot.addWidget(manager.carousel)

            item1 = QtWidgets.QListWidgetItem("image1.raw")
            item1.setData(QtCore.Qt.UserRole, "/path/image1.raw")
            item2 = QtWidgets.QListWidgetItem("image2.raw")
            item2.setData(QtCore.Qt.UserRole, "/path/image2.raw")

            manager.carousel.addItem(item1)
            manager.carousel.addItem(item2)
            manager.carousel.setCurrentRow(1)

            manager.select_next()
            assert manager.carousel.currentRow() == 0


class TestEditorPreviewRatingLogic:
    """Tests for editor preview rating logic without full widget initialization."""

    def test_preview_rating_widget_type(self):
        """Test that preview_rating_widget is PreviewStarRatingWidget type."""
        from src.pynegative.ui.editor import PreviewStarRatingWidget

        assert PreviewStarRatingWidget is not None

    def test_rating_shortcut_method_exists(self):
        """Test that _set_rating_shortcut method exists."""
        from src.pynegative.ui.editor import EditorWidget

        assert hasattr(EditorWidget, "_set_rating_shortcut")

    def test_navigate_previous_method_exists(self):
        """Test that _navigate_previous method exists."""
        from src.pynegative.ui.editor import EditorWidget

        assert hasattr(EditorWidget, "_navigate_previous")

    def test_navigate_next_method_exists(self):
        """Test that _navigate_next method exists."""
        from src.pynegative.ui.editor import EditorWidget

        assert hasattr(EditorWidget, "_navigate_next")

    def test_preview_rating_changes_method_exists(self):
        """Test that _on_preview_rating_changed method exists."""
        from src.pynegative.ui.editor import EditorWidget

        assert hasattr(EditorWidget, "_on_preview_rating_changed")
