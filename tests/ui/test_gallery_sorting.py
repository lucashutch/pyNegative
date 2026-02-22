import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PySide6 import QtCore, QtWidgets

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pynegative.ui.gallery import GalleryWidget  # noqa: E402


class TestGallerySorting:
    def test_apply_sort_filename(self, qtbot):
        thread_pool = MagicMock()
        # Mock QSettings to avoid touching real settings
        with patch("PySide6.QtCore.QSettings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.value.side_effect = lambda key, default, type=None: default
            widget = GalleryWidget(thread_pool)
            qtbot.addWidget(widget)

            # Add items in unsorted order
            items = [
                ("c.raw", "/path/c.raw", 1, "2024-01-03", 300),
                ("a.raw", "/path/a.raw", 3, "2024-01-01", 100),
                ("b.raw", "/path/b.raw", 2, "2024-01-02", 200),
            ]

            for name, path, rating, date, mtime in items:
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, path)
                item.setData(QtCore.Qt.UserRole + 1, rating)
                item.setData(QtCore.Qt.UserRole + 2, date)
                item.setData(QtCore.Qt.UserRole + 3, mtime)
                widget.list_widget.addItem(item)

            # Sort by Filename Ascending
            widget._sort_by = "Filename"
            widget._sort_ascending = True
            widget._apply_sort()

            assert widget.list_widget.item(0).text() == "a.raw"
            assert widget.list_widget.item(1).text() == "b.raw"
            assert widget.list_widget.item(2).text() == "c.raw"

            # Sort by Filename Descending
            widget._sort_ascending = False
            widget._apply_sort()

            assert widget.list_widget.item(0).text() == "c.raw"
            assert widget.list_widget.item(1).text() == "b.raw"
            assert widget.list_widget.item(2).text() == "a.raw"

    def test_apply_sort_rating(self, qtbot):
        thread_pool = MagicMock()
        with patch("PySide6.QtCore.QSettings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.value.side_effect = lambda key, default, type=None: default
            widget = GalleryWidget(thread_pool)
            qtbot.addWidget(widget)

            items = [
                ("c.raw", "/path/c.raw", 1, "2024-01-03", 300),
                ("a.raw", "/path/a.raw", 3, "2024-01-01", 100),
                ("b.raw", "/path/b.raw", 2, "2024-01-02", 200),
            ]

            for name, path, rating, date, mtime in items:
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, path)
                item.setData(QtCore.Qt.UserRole + 1, rating)
                item.setData(QtCore.Qt.UserRole + 2, date)
                item.setData(QtCore.Qt.UserRole + 3, mtime)
                widget.list_widget.addItem(item)

            # Sort by Rating Ascending
            widget._sort_by = "Rating"
            widget._sort_ascending = True
            widget._apply_sort()

            assert widget.list_widget.item(0).text() == "c.raw"  # Rating 1
            assert widget.list_widget.item(1).text() == "b.raw"  # Rating 2
            assert widget.list_widget.item(2).text() == "a.raw"  # Rating 3

    def test_apply_sort_date_taken(self, qtbot):
        thread_pool = MagicMock()
        with patch("PySide6.QtCore.QSettings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.value.side_effect = lambda key, default, type=None: default
            widget = GalleryWidget(thread_pool)
            qtbot.addWidget(widget)

            items = [
                ("c.raw", "/path/c.raw", 1, "2024-01-03", 300),
                ("a.raw", "/path/a.raw", 3, "2024-01-01", 100),
                ("b.raw", "/path/b.raw", 2, "2024-01-02", 200),
            ]

            for name, path, rating, date, mtime in items:
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, path)
                item.setData(QtCore.Qt.UserRole + 1, rating)
                item.setData(QtCore.Qt.UserRole + 2, date)
                item.setData(QtCore.Qt.UserRole + 3, mtime)
                widget.list_widget.addItem(item)

            # Sort by Date Taken Ascending
            widget._sort_by = "Date Taken"
            widget._sort_ascending = True
            widget._apply_sort()

            assert widget.list_widget.item(0).text() == "a.raw"  # 2024-01-01
            assert widget.list_widget.item(1).text() == "b.raw"  # 2024-01-02
            assert widget.list_widget.item(2).text() == "c.raw"  # 2024-01-03

    def test_apply_sort_last_edited(self, qtbot):
        thread_pool = MagicMock()
        with patch("PySide6.QtCore.QSettings") as mock_settings_class:
            mock_settings = mock_settings_class.return_value
            mock_settings.value.side_effect = lambda key, default, type=None: default
            widget = GalleryWidget(thread_pool)
            qtbot.addWidget(widget)

            items = [
                ("c.raw", "/path/c.raw", 1, "2024-01-03", 300),
                ("a.raw", "/path/a.raw", 3, "2024-01-01", 100),
                ("b.raw", "/path/b.raw", 2, "2024-01-02", 200),
            ]

            for name, path, rating, date, mtime in items:
                item = QtWidgets.QListWidgetItem(name)
                item.setData(QtCore.Qt.UserRole, path)
                item.setData(QtCore.Qt.UserRole + 1, rating)
                item.setData(QtCore.Qt.UserRole + 2, date)
                item.setData(QtCore.Qt.UserRole + 3, mtime)
                widget.list_widget.addItem(item)

            # Sort by Last Edited Ascending
            widget._sort_by = "Last Edited"
            widget._sort_ascending = True
            widget._apply_sort()

            assert widget.list_widget.item(0).text() == "a.raw"  # 100
            assert widget.list_widget.item(1).text() == "b.raw"  # 200
            assert widget.list_widget.item(2).text() == "c.raw"  # 300

    def test_on_rating_changed_recursion_fix(self, qtbot):
        thread_pool = MagicMock()
        with (
            patch("PySide6.QtCore.QSettings") as mock_settings_class,
            patch("src.pynegative.ui.gallery.pynegative.save_sidecar") as mock_save,
            patch("src.pynegative.ui.gallery.pynegative.load_sidecar"),
            patch("src.pynegative.ui.gallery.pynegative.get_sidecar_mtime"),
        ):
            mock_settings = mock_settings_class.return_value
            mock_settings.value.side_effect = lambda key, default, type=None: default
            widget = GalleryWidget(thread_pool)
            qtbot.addWidget(widget)

            # Add item
            item = QtWidgets.QListWidgetItem("test.raw")
            item.setData(QtCore.Qt.UserRole, "/path/test.raw")
            item.setData(QtCore.Qt.UserRole + 1, 0)
            widget.list_widget.addItem(item)

            index = widget.list_widget.model().index(0, 0)

            # 1. Trigger with wrong role (e.g. UserRole + 3 / Last Edited)
            # This simulates the recursive call
            widget._on_rating_changed(index, index, [QtCore.Qt.UserRole + 3])

            # Should NOT save sidecar
            mock_save.assert_not_called()

            # 2. Trigger with Rating role (UserRole + 1)
            widget._on_rating_changed(index, index, [QtCore.Qt.UserRole + 1])

            # Should save sidecar
            mock_save.assert_called_once()
