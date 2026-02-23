import pytest
from unittest.mock import patch
from PySide6 import QtWidgets, QtCore, QtGui
from pynegative.ui.exportgallerymanager import ExportGalleryManager


@pytest.fixture
def thread_pool():
    return QtCore.QThreadPool()


@pytest.fixture
def manager(qtbot, thread_pool):
    mgr = ExportGalleryManager(thread_pool)
    return mgr


def test_load_folder(manager, tmp_path):
    img1 = tmp_path / "img1.ARW"
    img1.touch()
    img2 = tmp_path / "img2.jpg"
    img2.touch()

    with patch("pynegative.core.load_sidecar", return_value={"rating": 5}), patch(
        "pynegative.ui.exportgallerymanager.ThumbnailLoader"
    ) as mock_loader:
        manager.load_folder(str(tmp_path))

        assert manager.list_widget.count() == 2
        mock_loader.assert_called()


def test_load_folder_with_filter(manager, tmp_path):
    img1 = tmp_path / "img1.ARW"
    img1.touch()
    img2 = tmp_path / "img2.ARW"
    img2.touch()

    def mock_load_sidecar(path):
        if "img1" in str(path):
            return {"rating": 5}
        return {"rating": 3}

    with patch("pynegative.core.load_sidecar", side_effect=mock_load_sidecar):
        manager.load_folder(str(tmp_path), filter_mode="Greater", filter_rating=4)
        assert manager.list_widget.count() == 1
        assert "img1" in manager.list_widget.item(0).text()


def test_selection(manager):
    item1 = QtWidgets.QListWidgetItem("1")
    item1.setData(QtCore.Qt.UserRole, "path1")
    manager.list_widget.addItem(item1)

    item2 = QtWidgets.QListWidgetItem("2")
    item2.setData(QtCore.Qt.UserRole, "path2")
    manager.list_widget.addItem(item2)

    manager.select_all()
    assert manager.get_selected_count() == 2
    assert "path1" in manager.get_selected_paths()

    manager.clear_selection()
    assert manager.get_selected_count() == 0


def test_on_thumbnail_loaded(manager):
    item = QtWidgets.QListWidgetItem("img1.ARW")
    item.setData(QtCore.Qt.UserRole, "path/img1.ARW")
    manager.list_widget.addItem(item)

    pix = QtGui.QPixmap(10, 10)
    manager._on_thumbnail_loaded("path/img1.ARW", pix)
    # Passed if it doesn't crash
