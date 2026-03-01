from unittest.mock import MagicMock, patch
from PySide6 import QtCore

from pynegative.ui.carouselmanager import CarouselManager


@patch("PySide6.QtCore.QThreadPool.start")
def test_carouselmanager_coverage(mock_start, qapp):
    thread_pool = QtCore.QThreadPool()
    manager = CarouselManager(thread_pool)

    # set_carousel_height & _do_deferred_resize
    manager.set_carousel_height(250)
    manager._pending_height = 250
    manager._do_deferred_resize()

    # load_folder
    with patch("pynegative.ui.carouselmanager.Path.iterdir") as mock_iterdir:
        fake_file = MagicMock()
        fake_file.is_file.return_value = True
        fake_file.suffix = ".jpg"
        fake_file.name = "test.jpg"
        fake_file.__str__.return_value = "test.jpg"
        mock_iterdir.return_value = [fake_file]

        manager.load_folder("/tmp/fake_folder")

    # set_images
    manager.set_images(["/tmp/img1.jpg", "/tmp/img2.jpg"], "/tmp/img1.jpg")
    manager.set_images(["/tmp/img1.jpg", "/tmp/img2.jpg"], "/tmp/img1.jpg")

    # select_image
    manager.select_image("/tmp/img2.jpg")

    # select_next & select_previous
    manager.select_next()
    manager.select_previous()

    # get_selected_paths / get_current_path
    manager.get_selected_paths()
    manager.get_current_path()

    # _on_thumbnail_loaded
    manager._on_thumbnail_loaded("/tmp/img1.jpg", None, {})

    # _on_item_clicked
    item = manager.carousel.item(0)
    if item:
        manager._on_item_clicked(item)

    # _on_selection_changed
    manager._on_selection_changed()

    # context menu
    manager._show_context_menu(QtCore.QPoint(0, 0))

    manager.clear()


def test_show_context_menu_falls_back_to_hovered_item(qapp):
    thread_pool = QtCore.QThreadPool()
    manager = CarouselManager(thread_pool)
    manager.set_images(["/tmp/img1.jpg"], "/tmp/img1.jpg")

    item = manager.carousel.item(0)
    manager.carousel.itemAt = MagicMock(return_value=None)
    manager.carousel.get_hovered_item = MagicMock(return_value=item)

    captured = []
    manager.contextMenuRequested.connect(
        lambda kind, data: captured.append((kind, data))
    )

    manager._show_context_menu(QtCore.QPoint(0, 0))

    assert captured
    assert captured[0][0] == "carousel"


def test_show_context_menu_falls_back_to_current_item(qapp):
    thread_pool = QtCore.QThreadPool()
    manager = CarouselManager(thread_pool)
    manager.set_images(["/tmp/img1.jpg"], "/tmp/img1.jpg")

    item = manager.carousel.item(0)
    manager.carousel.itemAt = MagicMock(return_value=None)
    manager.carousel.get_hovered_item = MagicMock(return_value=None)
    manager.carousel.currentItem = MagicMock(return_value=item)

    captured = []
    manager.contextMenuRequested.connect(
        lambda kind, data: captured.append((kind, data))
    )

    manager._show_context_menu(QtCore.QPoint(0, 0))

    assert captured
    assert captured[0][0] == "carousel"
