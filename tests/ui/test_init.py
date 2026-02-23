import sys
import pytest
from unittest.mock import MagicMock, patch
from pynegative.ui import main


def test_ui_main_help():
    # Test that --help works (it will raise SystemExit)
    with patch.object(sys, "argv", ["pynegative", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


def test_ui_main_version():
    with patch.object(sys, "argv", ["pynegative", "--version"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0


@patch("PySide6.QtWidgets.QApplication")
@patch("PySide6.QtWidgets.QSplashScreen")
@patch("PySide6.QtGui.QPixmap")
@patch("PySide6.QtGui.QPainter")
@patch("PySide6.QtGui.QGuiApplication")
@patch("pynegative.utils.numba_warmup.warmup_kernels")
@patch("pynegative.io.lens_db_xml.load_database")
def test_ui_main_flow(
    mock_load_db,
    mock_warmup,
    mock_qgui,
    mock_painter,
    mock_pixmap,
    mock_splash,
    mock_app,
):
    mock_warmup.return_value = (True, 100)

    # Mock QPixmap to return something with a rect() method for splash.move
    mock_pixmap.return_value.rect.return_value = MagicMock()
    mock_pixmap.return_value.isNull.return_value = False

    # Mock QGuiApplication.primaryScreen().geometry()
    mock_screen = MagicMock()
    mock_qgui.primaryScreen.return_value = mock_screen
    mock_screen.geometry.return_value = MagicMock()

    with patch.object(sys, "argv", ["pynegative"]):
        # Mock sys.exit to avoid exiting the test runner
        with patch.object(sys, "exit") as mock_exit:
            # We also need to avoid MainWindow init if it's too complex
            with patch("pynegative.ui.MainWindow") as mock_mw_class:
                main()

                mock_app.assert_called()
                mock_warmup.assert_called()
                mock_load_db.assert_called()
                mock_mw_class.assert_called()
                mock_exit.assert_called()
