import sys
import logging
import argparse
from pathlib import Path

# Configure logging EARLY before importing internal modules
# to capture debug logs during the import phase.
log_level = logging.DEBUG if "--debug" in sys.argv else logging.ERROR
logging.basicConfig(
    level=log_level,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress verbose third-party logs
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("llvmlite").setLevel(logging.WARNING)
logging.getLogger("exifread").setLevel(logging.WARNING)

from PySide6 import QtWidgets, QtGui, QtCore  # noqa: E402
from .main_window import MainWindow  # noqa: E402
from .. import __version__  # noqa: E402


logger = logging.getLogger(__name__)
logger.debug(f"pyNegative v{__version__} starting up...")


def main():
    parser = argparse.ArgumentParser(
        description="pyNegative - A simple RAW image processor"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "path", nargs="?", help="Path to an image file or directory to open"
    )
    args, unknown = parser.parse_known_args()

    # Note: We pass unknown args to QApplication so it can handle standard Qt flags
    app = QtWidgets.QApplication([sys.argv[0]] + unknown)

    # Show Splash Screen
    # Path is relative to src/pynegative/ui/__init__.py -> go up 4 levels to reach root
    icon_path = Path(__file__).parent.parent.parent.parent / "pynegative_icon.png"
    raw_icon = QtGui.QPixmap(str(icon_path))

    # Create a nice canvas for the splash screen
    splash_width, splash_height = 600, 400
    canvas = QtGui.QPixmap(splash_width, splash_height)
    canvas.fill(QtGui.QColor("#1a1a1a"))  # Match the main app background

    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

    if not raw_icon.isNull():
        # Scale icon to a modest size (e.g. 280px)
        icon_size = 280
        scaled_icon = raw_icon.scaled(
            icon_size,
            icon_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        icon_x = (splash_width - scaled_icon.width()) // 2
        icon_y = (splash_height - scaled_icon.height()) // 2 - 40
        painter.drawPixmap(icon_x, icon_y, scaled_icon)
    else:
        # Fallback text if icon missing
        painter.setPen(QtGui.QColor("#ffffff"))
        font = painter.font()
        font.setPointSize(24)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(canvas.rect(), QtCore.Qt.AlignCenter, "pyNegative")

    painter.end()

    splash = QtWidgets.QSplashScreen(canvas)
    # Set flags to ensure visibility and prevent "Not Responding" warnings
    splash.setWindowFlags(
        QtCore.Qt.WindowStaysOnTopHint
        | QtCore.Qt.FramelessWindowHint
        | QtCore.Qt.SplashScreen
    )

    # Center splash screen on the primary screen
    screen_geometry = QtGui.QGuiApplication.primaryScreen().geometry()
    splash.move(screen_geometry.center() - canvas.rect().center())

    splash.show()
    splash.raise_()
    splash.activateWindow()

    # Apply some styling to splash message
    def update_splash_status(msg):
        # Add a trailing newline to create some padding from the bottom edge
        splash.showMessage(
            f"pyNegative v{__version__}\n{msg}\n",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignCenter,
            QtGui.QColor("#cdd6f4"),
        )
        # Process events to keep UI updated
        app.processEvents()

    update_splash_status("Initializing...")

    # Force initial events processing to make sure splash is visible
    for _ in range(30):
        app.processEvents()
        QtCore.QThread.msleep(10)

    # Pre-compile Numba kernels (first launch may take 15-30s)
    update_splash_status("Preparing image engine...")
    app.processEvents()
    from ..utils.numba_warmup import warmup_kernels

    is_first_run, warmup_ms = warmup_kernels()
    if is_first_run:
        logger.info(f"First-launch kernel compilation took {warmup_ms:.0f}ms")
    else:
        logger.debug(f"Kernel cache warm ({warmup_ms:.0f}ms)")

    update_splash_status("Loading UI...")
    app.processEvents()

    window = MainWindow(initial_path=args.path)
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
