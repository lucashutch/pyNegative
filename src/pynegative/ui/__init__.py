import sys
import logging
import argparse
from PySide6 import QtWidgets
from .main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(
        description="pyNegative - A simple RAW image processor"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "path", nargs="?", help="Path to an image file or directory to open"
    )
    args, unknown = parser.parse_known_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Note: We pass unknown args to QApplication so it can handle standard Qt flags
    app = QtWidgets.QApplication([sys.argv[0]] + unknown)
    window = MainWindow(initial_path=args.path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
