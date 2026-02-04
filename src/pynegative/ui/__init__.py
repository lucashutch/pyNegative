import sys
import logging
from PySide6 import QtWidgets
from .main_window import MainWindow


def main():
    # Configure logging to show time, level, and message
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
