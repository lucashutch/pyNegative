from PySide6 import QtWidgets, QtCore


class ToastWidget(QtWidgets.QWidget):
    """Toast notification widget that appears at bottom of screen."""

    def __init__(self, parent=None, duration=3000):
        super().__init__(parent)
        self._duration = duration
        self._opacity = 0.0

        # Set up widget properties
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # Create background frame
        self._background_frame = QtWidgets.QFrame()
        self._background_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(64, 64, 64, 0.95);
                border: 1px solid #6366f1;
                border-radius: 8px;
            }
        """)

        # Create layout
        self._layout = QtWidgets.QHBoxLayout(self._background_frame)
        self._layout.setContentsMargins(16, 8, 16, 8)

        # Create label
        self._label = QtWidgets.QLabel()
        self._label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: 500;
                border: none;
            }
        """)

        self._layout.addWidget(self._label)
        self._background_frame.setLayout(self._layout)

        # Set main layout
        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.addWidget(self._background_frame)

        # Animation timers
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._hide_toast)

        self._fade_timer = QtCore.QTimer(self)
        self._fade_timer.timeout.connect(self._update_opacity)

        # Initialize hidden and transparent
        self.setWindowOpacity(0.0)
        self.hide()

    def show_message(self, message):
        """Show a toast message."""
        self._label.setText(message)
        self.ensurePolished()
        self.adjustSize()

        # Position at bottom center of parent
        if self.parent():
            parent_rect = self.parent().rect()
            # Use the actual width/height after adjustSize
            x = (parent_rect.width() - self.width()) // 2
            y = parent_rect.height() - self.height() - 40
            self.move(x, y)

            # Ensure toast is visible and properly positioned
            self.raise_()

        # Show with fade in
        self._opacity = 0.0
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        # Start fade in
        self._fade_direction = 1  # 1 = fade in, -1 = fade out
        self._fade_timer.start(16)  # ~60fps

        # Start hide timer
        self._timer.start(self._duration)

    def _update_opacity(self):
        """Update fade animation."""
        if self._fade_direction == 1:  # Fade in
            self._opacity += 0.1
            if self._opacity >= 1.0:
                self._opacity = 1.0
                self._fade_timer.stop()
        else:  # Fade out
            self._opacity -= 0.1
            if self._opacity <= 0.0:
                self._opacity = 0.0
                self._fade_timer.stop()
                self.hide()

        self.setWindowOpacity(self._opacity)

    def _hide_toast(self):
        """Start fade out animation."""
        self._fade_direction = -1
        self._fade_timer.start(16)  # ~60fps
