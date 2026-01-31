from PySide6 import QtWidgets


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header and a content area."""

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Header
        self.header = QtWidgets.QPushButton(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self.toggle)
        self.layout.addWidget(self.header)

        # Content Area
        self.content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 5, 4, 5)
        self.content_layout.setSpacing(2)
        self.layout.addWidget(self.content)

        if not expanded:
            self.content.hide()

    def toggle(self):
        if self.header.isChecked():
            self.content.show()
        else:
            self.content.hide()

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
