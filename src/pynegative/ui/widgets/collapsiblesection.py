from PySide6 import QtWidgets, QtCore


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header and a content area."""

    expandedChanged = QtCore.Signal(bool)

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header
        self.header = QtWidgets.QPushButton(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self.toggle)
        self.main_layout.addWidget(self.header)

        # Content Area
        self.content = QtWidgets.QWidget()
        self.content.setObjectName("SectionContent")
        self.content_layout = QtWidgets.QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 5, 4, 5)
        self.content_layout.setSpacing(2)
        self.main_layout.addWidget(self.content)

        if not expanded:
            self.content.hide()

    def toggle(self):
        expanded = self.header.isChecked()
        if expanded:
            self.content.show()
        else:
            self.content.hide()
        self.expandedChanged.emit(expanded)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def minimumSizeHint(self):
        hint = self.header.sizeHint()
        if self.header.isChecked():
            hint.setHeight(hint.height() + self.content.minimumSizeHint().height())
        return hint
