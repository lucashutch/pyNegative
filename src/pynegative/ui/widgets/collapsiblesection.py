from PySide6 import QtCore, QtWidgets


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header and a content area."""

    expandedChanged = QtCore.Signal(bool)
    resetClicked = QtCore.Signal()

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header Container
        self.header = QtWidgets.QPushButton(title)
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.clicked.connect(self.toggle)

        # Reset Button (Compact, inside header)
        self.reset_btn = QtWidgets.QPushButton("Reset", self.header)
        self.reset_btn.setObjectName("SectionReset")
        self.reset_btn.setFixedSize(30, 14)
        self.reset_btn.setToolTip(f"Reset all {title} parameters")
        self.reset_btn.setStyleSheet("""
            QPushButton#SectionReset {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                color: #bbb;
                font-size: 9px;
                font-weight: normal;
                padding: 0px;
                margin: 0px;
                min-height: 0px;
                max-height: 14px;
                text-transform: none;
                letter-spacing: 0px;
            }
            QPushButton#SectionReset:hover {
                background-color: rgba(139, 92, 246, 0.2);
                border: 1px solid #8b5cf6;
                color: #eee;
            }
        """)
        self.reset_btn.clicked.connect(self.resetClicked.emit)

        # Positioning reset button on the right
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(5, 0, 8, 2)
        header_layout.addStretch()
        header_layout.addWidget(self.reset_btn, 0, QtCore.Qt.AlignVCenter)

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

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

    def minimumSizeHint(self):
        hint = self.header.sizeHint()
        if self.header.isChecked():
            hint.setHeight(hint.height() + self.content.minimumSizeHint().height())
        return hint
