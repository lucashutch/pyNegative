"""Tabbed right panel container holding Info (metadata) and History tabs."""

from PySide6 import QtWidgets


class RightPanel(QtWidgets.QTabWidget):
    """QTabWidget hosting the MetadataPanel and HistoryPanel in tabs."""

    INFO_TAB = 0
    HISTORY_TAB = 1

    def __init__(self, metadata_panel, history_panel, parent=None):
        super().__init__(parent)
        self.setObjectName("RightPanel")
        self.setFixedWidth(280)
        self.setVisible(False)

        self.metadata_panel = metadata_panel
        self.history_panel = history_panel

        # Ensure child panels are always visible when inside the tab widget
        metadata_panel.setVisible(True)
        history_panel.setVisible(True)

        # Remove the fixed width from children since the tab widget controls it
        metadata_panel.setMinimumWidth(0)
        metadata_panel.setMaximumWidth(16777215)
        history_panel.setMinimumWidth(0)
        history_panel.setMaximumWidth(16777215)

        self.addTab(metadata_panel, "ⓘ Info")
        self.addTab(history_panel, "🕓 History")

    def show_info_tab(self):
        """Activate the Info/metadata tab."""
        self.setCurrentIndex(self.INFO_TAB)
        self.setVisible(True)

    def show_history_tab(self):
        """Activate the History tab."""
        self.setCurrentIndex(self.HISTORY_TAB)
        self.setVisible(True)
