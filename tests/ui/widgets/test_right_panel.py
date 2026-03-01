"""Tests for the RightPanel tabbed container."""

import pytest

from pynegative.ui.widgets.history_panel import HistoryPanel
from pynegative.ui.widgets.metadata_panel import MetadataPanel
from pynegative.ui.widgets.right_panel import RightPanel


@pytest.fixture()
def panels(qapp):
    mp = MetadataPanel()
    hp = HistoryPanel()
    rp = RightPanel(mp, hp)
    return rp, mp, hp


class TestRightPanel:
    def test_starts_hidden(self, panels):
        rp, _, _ = panels
        assert not rp.isVisible()

    def test_show_info_tab(self, panels):
        rp, _, _ = panels
        rp.show_info_tab()
        assert rp.isVisible()
        assert rp.currentIndex() == RightPanel.INFO_TAB

    def test_show_history_tab(self, panels):
        rp, _, _ = panels
        rp.show_history_tab()
        assert rp.isVisible()
        assert rp.currentIndex() == RightPanel.HISTORY_TAB

    def test_has_two_tabs(self, panels):
        rp, _, _ = panels
        assert rp.count() == 2

    def test_child_panels_accessible(self, panels):
        rp, mp, hp = panels
        assert rp.metadata_panel is mp
        assert rp.history_panel is hp
