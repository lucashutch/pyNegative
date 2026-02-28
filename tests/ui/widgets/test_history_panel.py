"""Tests for the HistoryPanel widget."""

import time

import pytest
from PySide6.QtCore import Qt

from pynegative.ui.widgets.history_panel import HistoryPanel, SnapshotItemWidget


def _make_snapshot(
    snap_id="s1",
    label=None,
    is_auto=True,
    is_tagged=False,
    settings=None,
    timestamp=None,
):
    return {
        "id": snap_id,
        "timestamp": timestamp or time.time(),
        "label": label,
        "is_auto": is_auto,
        "is_tagged": is_tagged,
        "settings": settings or {"exposure": 0.5},
    }


@pytest.fixture()
def panel(qapp):
    return HistoryPanel()


class TestHistoryPanelEmpty:
    def test_starts_empty(self, panel):
        assert panel.list_widget.count() == 0
        assert not panel._empty_label.isHidden()

    def test_clear(self, panel):
        panel.set_snapshots([_make_snapshot()])
        panel.clear()
        assert panel.list_widget.count() == 0
        assert not panel._empty_label.isHidden()


class TestHistoryPanelPopulation:
    def test_set_snapshots(self, panel):
        snaps = [_make_snapshot("s1"), _make_snapshot("s2")]
        panel.set_snapshots(snaps)
        assert panel.list_widget.count() == 2
        assert panel._empty_label.isHidden()

    def test_items_store_snapshot_id(self, panel):
        panel.set_snapshots([_make_snapshot("abc123")])
        item = panel.list_widget.item(0)
        assert item.data(Qt.UserRole) == "abc123"

    def test_repopulate_clears_old(self, panel):
        panel.set_snapshots([_make_snapshot("s1")])
        panel.set_snapshots([_make_snapshot("s2"), _make_snapshot("s3")])
        assert panel.list_widget.count() == 2


class TestHistoryPanelSelection:
    def test_emits_snapshot_selected(self, panel, qtbot):
        panel.set_snapshots([_make_snapshot("s1")])
        with qtbot.waitSignal(panel.snapshotSelected, timeout=1000) as blocker:
            panel.list_widget.setCurrentRow(0)
        assert blocker.args == ["s1"]

    def test_restore_button_disabled_by_default(self, panel):
        panel.set_snapshots([_make_snapshot()])
        assert not panel.restore_btn.isEnabled()

    def test_restore_button_enabled_in_preview_mode(self, panel):
        panel.set_snapshots([_make_snapshot()])
        panel.set_previewing(True)
        assert panel.restore_btn.isEnabled()

    def test_cancel_emits_empty_id(self, panel, qtbot):
        panel.set_snapshots([_make_snapshot()])
        panel.set_previewing(True)
        with qtbot.waitSignal(panel.snapshotSelected, timeout=1000) as blocker:
            panel.cancel_btn.click()
        assert blocker.args == [""]

    def test_right_click_does_not_change_selection(self, panel, qtbot):
        panel.set_snapshots([_make_snapshot("s1"), _make_snapshot("s2")])
        panel.list_widget.setCurrentRow(0)

        first_item = panel.list_widget.item(0)
        second_item = panel.list_widget.item(1)
        assert first_item.isSelected()

        second_rect = panel.list_widget.visualItemRect(second_item)
        qtbot.mouseClick(
            panel.list_widget.viewport(),
            Qt.RightButton,
            pos=second_rect.center(),
        )

        assert first_item.isSelected()


class TestHistoryPanelRestore:
    def test_emits_restore_requested(self, panel, qtbot):
        panel.set_snapshots([_make_snapshot("s1")])
        panel.list_widget.setCurrentRow(0)
        panel.set_previewing(True)
        with qtbot.waitSignal(panel.restoreRequested, timeout=1000) as blocker:
            panel.restore_btn.click()
        assert blocker.args == ["s1"]


class TestGetSnapshotById:
    def test_found(self, panel):
        snaps = [_make_snapshot("s1"), _make_snapshot("s2")]
        panel.set_snapshots(snaps)
        assert panel.get_snapshot_by_id("s2")["id"] == "s2"

    def test_not_found(self, panel):
        panel.set_snapshots([_make_snapshot("s1")])
        assert panel.get_snapshot_by_id("missing") is None


class TestSnapshotItemWidget:
    def test_tagged_shows_star(self, qapp):
        snap = _make_snapshot(is_tagged=True, label="My Version")
        w = SnapshotItemWidget(snap)
        assert w is not None

    def test_auto_shows_clock(self, qapp):
        snap = _make_snapshot(is_auto=True)
        w = SnapshotItemWidget(snap)
        assert w is not None

    def test_manual_shows_disk(self, qapp):
        snap = _make_snapshot(is_auto=False, is_tagged=False)
        w = SnapshotItemWidget(snap)
        assert w is not None
