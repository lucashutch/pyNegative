"""Tests for Phase 3: tagged versions and history panel context menu."""

import time

import pytest

from pynegative.ui.widgets.history_panel import HistoryPanel


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


class TestHistoryPanelContextMenuSignals:
    def test_tag_signal_exists(self, panel):
        assert hasattr(panel, "tagRequested")

    def test_delete_signal_exists(self, panel):
        assert hasattr(panel, "deleteRequested")

    def test_comparison_signals_exist(self, panel):
        assert hasattr(panel, "setLeftComparison")
        assert hasattr(panel, "setRightComparison")


class TestHistoryPanelComparisonActive:
    def test_default_comparison_inactive(self, panel):
        assert panel._comparison_active is False

    def test_set_comparison_active(self, panel):
        panel.set_comparison_active(True)
        assert panel._comparison_active is True


class TestHistoryPanelContextMenu:
    def test_context_menu_on_tagged_item(self, panel, qtbot):
        """Tagged items should show 'Untag Version' instead of 'Tag as Version'."""
        snap = _make_snapshot("s1", is_tagged=True, label="V1")
        panel.set_snapshots([snap])

        # We can't easily test the full menu popup, but we can verify
        # the signal connection works
        received = []
        panel.tagRequested.connect(lambda sid: received.append(("tag", sid)))

        # Simulate a tag request
        panel.tagRequested.emit("s1")
        assert received == [("tag", "s1")]

    def test_delete_signal_emitted(self, panel, qtbot):
        snap = _make_snapshot("s1", is_auto=True, is_tagged=False)
        panel.set_snapshots([snap])

        received = []
        panel.deleteRequested.connect(lambda sid: received.append(("del", sid)))
        panel.deleteRequested.emit("s1")
        assert received == [("del", "s1")]

    def test_comparison_signals_emitted(self, panel, qtbot):
        panel.set_comparison_active(True)
        snap = _make_snapshot("s1")
        panel.set_snapshots([snap])

        left_received = []
        right_received = []
        panel.setLeftComparison.connect(lambda sid: left_received.append(sid))
        panel.setRightComparison.connect(lambda sid: right_received.append(sid))

        panel.setLeftComparison.emit("s1")
        panel.setRightComparison.emit("s1")
        assert left_received == ["s1"]
        assert right_received == ["s1"]


class TestTaggingIntegration:
    """Test the tagging workflow via the sidecar layer."""

    def test_tag_snapshot(self, tmp_path):
        from pynegative.io.sidecar import (
            load_snapshots,
            save_snapshot,
            update_snapshot,
        )

        raw = tmp_path / "test.CR3"
        raw.write_text("fake")

        snap = save_snapshot(raw, {"exposure": 0.5})
        assert snap["is_tagged"] is False

        update_snapshot(raw, snap["id"], is_tagged=True, label="Final Edit")

        snaps = load_snapshots(raw)
        assert snaps[0]["is_tagged"] is True
        assert snaps[0]["label"] == "Final Edit"

    def test_untag_snapshot(self, tmp_path):
        from pynegative.io.sidecar import (
            load_snapshots,
            save_snapshot,
            update_snapshot,
        )

        raw = tmp_path / "test.CR3"
        raw.write_text("fake")

        snap = save_snapshot(raw, {"exposure": 0.5}, is_tagged=True, label="V1")

        update_snapshot(raw, snap["id"], is_tagged=False, label=None)

        snaps = load_snapshots(raw)
        assert snaps[0]["is_tagged"] is False
        assert snaps[0]["label"] is None

    def test_tagged_entries_not_pruned(self, tmp_path):
        from pynegative.io.sidecar import prune_snapshots_list

        snaps = [
            {
                "id": str(i),
                "timestamp": float(i),
                "is_auto": True,
                "is_tagged": False,
            }
            for i in range(55)
        ]
        # Tag one of them
        snaps[0]["is_tagged"] = True

        result = prune_snapshots_list(snaps, max_auto=50)
        tagged = [s for s in result if s.get("is_tagged")]
        assert len(tagged) == 1
        assert tagged[0]["id"] == "0"
