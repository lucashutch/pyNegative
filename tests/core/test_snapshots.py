"""Tests for snapshot/version persistence in sidecar."""

import json
import time

import pytest

from pynegative.io.sidecar import (
    delete_snapshot,
    format_snapshot_timestamp,
    load_snapshots,
    prune_snapshots_list,
    save_sidecar,
    save_snapshot,
    update_snapshot,
)


@pytest.fixture()
def raw_file(tmp_path):
    """Create a dummy raw file and return its path."""
    f = tmp_path / "test.CR3"
    f.write_text("fake raw")
    return f


class TestSaveSnapshot:
    def test_creates_snapshot_entry(self, raw_file):
        settings = {"exposure": 0.5, "contrast": 0.0}
        snap = save_snapshot(raw_file, settings)

        assert "id" in snap
        assert snap["is_auto"] is True
        assert snap["is_tagged"] is False
        assert snap["label"] is None
        assert snap["settings"] == settings

    def test_snapshot_stored_in_sidecar(self, raw_file):
        save_snapshot(raw_file, {"exposure": 1.0})
        snaps = load_snapshots(raw_file)
        assert len(snaps) == 1
        assert snaps[0]["settings"]["exposure"] == 1.0

    def test_multiple_snapshots(self, raw_file):
        save_snapshot(raw_file, {"exposure": 0.1})
        save_snapshot(raw_file, {"exposure": 0.2})
        save_snapshot(raw_file, {"exposure": 0.3})
        snaps = load_snapshots(raw_file)
        assert len(snaps) == 3

    def test_snapshots_sorted_newest_first(self, raw_file):
        save_snapshot(raw_file, {"exposure": 0.1})
        time.sleep(0.01)
        save_snapshot(raw_file, {"exposure": 0.2})
        snaps = load_snapshots(raw_file)
        assert snaps[0]["settings"]["exposure"] == 0.2
        assert snaps[1]["settings"]["exposure"] == 0.1

    def test_tagged_snapshot(self, raw_file):
        snap = save_snapshot(
            raw_file,
            {"exposure": 0.5},
            label="Final Edit",
            is_auto=False,
            is_tagged=True,
        )
        assert snap["is_tagged"] is True
        assert snap["label"] == "Final Edit"

    def test_snapshots_coexist_with_settings(self, raw_file):
        save_sidecar(raw_file, {"exposure": 0.5})
        save_snapshot(raw_file, {"exposure": 1.0})

        sidecar_path = raw_file.parent / ".pyNegative" / f"{raw_file.name}.json"
        with open(sidecar_path) as f:
            data = json.load(f)

        assert "settings" in data
        assert "snapshots" in data
        assert data["settings"]["exposure"] == 0.5
        assert len(data["snapshots"]) == 1

    def test_save_sidecar_preserves_existing_snapshots(self, raw_file):
        save_snapshot(raw_file, {"exposure": 0.1}, is_auto=False)
        save_snapshot(raw_file, {"exposure": 0.2}, is_auto=False)

        save_sidecar(raw_file, {"exposure": 1.5})

        snaps = load_snapshots(raw_file)
        assert len(snaps) == 2
        assert snaps[0]["settings"]["exposure"] in (0.1, 0.2)


class TestLoadSnapshots:
    def test_empty_when_no_sidecar(self, raw_file):
        assert load_snapshots(raw_file) == []

    def test_empty_when_no_snapshots_key(self, raw_file):
        save_sidecar(raw_file, {"exposure": 0.5})
        assert load_snapshots(raw_file) == []


class TestDeleteSnapshot:
    def test_deletes_by_id(self, raw_file):
        snap = save_snapshot(raw_file, {"exposure": 0.5})
        assert delete_snapshot(raw_file, snap["id"]) is True
        assert load_snapshots(raw_file) == []

    def test_returns_false_for_missing_id(self, raw_file):
        save_snapshot(raw_file, {"exposure": 0.5})
        assert delete_snapshot(raw_file, "nonexistent") is False

    def test_only_deletes_target(self, raw_file):
        snap1 = save_snapshot(raw_file, {"exposure": 0.1})
        save_snapshot(raw_file, {"exposure": 0.2})
        delete_snapshot(raw_file, snap1["id"])
        snaps = load_snapshots(raw_file)
        assert len(snaps) == 1
        assert snaps[0]["settings"]["exposure"] == 0.2


class TestUpdateSnapshot:
    def test_update_label(self, raw_file):
        snap = save_snapshot(raw_file, {"exposure": 0.5})
        result = update_snapshot(raw_file, snap["id"], label="My Version")
        assert result is True
        snaps = load_snapshots(raw_file)
        assert snaps[0]["label"] == "My Version"

    def test_update_tagged(self, raw_file):
        snap = save_snapshot(raw_file, {"exposure": 0.5})
        update_snapshot(raw_file, snap["id"], is_tagged=True)
        snaps = load_snapshots(raw_file)
        assert snaps[0]["is_tagged"] is True

    def test_update_missing_returns_false(self, raw_file):
        assert update_snapshot(raw_file, "nonexistent", label="X") is False


class TestPruneSnapshots:
    def test_prunes_oldest_auto(self):
        snaps = [
            {"id": str(i), "timestamp": float(i), "is_auto": True, "is_tagged": False}
            for i in range(60)
        ]
        result = prune_snapshots_list(snaps, max_auto=50)
        auto_result = [s for s in result if s["is_auto"]]
        assert len(auto_result) == 50
        # Should keep the newest 50 (ids 10-59)
        ids = {s["id"] for s in auto_result}
        assert "0" not in ids
        assert "59" in ids

    def test_tagged_always_kept(self):
        snaps = [
            {"id": str(i), "timestamp": float(i), "is_auto": True, "is_tagged": False}
            for i in range(55)
        ]
        snaps.append(
            {"id": "tagged1", "timestamp": 0.0, "is_auto": False, "is_tagged": True}
        )
        result = prune_snapshots_list(snaps, max_auto=50)
        tagged = [s for s in result if s.get("is_tagged")]
        assert len(tagged) == 1

    def test_manual_snapshots_not_pruned(self):
        snaps = [
            {"id": f"m{i}", "timestamp": float(i), "is_auto": False, "is_tagged": False}
            for i in range(200)
        ]
        result = prune_snapshots_list(snaps, max_auto=50)
        assert len(result) == 200


class TestFormatSnapshotTimestamp:
    def test_formats_correctly(self):
        ts = 1835356200.0  # 2028-02-28 14:30 UTC (approx)
        result = format_snapshot_timestamp(ts)
        assert "2028" in result
        assert "-" in result
        assert ":" in result
