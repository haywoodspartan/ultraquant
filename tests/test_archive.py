"""Tests for ultraquant.archive.artchive (ArTchive snapshots)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ultraquant.archive.artchive import ArTchive, IntegrityError


class TestArTchive(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "store"
        self.arc = ArTchive(self.root)

    def test_init_creates_layout(self) -> None:
        self.assertTrue(self.root.is_dir())
        self.assertTrue((self.root / "snapshots").is_dir())
        self.assertTrue((self.root / "manifest.json").is_file())
        with open(self.root / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["versions"], [])

    def test_commit_restore_round_trip(self) -> None:
        payload = {"weights": [1.0, -0.5, 0.25], "meta": {"epoch": 3}, "name": "net"}
        vid = self.arc.commit("first", payload)
        self.assertEqual(vid, "T-0001")
        restored = self.arc.restore(vid)
        self.assertEqual(restored, payload)

    def test_ids_increment(self) -> None:
        v1 = self.arc.commit("a", {"x": 1})
        v2 = self.arc.commit("b", {"x": 2})
        v3 = self.arc.commit("c", {"x": 3})
        self.assertEqual([v1, v2, v3], ["T-0001", "T-0002", "T-0003"])
        self.assertEqual([e["id"] for e in self.arc.versions()], [v1, v2, v3])
        self.assertEqual(self.arc.latest(), v3)

    def test_latest_none_when_empty(self) -> None:
        self.assertIsNone(self.arc.latest())
        self.assertEqual(self.arc.versions(), [])

    def test_tamper_raises_integrity_error(self) -> None:
        vid = self.arc.commit("orig", {"secret": 42})
        snap = self.root / "snapshots" / f"{vid}.json"
        raw = snap.read_bytes()
        snap.write_bytes(raw.replace(b"42", b"43"))
        with self.assertRaises(IntegrityError):
            self.arc.restore(vid)

    def test_untampered_restore_after_other_tamper(self) -> None:
        v1 = self.arc.commit("a", {"k": 1})
        v2 = self.arc.commit("b", {"k": 2})
        snap = self.root / "snapshots" / f"{v1}.json"
        snap.write_bytes(b'{"k":999}')
        with self.assertRaises(IntegrityError):
            self.arc.restore(v1)
        self.assertEqual(self.arc.restore(v2), {"k": 2})

    def test_diff_reports_added_removed_changed(self) -> None:
        v1 = self.arc.commit("a", {"keep": 1, "gone": 2, "mod": [1, 2]})
        v2 = self.arc.commit("b", {"keep": 1, "new": 3, "mod": [1, 2, 3]})
        d = self.arc.diff(v1, v2)
        self.assertEqual(d["added"], ["new"])
        self.assertEqual(d["removed"], ["gone"])
        self.assertEqual(d["changed"], ["mod"])

    def test_diff_identical_payloads(self) -> None:
        v1 = self.arc.commit("a", {"x": 1, "y": 2})
        v2 = self.arc.commit("b", {"x": 1, "y": 2})
        self.assertEqual(
            self.arc.diff(v1, v2), {"added": [], "removed": [], "changed": []}
        )

    def test_restore_unknown_id_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            self.arc.restore("T-9999")

    def test_manifest_entry_shape(self) -> None:
        vid = self.arc.commit("labelled", {"a": 1})
        entry = self.arc.versions()[0]
        self.assertEqual(entry["id"], vid)
        self.assertEqual(entry["label"], "labelled")
        self.assertEqual(entry["path"], f"snapshots/{vid}.json")
        self.assertEqual(len(entry["sha256"]), 64)
        self.assertIn("timestamp", entry)

    def test_reopen_continues_numbering(self) -> None:
        self.arc.commit("a", {"x": 1})
        self.arc.commit("b", {"x": 2})
        reopened = ArTchive(self.root)
        v3 = reopened.commit("c", {"x": 3})
        self.assertEqual(v3, "T-0003")
        self.assertEqual(reopened.latest(), "T-0003")
        self.assertEqual(reopened.restore("T-0001"), {"x": 1})


if __name__ == "__main__":
    unittest.main()
