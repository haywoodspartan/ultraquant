"""Access hints stay in memory until a real write or an explicit flush."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.shards.vault import ShardVault


class TestReads(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="uq_test_reads_")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.vault = ShardVault(self.root)
        self.payload = {"value": 42}
        self.vault.add_shard("first", "facts", self.payload)
        writer = mock.patch.object(
            ShardVault, "_write_catalog", autospec=True,
            side_effect=ShardVault._write_catalog,
        )
        self.writes = writer.start()
        self.addCleanup(writer.stop)

    def disk_entry(self):
        catalog = json.loads(self.vault.catalog_path.read_text(encoding="utf-8"))
        return next(e for e in catalog["shards"] if e["shard_id"] == "first")

    def test_get_does_not_write(self):
        committed = self.vault.catalog_path.read_bytes()
        with mock.patch("ultraquant.shards.vault._utc_now", return_value="read-time"):
            self.assertEqual(self.vault.get("first"), self.payload)
        self.assertEqual(self.vault.entry("first")["access_count"], 1)
        self.assertEqual(self.vault.entry("first")["last_access"], "read-time")
        self.assertEqual(self.vault.catalog_path.read_bytes(), committed)
        self.writes.assert_not_called()

    def test_read_only_nested_batch_does_not_write(self):
        committed = self.vault.catalog_path.read_bytes()
        with self.vault.batch():
            self.assertEqual(self.vault.get("first"), self.payload)
            with self.vault.batch():
                self.assertEqual(self.vault.get("first"), self.payload)
            self.writes.assert_not_called()
        self.writes.assert_not_called()
        self.assertEqual(self.vault.entry("first")["access_count"], 2)
        self.assertEqual(self.vault.catalog_path.read_bytes(), committed)

    def test_flush_persists_counts_and_reopen_sees_them(self):
        self.vault.flush_stats()
        self.writes.assert_not_called()
        self.vault.get("first")
        self.vault.get("first")
        expected = self.vault.entry("first")
        self.vault.flush_stats()
        self.writes.assert_called_once_with(self.vault)
        self.assertEqual(self.disk_entry(), expected)
        self.assertEqual(expected["access_count"], 2)
        reopened = ShardVault(self.root)
        self.assertEqual(reopened.entry("first"), expected)
        self.vault.flush_stats()
        reopened.flush_stats()
        self.writes.assert_called_once_with(self.vault)

    def test_flush_inside_batch_does_not_schedule_write(self):
        with self.vault.batch():
            self.vault.get("first")
            with self.vault.batch():
                self.vault.flush_stats()
            self.writes.assert_not_called()
        self.writes.assert_not_called()
        self.assertEqual(self.disk_entry()["access_count"], 0)
        self.vault.flush_stats()
        self.writes.assert_called_once_with(self.vault)
        self.assertEqual(self.disk_entry()["access_count"], 1)

    def test_add_shard_persists_earlier_counts(self):
        self.vault.get("first")
        expected = self.vault.entry("first")
        self.vault.add_shard("second", "facts", {"value": 43})
        self.writes.assert_called_once_with(self.vault)
        self.assertEqual(self.disk_entry(), expected)
        self.vault.flush_stats()
        self.writes.assert_called_once_with(self.vault)

    def test_real_batch_commit_persists_counts(self):
        self.vault.get("first")
        with self.vault.batch():
            self.vault.add_shard("second", "facts", {"value": 43})
            self.vault.get("first")
            self.writes.assert_not_called()
        self.writes.assert_called_once_with(self.vault)
        self.assertEqual(self.disk_entry()["access_count"], 2)
        self.vault.flush_stats()
        self.writes.assert_called_once_with(self.vault)

    def test_failed_batch_reloads_committed_catalog(self):
        self.vault.get("first")
        self.vault.flush_stats()
        committed = self.vault.catalog()
        self.writes.reset_mock()
        self.vault.get("first")
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with self.vault.batch():
                self.vault.get("first")
                self.vault.add_shard("second", "facts", {"value": 43})
                raise RuntimeError("abort")
        self.assertEqual(self.vault.catalog(), committed)
        self.assertEqual(ShardVault(self.root).catalog(), committed)
        self.assertEqual(self.vault.load_bytes("first"), ShardVault(self.root).load_bytes("first"))
        self.vault.flush_stats()
        self.writes.assert_not_called()

    def test_failed_flush_can_retry_counts(self):
        self.vault.get("first")
        with mock.patch.object(ShardVault, "_write_catalog", side_effect=OSError("disk")):
            with self.assertRaisesRegex(OSError, "disk"):
                self.vault.flush_stats()
        self.assertEqual(self.disk_entry()["access_count"], 0)
        self.vault.flush_stats()
        self.writes.assert_called_once_with(self.vault)
        self.assertEqual(self.disk_entry()["access_count"], 1)


if __name__ == "__main__":
    unittest.main()
