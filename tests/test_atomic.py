"""A store survives the crash that interrupts it (§11.176).

Every store here was once written in place: ``open(path, "w")`` truncates
first, so a process killed mid-write left a file the next start could not
parse. ``build_session`` and ``ArTchive(root)`` raised JSONDecodeError, and a
crash during the second compaction of a library destroyed every shard packed
into it. These pins inject a failure at the final rename and require the
previous file back, byte for byte, with no temporary file left behind.

The exam was frozen before the code (sha256 d46f0020..., Amendment A
corrected one keyword after run 1) and copied here verbatim.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _no_temp_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.rglob("*") if p.is_file() and ".tmp" in p.name)


class AtomicWriterTests(unittest.TestCase):
    """Criterion 2, and the helper's own contract."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_atomic_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_write_json_serialises_before_touching_disk(self) -> None:
        from ultraquant import atomic
        path = self.tmp / "store.json"
        atomic.write_json(path, {"a": 1})
        before = path.read_bytes()
        with self.assertRaises(TypeError):
            atomic.write_json(path, {"a": object()})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_no_temp_files(self.tmp), [])

    def test_a_failure_inside_replacing_leaves_the_previous_file(self) -> None:
        from ultraquant import atomic
        path = self.tmp / "blob.bin"
        atomic.write_bytes(path, b"first")
        with self.assertRaises(RuntimeError):
            with atomic.replacing(path, "wb") as handle:
                handle.write(b"half")
                raise RuntimeError("simulated crash mid-write")
        self.assertEqual(path.read_bytes(), b"first")
        self.assertEqual(_no_temp_files(self.tmp), [])
        atomic.write_bytes(path, b"second")
        self.assertEqual(path.read_bytes(), b"second")

    def test_write_text_lands_and_reads_back(self) -> None:
        from ultraquant import atomic
        path = self.tmp / "nested" / "note.txt"
        atomic.write_text(path, "café\n")
        self.assertEqual(path.read_text(encoding="utf-8"), "café\n")


class InterruptedStoreWritesTests(unittest.TestCase):
    """Criteria 1 and 3: a replace that fails leaves the previous file intact."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_interrupt_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _failing_replace(self):
        from ultraquant import atomic
        return mock.patch.object(atomic.os, "replace",
                                 side_effect=OSError("simulated crash at replace"))

    def test_memory_json(self) -> None:
        from ultraquant.memory.systematic import SystematicMemory
        path = self.tmp / "memory.json"
        mem = SystematicMemory(path=path)
        mem.remember_fact("sky", "blue")
        mem.save()
        before = path.read_bytes()
        mem.remember_fact("grass", "green")
        with self._failing_replace():
            with self.assertRaises(OSError):
                mem.save()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_no_temp_files(self.tmp), [])
        reloaded = SystematicMemory(path=path)
        self.assertEqual(reloaded.recall_fact("sky")["value"], "blue")
        self.assertIsNone(reloaded.recall_fact("grass"))

    def test_router_json(self) -> None:
        from ultraquant.shards.router import CategoryRouter
        from ultraquant.shards.vault import ShardVault
        vault = ShardVault(self.tmp / "vault")
        path = self.tmp / "router.json"
        router = CategoryRouter(vault, path=path)
        router.save()
        before = path.read_bytes()
        router._learned.setdefault("math", {})["algebra"] = 1.0
        with self._failing_replace():
            with self.assertRaises(OSError):
                router.save()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_no_temp_files(self.tmp), [])
        CategoryRouter(vault, path=path)          # still loads

    def test_archive_manifest_and_snapshot(self) -> None:
        from ultraquant.archive.artchive import ArTchive
        root = self.tmp / "arch"
        arc = ArTchive(root)
        arc.commit("a", {"x": 1})
        manifest = root / "manifest.json"
        before = manifest.read_bytes()
        with self._failing_replace():
            with self.assertRaises(OSError):
                arc.commit("b", {"x": 2})
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(_no_temp_files(root), [])
        self.assertEqual([e["id"] for e in ArTchive(root).versions()], ["T-0001"])
        self.assertEqual(ArTchive(root).restore("T-0001"), {"x": 1})

    def test_pack_onto_an_existing_library(self) -> None:
        from ultraquant.shards.vault import ShardVault
        vault = ShardVault(self.tmp / "vault")
        payloads = {f"s{i}": {"w": [i, i + 1]} for i in range(3)}
        for sid, payload in payloads.items():
            vault.add_shard(sid, "math", payload)
        lib = self.tmp / "lib.uql"
        vault.pack(lib, prune_loose=True)
        before = lib.read_bytes()
        with self._failing_replace():
            with self.assertRaises(OSError):
                vault.pack(lib)
        self.assertEqual(lib.read_bytes(), before)
        self.assertEqual(_no_temp_files(self.tmp), [])
        for sid, payload in payloads.items():
            self.assertEqual(vault.get(sid), payload)

    def test_dispatch_experience(self) -> None:
        from ultraquant.native.scheduler import DispatchExperience
        path = self.tmp / "dispatch.json"
        exp = DispatchExperience(path)
        exp.save()
        before = path.read_bytes()
        exp.verdicts["probe"] = "python"
        with self._failing_replace():
            with self.assertRaises(OSError):
                exp.save()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_no_temp_files(self.tmp), [])

    def test_distillation_ledger(self) -> None:
        from ultraquant.forge.train_from_llm import _catalogue_state, _save_catalogue_state
        home = self.tmp / "home"
        _save_catalogue_state(home, {"teachers": ["a"], "phrasings": {}})
        path = home / "vault" / "distilled.json"
        before = path.read_bytes()
        with self._failing_replace():
            with self.assertRaises(OSError):
                _save_catalogue_state(home, {"teachers": ["a", "b"], "phrasings": {}})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(_no_temp_files(home), [])
        self.assertEqual(_catalogue_state(home)["teachers"], ["a"])


if __name__ == "__main__":
    unittest.main()
