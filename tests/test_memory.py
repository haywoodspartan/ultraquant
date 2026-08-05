"""Tests for ultraquant.memory.systematic (SystematicMemory)."""

from __future__ import annotations

import os
import tempfile
import unittest

from ultraquant.memory.systematic import SystematicMemory


class TestEpisodic(unittest.TestCase):
    def test_ids_increment_and_fields(self) -> None:
        mem = SystematicMemory()
        first = mem.remember_episode("obs", {"x": 1}, tags=["a"])
        second = mem.remember_episode("obs", {"x": 2})
        self.assertEqual(second, first + 1)
        episodes = mem.recall_episodes()
        self.assertEqual(len(episodes), 2)
        newest = episodes[0]
        self.assertEqual(newest["id"], second)
        self.assertEqual(newest["kind"], "obs")
        self.assertEqual(newest["content"], {"x": 2})
        self.assertEqual(newest["tags"], [])
        self.assertIn("t", newest)

    def test_recall_filters_and_limit(self) -> None:
        mem = SystematicMemory()
        mem.remember_episode("a", {"i": 0}, tags=["red"])
        mem.remember_episode("b", {"i": 1}, tags=["blue"])
        mem.remember_episode("a", {"i": 2}, tags=["red", "blue"])

        by_kind = mem.recall_episodes(kind="a")
        self.assertEqual([e["content"]["i"] for e in by_kind], [2, 0])

        by_tag = mem.recall_episodes(tags=["blue"])
        self.assertEqual([e["content"]["i"] for e in by_tag], [2, 1])

        limited = mem.recall_episodes(limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["content"]["i"], 2)

    def test_working_memory_bounded_fifo_oldest_first(self) -> None:
        mem = SystematicMemory(working_capacity=3)
        for i in range(5):
            mem.remember_episode("k", {"i": i})
        working = mem.working()
        self.assertEqual(len(working), 3)
        self.assertEqual([e["content"]["i"] for e in working], [2, 3, 4])


class TestSemantic(unittest.TestCase):
    def test_new_fact_stored(self) -> None:
        mem = SystematicMemory()
        mem.remember_fact("sky", "blue", confidence=0.4)
        fact = mem.recall_fact("sky")
        self.assertIsNotNone(fact)
        assert fact is not None
        self.assertEqual(fact["value"], "blue")
        self.assertAlmostEqual(fact["confidence"], 0.4)
        self.assertEqual(fact["reinforcements"], 0)
        self.assertIn("first_seen", fact)
        self.assertIn("last_seen", fact)

    def test_reinforcement_raises_confidence_and_count(self) -> None:
        mem = SystematicMemory()
        mem.remember_fact("sky", "blue", confidence=0.5)
        mem.remember_fact("sky", "blue")
        fact = mem.recall_fact("sky")
        assert fact is not None
        self.assertAlmostEqual(fact["confidence"], 0.6)
        self.assertEqual(fact["reinforcements"], 1)

        mem.remember_fact("sky", "blue")
        fact = mem.recall_fact("sky")
        assert fact is not None
        self.assertAlmostEqual(fact["confidence"], 0.7)
        self.assertEqual(fact["reinforcements"], 2)

    def test_confidence_capped_at_one(self) -> None:
        mem = SystematicMemory()
        mem.remember_fact("k", 1, confidence=0.95)
        for _ in range(5):
            mem.remember_fact("k", 1)
        fact = mem.recall_fact("k")
        assert fact is not None
        self.assertAlmostEqual(fact["confidence"], 1.0)
        self.assertEqual(fact["reinforcements"], 5)

    def test_conflicting_fact_logs_revision_episode(self) -> None:
        mem = SystematicMemory()
        mem.remember_fact("sky", "blue", confidence=0.9)
        mem.remember_fact("sky", "grey", confidence=0.3)

        fact = mem.recall_fact("sky")
        assert fact is not None
        self.assertEqual(fact["value"], "grey")
        self.assertAlmostEqual(fact["confidence"], 0.3)

        revisions = mem.recall_episodes(kind="revision")
        self.assertEqual(len(revisions), 1)
        content = revisions[0]["content"]
        self.assertEqual(content["key"], "sky")
        self.assertEqual(content["old_value"], "blue")
        self.assertEqual(content["new_value"], "grey")

    def test_recall_missing_fact_is_none(self) -> None:
        mem = SystematicMemory()
        self.assertIsNone(mem.recall_fact("nope"))


class TestSignatures(unittest.TestCase):
    def test_nearest_signature_exact_and_near(self) -> None:
        mem = SystematicMemory()
        mem.store_signature("plus", [0, 1, 0, 1])
        mem.store_signature("cross", [1, 0, 1, 0])

        label, sim = mem.nearest_signature([0, 1, 0, 1])
        self.assertEqual(label, "plus")
        self.assertAlmostEqual(sim, 1.0)

        label, sim = mem.nearest_signature([1, 0, 1, 1])
        self.assertEqual(label, "cross")
        self.assertAlmostEqual(sim, 0.75)

    def test_nearest_signature_skips_length_mismatch(self) -> None:
        mem = SystematicMemory()
        mem.store_signature("short", [1, 1])
        self.assertIsNone(mem.nearest_signature([1, 1, 1]))
        mem.store_signature("long", [1, 0, 1])
        label, sim = mem.nearest_signature([1, 1, 1])
        self.assertEqual(label, "long")
        self.assertAlmostEqual(sim, 2.0 / 3.0)

    def test_nearest_signature_empty_store(self) -> None:
        mem = SystematicMemory()
        self.assertIsNone(mem.nearest_signature([0, 1]))


class TestPersistence(unittest.TestCase):
    def test_save_load_round_trip_preserves_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            mem = SystematicMemory(path=path, working_capacity=4)
            mem.remember_episode("obs", {"x": 1}, tags=["a"])
            mem.remember_episode("obs", {"x": 2})
            mem.remember_fact("sky", "blue", confidence=0.5)
            mem.remember_fact("sky", "blue")
            mem.store_signature("plus", [0, 1, 0, 1])
            mem.save()

            reloaded = SystematicMemory(path=path, working_capacity=4)
            self.assertEqual(reloaded.stats(), mem.stats())

            fact = reloaded.recall_fact("sky")
            assert fact is not None
            self.assertAlmostEqual(fact["confidence"], 0.6)
            self.assertEqual(fact["reinforcements"], 1)

            result = reloaded.nearest_signature([0, 1, 0, 1])
            assert result is not None
            self.assertEqual(result[0], "plus")

            episodes = reloaded.recall_episodes(kind="obs")
            self.assertEqual([e["content"]["x"] for e in episodes], [2, 1])
            self.assertEqual(
                [e["content"]["x"] for e in reloaded.working()], [1, 2]
            )

    def test_ids_continue_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            mem = SystematicMemory(path=path)
            first = mem.remember_episode("k", {})
            mem.save()
            reloaded = SystematicMemory(path=path)
            second = reloaded.remember_episode("k", {})
            self.assertEqual(second, first + 1)

    def test_save_without_path_raises(self) -> None:
        mem = SystematicMemory()
        with self.assertRaises(ValueError):
            mem.save()

    def test_missing_path_does_not_autoload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does_not_exist.json")
            mem = SystematicMemory(path=path)
            self.assertEqual(mem.stats()["episodic"], 0)


class TestStats(unittest.TestCase):
    def test_counts_per_store(self) -> None:
        mem = SystematicMemory(working_capacity=2)
        mem.remember_episode("a", {})
        mem.remember_episode("b", {})
        mem.remember_episode("c", {})
        mem.remember_fact("k", 1)
        mem.store_signature("s", [1, 0])
        stats = mem.stats()
        self.assertEqual(stats["episodic"], 3)
        self.assertEqual(stats["semantic"], 1)
        self.assertEqual(stats["signatures"], 1)
        self.assertEqual(stats["working"], 2)


if __name__ == "__main__":
    unittest.main()
