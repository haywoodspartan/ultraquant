"""Reading until you know enough — and the measurement that it does not pay.

§11.104 built the loop the thesis asks for: consult shards in router
order, stop when the answer knows enough. It FAILED against a plain
constant. These pins hold the loop's mechanics, the read-accounting
that made the saving measurable, and the recorded verdict — because
a negative result that nothing pins is a negative result that gets
rediscovered.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from ultraquant.model.uncertainty import describe
from ultraquant.shards import enough
from ultraquant.shards.budget import ShardCache


@dataclass
class _Fake:
    """A pool whose experts are scripted, so the loop can be tested."""

    scripted: dict
    cache: ShardCache

    def shard_id(self, category: str) -> str:
        return f"expert:{category}"

    def predict_uncertain(self, category, features, floor=0.0,
                          margin_floor=0.0):
        if category not in self.scripted:
            raise KeyError(category)
        label, probs = self.scripted[category]
        # Consulting pages the shard in, exactly as the real pool does.
        self.cache.get(self.shard_id(category), lambda: ({"x": 1}, 8))
        return label, describe(probs, floor, margin_floor)


def _peaked(top: float, count: int = 4) -> list:
    rest = (1.0 - top) / (count - 1)
    return [top] + [rest] * (count - 1)


class LoopTests(unittest.TestCase):
    """What the loop reads, and when it stops."""

    def _pool(self, script: dict) -> _Fake:
        return _Fake(scripted=script, cache=ShardCache(1 << 20))

    def test_a_confident_first_expert_still_costs_a_confirming_read(self):
        """Run two's fix: the best answer must be confirmed, not assumed."""
        pool = self._pool({"a": ("alpha", _peaked(0.97)),
                           "b": ("beta", _peaked(0.30)),
                           "c": ("gamma", _peaked(0.30))})
        recall = enough.recall_until_enough(pool, ["a", "b", "c"],
                                            [0.0], floor=1.0)
        self.assertEqual(recall.label, "alpha")
        self.assertEqual(recall.stopped, "confident")
        self.assertEqual(recall.reads, 2,
                         "one read to find it, one to confirm nothing "
                         "better follows")

    def test_the_best_answer_wins_not_the_first_adequate_one(self) -> None:
        """The defect the sweep found in run one."""
        pool = self._pool({"a": ("alpha", _peaked(0.80)),
                           "b": ("beta", _peaked(0.99)),
                           "c": ("gamma", _peaked(0.30))})
        recall = enough.recall_until_enough(pool, ["a", "b", "c"],
                                            [0.0], floor=1.0)
        self.assertEqual(recall.label, "beta")

    def test_unread_experts_are_counted_as_spared(self) -> None:
        pool = self._pool({f"c{i}": ("alpha", _peaked(0.97))
                           for i in range(10)})
        recall = enough.recall_until_enough(
            pool, [f"c{i}" for i in range(10)], [0.0], floor=1.0)
        self.assertEqual(recall.available, 10)
        self.assertLess(recall.reads, 10)
        self.assertEqual(recall.spared, 10 - recall.reads)

    def test_a_resident_shard_is_not_counted_as_a_paid_read(self) -> None:
        """A saving measured in cache hits is not a saving."""
        pool = self._pool({"a": ("alpha", _peaked(0.97)),
                           "b": ("beta", _peaked(0.30))})
        first = enough.recall_until_enough(pool, ["a", "b"], [0.0],
                                           floor=1.0)
        self.assertEqual(first.paid_reads, first.reads)
        again = enough.recall_until_enough(pool, ["a", "b"], [0.0],
                                           floor=1.0)
        self.assertEqual(again.paid_reads, 0,
                         "everything was already resident")

    def test_nothing_confident_returns_no_label(self) -> None:
        """Running out of budget is not an answer."""
        pool = self._pool({f"c{i}": ("alpha", _peaked(0.28))
                           for i in range(4)})
        recall = enough.recall_until_enough(
            pool, [f"c{i}" for i in range(4)], [0.0], floor=1.9)
        self.assertIsNone(recall.label)
        self.assertIn(recall.stopped, ("plateau", "budget"))

    def test_a_missing_expert_is_skipped_not_fatal(self) -> None:
        pool = self._pool({"b": ("beta", _peaked(0.97))})
        recall = enough.recall_until_enough(pool, ["missing", "b"],
                                            [0.0], floor=1.0)
        self.assertEqual(recall.label, "beta")

    def test_max_reads_caps_the_spend(self) -> None:
        pool = self._pool({f"c{i}": ("alpha", _peaked(0.28))
                           for i in range(10)})
        recall = enough.recall_until_enough(
            pool, [f"c{i}" for i in range(10)], [0.0], floor=1.9,
            max_reads=3)
        self.assertLessEqual(recall.reads, 3)


class ResidencyTests(unittest.TestCase):
    """is_resident must not change what it is asked about."""

    def test_probing_does_not_page_anything_in(self) -> None:
        cache = ShardCache(1 << 20)
        self.assertFalse(cache.is_resident("x"))
        self.assertEqual(cache.stats()["misses"], 0)

    def test_probing_does_not_disturb_recency(self) -> None:
        cache = ShardCache(64)
        cache.get("a", lambda: ({"v": 1}, 16))
        cache.get("b", lambda: ({"v": 2}, 16))
        order_before = list(cache._entries)
        cache.is_resident("a")
        self.assertEqual(list(cache._entries), order_before)


class GateVerdictTests(unittest.TestCase):
    """The negative result, and the two halves it separates."""

    def setUp(self) -> None:
        from ultraquant.experiments import enough_gate
        self.doc = " ".join(enough_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Cost scales with the question",
                       "Accuracy holds against reading everything",
                       "It beats a fixed top-k matched to its own mean spend",
                       "The savings are real and reported",
                       "Stopping early is not giving up"):
            self.assertIn(phrase, self.doc)

    def test_the_sharding_result_is_recorded(self) -> None:
        """The half that passed, and it is the bigger claim."""
        self.assertIn("A quarter of the library buys all of the accuracy",
                      self.doc)

    def test_the_stopping_result_is_recorded_as_a_loss(self) -> None:
        self.assertIn("beaten by a plain constant", self.doc)
        self.assertIn("lost at **11 of 12**", self.doc)

    def test_the_unfair_comparison_is_corrected_not_kept(self) -> None:
        """round(2.54)=3 gave the constant a bigger budget."""
        self.assertIn("unfair in adaptive's FAVOUR", self.doc)
        self.assertIn("0.9823", self.doc)

    def test_the_limitation_is_stated_as_a_limitation(self) -> None:
        self.assertIn("stated as a limitation and not as a defence",
                      self.doc)
        self.assertIn("searching for a setting that passes", self.doc)


if __name__ == "__main__":
    unittest.main()
