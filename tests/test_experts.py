"""Tests for :mod:`ultraquant.experts.moe` (ExpertPool) and the associative
:class:`ultraquant.shards.router.CategoryRouter`.

Covers, per SPEC-INTERPRETER Section 11:

- ensure/predict on-demand load (cache miss then hit);
- ``train_expert`` persistence (a fresh pool + fresh cache over the same vault
  reloads the trained weights and keeps identical accuracy);
- two experts under a one-expert budget swap correctly (LRU eviction);
- router register + route ranks the right category, and ``learn`` flips a tie.

Everything is seeded and deterministic; temp dirs are cleaned up.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardVault

_HUGE = 1_000_000_000  # effectively-unbounded cache budget


def _stored_nbytes(vault: ShardVault, shard_id: str) -> int:
    """Read a shard's stored (compressed) size from the vault catalog."""
    for entry in vault.catalog():
        if entry["shard_id"] == shard_id:
            return int(entry["nbytes"])
    raise KeyError(shard_id)


def _toy_dataset(
    num_classes: int = 3, per_class: int = 12, dim: int = 6, seed: int = 1
) -> tuple[list[list[float]], list[int]]:
    """Build a small, linearly separable one-hot-plus-noise dataset.

    Class ``c`` places a unit bump at coordinate ``c`` with small seeded noise
    on every coordinate, so the classes are cleanly separable.
    """
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for c in range(num_classes):
        for _ in range(per_class):
            vec = [rng.uniform(-0.12, 0.12) for _ in range(dim)]
            vec[c % dim] += 1.0
            xs.append(vec)
            ys.append(c)
    return xs, ys


class ExpertPoolTest(unittest.TestCase):
    """On-demand paging, persistence and budgeted swapping of experts."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_experts_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ensure_and_predict_on_demand(self) -> None:
        """First predict is a cache miss (paged in); the second is a hit."""
        vault = ShardVault(self.tmp / "vault")
        cache = ShardCache(_HUGE)
        pool = ExpertPool(vault, cache, input_dim=6, hidden=(8,), seed=0)

        entry = pool.ensure_expert("energy", ["low", "high"])
        self.assertEqual(entry["shard_id"], "expert:energy")
        # ensure_expert is idempotent: no duplicate shard is created.
        pool.ensure_expert("energy", ["low", "high"])
        experts = [e for e in vault.catalog() if e["shard_id"] == "expert:energy"]
        self.assertEqual(len(experts), 1)

        feats = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]
        base = cache.stats()
        self.assertEqual(base["hits"], 0)
        self.assertEqual(base["misses"], 0)

        label, conf = pool.predict("energy", feats)
        self.assertIn(label, {"low", "high"})
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

        after_miss = cache.stats()
        self.assertEqual(after_miss["misses"], 1)
        self.assertEqual(after_miss["hits"], 0)
        self.assertIn("expert:energy", cache.resident())

        label2, conf2 = pool.predict("energy", feats)
        after_hit = cache.stats()
        self.assertEqual(after_hit["misses"], 1)  # no new page-in
        self.assertEqual(after_hit["hits"], 1)  # served from RAM
        self.assertEqual(label, label2)
        self.assertAlmostEqual(conf, conf2, places=12)

    def test_predict_without_expert_raises(self) -> None:
        """Predicting an unknown category is a clear error, not a silent load."""
        vault = ShardVault(self.tmp / "vault")
        cache = ShardCache(_HUGE)
        pool = ExpertPool(vault, cache, input_dim=6, hidden=(8,), seed=0)
        with self.assertRaises(KeyError):
            pool.predict("missing", [0.0] * 6)

    def test_train_expert_persists(self) -> None:
        """A fresh pool + fresh cache over the same vault reloads trained weights.

        Because ``load_state_dict`` reproduces identical predictions, the fresh
        pool's accuracy must exactly equal the accuracy ``train_expert``
        reported, and both must clear the learnability bar.
        """
        root = self.tmp / "vault"
        vault = ShardVault(root)
        cache = ShardCache(_HUGE)
        pool = ExpertPool(vault, cache, input_dim=6, hidden=(16,), seed=0)

        labels = ["c0", "c1", "c2"]
        xs, ys = _toy_dataset(num_classes=3, per_class=12, dim=6, seed=1)
        result = pool.train_expert("shapes", xs, ys, labels, epochs=80, lr=0.1)

        self.assertIn("loss", result)
        self.assertIn("accuracy", result)
        self.assertGreaterEqual(result["accuracy"], 0.8)
        self.assertGreaterEqual(result["loss"], 0.0)

        # trained_examples was recorded on the persisted shard.
        stored = vault.get("expert:shapes")
        self.assertEqual(stored["trained_examples"], len(xs))
        self.assertEqual(stored["labels"], labels)

        # Fresh pool + fresh cache, same on-disk vault (catalog auto-loaded).
        vault2 = ShardVault(root)
        cache2 = ShardCache(_HUGE)
        pool2 = ExpertPool(vault2, cache2, input_dim=6, hidden=(16,), seed=0)

        correct = 0
        for x, y in zip(xs, ys):
            label, _ = pool2.predict("shapes", x)
            if label == labels[y]:
                correct += 1
        fresh_acc = correct / len(xs)

        self.assertAlmostEqual(fresh_acc, result["accuracy"], places=9)
        self.assertGreaterEqual(fresh_acc, 0.8)

    def test_two_experts_swap_under_one_expert_budget(self) -> None:
        """With room for exactly one expert, alternating predicts swap them."""
        vault = ShardVault(self.tmp / "vault")
        cache = ShardCache(_HUGE)
        pool = ExpertPool(vault, cache, input_dim=6, hidden=(8,), seed=0)

        pool.ensure_expert("alpha", ["a", "b"])
        pool.ensure_expert("beta", ["a", "b"])

        size_a = _stored_nbytes(vault, "expert:alpha")
        size_b = _stored_nbytes(vault, "expert:beta")
        budget = max(size_a, size_b)  # fits one expert, never two
        cache.set_budget(budget)

        feats = [0.1, 0.2, -0.1, 0.0, 0.3, -0.2]
        pool.predict("alpha", feats)
        self.assertEqual(cache.resident(), ["expert:alpha"])
        pool.predict("beta", feats)
        self.assertEqual(cache.resident(), ["expert:beta"])
        pool.predict("alpha", feats)
        self.assertEqual(cache.resident(), ["expert:alpha"])

        stats = cache.stats()
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["evictions"], 2)
        # The budget must never have been exceeded (paging invariant).
        self.assertLessEqual(stats["current_bytes"], budget)
        self.assertLessEqual(stats["peak_bytes"], budget)


class CategoryRouterTest(unittest.TestCase):
    """Associative routing: register + route ranking, and learn flipping a tie."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_router_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_route_ranks_registered_category(self) -> None:
        """A text full of one category's keywords routes to that category."""
        vault = ShardVault(self.tmp / "vault")
        router = CategoryRouter(vault)
        router.register("weather", ["rain", "sun", "cloud"])
        router.register("finance", ["stock", "market", "price"])

        ranked = router.route("the stock market set a new price today")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], "finance")

    def test_learn_flips_a_tie(self) -> None:
        """Two categories tie on a shared keyword; ``learn`` breaks the tie."""
        vault = ShardVault(self.tmp / "vault")
        router = CategoryRouter(vault)
        router.register("aa_cat", ["signal"])
        router.register("zz_cat", ["signal"])

        text = "a strong signal appeared"
        ranked = router.route(text)
        scores = {cat: score for cat, score in ranked}
        self.assertIn("aa_cat", scores)
        self.assertIn("zz_cat", scores)
        # Genuine tie before learning.
        self.assertAlmostEqual(scores["aa_cat"], scores["zz_cat"], places=12)

        router.learn(text, "zz_cat")
        ranked_after = router.route(text)
        scores_after = {cat: score for cat, score in ranked_after}
        self.assertEqual(ranked_after[0][0], "zz_cat")
        self.assertGreater(scores_after["zz_cat"], scores_after["aa_cat"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
