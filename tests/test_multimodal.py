"""One library, more than one kind of thing in it.

The pool used to impose a single ``input_dim`` on every expert it created, so a
library could hold only one modality: 5x5 glyphs, 30 features, forever. Every
other piece of the architecture — cataloguing, sharding, paging, routing,
reinforcement, the archive — is indifferent to what a feature vector means, so
that limit was in one place and bought nothing.

These tests hold the generalisation in place. A library may hold experts over
different feature widths at once; a pattern reaches only experts that perceive
its kind; and a width mismatch is an error rather than a confident wrong answer.

This is a necessary step toward generality, not a sufficient one. Handling two
feature spaces is not the same as understanding two domains — nothing here
transfers anything learned in one modality to the other.
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

#: A second "modality": 12-dimensional vectors standing in for, say, token
#: statistics. Nothing about it resembles a 30-feature glyph.
WIDE = 12
GLYPHY = 30


def _blob(rng: random.Random, width: int, centre: float) -> list[float]:
    """A vector clustered around ``centre``."""
    return [min(1.0, max(0.0, rng.gauss(centre, 0.05))) for _ in range(width)]


class MixedWidthLibraryTests(unittest.TestCase):
    """Experts of different feature widths coexisting in one vault."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_multi_"))
        self.vault = ShardVault(self.dir)
        self.cache = ShardCache(4 * 1024 * 1024)
        self.pool = ExpertPool(self.vault, self.cache, input_dim=GLYPHY, hidden=(8,))
        self.router = CategoryRouter(self.vault)
        self.rng = random.Random(3)

        # Two categories over 30 features, two over 12.
        self.built: dict[str, int] = {}
        for category, width, centre in (
            ("shapes", GLYPHY, 0.8), ("marks", GLYPHY, 0.2),
            ("prose", WIDE, 0.75), ("numbers", WIDE, 0.25),
        ):
            labels = [f"{category}_a", f"{category}_b"]
            xs, ys = [], []
            for index, label_centre in enumerate((centre, 1.0 - centre)):
                for _ in range(24):
                    xs.append(_blob(self.rng, width, label_centre))
                    ys.append(index)
            self.pool.train_expert(category, xs, ys, labels, epochs=30)
            self.vault.set_signature(
                ExpertPool.shard_id(category), [_blob(self.rng, width, centre)]
            )
            self.built[category] = width

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- the library holds both --------------------------------------------

    def test_experts_keep_their_own_feature_width(self) -> None:
        for category, width in self.built.items():
            with self.subTest(category=category):
                config = self.vault.get(ExpertPool.shard_id(category))["net"]["config"]
                self.assertEqual(int(config["input_dim"]), width)

    def test_the_vault_reports_both_widths(self) -> None:
        self.assertEqual(set(self.vault.signature_widths()), {WIDE, GLYPHY})

    # -- and never confuses them -------------------------------------------

    def test_a_pattern_only_reaches_experts_that_perceive_its_kind(self) -> None:
        wide_query = _blob(self.rng, WIDE, 0.75)
        routed = [category for category, _ in self.router.route_pattern(wide_query, top_k=4)]
        self.assertTrue(routed)
        for category in routed:
            self.assertEqual(self.built[category], WIDE,
                             f"{category} takes {self.built[category]} features")

    def test_the_other_modality_is_reached_symmetrically(self) -> None:
        narrow_query = _blob(self.rng, GLYPHY, 0.8)
        routed = [category for category, _ in self.router.route_pattern(narrow_query, top_k=4)]
        self.assertTrue(routed)
        for category in routed:
            self.assertEqual(self.built[category], GLYPHY)

    def test_a_width_nothing_was_trained_on_routes_nowhere(self) -> None:
        self.assertEqual(self.router.route_pattern([0.5] * 7, top_k=4), [])

    def test_feeding_an_expert_the_wrong_modality_is_an_error(self) -> None:
        """Not a near-miss — a confident answer here would be meaningless."""
        with self.assertRaises(ValueError) as caught:
            self.pool.predict("prose", _blob(self.rng, GLYPHY, 0.8))
        self.assertIn("different modality", str(caught.exception))

    # -- the rest of the machinery is genuinely indifferent -----------------

    def test_each_modality_still_classifies_correctly(self) -> None:
        for category, width in self.built.items():
            with self.subTest(category=category):
                centre = 0.8 if category in ("shapes",) else (
                    0.2 if category == "marks" else
                    0.75 if category == "prose" else 0.25
                )
                label, _confidence = self.pool.predict(
                    category, _blob(self.rng, width, centre)
                )
                self.assertEqual(label, f"{category}_a")

    def test_paging_and_sharding_do_not_care_what_a_feature_means(self) -> None:
        library = self.dir / "mixed.uql"
        self.vault.pack(library, prune_loose=True)

        fresh = ShardVault(self.dir / "elsewhere")
        fresh.attach(library)
        self.assertEqual(set(fresh.signature_widths()), {WIDE, GLYPHY})

        pool = ExpertPool(fresh, ShardCache(1024 * 1024), input_dim=GLYPHY, hidden=(8,))
        label, _confidence = pool.predict("prose", _blob(self.rng, WIDE, 0.75))
        self.assertEqual(label, "prose_a")


if __name__ == "__main__":
    unittest.main()
