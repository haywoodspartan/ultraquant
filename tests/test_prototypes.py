"""Prototype consolidation: episodic traces becoming semantic prototypes.

Every lesson appends what it saw. Measured at 200 lessons into one category
that meant 202 prototypes, 25.6 KB of catalog-resident signature, and 377 us
per routing call. Consolidation clusters the traces (Stage 3's machinery — k
chosen by silhouette, never supplied) and keeps **medoids**, real stored
traces, because mean-pooling was measured harmful in §4.1.1.

Also recorded here in test form: an honest correction. The first stress driver
mislabelled every lesson (rows and labels inverted), which produced a
perfectly confident label swap that was misdiagnosed as catastrophic drift. A
rehearsal mechanism built against that phantom was A/B measured in both the
alternating and the one-label regime, changed nothing in either, and was
deleted — the same standard that removed the escalation heuristic.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.shards.prototypes import PROTOTYPE_BUDGET, consolidate_signatures
from ultraquant.shards.vault import ShardVault


def _vector(rng: random.Random, centre: float) -> list[float]:
    return [min(1.0, max(0.0, rng.gauss(centre, 0.05))) for _ in range(25)]


class ConsolidateSignaturesTests(unittest.TestCase):
    """The mechanics, on a bare vault."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_protos_"))
        self.vault = ShardVault(self.dir)
        self.rng = random.Random(0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _grow(self, shard: str, category: str, centres: list[float],
              per_centre: int) -> list[list[float]]:
        self.vault.add_shard(shard, category, {"w": [0.0]}, kind="expert-net")
        traces = [
            _vector(self.rng, centre)
            for centre in centres for _ in range(per_centre)
        ]
        self.vault.set_signature(shard, traces)
        return traces

    def test_an_oversized_signature_is_reduced_to_the_budget(self) -> None:
        self._grow("expert:a", "a", [0.2, 0.8], 40)
        report = consolidate_signatures(self.vault)
        self.assertEqual(report["categories"], 1)
        self.assertEqual(report["before"], 80)
        kept = self.vault.entry("expert:a")["signature"]
        self.assertLessEqual(len(kept), PROTOTYPE_BUDGET)
        self.assertEqual(report["after"], len(kept))

    def test_kept_prototypes_are_real_traces_not_means(self) -> None:
        """Mean-pooling measured 7/8 vs 8/8 in §4.1.1; medoids or nothing."""
        traces = self._grow("expert:a", "a", [0.2, 0.8], 40)
        consolidate_signatures(self.vault)
        originals = {tuple(round(v, 4) for v in t) for t in traces}
        for kept in self.vault.entry("expert:a")["signature"]:
            self.assertIn(tuple(round(v, 4) for v in kept), originals,
                          "a synthetic average crept into the signature")

    def test_every_cluster_keeps_a_representative(self) -> None:
        """Two well-separated groups must both survive consolidation."""
        from ultraquant.reason.discovery import distance

        self._grow("expert:a", "a", [0.15, 0.85], 40)
        consolidate_signatures(self.vault)
        kept = self.vault.entry("expert:a")["signature"]
        near_low = any(distance(k, [0.15] * 25) < 0.5 for k in kept)
        near_high = any(distance(k, [0.85] * 25) < 0.5 for k in kept)
        self.assertTrue(near_low and near_high,
                        "consolidation dropped a whole mode of the category")

    def test_a_signature_within_budget_is_untouched(self) -> None:
        traces = self._grow("expert:a", "a", [0.5], PROTOTYPE_BUDGET)[:PROTOTYPE_BUDGET]
        before = [list(v) for v in self.vault.entry("expert:a")["signature"]]
        report = consolidate_signatures(self.vault)
        self.assertEqual(report["categories"], 0)
        self.assertEqual(self.vault.entry("expert:a")["signature"], before)
        self.assertEqual(len(traces), PROTOTYPE_BUDGET)

    def test_consolidation_is_idempotent(self) -> None:
        self._grow("expert:a", "a", [0.2, 0.8], 40)
        consolidate_signatures(self.vault)
        once = [list(v) for v in self.vault.entry("expert:a")["signature"]]
        report = consolidate_signatures(self.vault)
        self.assertEqual(report["categories"], 0)
        self.assertEqual(self.vault.entry("expert:a")["signature"], once)


class TeachingGrowthTests(unittest.TestCase):
    """The live scenario: a long teaching run must not degrade the library."""

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.forge.corpus import (
            BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
        )
        from ultraquant.forge.forge import ModelForge
        from ultraquant.interpreter.selflearn import SelfLearner
        from ultraquant.interpreter.thoughts import build_session
        from ultraquant.pattern.recognition import PATTERNS
        from ultraquant.experts.moe import ExpertPool

        cls.dir = Path(tempfile.mkdtemp(prefix="uq_protogrow_"))
        forge = ModelForge(cls.dir / "home", seed=0, hidden=16, tier="auto")
        forge.build(
            build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                          keywords=BUILTIN_KEYWORDS),
            epochs=25,
        )
        cls.session = build_session(cls.dir / "home", seed=0)
        rng = random.Random(0)
        labels = list(cls.session.vault.get(
            ExpertPool.shard_id("crosses"))["labels"])
        learner = SelfLearner(cls.session)
        for i in range(60):
            glyph = "cross" if i % 2 else "plus"
            rows = list(PATTERNS[glyph])
            r, c = rng.randrange(5), rng.randrange(5)
            row = list(rows[r])
            row[c] = "#" if row[c] == "." else "."
            rows[r] = "".join(row)
            learner.teach_glyph("crosses", labels, rows, glyph,
                                epochs=3, samples=6)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_tripwire_keeps_prototypes_bounded(self) -> None:
        from ultraquant.experts.moe import ExpertPool

        signature = self.session.vault.entry(
            ExpertPool.shard_id("crosses")).get("signature") or []
        self.assertLessEqual(len(signature), 2 * PROTOTYPE_BUDGET,
                             "auto-consolidation never fired during teaching")

    def test_the_whole_library_still_routes_after_the_marathon(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline
        from ultraquant.pattern.recognition import PATTERNS

        correct = sum(
            f"'{glyph}'" in run_pipeline("\n".join(PATTERNS[glyph]),
                                         self.session)[0]
            for glyph in PATTERNS
        )
        self.assertEqual(correct, len(PATTERNS))

    def test_noisy_copies_of_the_taught_category_still_clear_the_floor(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline
        from ultraquant.pattern.recognition import PATTERNS

        rng = random.Random(7)
        accepted = 0
        for trial in range(20):
            rows = list(PATTERNS["plus" if trial % 2 else "cross"])
            r, c = rng.randrange(5), rng.randrange(5)
            row = list(rows[r])
            row[c] = "#" if row[c] == "." else "."
            rows[r] = "".join(row)
            response, _ = run_pipeline("\n".join(rows), self.session)
            accepted += "do not recognise" not in response
        self.assertEqual(accepted, 20)

    def test_consolidation_runs_inside_the_housekeeping_pass(self) -> None:
        from ultraquant.interpreter.selflearn import SelfLearner

        report = SelfLearner(self.session).consolidate()
        self.assertIn("prototypes", report)


if __name__ == "__main__":
    unittest.main()
