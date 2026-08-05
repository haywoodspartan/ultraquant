"""Sketch screening: it must be fast, compact, and not change the answer.

The screen exists because content routing had quietly reintroduced an O(library)
scan — measured at 128 ms per pattern on a 20,000-category library, against
7.6 us for the keyword path that has an inverted index behind it.

An approximate index earns its place only if its errors are bounded and known,
so the tests below are about *agreement with the exact scan*, not about the
screen in isolation. The screen may reorder work; it may not change results.
"""

from __future__ import annotations

import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.sketch import (
    SKETCH_BITS,
    estimate_cosine,
    hamming,
    hyperplanes,
    sketch,
)
from ultraquant.shards.vault import ShardVault


def _cosine(left: list[float], right: list[float]) -> float:
    """Exact cosine similarity."""
    dot = sum(a * b for a, b in zip(left, right))
    la = math.sqrt(sum(a * a for a in left))
    lb = math.sqrt(sum(b * b for b in right))
    return dot / (la * lb) if la and lb else 0.0


class SketchTests(unittest.TestCase):
    """The projection sketch itself."""

    def test_sketches_are_deterministic_across_calls(self) -> None:
        """A stored sketch must still mean the same thing next month."""
        rng = random.Random(1)
        vector = [rng.random() for _ in range(25)]
        self.assertEqual(sketch(vector), sketch(list(vector)))

    def test_hyperplanes_are_cached_per_dimension(self) -> None:
        self.assertIs(hyperplanes(25), hyperplanes(25))
        self.assertEqual(len(hyperplanes(25)), SKETCH_BITS)
        self.assertEqual(len(hyperplanes(25)[0]), 25)

    def test_identical_vectors_have_zero_distance(self) -> None:
        rng = random.Random(2)
        vector = [rng.random() for _ in range(25)]
        self.assertEqual(hamming(sketch(vector), sketch(vector)), 0)

    def test_scaling_a_vector_does_not_change_its_sketch(self) -> None:
        """Cosine is scale-free, so the sketch of it must be too."""
        rng = random.Random(3)
        vector = [rng.random() for _ in range(25)]
        self.assertEqual(sketch(vector), sketch([v * 7.5 for v in vector]))

    def test_an_empty_vector_sketches_to_zero(self) -> None:
        self.assertEqual(sketch([]), 0)

    def test_distance_tracks_cosine_similarity(self) -> None:
        """The estimate is coarse, but it must not be biased.

        With 64 bits the standard error is about 1/sqrt(64) = 0.125, so the
        assertion is deliberately loose: what matters is that closer vectors
        really do sketch closer, which is the whole basis of the screen.
        """
        rng = random.Random(4)
        errors = []
        for _ in range(200):
            a = [rng.gauss(0, 1) for _ in range(25)]
            b = [x + rng.gauss(0, rng.choice([0.1, 0.5, 1.5])) for x in a]
            errors.append(estimate_cosine(hamming(sketch(a), sketch(b))) - _cosine(a, b))
        mean = sum(errors) / len(errors)
        self.assertLess(abs(mean), 0.05, f"estimator is biased: {mean:+.3f}")
        self.assertLess(max(abs(e) for e in errors), 0.45)

    def test_closer_vectors_sketch_closer_on_average(self) -> None:
        rng = random.Random(5)
        near, far = [], []
        for _ in range(150):
            a = [rng.gauss(0, 1) for _ in range(25)]
            near.append(hamming(sketch(a), sketch([x + rng.gauss(0, 0.1) for x in a])))
            far.append(hamming(sketch(a), sketch([rng.gauss(0, 1) for _ in range(25)])))
        self.assertLess(sum(near) / len(near), sum(far) / len(far))


class ScreenAgreesWithExactTests(unittest.TestCase):
    """The screen must not change what routing decides."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_sketch_"))
        self.vault = ShardVault(self.dir)
        self.router = CategoryRouter(self.vault)
        self.rng = random.Random(11)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fill(self, count: int, prototypes: int = 3) -> None:
        with self.vault.batch():
            for i in range(count):
                shard_id = f"expert:c{i}"
                self.vault.add_shard(shard_id, f"c{i}", {"w": [0.0]}, kind="expert-net")
                self.vault.set_signature(shard_id, [
                    [self.rng.random() for _ in range(25)] for _ in range(prototypes)
                ])

    def _queries(self, n: int) -> list[list[float]]:
        """Perturbed copies of stored prototypes, so a true nearest exists."""
        protos = [entry["signature"][0] for entry in self.vault.catalog()]
        return [
            [min(1.0, max(0.0, x + self.rng.gauss(0, 0.06)))
             for x in self.rng.choice(protos)]
            for _ in range(n)
        ]

    def test_a_small_library_is_scored_exactly_and_not_screened(self) -> None:
        """The common case must carry no approximation risk at all."""
        self._fill(20)
        pixels = self._queries(1)[0]
        screened = self.vault.screen_signatures(pixels, limit=4)
        self.assertEqual(len(screened), 20, "small libraries are returned whole")

    def test_screened_routing_matches_exact_routing_at_scale(self) -> None:
        self._fill(1200)
        for pixels in self._queries(40):
            exact = self.router.route_pattern(pixels, top_k=1, screen=10 ** 9)
            screened = self.router.route_pattern(pixels, top_k=1)
            self.assertEqual(exact[0][0], screened[0][0])
            self.assertAlmostEqual(exact[0][1], screened[0][1], places=9)

    def test_the_similarity_reported_is_the_true_cosine(self) -> None:
        """The sketch chooses what to score; it never becomes the score."""
        self._fill(1200)
        pixels = self._queries(1)[0]
        category, similarity = self.router.route_pattern(pixels, top_k=1)[0]
        best = max(
            _cosine(pixels, vector)
            for entry in self.vault.catalog()
            if entry["category"] == category
            for vector in entry["signature"]
        )
        self.assertAlmostEqual(similarity, best, places=9)

    def test_the_index_is_small_next_to_what_it_screens(self) -> None:
        """An index bigger than its data would be worse than no index."""
        self._fill(1200)
        signature_floats = sum(
            len(vector) for entry in self.vault.catalog()
            for vector in entry.get("signature") or []
        )
        self.assertLess(self.vault.sketch_index_bytes(), signature_floats * 8 / 4)

    # -- staleness is the failure mode of any derived index ----------------

    def test_a_new_signature_is_visible_immediately(self) -> None:
        self._fill(600)
        pixels = [1.0] * 25
        self.vault.add_shard("expert:exact", "exact", {"w": [0.0]}, kind="expert-net")
        self.vault.set_signature("expert:exact", [pixels])
        top = self.router.route_pattern(pixels, top_k=1)
        self.assertEqual(top[0][0], "exact")
        self.assertAlmostEqual(top[0][1], 1.0, places=9)

    def test_retraining_a_shard_does_not_strand_the_index(self) -> None:
        self._fill(600)
        pixels = [1.0] * 25
        self.vault.add_shard("expert:exact", "exact", {"w": [0.0]}, kind="expert-net")
        self.vault.set_signature("expert:exact", [pixels])
        self.router.route_pattern(pixels, top_k=1)  # build the index

        self.vault.add_shard("expert:exact", "exact", {"w": [1.0]}, kind="expert-net")
        top = self.router.route_pattern(pixels, top_k=1)
        self.assertEqual(top[0][0], "exact", "the signature survives a re-add")

    def test_reopening_the_vault_rebuilds_the_index(self) -> None:
        self._fill(600)
        pixels = [1.0] * 25
        self.vault.add_shard("expert:exact", "exact", {"w": [0.0]}, kind="expert-net")
        self.vault.set_signature("expert:exact", [pixels])

        reopened = ShardVault(self.dir)
        top = CategoryRouter(reopened).route_pattern(pixels, top_k=1)
        self.assertEqual(top[0][0], "exact")


if __name__ == "__main__":
    unittest.main()
