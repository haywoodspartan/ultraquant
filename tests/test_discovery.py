"""Concept discovery: the model choosing what is worth calling a category.

Every other kind of learning here starts from a name a human supplied. This is
the one where the model looks at what it could not place, decides some of it
belongs together, and asks only what to call the group.

The gate's verdict lives in ARCHITECTURE.md §11.8. What is asserted here is the
machinery and, more importantly, the properties that keep the measurement
honest: that ``k`` is never supplied, that the score is chance-corrected and
invariant to how clusters are numbered, and that the system stays quiet rather
than inventing categories out of noise.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments import discovery_gate as gate
from ultraquant.reason.discovery import (
    SILHOUETTE_FLOOR,
    ConceptDiscovery,
    adjusted_rand_index,
    discover,
    kmeans,
    silhouette,
)


class AdjustedRandTests(unittest.TestCase):
    """The score has to be invariant to naming and corrected for chance."""

    def test_identical_groupings_score_one(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(adjusted_rand_index(labels, labels), 1.0)

    def test_renaming_clusters_changes_nothing(self) -> None:
        """Cluster 0 here and cluster 3 there may be the same set."""
        truth = [0, 0, 1, 1, 2, 2]
        renamed = [7, 7, 3, 3, 9, 9]
        self.assertAlmostEqual(adjusted_rand_index(truth, renamed), 1.0)

    def test_unrelated_groupings_score_about_zero(self) -> None:
        rng = random.Random(0)
        scores = []
        for _ in range(30):
            truth = [rng.randrange(4) for _ in range(80)]
            other = [rng.randrange(4) for _ in range(80)]
            scores.append(adjusted_rand_index(truth, other))
        mean = sum(scores) / len(scores)
        self.assertLess(abs(mean), 0.05, f"chance correction is off: {mean:+.3f}")

    def test_everything_in_one_cluster_is_not_rewarded(self) -> None:
        """The uncorrected Rand index rewards this; the adjusted one must not."""
        truth = [0, 0, 1, 1, 2, 2]
        lumped = [0] * 6
        self.assertLess(adjusted_rand_index(truth, lumped), 0.05)

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            adjusted_rand_index([0, 1], [0, 1, 2])


class ClusteringTests(unittest.TestCase):
    """k-means and the criterion that picks k."""

    def _blobs(self, seed: int = 0):
        rng = random.Random(seed)
        vectors, labels = [], []
        for label, centre in enumerate(([0.0, 0.0], [8.0, 8.0], [0.0, 8.0])):
            for _ in range(12):
                vectors.append([centre[0] + rng.gauss(0, 0.4),
                                centre[1] + rng.gauss(0, 0.4)])
                labels.append(label)
        return vectors, labels

    def test_it_separates_obvious_blobs(self) -> None:
        vectors, labels = self._blobs()
        assignment, _inertia = kmeans(vectors, 3, seed=0)
        self.assertAlmostEqual(adjusted_rand_index(labels, assignment), 1.0)

    def test_clustering_is_reproducible(self) -> None:
        vectors, _labels = self._blobs()
        self.assertEqual(kmeans(vectors, 3, seed=1), kmeans(vectors, 3, seed=1))

    def test_silhouette_prefers_the_right_number_of_groups(self) -> None:
        vectors, _labels = self._blobs()
        scores = {
            k: silhouette(vectors, kmeans(vectors, k, seed=0)[0])
            for k in (2, 3, 4, 5)
        }
        self.assertEqual(max(scores, key=scores.get), 3)

    def test_discover_is_never_told_how_many_groups_there_are(self) -> None:
        """Supplying k would leak the answer into every result downstream."""
        vectors, labels = self._blobs()
        found = discover(vectors, k_range=(2, 8), seed=0)
        self.assertEqual(found["k"], 3)
        self.assertAlmostEqual(adjusted_rand_index(labels, found["assignment"]), 1.0)

    def test_discover_survives_too_little_data(self) -> None:
        self.assertEqual(discover([[0.0], [1.0]])["k"], 1)


class SilhouetteFloorTests(unittest.TestCase):
    """The system must decline to invent categories."""

    def test_the_floor_lies_between_structure_and_noise(self) -> None:
        """Chosen from measurement, so the data has to keep supporting it.

        Measured over 20 datasets each: real structure never scored below
        0.314 and structureless data never above 0.104.
        """
        real, none = [], []
        for seed in range(6):
            rng = random.Random(seed)
            vectors, _labels = gate.make_families(rng, classes=4, flips=3)
            real.append(discover(vectors, seed=seed)["silhouette"])
            rng2 = random.Random(1000 + seed)
            flat, _ = gate.make_families(rng2, classes=4, flips=3, structured=False)
            none.append(discover(flat, seed=seed)["silhouette"])
        self.assertGreater(min(real), SILHOUETTE_FLOOR)
        self.assertLess(max(none), SILHOUETTE_FLOOR)


class ConceptDiscoveryTests(unittest.TestCase):
    """Proposing concepts from what a session actually met."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_discover_"))
        from ultraquant.forge.corpus import (
            BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
        )
        from ultraquant.forge.forge import ModelForge
        from ultraquant.interpreter.thoughts import build_session

        forge = ModelForge(self.dir / "home", seed=0, hidden=16, tier="auto")
        forge.build(
            build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                          keywords=BUILTIN_KEYWORDS),
            epochs=25,
        )
        self.session = build_session(self.dir / "home", seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _rows(lit: set[int]) -> list[str]:
        return ["".join("#" if r * 5 + c in lit else "." for c in range(5))
                for r in range(5)]

    def _show_two_hidden_families(self, count: int = 14) -> None:
        """Feed patterns from two families the library was never taught."""
        from ultraquant.interpreter.thoughts import run_pipeline

        rng = random.Random(0)
        first = set(gate.STROKES["slash"]) | set(gate.STROKES["top"])
        second = set(gate.STROKES["left"]) | set(gate.STROKES["bottom"])
        for family in (first, second):
            for _ in range(count):
                lit = set(family)
                for position in rng.sample(range(25), 3):
                    lit.symmetric_difference_update({position})
                run_pipeline("\n".join(self._rows(lit)), self.session)

    def test_nothing_is_proposed_without_observations(self) -> None:
        self.assertEqual(ConceptDiscovery(self.session).propose(), [])

    def test_it_proposes_one_group_per_hidden_family(self) -> None:
        self._show_two_hidden_families()
        proposals = ConceptDiscovery(self.session).propose()
        self.assertEqual(len(proposals), 2)
        for proposal in proposals:
            self.assertGreaterEqual(proposal["size"], 4)
            self.assertGreaterEqual(proposal["silhouette"], SILHOUETTE_FLOOR)

    def test_it_stays_quiet_when_the_grouping_is_weak(self) -> None:
        """Refusing is the right answer; a named category is expensive to undo."""
        self._show_two_hidden_families()
        strict = ConceptDiscovery(self.session, min_silhouette=0.99)
        self.assertEqual(strict.propose(), [])

    def test_learning_mode_asks_what_to_call_the_group(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        self._show_two_hidden_families()
        asked = [q for q in LearningSession(self.session).survey()
                 if q.kind == "proposed-concept"]
        self.assertTrue(asked, "a discovered group should become a question")
        self.assertIn("no name for them", asked[0].prompt)
        self.assertGreaterEqual(asked[0].context["size"], 4)

    def test_naming_a_group_trains_it_into_the_library(self) -> None:
        from ultraquant.interpreter.learning import LearningSession
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        self._show_two_hidden_families()
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "proposed-concept")
        answer = learner.answer(question, "diagonals")

        self.assertTrue(answer.accepted, answer.detail)
        self.assertEqual(answer.learned["category"], "diagonals")
        self.assertIn("diagonals", self.session.vault.signatures())

        reopened = build_session(self.dir / "home", seed=0)
        response, _trace = run_pipeline(
            "\n".join(question.context["rows"][0]), reopened
        )
        self.assertIn("diagonals", response.lower())

    def test_the_original_categories_still_work_afterwards(self) -> None:
        from ultraquant.interpreter.learning import LearningSession
        from ultraquant.interpreter.thoughts import build_session, run_pipeline
        from ultraquant.pattern.recognition import PATTERNS

        self._show_two_hidden_families()
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "proposed-concept")
        learner.answer(question, "diagonals")

        reopened = build_session(self.dir / "home", seed=0)
        for glyph in ("plus", "square", "arrow_up", "stripes_h"):
            with self.subTest(glyph=glyph):
                response, _ = run_pipeline("\n".join(PATTERNS[glyph]), reopened)
                self.assertIn(f"'{glyph}'", response)


class GateIntegrityTests(unittest.TestCase):
    """The experiment must stay capable of an honest answer."""

    def test_the_control_data_really_has_no_structure(self) -> None:
        rng = random.Random(0)
        vectors, labels = gate.make_families(rng, classes=4, structured=False)
        found = discover(vectors, seed=0)
        self.assertLess(abs(adjusted_rand_index(labels, found["assignment"])), 0.15)

    def test_the_structured_data_really_does(self) -> None:
        rng = random.Random(0)
        vectors, labels = gate.make_families(rng, classes=4, structured=True)
        found = discover(vectors, seed=0)
        self.assertGreater(adjusted_rand_index(labels, found["assignment"]), 0.5)

    def test_the_gate_never_passes_k_to_the_clusterer(self) -> None:
        """The single property that would invalidate the whole result."""
        import inspect

        source = inspect.getsource(gate.one_trial)
        self.assertIn("k_range=(2, 8)", source)
        self.assertNotIn("k=classes", source)


if __name__ == "__main__":
    unittest.main()
