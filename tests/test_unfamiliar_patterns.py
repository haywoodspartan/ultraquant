"""Saying "I don't know", and turning that into a question worth answering.

Two things are tested here, and it is worth being precise about which is which.

**The floor is a dissimilarity test, not novelty detection.** It reports a
pattern as unfamiliar when it resembles *nothing* stored. It does not, and
cannot, detect a genuinely new pattern that happens to look like a known one --
measurement showed novel glyphs reaching 0.97 cosine similarity against stored
prototypes, and expert confidence separating novel from known by only 0.876 vs
0.942. No threshold on either signal works. So what is asserted below is what
was actually measured: known glyphs and noisy copies of them are never rejected
(the false-alarm rate is what matters most), and patterns unlike anything held
are flagged.

**The loop closes.** An unfamiliar pattern becomes an ``unknown-pattern``
episode, learning mode surfaces it as a question, answering it trains a category
-- inventing one if needed -- and the same input then routes correctly in a
fresh session. That is the whole self-learning cycle in one test.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.forge.corpus import BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy
from ultraquant.forge.forge import ModelForge
from ultraquant.interpreter.learning import LearningSession
from ultraquant.interpreter.thoughts import UNFAMILIAR_BELOW, build_session, run_pipeline
from ultraquant.pattern.recognition import PATTERNS, render

#: Shapes deliberately unlike anything in the built-in taxonomy.
NOVEL = {
    "backslash": ["....#", "...#.", "..#..", ".#...", "#...."],
    "checker": ["#.#.#", ".....", "##.##", ".....", "#.#.#"],
    "corners": ["##...", "##...", ".....", "...##", "...##"],
    "blob": [".###.", "#...#", "#.#.#", "#...#", ".###."],
}

_NOT_HELD = "do not recognise"


def _forge(root: Path) -> ModelForge:
    forge = ModelForge(root / "home", seed=0, hidden=16, tier="auto")
    forge.build(
        build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                      keywords=BUILTIN_KEYWORDS),
        epochs=25,
    )
    return forge


class UnfamiliarPatternTests(unittest.TestCase):
    """What the model says when it is shown something it does not hold."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = Path(tempfile.mkdtemp(prefix="uq_unfam_"))
        cls.forge = _forge(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self) -> None:
        self.session = build_session(self.dir / "home", seed=0)

    def _ask(self, rows: list[str]) -> str:
        return run_pipeline("\n".join(rows), self.session)[0]

    # -- false alarms are the expensive failure ----------------------------

    def test_known_prototypes_are_never_called_unfamiliar(self) -> None:
        for glyph, rows in PATTERNS.items():
            with self.subTest(glyph=glyph):
                self.assertNotIn(_NOT_HELD, self._ask(rows))

    def test_noisy_copies_of_known_glyphs_survive_the_floor(self) -> None:
        """One flipped pixel must not make a familiar shape unfamiliar.

        Measured: noisy copies never fall below 0.866 similarity, and the floor
        sits at 0.85, so this should hold with margin rather than by luck.
        """
        rng = random.Random(0)
        for glyph, rows in PATTERNS.items():
            for trial in range(6):
                with self.subTest(glyph=glyph, trial=trial):
                    variant = list(rows)
                    r, c = rng.randrange(5), rng.randrange(5)
                    row = list(variant[r])
                    row[c] = "#" if row[c] == "." else "."
                    variant[r] = "".join(row)
                    self.assertNotIn(_NOT_HELD, self._ask(variant))

    # -- and it does say so when nothing matches ---------------------------

    def test_patterns_unlike_anything_stored_are_flagged(self) -> None:
        flagged = sum(_NOT_HELD in self._ask(rows) for rows in NOVEL.values())
        self.assertEqual(flagged, len(NOVEL))

    def test_the_report_names_the_nearest_thing_it_does_hold(self) -> None:
        """A bare "I don't know" is less useful than a near miss."""
        response = self._ask(NOVEL["backslash"])
        self.assertIn(_NOT_HELD, response)
        self.assertIn("closest", response)
        self.assertTrue(
            any(f"'{category}'" in response for category in self.forge.vault.signatures()),
            f"expected a stored category to be named: {response}",
        )

    def test_an_unfamiliar_pattern_does_not_train_anything(self) -> None:
        """Guessing is bad; guessing *and* reinforcing the guess is worse."""
        shard = ExpertPool.shard_id("crosses")
        before = self.session.vault.entry(shard)["access_count"]
        self._ask(NOVEL["checker"])
        self.assertEqual(self.session.vault.entry(shard)["access_count"], before)

    def test_unfamiliar_patterns_are_recorded_for_learning_mode(self) -> None:
        self._ask(NOVEL["corners"])
        episodes = self.session.memory.recall_episodes(kind="unknown-pattern", limit=10)
        self.assertTrue(episodes)
        self.assertEqual(episodes[0]["content"]["rows"], NOVEL["corners"])

    # -- the mechanism itself ----------------------------------------------

    def test_the_floor_is_below_every_stored_prototype_similarity(self) -> None:
        """The threshold has to be justified by the data, not chosen by feel."""
        worst = 1.0
        for rows in PATTERNS.values():
            ranked = self.session.router.route_pattern(render(rows), top_k=1)
            self.assertTrue(ranked)
            worst = min(worst, ranked[0][1])
        self.assertGreater(worst, UNFAMILIAR_BELOW,
                           "the floor must not reject the library's own prototypes")

    def test_route_pattern_honours_an_explicit_floor(self) -> None:
        pixels = render(PATTERNS["plus"])
        self.assertTrue(self.session.router.route_pattern(pixels, min_similarity=0.5))
        self.assertEqual(self.session.router.route_pattern(pixels, min_similarity=1.01), [])


class UnfamiliarToLearnedTests(unittest.TestCase):
    """The full cycle: cannot place it -> asks -> taught -> places it."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_unfamloop_"))
        self.forge = _forge(self.dir)
        self.session = build_session(self.dir / "home", seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_learning_mode_asks_about_what_it_could_not_place(self) -> None:
        run_pipeline("\n".join(NOVEL["backslash"]), self.session)
        questions = LearningSession(self.session).survey()
        asked = [q for q in questions if q.kind == "unknown-pattern"]
        self.assertTrue(asked, "an unplaceable pattern should become a question")
        self.assertEqual(asked[0].context["rows"], NOVEL["backslash"])
        for row in NOVEL["backslash"]:
            self.assertIn(row, asked[0].prompt)

    def test_it_asks_once_per_distinct_pattern(self) -> None:
        for _ in range(3):
            run_pipeline("\n".join(NOVEL["checker"]), self.session)
        asked = [q for q in LearningSession(self.session).survey()
                 if q.kind == "unknown-pattern"]
        self.assertEqual(len(asked), 1)

    def test_answering_invents_the_category_and_teaches_it(self) -> None:
        run_pipeline("\n".join(NOVEL["backslash"]), self.session)
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "unknown-pattern")

        self.assertNotIn("diagonals", self.forge.vault.signatures())
        answer = learner.answer(question, "diagonals backslash")

        self.assertTrue(answer.accepted, answer.detail)
        self.assertEqual(answer.learned["category"], "diagonals")
        self.assertIn("diagonals", self.session.vault.signatures())

    def test_the_taught_pattern_is_recognised_in_a_fresh_session(self) -> None:
        """A lesson that does not survive a restart has not been learned."""
        run_pipeline("\n".join(NOVEL["backslash"]), self.session)
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "unknown-pattern")
        learner.answer(question, "diagonals backslash")

        reopened = build_session(self.dir / "home", seed=0)
        response, trace = run_pipeline("\n".join(NOVEL["backslash"]), reopened)
        route = next(step for step in trace if step["thought"] == "Route")
        self.assertEqual(route["routes"][0], "diagonals")
        self.assertIn("'backslash'", response)
        self.assertNotIn(_NOT_HELD, response)

    def test_teaching_a_new_category_leaves_the_old_ones_intact(self) -> None:
        run_pipeline("\n".join(NOVEL["backslash"]), self.session)
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "unknown-pattern")
        learner.answer(question, "diagonals backslash")

        reopened = build_session(self.dir / "home", seed=0)
        for glyph in ("plus", "square", "arrow_up", "stripes_h"):
            with self.subTest(glyph=glyph):
                response, _ = run_pipeline("\n".join(PATTERNS[glyph]), reopened)
                self.assertIn(f"'{glyph}'", response)

    def test_a_single_word_answer_names_both_category_and_label(self) -> None:
        run_pipeline("\n".join(NOVEL["blob"]), self.session)
        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "unknown-pattern")
        answer = learner.answer(question, "blobs")
        self.assertTrue(answer.accepted, answer.detail)
        self.assertEqual(answer.learned["category"], "blobs")
        self.assertEqual(answer.learned["label"], "blobs")


class TeachingStoresASignatureTests(unittest.TestCase):
    """Teaching must update what the category *looks like*, not just its weights.

    The bug this guards: ``teach_glyph`` trained the expert but stored no
    signature, so pattern routing could not see the new category at all. The
    lesson was real and completely unreachable -- showing the taught glyph back
    routed it by keywords it does not contain, to some other expert entirely.
    """

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_teachsig_"))
        _forge(self.dir)
        self.session = build_session(self.dir / "home", seed=0)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_teaching_appends_a_prototype_to_the_category_signature(self) -> None:
        from ultraquant.interpreter.selflearn import SelfLearner

        shard = ExpertPool.shard_id("crosses")
        before = len(self.session.vault.entry(shard).get("signature") or [])
        labels = list(self.session.vault.get(shard)["labels"])
        SelfLearner(self.session).teach_glyph(
            "crosses", labels, PATTERNS["plus"], labels[0]
        )
        after = self.session.vault.entry(shard).get("signature") or []
        self.assertEqual(len(after), before + 1)
        self.assertEqual(len(after[-1]), 25)

    def test_a_taught_shape_routes_to_the_category_that_learned_it(self) -> None:
        from ultraquant.interpreter.selflearn import SelfLearner

        rows = NOVEL["corners"]
        self.session.experts.ensure_expert("mymarks", ["corners", "not_corners"])
        SelfLearner(self.session).teach_glyph(
            "mymarks", ["corners", "not_corners"], rows, "corners"
        )
        response, trace = run_pipeline("\n".join(rows), self.session)
        route = next(step for step in trace if step["thought"] == "Route")
        self.assertEqual(route["routes"][0], "mymarks")
        self.assertIn("'corners'", response)


if __name__ == "__main__":
    unittest.main()
