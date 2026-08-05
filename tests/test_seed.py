"""The starter knowledge: enough to know, and enough to doubt.

Self-learning needs material — the survey walks the stores looking for gaps,
and a freshly forged library has none. ``seed_knowledge`` plants a small body
of facts through the ordinary pipeline, deliberately leaving a few at low
confidence so the first survey has genuine questions.

Also covered: the deployment vocabulary decisions that dodge measured failure
modes — above all the homonym rule, because one word for two atoms was
measured to cut both directions to 0.75.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.forge.languages import (
    BASIC_FACTS,
    WHOLE_LABEL_WORDS,
    deploy_languages,
    seed_knowledge,
)


class VocabularyDecisionTests(unittest.TestCase):
    """Choices that exist because a failure mode was measured."""

    def test_the_cross_homonym_is_dodged(self) -> None:
        """Whole-label 'cross' must not reuse the mark word 'saltire'."""
        self.assertNotEqual(WHOLE_LABEL_WORDS["coined"]["cross"], "saltire")

    def test_no_coined_whole_word_collides_with_a_mark_word(self) -> None:
        """The lexicon is a bijection; a collision breaks both directions."""
        mark_words = {"plus", "saltire", "speck", "dash"}
        for label, word in WHOLE_LABEL_WORDS["coined"].items():
            if label == "plus":
                continue  # same atom as the mark, shared by identity
            with self.subTest(label=label):
                self.assertNotIn(word, mark_words)

    def test_whole_plus_shares_the_mark_atom_by_identity(self) -> None:
        """Label 'plus' and mark 'plus' are the same atom, so the same word."""
        self.assertEqual(WHOLE_LABEL_WORDS["coined"]["plus"], "plus")

    def test_the_seed_mixes_solid_and_shaky(self) -> None:
        """Self-learning needs something to doubt, not only something to know."""
        solid = sum(1 for _k, _v, s in BASIC_FACTS if s)
        shaky = sum(1 for _k, _v, s in BASIC_FACTS if not s)
        self.assertGreaterEqual(solid, 10)
        self.assertGreaterEqual(shaky, 2)


class SeededSessionTests(unittest.TestCase):
    """The seed lands, and the loop it exists for actually closes."""

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.interpreter.thoughts import build_session

        cls.dir = Path(tempfile.mkdtemp(prefix="uq_seed_"))
        cls.session = build_session(cls.dir, seed=0)
        cls.report = seed_knowledge(cls.session)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_facts_land_in_the_sharded_store(self) -> None:
        self.assertEqual(self.report["taught"], len(BASIC_FACTS))
        buckets = [entry for entry in self.session.vault.catalog()
                   if entry["kind"] == "fact-bucket"]
        self.assertTrue(buckets, "seeded facts must live in the library")

    def test_a_seeded_fact_recalls_through_the_pipeline(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline(
            "what is the number of glyph pixels?", self.session
        )
        self.assertIn("25", response)

    def test_the_first_survey_has_questions_to_ask(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        questions = LearningSession(self.session).survey()
        kinds = {question.kind for question in questions}
        self.assertIn("shaky", kinds,
                      "the shaky seeds are the material self-learning starts on")

    def test_confirming_a_shaky_seed_raises_its_confidence(self) -> None:
        """The loop this exists for: doubt -> question -> answer -> knowledge."""
        from ultraquant.interpreter.learning import LearningSession

        learner = LearningSession(self.session)
        question = next(q for q in learner.survey() if q.kind == "shaky")
        before = self.session.memory.recall_fact(question.subject)["confidence"]
        answer = learner.answer(question, "yes")
        self.assertTrue(answer.accepted, answer.detail)
        after = self.session.memory.recall_fact(question.subject)["confidence"]
        self.assertGreater(after, before)


class DeployedLanguageTests(unittest.TestCase):
    """The deployment builder wires coverage correctly."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_deploylang_"))
        self.session = build_session(self.dir, seed=0)
        self.names = deploy_languages(
            self.session, ["box", "caps", "corners", "rails"],
            ["bar", "dot", "ex", "plus"],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_all_three_languages_are_stored(self) -> None:
        self.assertEqual(self.names, ["holding", "nested", "plain"])

    def test_each_covers_scenes_and_bare_labels(self) -> None:
        from ultraquant.reason.language import verbalise

        self.assertEqual(
            verbalise(self.session, {"shape": "box", "mark": "dot"},
                      language="nested"),
            "a speck inside a box",
        )
        self.assertEqual(
            verbalise(self.session, {"pattern": "square"}, language="nested"),
            "a quad",
        )
        self.assertEqual(
            verbalise(self.session, {"pattern": "square"}, language="plain"),
            "a square",
        )

    def test_the_reversed_language_reverses(self) -> None:
        from ultraquant.reason.language import verbalise

        self.assertEqual(
            verbalise(self.session, {"shape": "box", "mark": "dot"},
                      language="holding"),
            "a box holding a speck",
        )


if __name__ == "__main__":
    unittest.main()
