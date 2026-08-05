"""Tests for learning mode: gap discovery, question ranking, and applied answers."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter.learning import LearningSession
from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.pattern.recognition import PATTERNS


class LearningTests(unittest.TestCase):
    """The model finds its own gaps and learns from the answers."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_learn_"))
        self.session = build_session(self.dir / "home", seed=0)
        self.learner = LearningSession(self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _kinds(self) -> set[str]:
        return {q.kind for q in self.learner.survey()}

    # -- discovery --------------------------------------------------------

    def test_low_confidence_fact_is_questioned(self) -> None:
        self.session.memory.remember_fact("tower height", "324 metres", confidence=0.4)
        self.assertIn("shaky", self._kinds())

    def test_confident_fact_is_not_questioned(self) -> None:
        self.session.memory.remember_fact("sky color", "blue", confidence=0.95)
        shaky = [q for q in self.learner.survey() if q.kind == "shaky"]
        self.assertEqual([q.subject for q in shaky], [])

    def test_repeated_unknown_term_is_questioned(self) -> None:
        for _ in range(4):
            run_pipeline("tell me about photonics", self.session)
        questions = [q for q in self.learner.survey() if q.kind == "unknown-term"]
        self.assertIn("photonics", [q.subject for q in questions])

    def test_term_seen_once_is_not_questioned(self) -> None:
        run_pipeline("something about tachyons", self.session)
        questions = [q for q in self.learner.survey() if q.kind == "unknown-term"]
        self.assertNotIn("tachyons", [q.subject for q in questions])

    def test_stopwords_are_never_asked_about(self) -> None:
        for _ in range(6):
            run_pipeline("what is the thing that you know about", self.session)
        subjects = {q.subject for q in self.learner.survey()}
        self.assertFalse(subjects & {"the", "what", "you", "know", "about", "that"})

    def test_untrained_category_asks_for_a_demonstration(self) -> None:
        questions = [q for q in self.learner.survey() if q.kind == "untrained-category"]
        self.assertTrue(questions)
        self.assertEqual(questions[0].expects, "glyph")

    def test_questions_are_ranked_by_score(self) -> None:
        self.session.memory.remember_fact("a", "1", confidence=0.3)
        for _ in range(5):
            run_pipeline("discuss photonics research", self.session)
        questions = self.learner.survey()
        scores = [q.score for q in questions]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_survey_is_capped(self) -> None:
        learner = LearningSession(self.session, max_questions=2)
        for _ in range(4):
            run_pipeline("photonics and metamaterials research", self.session)
        self.assertLessEqual(len(learner.survey()), 2)

    def test_prompts_are_console_safe(self) -> None:
        """Regression guard: prompts reach a cp1252 console."""
        for _ in range(4):
            run_pipeline("photonics work", self.session)
        for question in self.learner.survey():
            with self.subTest(kind=question.kind):
                question.prompt.encode("cp1252")

    # -- answering --------------------------------------------------------

    def test_answering_an_unknown_term_stores_a_fact(self) -> None:
        for _ in range(4):
            run_pipeline("about photonics", self.session)
        self.learner.survey()
        question = next(q for q in self.learner.pending if q.subject == "photonics")
        answer = self.learner.answer(question, "the science of light")
        self.assertTrue(answer.accepted)
        fact = self.session.memory.recall_fact("photonics")
        self.assertEqual(fact["value"], "the science of light")

    def test_confirming_a_shaky_fact_raises_confidence(self) -> None:
        self.session.memory.remember_fact("tower height", "324 metres", confidence=0.4)
        self.learner.survey()
        question = next(q for q in self.learner.pending if q.kind == "shaky")
        self.learner.answer(question, "yes")
        self.assertGreaterEqual(
            self.session.memory.recall_fact("tower height")["confidence"], 0.85
        )

    def test_correcting_a_shaky_fact_replaces_the_value(self) -> None:
        self.session.memory.remember_fact("tower height", "wrong", confidence=0.4)
        self.learner.survey()
        question = next(q for q in self.learner.pending if q.kind == "shaky")
        self.learner.answer(question, "330 metres")
        self.assertEqual(
            self.session.memory.recall_fact("tower height")["value"], "330 metres"
        )

    def test_teaching_a_glyph_trains_an_expert(self) -> None:
        self.learner.survey()
        question = next(
            q for q in self.learner.pending if q.kind == "untrained-category"
        )
        category = question.context["category"]
        answer = self.learner.answer(question, "\n".join(PATTERNS["plus"]))
        self.assertTrue(answer.accepted, answer.detail)
        self.assertTrue(self.session.experts.has_expert(category))

    def test_malformed_glyph_is_rejected_with_guidance(self) -> None:
        self.learner.survey()
        question = next(
            q for q in self.learner.pending if q.kind == "untrained-category"
        )
        answer = self.learner.answer(question, "not a glyph")
        self.assertFalse(answer.accepted)
        self.assertIn("five rows", answer.detail)

    def test_empty_answer_skips(self) -> None:
        self.learner.survey()
        question = self.learner.next_question()
        answer = self.learner.answer(question, "   ")
        self.assertFalse(answer.accepted)
        self.assertIn(question.id, self.learner.skipped)

    def test_skip_advances_to_the_next_question(self) -> None:
        self.learner.survey()
        first = self.learner.next_question()
        self.learner.skip(first.id)
        second = self.learner.next_question()
        self.assertIsNotNone(second)
        self.assertNotEqual(first.id, second.id)

    def test_every_answer_is_recorded_as_an_episode(self) -> None:
        for _ in range(4):
            run_pipeline("photonics", self.session)
        self.learner.survey()
        question = next(q for q in self.learner.pending if q.subject == "photonics")
        self.learner.answer(question, "light science")
        episodes = self.session.memory.recall_episodes(kind="learning")
        self.assertTrue(episodes)
        self.assertEqual(episodes[0]["content"]["kind"], "unknown-term")

    def test_stats_track_progress(self) -> None:
        for _ in range(4):
            run_pipeline("photonics", self.session)
        self.learner.survey()
        question = next(q for q in self.learner.pending if q.subject == "photonics")
        self.learner.answer(question, "light science")
        stats = self.learner.stats()
        self.assertEqual(stats["asked"], 1)
        self.assertEqual(stats["accepted"], 1)
        self.assertIn("by_kind", stats)

    def test_a_failing_probe_does_not_stop_the_survey(self) -> None:
        """One broken store must not deny the whole survey."""
        class Exploding:
            def entries(self, *a, **k):
                raise RuntimeError("stash is broken")

        self.session.stash = Exploding()
        self.session.memory.remember_fact("k", "v", confidence=0.3)
        self.assertIn("shaky", {q.kind for q in self.learner.survey()})


class StashLearningTests(unittest.TestCase):
    """Disputed and unclassified web claims become questions."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_learn2_"))
        self.session = build_session(self.dir / "home", seed=0)
        self.learner = LearningSession(self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _stage(self, text: str) -> None:
        self.session.stash.add_page("http://a.test/x", "T", text)
        self.session.stash.analyze(self.session.memory)

    def test_disputed_claim_is_asked_about_first(self) -> None:
        self.session.memory.remember_fact("tower height", "1000 metres", confidence=0.9)
        self._stage("The tower height is 324 metres.")
        questions = self.learner.survey()
        self.assertEqual(questions[0].kind, "disputed")
        self.assertEqual(questions[0].expects, "choice")

    def test_adopting_the_new_claim_updates_memory(self) -> None:
        self.session.memory.remember_fact("tower height", "1000 metres", confidence=0.9)
        self._stage("The tower height is 324 metres.")
        question = next(q for q in self.learner.survey() if q.kind == "disputed")
        answer = self.learner.answer(question, "use the new claim")
        self.assertTrue(answer.accepted, answer.detail)
        self.assertIn("324", str(self.session.memory.recall_fact("tower height")["value"]))

    def test_keeping_the_existing_fact_rejects_the_claim(self) -> None:
        self.session.memory.remember_fact("tower height", "1000 metres", confidence=0.9)
        self._stage("The tower height is 324 metres.")
        question = next(q for q in self.learner.survey() if q.kind == "disputed")
        self.learner.answer(question, "keep what I have")
        self.assertEqual(
            self.session.memory.recall_fact("tower height")["value"], "1000 metres"
        )
        entry = self.session.stash.get(int(question.subject))
        self.assertEqual(entry["status"], "rejected")

    def test_unclassified_claim_can_be_marked_opinion(self) -> None:
        self._stage("Whatever happens next remains to be seen by everyone here.")
        questions = [q for q in self.learner.survey() if q.kind == "unclassified"]
        if not questions:
            self.skipTest("no unclassified claim produced by the classifier")
        answer = self.learner.answer(questions[0], "opinion")
        self.assertTrue(answer.accepted)
        self.assertEqual(self.session.memory.stats()["semantic"], 0)


if __name__ == "__main__":
    unittest.main()
