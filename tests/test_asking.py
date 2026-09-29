"""Section 11.152: confirmed readings teach indexed ways of asking."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ultraquant.interpreter.thoughts import (
    Perceive, ThoughtContext, build_session, run_pipeline,
)


BOOKS = (("Emma", "Jane Austen"), ("Dune", "Frank Herbert"),
         ("Ulysses", "James Joyce"))


class AskingTests(unittest.TestCase):
    def session(self, sharded, books=True):
        scratch = tempfile.TemporaryDirectory(prefix="uq_asking_test_")
        self.addCleanup(scratch.cleanup)
        session = build_session(Path(scratch.name), seed=0)
        if not sharded:
            session.memory.shards = None
        if books:
            for subject, author in BOOKS:
                session.memory.remember_fact(
                    f"author of {subject.lower()}", author, .9,
                    subject=subject, attribute="author")
                session.memory.learn_asking("author", subject, f"Who wrote {subject}?")
        return session

    def say(self, session, text):
        return run_pipeline(text, session)[0]

    def vocabulary(self, session):
        return deepcopy(session.memory._attribute_vocabulary())

    def facts(self, session):
        return {key: deepcopy(session.memory.recall_fact(key))
                for key in session.memory.fact_keys()}

    def test_arms_only_structured_facts_naming_the_indexed_subject(self):
        cases = [
            ({"subject": "Emma", "attribute": "author"}, True),
            ({"subject": "The Émma", "attribute": "author"}, True),
            ({"subject": "Dune", "attribute": "author"}, False),
            ({"subject": "Emma"}, False),
            ({"attribute": "author"}, False),
        ]
        for sharded in (False, True):
            for structure, armed in cases:
                with self.subTest(sharded=sharded, structure=structure):
                    session = self.session(sharded, books=False)
                    # Keys are opaque: only the explicit subject slot counts.
                    session.memory.remember_fact(
                        "author of emma", "Jane Austen", .9, **structure)
                    before = self.vocabulary(session)
                    question = "Emma was written by whom?"
                    self.assertIn("Nearest I hold:", self.say(session, question))
                    if armed:
                        self.assertEqual(session.pending_reading, {
                            "key": "author of emma", "attribute": "author",
                            "subject": structure["subject"], "question": question,
                        })
                    else:
                        self.assertIsNone(session.pending_reading)
                    self.assertEqual(self.vocabulary(session), before)

    def test_yes_learns_and_two_confirmed_subjects_answer_exactly(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                session = self.session(sharded)
                before = self.facts(session)
                for subject in ("Emma", "Dune"):
                    question = f"{subject} was written by whom?"
                    self.assertIn("Nearest I hold:", self.say(session, question))
                    self.assertEqual(self.say(session, "yes"),
                                     f"Noted: '{question}' asks for the author.")
                    self.assertIsNone(session.pending_reading)
                    asked = self.vocabulary(session)["author"]["asked_by"]
                    self.assertIn(subject.lower(), asked["written"])
                    if subject == "Emma":
                        self.assertIn("Nearest I hold:", self.say(
                            session, "Ulysses was written by whom?"))
                self.assertEqual(self.say(session, "Ulysses was written by whom?"),
                                 "author of ulysses is James Joyce (confidence 0.90).")
                self.assertIsNone(session.pending_reading)
                self.assertEqual(self.facts(session), before)

    def test_no_learns_nothing_and_changes_no_fact(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                session = self.session(sharded)
                vocabulary, facts = self.vocabulary(session), self.facts(session)
                for subject in ("Emma", "Dune"):
                    self.assertIn("Nearest I hold:", self.say(
                        session, f"{subject} was written by whom?"))
                    self.assertEqual(self.say(session, "no"),
                                     "Noted - that is not what you asked.")
                    self.assertIsNone(session.pending_reading)
                self.assertEqual(self.vocabulary(session), vocabulary)
                self.assertEqual(self.facts(session), facts)
                self.assertIn("Nearest I hold:", self.say(
                    session, "Ulysses was written by whom?"))

    def test_reading_expires_after_one_turn(self):
        for sharded in (False, True):
            for intervening in ("hello", "", "yes please", "no thanks"):
                with self.subTest(sharded=sharded, intervening=intervening):
                    session = self.session(sharded)
                    before = self.vocabulary(session)
                    self.say(session, "Emma was written by whom?")
                    self.assertIsNotNone(session.pending_reading)
                    self.say(session, intervening)
                    self.assertIsNone(session.pending_reading)
                    self.say(session, "yes")
                    self.assertEqual(self.vocabulary(session), before)

    def test_unstructured_nearest_held_arms_nothing(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                session = self.session(sharded, books=False)
                session.memory.remember_fact("author of emma", "Jane Austen", .9)
                before = self.facts(session)
                self.assertIn("Nearest I hold:", self.say(
                    session, "Emma was written by whom?"))
                self.assertIsNone(session.pending_reading)
                self.say(session, "yes")
                self.assertEqual(self.vocabulary(session), {})
                self.assertEqual(self.facts(session), before)

    def test_existing_reply_patterns_are_used(self):
        for sharded in (False, True):
            for reply, intent in ((" YES, that's correct! ", "reading_confirmed"),
                                  ("That's wrong.", "reading_declined")):
                with self.subTest(sharded=sharded, reply=reply):
                    session = self.session(sharded)
                    self.say(session, "Emma was written by whom?")
                    _, trace = run_pipeline(reply, session)
                    self.assertEqual(trace[0]["intent"], intent)
                    self.assertIsNone(session.pending_reading)

    def test_inference_and_testimony_keep_precedence(self):
        cases = (("pending_inference", "yes", "affirmation"),
                 ("pending_inference", "no", "declination"),
                 ("pending_confirmation", "yes", "confirmation"),
                 ("pending_confirmation", "no", "disconfirmation"))
        for sharded in (False, True):
            session = self.session(sharded)
            for field, reply, intent in cases:
                with self.subTest(sharded=sharded, field=field, reply=reply):
                    contexts = []
                    for reading in (None, {"question": "Emma was written by whom?"}):
                        setattr(session, field, {"key": "author of emma"})
                        session.pending_reading = reading
                        ctx = ThoughtContext(reply, session)
                        Perceive().run(ctx)
                        contexts.append(ctx)
                        self.assertEqual(ctx.data["intent"], intent)
                        self.assertIsNone(session.pending_reading)
                    self.assertEqual(contexts[0].data, contexts[1].data)
                    self.assertEqual(contexts[0].trace, contexts[1].trace)


if __name__ == "__main__":
    unittest.main()
