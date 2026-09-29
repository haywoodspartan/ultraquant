"""Catalogue names are opaque to arithmetic, in both memory stores."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.reason import calculate


class NamesTests:
    sharded = True

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uq_names_test_")
        self.addCleanup(self.temp.cleanup)
        self.session = build_session(Path(self.temp.name), seed=0)
        self.memory = self.session.memory
        if not self.sharded:
            self.memory.shards = None
        self.hold("Catch-22", "Joseph Heller", "author")

    def hold(self, name, value="fixture", attribute="label"):
        self.memory.remember_fact(
            f"{attribute} of {name}", value, 0.9,
            subject=name, attribute=attribute)

    def ask(self, text):
        return run_pipeline(text, self.session)[0]

    def test_author_through_pipeline(self):
        for question in ("Who wrote Catch-22?",
                         "Who wrote the novel Catch-22?",
                         "Who is the author of Catch-22?",
                         "What is the author of Catch-22?"):
            with self.subTest(question=question):
                self.assertIsNone(calculate.evaluate(question, self.memory))
                reply = self.ask(question)
                self.assertIn("Joseph Heller", reply)
                self.assertFalse(reply.startswith("I can't"), reply)

    def test_only_recorded_operator_names_present_in_text(self):
        self.hold("Plain Name")
        self.hold("Absent/Name")
        self.hold("Signal+Noise")
        self.hold("Other-Name")
        self.assertEqual(
            set(calculate._held_names(
                "CATCH-22, signal+noise, Plain Name, Unknown-7, Other Name",
                self.memory)),
            {"Catch-22", "Signal+Noise"})
        # A normalized catalogue match alone does not attest the spelling.
        self.assertEqual(calculate._held_names("Catch 22", self.memory), [])

    def test_every_tokenizer_operator_can_belong_to_a_name(self):
        for index, operator in enumerate("()+-*/%^"):
            name = f"Panel{index}{operator}Squared"
            self.hold(name)
            with self.subTest(name=name):
                self.assertEqual(calculate._held_names(name, self.memory),
                                 [name])
                self.assertIsNone(calculate.evaluate(
                    f"What is {name}?", self.memory))
                result = calculate.evaluate(f"What is {name} plus 3?",
                                            self.memory)
                self.assertIsNotNone(result)
                self.assertIn(name.lower(), result.refusal)

    def test_unheld_numeric_expression_still_computes(self):
        self.assertEqual(calculate._held_names("What is 10-2?", self.memory),
                         [])
        self.assertIn("= 8", self.ask("What is 10-2?"))
        self.hold("10-2")
        self.assertIsNone(calculate.evaluate("What is 10-2?", self.memory))

    def test_external_operator_refuses_whole_name(self):
        answer = self.ask("What is catch-22 plus 3?")
        self.assertNotIn("Joseph Heller", answer)
        self.assertTrue(answer.startswith("I can't compute"), answer)
        self.assertIn("'catch-22'", answer)

    def test_every_occurrence_and_longest_name_are_opaque(self):
        self.assertIsNone(calculate.evaluate(
            "Who wrote Catch-22 and CATCH-22?", self.memory))
        self.hold("Catch-22/Sequel")
        self.assertIsNone(calculate.evaluate(
            "Who wrote Catch-22/Sequel?", self.memory))
        result = calculate.evaluate("What is Catch-22/Sequel plus 3?",
                                    self.memory)
        self.assertIn("'catch-22/sequel'", result.refusal)

    def test_arithmetic_replies_are_byte_identical_without_names(self):
        self.memory.remember_fact("tower height", "300 meters", 0.9)
        for question in ("What is 22 - 7?", "What is 10-2?",
                         "What is the tower height times 3?"):
            with self.subTest(question=question):
                with patch.object(calculate, "_held_names", return_value=[]):
                    original = self.ask(question)
                self.assertEqual(self.ask(question), original)

    def test_evaluate_uses_patchable_catalogue_seam(self):
        with patch.object(calculate, "_held_names", return_value=[]):
            result = calculate.evaluate("Who wrote Catch-22?", self.memory)
            self.assertIn("'who wrote catch'", result.refusal)
        with patch.object(calculate, "_held_names", return_value=["10-2"]):
            self.assertIsNone(calculate.evaluate("What is 10-2?", self.memory))


class ShardedNamesTests(NamesTests, unittest.TestCase):
    pass


class RamNamesTests(NamesTests, unittest.TestCase):
    sharded = False


class NoCatalogueTests(unittest.TestCase):
    def test_absent_catalogue(self):
        for memory in (None, object()):
            self.assertEqual(calculate._held_names("Catch-22", memory), [])
        self.assertEqual(calculate.evaluate("What is 10-2?").shown, "8")


if __name__ == "__main__":
    unittest.main()
