"""Section 11.149: curiosity uses the catalogue's attribute vocabulary."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason import inference


class AskableTests(unittest.TestCase):
    def scratch(self):
        directory = tempfile.TemporaryDirectory(prefix="uq_askable_test_")
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def memories(self):
        yield "ram", SystematicMemory(self.scratch() / "memory.json")
        yield "sharded", build_session(self.scratch(), seed=0).memory

    def test_written_registers_no_curiosity_or_hint(self):
        session = build_session(self.scratch(), seed=0)
        session.memory.remember_fact(
            "author of emma", "Jane Austen", .9,
            subject="Emma", attribute="author")
        reply, _ = run_pipeline("Emma was written by whom?", session)
        self.assertEqual(session.curiosities, [])
        self.assertNotIn("If I knew", reply)

    def test_catalogued_bridge_asks_for_known_attribute(self):
        session = build_session(self.scratch(), seed=0)
        session.memory.remember_fact(
            "material of the tower", "steel", .9,
            subject="the tower", attribute="material")
        session.memory.remember_fact(
            "conductivity of copper", "high", .9,
            subject="copper", attribute="conductivity")
        reply, _ = run_pipeline("What is the conductivity of the tower?", session)
        # §11.150: a catalogued premise is keyed as the catalogue keys it.
        self.assertEqual([c["premise_key"] for c in session.curiosities],
                         ["conductivity of steel"])
        self.assertIn("If I knew the conductivity of steel", reply)

    def test_chat_bridge_does_not_consult_attribute_vocabulary(self):
        session = build_session(self.scratch(), seed=0)
        run_pipeline("tower material is steel", session)
        with mock.patch.object(inference, "_remainder_known",
                               side_effect=AssertionError("unstructured bridge")):
            reply, _ = run_pipeline("what is the tower conductivity?", session)
        self.assertEqual([c["premise_key"] for c in session.curiosities],
                         ["steel conductivity"])
        self.assertIn("If I knew the steel conductivity", reply)

    def test_remainder_uses_all_indexed_attributes_and_requires_every_word(self):
        for label, memory in self.memories():
            with self.subTest(store=label):
                self.assertTrue(inference._remainder_known(set(), memory))
                self.assertFalse(inference._remainder_known({"conductivity"}, memory))
                memory.remember_fact("copper conductivity", "high")
                self.assertFalse(inference._remainder_known({"conductivity"}, memory))
                for attribute in ("conductivity", "glimmer bands"):
                    memory.remember_fact(
                        f"{attribute} of copper", "high", .9,
                        subject="copper", attribute=attribute)
                self.assertTrue(inference._remainder_known(
                    inference._fold("conductivity glimmer bands"), memory))
                self.assertFalse(inference._remainder_known(
                    {"conductivity", "written"}, memory))

    def test_remainder_accepts_asking_words_only_after_attestation(self):
        for label, memory in self.memories():
            with self.subTest(store=label):
                for subject in ("Echo", "Foxtrot"):
                    memory.remember_fact(
                        f"author of {subject.lower()}", "Writer", .9,
                        subject=subject, attribute="author")
                remainder = inference._fold("scribes")
                memory.learn_asking("author", "Echo", "Which scribes for Echo?")
                self.assertFalse(inference._remainder_known(remainder, memory))
                memory.learn_asking("author", "Echo", "Which scribes for Echo?")
                self.assertFalse(inference._remainder_known(remainder, memory))
                memory.learn_asking("author", "Foxtrot", "Which scribes for Foxtrot?")
                self.assertTrue(inference._remainder_known(remainder, memory))
                self.assertFalse(inference._remainder_known(
                    remainder | {"written"}, memory))


if __name__ == "__main__":
    unittest.main()
