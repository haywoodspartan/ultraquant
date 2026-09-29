"""Section 11.153: indexed subjects shared across question parts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ultraquant.interpreter.thoughts import (
    Reason, _ambiguous_cover, _protected_parts, _shared_subject, build_session, run_pipeline,
)
from ultraquant.memory.systematic import SystematicMemory


class SharedSubjectTests(unittest.TestCase):
    def sessions(self):
        for sharded in (False, True):
            directory = tempfile.TemporaryDirectory(prefix="uq_shared_test_")
            self.addCleanup(directory.cleanup)
            session = build_session(Path(directory.name), seed=0)
            if not sharded:
                session.memory = SystematicMemory(session.root / "memory.json")
            yield session

    def hold(self, memory, subject, attribute, value, key=None):
        memory.remember_fact(key or f"{attribute} of {subject.lower()}", value,
                             .9, subject=subject, attribute=attribute)

    def world(self, memory):
        for subject, number, symbol in (("gold", "79", "Au"),
                                        ("silver", "47", "Ag"),
                                        ("copper", "29", "Cu")):
            self.hold(memory, subject, "atomic number", number)
            self.hold(memory, subject, "chemical symbol", symbol)
        self.hold(memory, "Kenya", "capital", "Nairobi")
        self.hold(memory, "Emma", "author", "Jane Austen")

    def ask(self, session, text):
        return run_pipeline(text, session)[0]

    def test_attribute_without_subject_names_ambiguity(self):
        for session in self.sessions():
            self.world(session.memory)
            for text in ("What is the chemical symbol?", "What is its chemical symbol?"):
                with self.subTest(sharded=session.memory.shards is not None, text=text):
                    self.assertEqual(_ambiguous_cover(session.memory, text), 3)
                    self.assertEqual(self.ask(session, text),
                                     "That depends on which one - I hold the chemical "
                                     "symbol for 3 subjects.")
            self.assertIsNone(_ambiguous_cover(session.memory, "What is the capital?"))
            self.assertIsNone(_ambiguous_cover(session.memory,
                                              "What is the chemical symbol of gold?"))
            self.assertIsNone(_ambiguous_cover(session.memory,
                                              "What is the number and symbol?"))
        self.assertIsNone(_ambiguous_cover(SimpleNamespace(), "What is the symbol?"))
        self.assertIsNone(_shared_subject(SimpleNamespace(), "What is gold?"))

    def test_shared_subject_in_both_attribute_and_subject_orders(self):
        for session in self.sessions():
            self.world(session.memory)
            for first, second in (("atomic number", "chemical symbol"),
                                  ("chemical symbol", "atomic number")):
                for text in (f"What is the {first} and {second} of gold?",
                             f"What is the {first} of gold and {second}?"):
                    with self.subTest(text=text):
                        reply = self.ask(session, text)
                        self.assertIn("atomic number of gold is 79", reply)
                        self.assertIn("chemical symbol of gold is Au", reply)
                        self.assertNotIn("silver", reply)
                        self.assertNotIn("unknown", reply)

    def test_kenya_capital_and_missing_author(self):
        for session in self.sessions():
            self.world(session.memory)
            reply = self.ask(session, "What are the capital and author of Kenya?")
            self.assertIn("capital of kenya is Nairobi", reply)
            self.assertIn("Still missing: author of kenya (unknown", reply)
            self.assertNotIn("Jane Austen", reply)
            self.assertEqual(self.ask(session, "What are the author and capital of Kenya?"),
                             "capital of kenya is Nairobi. Still missing: the author (unknown).")
            self.assertIn("capital of kenya is Nairobi",
                          self.ask(session, "What is the capital and author of Kenya?"))

    def test_names_containing_and_answer_exactly(self):
        for session in self.sessions():
            memory = session.memory
            self.hold(memory, "Pride and Prejudice", "author", "Jane Austen")
            memory.learn_asking("author", "Pride and Prejudice",
                                "Who wrote the novel Pride and Prejudice?")
            self.hold(memory, "Emma", "author", "Jane Austen")
            memory.learn_asking("author", "Emma", "Who wrote the novel Emma?")
            self.hold(memory, "Newfoundland and Labrador", "capital", "St. John's")
            for text, expected in (
                    ("Who wrote Pride and Prejudice?",
                     "author of pride and prejudice is Jane Austen (confidence 0.90)."),
                    ("What is the capital of Newfoundland and Labrador?",
                     "capital of newfoundland and labrador is St. John's (confidence 0.90).")):
                with self.subTest(sharded=memory.shards is not None, text=text):
                    self.assertIsNone(_protected_parts(memory, text))
                    with mock.patch.object(Reason, "_compound_parts",
                                           side_effect=AssertionError("decomposed name")):
                        self.assertEqual(self.ask(session, text), expected)

    def test_protected_parts_keep_names_and_split_outside_them(self):
        for session in self.sessions():
            memory = session.memory
            self.world(memory)
            self.hold(memory, "Pride", "author", "Someone")
            self.hold(memory, "Pride and Prejudice", "author", "Jane Austen")
            for text, expected in (
                    ("Who wrote Pride, and Prejudice?", None),
                    ("What is the author and marker of Pride and Prejudice?",
                     ["the author", "marker of pride and prejudice"]),
                    ("What is the author of Pride and Prejudice and Emma?",
                     ["the author of pride and prejudice", "emma"]),
                    ("What is Pride and Prejudice and Pride and Prejudice?",
                     ["pride and prejudice", "pride and prejudice"]),
                    ("What is the atomic number and chemical symbol of gold?",
                     ["the atomic number", "chemical symbol of gold"]),
                    ("What is the capital and author of Kenya?",
                     ["the capital", "author of kenya"]),
                    ("What is and Pride and Prejudice?", None)):
                with self.subTest(sharded=memory.shards is not None, text=text):
                    self.assertEqual(_protected_parts(memory, text), expected)

    def test_two_and_three_subjects(self):
        for session in self.sessions():
            self.world(session.memory)
            for names in (("gold", "silver"), ("gold", "silver", "copper")):
                text = "What is the chemical symbol of " + " and ".join(names) + "?"
                answers = session.memory.catalogue_answers(text)
                self.assertEqual({answer["subject"] for answer in answers}, set(names))
                with mock.patch.object(Reason, "_compound_parts",
                                       side_effect=AssertionError("decomposed")):
                    reply = self.ask(session, text)
                for name, symbol in (("gold", "Au"), ("silver", "Ag"), ("copper", "Cu")):
                    if name in names:
                        self.assertIn(f"chemical symbol of {name} is {symbol}", reply)

    def test_named_subject_without_attribute(self):
        for session in self.sessions():
            self.world(session.memory)
            text = "What is the chemical symbol of gold and Kenya?"
            answers = session.memory.catalogue_answers(text)
            self.assertEqual(answers[1], {"subject": "kenya", "attribute": "chemical symbol",
                                         "key": None, "record": None})
            reply = self.ask(session, text)
            self.assertIn("chemical symbol of gold is Au", reply)
            self.assertIn("chemical symbol of kenya (unknown)", reply)

    def test_unstructured_compound_is_unchanged(self):
        for session in self.sessions():
            memory = session.memory
            memory.remember_fact("tower material", "steel")
            memory.remember_fact("bridge length", "2 kilometers")
            text = "What is the tower material and the bridge length?"
            self.assertIsNone(_shared_subject(memory, text))
            self.assertIsNone(memory.catalogue_answers(text))
            self.assertEqual(Reason._compound_parts(text),
                             ["the tower material", "the bridge length"])
            self.assertEqual(self.ask(session, text),
                             "tower material is steel; bridge length is 2 kilometers.")

    def test_nested_subject_uses_single_catalogue_resolution(self):
        for session in self.sessions():
            memory = session.memory
            self.hold(memory, "Australia", "capital", "Canberra")
            self.hold(memory, "South Australia", "capital", "Adelaide")
            self.hold(memory, "South Australia", "marker", "S")
            text = "What is the capital and marker of South Australia?"
            self.assertEqual(_shared_subject(memory, text), "south australia")
            self.assertIsNone(memory.catalogue_answers("What is the capital of South Australia?"))
            self.assertIn("capital of south australia is Adelaide", self.ask(session, text))

    def test_opaque_keys_and_arbitrary_attributes_use_catalogue(self):
        for session in self.sessions():
            memory = session.memory
            self.hold(memory, "Nacre", "velar", "Iris", key="entry one")
            self.hold(memory, "Quartz", "velar", "Opal", key="entry two")
            self.hold(memory, "Nacre", "surn", "Jade")
            text = "What is the velar and surn of Nacre?"
            self.assertIn("entry one is Iris", self.ask(session, text))
            self.assertEqual(_ambiguous_cover(memory, "What is the velar?"), 2)
            self.assertIsNone(_shared_subject(memory, "Nacre and Quartz"))
            with mock.patch.object(memory, "fact_keys", side_effect=AssertionError("scan")), \
                    mock.patch.object(memory, "find_facts", side_effect=AssertionError("search")):
                answers = memory.catalogue_answers("What is the velar of Nacre and Quartz?")
            self.assertEqual([answer["key"] for answer in answers], ["entry one", "entry two"])
            self.assertIsNone(memory.catalogue_answers("What is the surn and velar of Nacre and Quartz?"))


if __name__ == "__main__":
    unittest.main()
