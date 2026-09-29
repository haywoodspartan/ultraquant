"""Section 11.150: two catalogue hops and catalogue-keyed curiosity."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ultraquant.interpreter.thoughts import Reason, ThoughtContext, build_session, run_pipeline
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import missing_premise
from ultraquant.shards.vault import ShardVault


class HopsTests(unittest.TestCase):
    def scratch(self):
        directory = tempfile.TemporaryDirectory(prefix="uq_hops_test_")
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def memories(self):
        for sharded in (False, True):
            root = self.scratch()
            shards = FactShards(ShardVault(root / "vault")) if sharded else None
            yield sharded, SystematicMemory(shards=shards)

    def hold(self, memory, subject, attribute, value, confidence=.9, key=None):
        key = key or f"{attribute} of {subject.lower()}"
        memory.remember_fact(key, value, confidence, subject=subject, attribute=attribute)
        return key

    def world(self, memory, birthplace=True):
        self.hold(memory, "Emma", "author", "Jane Austen", .7)
        self.hold(memory, "Dune", "author", "Frank Herbert")
        self.hold(memory, "Frank Herbert", "birthplace", "Tacoma", .6)
        if birthplace:
            self.hold(memory, "Jane Austen", "birthplace", "Steventon")

    def test_chain_and_sentence_in_both_stores(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                session = build_session(self.scratch(), seed=0)
                if not sharded:
                    session.memory = SystematicMemory(session.root / "memory.json")
                memory = session.memory
                self.world(memory)
                for book, author, place, confidence in (
                        ("Emma", "Jane Austen", "Steventon", .7),
                        ("Dune", "Frank Herbert", "Tacoma", .6)):
                    text = f"What is the birthplace of the author of {book}?"
                    key1, key2 = f"author of {book.lower()}", f"birthplace of {author.lower()}"
                    self.assertEqual(memory.catalogue_answer(text), {
                        "form": "chain", "key": key2, "record": memory.recall_fact(key2),
                        "via": {"key": key1, "record": memory.recall_fact(key1)}})
                    reply, _ = run_pipeline(text, session)
                    self.assertEqual(reply, f"{key2} is {place} (confidence {confidence:.2f}), "
                                     f"through {key1} is {author}.")
                reply, _ = run_pipeline("What is the birthplace of Jane Austen?", session)
                self.assertEqual(reply, "birthplace of jane austen is Steventon (confidence 0.90).")
                self.assertEqual(session.curiosities, [])

    def test_missing_second_hop_has_no_reading_and_catalogue_curiosity(self):
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                session = build_session(self.scratch(), seed=0)
                if not sharded:
                    session.memory = SystematicMemory(session.root / "memory.json")
                memory = session.memory
                self.world(memory, birthplace=False)
                text = "What is the birthplace of the author of Emma?"
                self.assertIsNone(memory.catalogue_answer(text))
                self.assertEqual(missing_premise(text, memory)["premise_key"],
                                 "birthplace of jane austen")
                reply, _ = run_pipeline(text, session)
                self.assertFalse(reply.startswith("Reading that as"))
                self.assertEqual([c["premise_key"] for c in session.curiosities],
                                 ["birthplace of jane austen"])

    def test_direct_answer_stays_exact_and_chat_rejects_chain(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.world(memory)
                key = "birthplace of jane austen"
                text = "What is the birthplace of Jane Austen?"
                expected = {"form": "exact", "key": key, "record": memory.recall_fact(key)}
                self.assertEqual(memory.catalogue_answer(text), expected)
                self.assertEqual(memory.catalogue_request(text), expected)
                chain = "Tell me the birthplace of the author of Emma."
                self.assertIsNone(memory.catalogue_request(chain))
                ctx = ThoughtContext(chain, SimpleNamespace(memory=memory))
                Reason()._chat(ctx)
                self.assertEqual(ctx.response_parts,
                                 ["I have nothing on that yet. ':help' lists what I can do."])

    def test_second_hop_uses_subject_and_attribute_not_key(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                key = self.hold(memory, "The Café", "BirthPlace", "Here", key="opaque entry")
                self.hold(memory, "Cafeteria", "birthplace", "Wrong")
                self.hold(memory, "The Café", "author", "Wrong")
                memory.remember_fact("cafe birthplace", "Unstructured decoy")
                with mock.patch.object(memory, "fact_keys", side_effect=AssertionError("scan")), \
                        mock.patch.object(memory, "find_facts", side_effect=AssertionError("search")):
                    self.assertEqual(memory._second_hop("CAFE", "birthplace"),
                                     (key, memory.recall_fact(key)))
                    self.assertIsNone(memory._second_hop("missing", "birthplace"))
                    self.assertIsNone(memory._second_hop("cafe", "missing"))
                self.hold(memory, "123", "marker", "Found", key="numeric subject")
                self.assertEqual(memory._second_hop(123, "marker")[0], "numeric subject")

    def test_key_form_frequency_ties_and_ineligible_records(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.assertIsNone(memory.key_form("birthplace"))
                self.hold(memory, "Alpha", "birthplace", "A", key="alpha birthplace")
                self.hold(memory, "Beta", "birthplace", "B")
                self.assertEqual(memory.key_form("birthplace"), "birthplace of {subject}")
                self.hold(memory, "Gamma", "birthplace", "C", key="gamma birthplace")
                self.assertEqual(memory.key_form("birthplace"), "{subject} birthplace")
                self.hold(memory, "Delta", "birthplace", "D", key="delta delta birthplace")
                self.hold(memory, "Epsilon", "birthplace", "E", key="opaque")
                memory.remember_fact("birthplace of zeta", "Z", attribute="birthplace")
                self.hold(memory, "Eta", "author", "Writer")
                self.assertEqual(memory.key_form("birthplace"), "{subject} birthplace")
                self.assertIsNone(memory.key_form("absent"))

    def test_other_attribute_uses_attested_asking_words(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.world(memory, birthplace=False)
                memory.learn_asking("birthplace", "Alpha", "Where was Alpha cradled?")
                words = {"author", "cradled"}
                self.assertFalse(memory._names_other_attribute(words, {"author"}))
                memory.learn_asking("birthplace", "Beta", "Where was Beta cradled?")
                self.assertTrue(memory._names_other_attribute(words, {"author"}))
                self.assertFalse(memory._names_other_attribute(words, words))
                text = "Where was the author of Emma cradled?"
                self.assertIsNone(memory.catalogue_answer(text))
                self.assertEqual(missing_premise(text, memory)["premise_key"],
                                 "birthplace of jane austen")
                self.hold(memory, "Jane Austen", "birthplace", "Steventon")
                self.assertEqual(memory.catalogue_answer(text)["form"], "chain")

    def test_arbitrary_attributes_and_ambiguous_remainder(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.hold(memory, "Nacre", "velar", "Iris")
                self.hold(memory, "Iris", "surn", "Quartz", key="iris dossier")
                text = "What is the surn of the velar of Nacre?"
                self.assertEqual(memory.catalogue_answer(text)["key"], "iris dossier")
                self.hold(memory, "Elsewhere", "surn label", "Decoy")
                self.assertIsNone(memory.catalogue_answer(text))

    def test_curiosity_falls_back_without_unique_attribute_or_key_form(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.hold(memory, "Emma", "author", "Jane Austen")
                self.hold(memory, "Dune", "birthplace", "Somewhere", key="birthplace dossier")
                text = "What is the birthplace of the author of Emma?"
                self.assertEqual(missing_premise(text, memory)["premise_key"],
                                 "jane austen birthplace")
                self.hold(memory, "Dune", "birthplace", "Somewhere")
                self.hold(memory, "Echo", "birthplace label", "Elsewhere")
                self.assertEqual(missing_premise(text, memory)["premise_key"],
                                 "jane austen birthplace")
                memory.remember_fact("tower material", "steel")
                self.hold(memory, "Copper", "conductivity", "High")
                self.assertEqual(missing_premise("What is the tower conductivity?", memory)
                                 ["premise_key"], "steel conductivity")


if __name__ == "__main__":
    unittest.main()
