"""Section 11.145: chat mentions, vocabulary ambiguity, and indexed absence."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ultraquant.interpreter.thoughts import Reason, ThoughtContext
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class Review9Tests(unittest.TestCase):
    def memories(self):
        for sharded in (False, True):
            scratch = tempfile.TemporaryDirectory(prefix="uq_review9_test_")
            self.addCleanup(scratch.cleanup)
            root = Path(scratch.name)
            memory = SystematicMemory(path=root / "memory.json")
            if sharded:
                memory.shards = FactShards(ShardVault(root / "vault"))
            yield sharded, memory

    def hold(self, memory, subject, attribute, value):
        key = f"{attribute} of {subject.lower()}"
        memory.remember_fact(key, value, .9, subject=subject, attribute=attribute)
        return key

    def test_chat_mentions_catalogue_fact_and_preserves_exact_dict(self):
        for sharded, memory in self.memories():
            self.hold(memory, "Dune", "author", "Frank Herbert")
            self.hold(memory, "Emma", "author", "Jane Austen")
            for subject in ("Dune", "Emma"):
                memory.learn_asking("author", subject, f"Who wrote {subject}?")
            self.hold(memory, "Kenya", "capital", "Nairobi")
            for text, key in (("I wrote Dune.", "author of dune"),
                              ("I know the capital of Kenya.", "capital of kenya")):
                with self.subTest(sharded=sharded, text=text):
                    record = memory.recall_fact(key)
                    expected = {"form": "exact", "key": key, "record": record}
                    self.assertEqual(memory.catalogue_request(text), expected)
                    ctx = ThoughtContext(text, SimpleNamespace(memory=memory))
                    ctx.data["facts"] = [("distractor", {"value": "wrong"})]
                    Reason()._chat(ctx)
                    reply = "".join(ctx.response_parts)
                    self.assertNotIn("(confidence", reply)
                    self.assertEqual(reply, f"That lands near '{key}', which I hold as: "
                                     f"{record['value']}.")
                    self.assertIn("catalogue mention", ctx.trace[-1]["summary"])
                    self.assertEqual(memory.catalogue_request(text), expected)

    def test_chat_fallback_replies(self):
        for sharded, memory in self.memories():
            cases = [
                ({"facts": [("dome", {"value": "steel", "negated": True})]},
                 "That lands near 'dome', which I hold as: not steel."),
                ({"routes": [("world", 1.0)]},
                 "That reads as world. I have nothing stored on it yet."),
                ({}, "I have nothing on that yet. ':help' lists what I can do."),
            ]
            for data, expected in cases:
                with self.subTest(sharded=sharded, data=data):
                    ctx = ThoughtContext("hello", SimpleNamespace(memory=memory), data=data)
                    Reason()._chat(ctx)
                    self.assertEqual(ctx.response_parts, [expected])
                    self.assertEqual(ctx.trace[-1]["summary"], "conversational reply")

    def test_ambiguity_uses_whole_attribute_vocabulary(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                for subject in ("Echo", "Foxtrot"):
                    self.hold(memory, subject, "Author", "Main Writer")
                    self.hold(memory, subject, "Preface Author", "Preface Writer")
                    memory.learn_asking("Author", subject, f"Who wrote {subject}?")
                    memory.learn_asking("Preface Author", subject,
                                        f"Who wrote the preface of {subject}?")
                key = self.hold(memory, "Dune", "preface author", "Guest Writer")
                self.assertEqual(memory._asked_attributes(set()), set())
                self.assertEqual(memory._asked_attributes({"unexplained"}), set())
                self.assertEqual(memory._asked_attributes({"wrote"}),
                                 {"author", "preface author"})
                self.assertEqual(memory._asked_attributes({"wrote", "preface"}),
                                 {"preface author"})
                answer = memory.catalogue_answer("Who wrote Dune?")
                self.assertEqual(answer["form"], "reading")
                self.assertEqual(answer["record"]["value"], "Guest Writer")
                self.assertIsNone(memory.catalogue_request("Who wrote Dune?"))
                self.assertEqual(memory.catalogue_answer("Who wrote the preface of Dune?"),
                                 {"form": "exact", "key": key,
                                  "record": memory.recall_fact(key)})

    def test_absence_is_exhaustive_and_uses_index_tokens(self):
        for sharded, memory in self.memories():
            self.hold(memory, "Kenya", "capital", "Nairobi")
            memory.remember_fact("freedonia capital", "Fredville", .6)
            memory.remember_fact("café capital", "Cafeville", .6)
            for n in range(10):
                memory.remember_fact(f"freedonia dossier {n}", f"file {n}", .9,
                                     subject=f"Holder {n}", attribute="dossier")
            for persisted in (False, True):
                if persisted:
                    memory.save()
                    if sharded:
                        memory.shards = FactShards(ShardVault(memory.shards.vault.root))
                with self.subTest(sharded=sharded, persisted=persisted), mock.patch.object(
                        memory, "find_facts", side_effect=AssertionError("ranked lookup")):
                    for subject in ("Freedonia", "Café"):
                        self.assertIsNone(memory.catalogue_answer(
                            f"What is the capital of {subject}?"))
                    self.assertEqual(memory.catalogue_answer(
                        "What is the capital of the US state of Kessaway?"),
                        {"form": "unknown-subject"})
                    with mock.patch.object(memory, "_unheld_subject",
                                           wraps=memory._unheld_subject) as check:
                        memory.catalogue_answer("What is the capital of Café?")
                        self.assertEqual(check.call_args.args[0], {"capital", "caf"})

    def test_uninformative_remainder_is_inconclusive(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.hold(memory, "Kenya", "capital", "Nairobi")
                self.assertFalse(memory._unheld_subject({"capital", "us"}, {"capital"}))
                self.assertFalse(memory._unheld_subject({"capital"}, {"capital"}))
                self.assertIsNone(memory.catalogue_answer("What is the capital of US?"))

    def test_keys_covering_pages_every_matching_bucket_and_staged_facts(self):
        _, memory = list(self.memories())[1]
        shards = memory.shards
        expected = []
        for n in range(20):
            key = f"freedonia capital {n}"
            expected.append(key)
            memory.remember_fact(key, str(n), subject=f"Holder {n}", attribute="capital")
        memory.remember_fact("freedonia dossier", "unrelated")
        memory.save()
        shards = FactShards(ShardVault(shards.vault.root))
        wanted = {"freedonia", "capital"}
        categories = set.intersection(*(set(shards.vault.association_scores({token}))
                                        for token in wanted))
        self.assertGreater(len(categories), 8)
        with mock.patch.object(shards, "_load", wraps=shards._load) as page:
            self.assertEqual(shards.keys_covering(wanted), sorted(expected))
            self.assertEqual({call.args[0] for call in page.call_args_list}, categories)
        shards.put("freedonia capital staged", {"value": "staged"})
        self.assertEqual(shards.keys_covering(wanted), sorted(expected + ["freedonia capital staged"]))
        self.assertEqual(shards.keys_covering(set()), [])


if __name__ == "__main__":
    unittest.main()
