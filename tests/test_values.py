"""Section 11.154: indexed values and evidence that names them as values."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.memory.factshards import FactShards, question_words
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class ValuesTests(unittest.TestCase):
    def scratch(self):
        directory = tempfile.TemporaryDirectory(prefix="uq_values_test_")
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def memories(self):
        for sharded in (False, True):
            root = self.scratch()
            shards = FactShards(ShardVault(root / "vault")) if sharded else None
            yield sharded, SystematicMemory(root / "memory.json", shards=shards)

    def hold(self, memory, subject, attribute, value, key=None):
        key = key or f"{attribute} of {subject.lower()}"
        memory.remember_fact(key, value, .8, subject=subject, attribute=attribute)
        return key

    def reopen(self, memory):
        memory.save()
        shards = (FactShards(ShardVault(memory.shards.vault.root))
                  if memory.shards is not None else None)
        return SystematicMemory(memory.path, shards=shards)

    def test_membership_follows_put_value_change_structure_change_and_delete(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                key = self.hold(memory, "Kenya", "capital", "The NÁIROBI!")
                second = self.hold(memory, "Elsewhere", "seat", "NAIROBI")
                memory.remember_fact("unstructured", "Nairobi")
                memory.remember_fact("subject only", "Nairobi", subject="Other")
                memory.remember_fact("attribute only", "Nairobi", attribute="capital")
                self.assertEqual(memory.value_keys("nairobi"), sorted([key, second]))
                if sharded:
                    self.assertEqual(memory.shards._index_data("values", "nairobi")["nairobi"],
                                     {"keys": sorted([key, second])})
                memory.remember_fact(key, "Mombasa")
                self.assertEqual(memory.value_keys("nairobi"), [second])
                self.assertEqual(memory.value_keys("mombasa"), [key])
                memory = self.reopen(memory)
                self.assertEqual(memory.value_keys("mombasa"), [key])
                record = memory.recall_fact(second)
                record.pop("attribute")
                memory.restore_fact(second, record)
                self.assertEqual(memory.value_keys("nairobi"), [])
                memory.restore_fact(key, None)
                self.assertEqual(memory.value_keys("mombasa"), [])
                memory = self.reopen(memory)
                self.assertEqual(memory.value_keys("mombasa"), [])

    def test_old_sharded_library_migrates_once_and_persists_marker(self):
        for structured in (False, True):
            for derivations in (False, True):
                with self.subTest(structured=structured, derivations=derivations):
                    root = self.scratch()
                    vault = ShardVault(root / "vault")
                    shards = FactShards(vault)
                    key = "capital of kenya"
                    record = {"value": "Nairobi", "confidence": .8}
                    if structured:
                        record.update(subject="Kenya", attribute="capital")
                    bucket = shards.bucket_of(key)
                    vault.add_shard(bucket, bucket, {"facts": {key: record}}, kind="fact-bucket")
                    if derivations:
                        vault.add_shard("index:derivations:00", "index:derivations:00",
                                        {"derived": {}}, kind="fact-index")
                    memory = SystematicMemory(root / "memory.json", shards=shards)
                    load = shards._load

                    def in_batch(sid):
                        self.assertGreater(vault._defer_save, 0)
                        return load(sid)

                    with mock.patch.object(shards, "_load", side_effect=in_batch) as scan:
                        expected = [key] if structured else []
                        self.assertEqual(memory.value_keys("nairobi"), expected)
                        self.assertEqual(memory.value_keys("nairobi"), expected)
                        self.assertEqual(scan.call_count, 1)
                    self.assertFalse(vault.has("index:values:00"))
                    shards.flush()
                    self.assertTrue(vault.has("index:values:00"))
                    fresh = FactShards(ShardVault(root / "vault"))
                    memory = SystematicMemory(root / "memory.json", shards=fresh)
                    with mock.patch.object(fresh, "_load", side_effect=AssertionError("rescanned")):
                        self.assertEqual(memory.value_keys("nairobi"), expected)
                        fresh.flush()
                    self.assertTrue(fresh.vault.has("index:values:00"))

    def test_value_attributes_require_names_or_informativeness(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                for subject, value in (("Indium", "In"), ("Helium", "He")):
                    self.hold(memory, subject, "Chemical Symbol", value)
                self.hold(memory, "Beryllium", "atomic number", 4)
                self.hold(memory, "Kenya", "capital", "Nairobi")
                for value, question in (("in", "What is in the box?"),
                                        ("he", "Where did he go?"),
                                        ("4", "Which planet is number 4?")):
                    self.assertIsNone(memory.value_attributes(value, question_words(question, value)))
                    self.assertIsNone(memory.catalogue_by_value(question))
                self.assertEqual(memory.value_attributes("in", {"chemical", "symbol"}),
                                 {"chemical symbol"})
                self.assertIsNone(memory.value_attributes("in", {"symbol"}))
                memory.learn_asking("Chemical Symbol", "Indium", "Which glyph labels Indium?")
                self.assertIsNone(memory.value_attributes("he", {"glyph"}))
                memory.learn_asking("Chemical Symbol", "Indium", "Which glyph labels Indium?")
                self.assertIsNone(memory.value_attributes("he", {"glyph"}))
                memory.learn_asking("Chemical Symbol", "Helium", "Which glyph labels Helium?")
                self.assertEqual(memory.value_attributes("he", {"glyph"}), {"chemical symbol"})
                self.assertEqual(memory.value_attributes("nairobi", {"unrelated"}), {"capital"})
                self.assertEqual(memory.value_attributes("4", {"atomic", "number"}), {"atomic number"})

    def test_normalized_value_spans_do_not_lose_a_second_article(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                key = self.hold(memory, "First item", "title", "The A Tower")
                self.hold(memory, "Second item", "title", "Tower")
                self.assertEqual(memory.value_keys("a tower"), [key])
                answer = memory.catalogue_by_value("Which title is the a tower?")
                self.assertEqual(answer["value"], "a tower")
                self.assertEqual([item["key"] for item in answer["records"]], [key])

    def test_catalogue_nested_values_holders_filter_and_subject_precedence(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.hold(memory, "Short title", "author", "Austen")
                emma = self.hold(memory, "Emma", "author", "Jane Austen")
                pride = self.hold(memory, "Pride and Prejudice", "author", "Jane Austen")
                editor = self.hold(memory, "Collected letters", "editor", "Jane Austen")
                self.hold(memory, "Brave New World", "author", "Aldous Huxley")
                answer = memory.catalogue_by_value("What did Jane Austen write?")
                self.assertEqual(answer["form"], "value")
                self.assertEqual(answer["value"], "jane austen")
                self.assertEqual(answer["attributes"], ["author", "editor"])
                self.assertEqual([item["key"] for item in answer["records"]], sorted([emma, pride, editor]))
                self.assertEqual(answer["records"][0]["record"], memory.recall_fact(emma))
                answer = memory.catalogue_by_value("Which author is Jane Austen?")
                self.assertEqual(answer["attributes"], ["author"])
                self.assertEqual([item["key"] for item in answer["records"]], sorted([emma, pride]))
                self.assertIsNone(memory.catalogue_by_value("What links Jane Austen and Aldous Huxley?"))
                self.assertIsNone(memory.catalogue_by_value("Did Emma have author Jane Austen?"))
                self.assertIsNone(memory.catalogue_by_value("What about an absent value?"))

    def test_chat_lists_returned_records_exactly_and_obeys_truncation(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                session = build_session(self.scratch(), seed=0)
                session.memory = memory
                emma = self.hold(memory, "Emma", "author", "Jane Austen")
                pride = self.hold(memory, "Pride and Prejudice", "author", "Jane Austen")
                question = "What did Jane Austen write?"
                expected = (f"Emma: {emma} is Jane Austen (confidence 0.80); "
                            f"Pride and Prejudice: {pride} is Jane Austen (confidence 0.80).")
                self.assertEqual(run_pipeline(question, session)[0], expected)
                answer = memory.catalogue_by_value(question)
                answer["records"] = answer["records"][:1]
                with mock.patch.object(memory, "catalogue_by_value", return_value=answer):
                    self.assertEqual(run_pipeline(question, session)[0],
                                     f"Emma: {emma} is Jane Austen (confidence 0.80).")

    def test_named_value_avoids_refusal_but_unknown_value_is_refused(self):
        for sharded, memory in self.memories():
            with self.subTest(sharded=sharded):
                self.hold(memory, "Dysprosium", "chemical symbol", "Dy")
                question = "Which element has the chemical symbol Dy?"
                self.assertIsNone(memory.catalogue_answer(question))
                with mock.patch.object(memory, "catalogue_by_value", return_value=None):
                    self.assertEqual(memory.catalogue_answer(question), {"form": "unknown-subject"})
                unknown = "Which element has the chemical symbol Qx?"
                self.assertEqual(memory.catalogue_answer(unknown), {"form": "unknown-subject"})
                session = build_session(self.scratch(), seed=0)
                session.memory = memory
                self.assertIn("Dysprosium:", run_pipeline(question, session)[0])
                self.assertEqual(run_pipeline(unknown, session)[0],
                                 "I don't hold that: nothing it names is in my catalogue.")


if __name__ == "__main__":
    unittest.main()
