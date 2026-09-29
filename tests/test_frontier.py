"""Shape-driven study using fake teachers and scratch stores only."""

from dataclasses import FrozenInstanceError
import json
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ultraquant.distill import elicit, file, frontier, sources, targets
from ultraquant.distill.teachers import TeacherSpec
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class FakeTeacher:
    def __init__(self, home, replies=None):
        gguf = home / "fake.gguf"
        name, value = b"general.name", b"Fake teacher"
        gguf.write_bytes(struct.pack("<4sIQQQ", b"GGUF", 3, 0, 1, len(name))
                         + name + struct.pack("<IQ", 8, len(value)) + value)
        self.spec = TeacherSpec("c4ai-command-r-08-2024", gguf)
        self.replies = replies or {}
        self.calls = []

    def ask(self, questions, **kwargs):
        self.calls.append((list(questions), kwargs))
        return [self.replies[q] for q in questions]


class DenseWindowTests(unittest.TestCase):
    def test_comes_back_requires_normalized_equality(self):
        self.assertTrue(frontier.comes_back("The Alpha.", "alpha"))
        self.assertTrue(frontier.comes_back(39, "39"))
        self.assertFalse(frontier.comes_back(None, "39"))
        self.assertFalse(frontier.comes_back("19", "39"))
        self.assertFalse(frontier.comes_back("named alpha", "alpha"))

    def test_outlier_does_not_extend_the_dense_range(self):
        values = [n for n in range(1, 119) if n not in (21, 23, 37, 39, 40)]
        self.assertEqual(frontier.dense_window(values + [276]), (1, 118))

    def test_support_density_ties_and_distinct_values(self):
        self.assertIsNone(frontier.dense_window([]))
        self.assertIsNone(frontier.dense_window([1, 2, 3, 4]))
        self.assertEqual(frontier.dense_window([1, 2, 3, 4, 6]), (1, 6))
        self.assertEqual(frontier.dense_window([1, 2, 3, 4, 7]), (1, 4))
        self.assertEqual(frontier.dense_window(
            [1, 2, 3, 4, 20, 21, 22, 24]), (20, 24))
        self.assertEqual(frontier.dense_window(
            [1, 2, 3, 5, 20, 21, 22, 24]), (1, 5))
        self.assertEqual(frontier.dense_window(
            [1, 2, 3, 4, 5] + [100] * 30), (1, 5))


class MemoryFrontierTests(unittest.TestCase):
    sharded = False

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="uq_frontier_test_")
        self.addCleanup(scratch.cleanup)
        self.home = Path(scratch.name)
        self.memory = self.open_memory()
        self.stash = ContemporaryStash(self.home / "stash.json")
        self.teacher = FakeTeacher(self.home)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(sources.LMStudioTeacher, "ask",
                          side_effect=AssertionError("Live teacher forbidden")).start()

    def open_memory(self):
        shards = (FactShards(ShardVault(self.home / "vault"))
                  if self.sharded else None)
        return SystematicMemory(self.home / "memory.json", shards=shards)

    def fact(self, number, attribute="atomic number", value=None):
        subject = f"Member {number}"
        self.memory.remember_fact(
            f"{attribute} of {subject.lower()}",
            str(number) if value is None else value,
            subject=subject, attribute=attribute)

    def form(self, attribute="atomic number", category="number"):
        subject, value = "Template", "Label"
        qid = f"{category}:{subject}"
        entry_id = self.stash.add_claim(
            f"https://distill.invalid/seed/{qid}",
            f"What is the {attribute} of {subject}?",
            f"The {attribute} of {subject} is {value}.",
            provenance={"run_id": "seed", "question_id": qid},
            fields={"subject": subject, "attribute": attribute, "value": value,
                    "key": f"{attribute} of {subject.lower()}"})
        self.stash._entries[entry_id]["status"] = "promoted"
        self.stash.save()
        return entry_id

    def kind_reply(self, answers=None):
        subjects = sorted({self.memory.recall_fact(key)["subject"]
                           for key in self.memory.fact_keys()
                           if self.memory.recall_fact(key).get("attribute") == "atomic number"})
        a, b, c = random.Random(158).sample(subjects, 3)
        question = frontier._seed_questions()["kind"].format(a=a, b=b, c=c)
        self.teacher.replies[question] = answers or ["element"] * 5
        return question

    def test_sequence_gaps_and_verification_queue(self):
        missing = [21, 23, 37, 39, 40]
        for number in range(1, 119):
            if number not in missing:
                self.fact(number)
        self.fact(276)
        expected = {"attribute": "atomic number", "window": (1, 118),
                    "missing": missing,
                    "outliers": [("atomic number of member 276", "276")]}
        self.assertEqual(frontier.sequence_gaps(self.memory), [expected])
        self.assertEqual(frontier.verification_queue(self.memory), expected["outliers"])
        self.assertEqual(self.memory.recall_fact(
            "atomic number of member 276")["value"], "276")
        with mock.patch.object(frontier, "dense_window", return_value=(1, 276)):
            self.assertEqual(frontier.verification_queue(self.memory), [])

    def test_forward_value_uses_normalized_catalogue_in_both_stores(self):
        self.memory.remember_fact("unrelated key", "19", subject="Potassium",
                                  attribute="Atomic Number")
        self.assertEqual(frontier.forward_value(
            self.memory, " POTASSIUM ", " ATOMIC NUMBER "), "19")
        self.assertIsNone(frontier.forward_value(self.memory, "Yttrium", "atomic number"))
        self.assertIsNone(frontier.forward_value(self.memory, "Potassium", "chemical symbol"))
        self.memory.save()
        self.assertEqual(frontier.forward_value(
            self.open_memory(), "potassium", "atomic number"), "19")

    def test_ask_forward_learned_forms_order_deduplication_and_missing_form(self):
        entry_id = self.form()
        self.stash._entries[entry_id]["title"] = "Stored form <Template> {literal}"
        self.stash.save()
        pairs = [("Second", "atomic number"), ("First", "atomic number"),
                 ("Second", "atomic number"), ("Missing", "no form")]
        questions = [f"Stored form <{subject}> {{literal}}" for subject in ("Second", "First")]
        self.teacher.replies.update({questions[0]: ["39"] * 5,
                                     questions[1]: ["UNKNOWN"] * 5})
        path = self.home / "forward.jsonl"
        path.write_text("", encoding="utf-8")
        with mock.patch.object(elicit, "elicit", wraps=elicit.elicit) as ask, \
                mock.patch.object(sources, "single_source_decide",
                                  wraps=sources.single_source_decide) as decide:
            answers = frontier.ask_forward(self.stash, self.teacher, "sole source", pairs, path)
        self.assertEqual(answers, {pairs[0]: "39", pairs[1]: None})
        self.assertEqual(self.teacher.calls[0][0], questions)
        ask.assert_called_once()
        decide.assert_called_once()
        self.assertTrue(all(r.lineage == "sole source" for r in elicit.load_records(path)))
        self.assertEqual(frontier.ask_forward(
            self.stash, self.teacher, "sole source", [("X", "no form")], path), {})
        self.assertEqual(len(self.teacher.calls), 1)

    def test_held_wrong_reverse_answer_is_queued_without_forward_question(self):
        for number in (1, 2, 3, 4, 6):
            self.fact(number)
        self.form()
        self.memory.learn_kind("atomic number", "element")
        target = frontier.reverse_targets(self.memory, self.stash, self.teacher)[0]
        self.teacher.replies[target.question] = ["Member 2"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        with mock.patch.object(frontier, "ask_forward", wraps=frontier.ask_forward) as ask:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id="held", records_path=self.home / "held.jsonl")
        ask.assert_not_called()
        self.assertEqual(result, {"asked": 1, "filed": 0, "queued": 1, "used_up": True})
        self.assertEqual(frontier.forward_value(self.memory, "Member 2", "atomic number"), "2")
        self.assertEqual(ledger.history("source"), [{
            "question_id": "number:5", "promoted": False, "queued": {
                "key": "atomic number of member 2", "value": "5", "subject": "Member 2",
                "attribute": "atomic number", "forward": "2", "forward_from": "index"}}])
        self.assertEqual(len(self.stash.entries()), 1)
        self.assertEqual(self.teacher.calls[0][0], [target.question])
        self.assertEqual(frontier.pending(self.memory, self.stash, self.teacher, ledger, "source"), [])
        self.assertEqual(frontier.pending(self.memory, self.stash, self.teacher, ledger, "next"), [target])
        self.assertIn(("atomic number of member 2", "5"), frontier.verification_queue(self.memory, ledger))

    def test_two_gaps_same_unheld_subject_ask_once_and_only_matching_gap_files(self):
        for number in range(1, 13):
            if number not in (6, 9):
                self.fact(number)
        self.form()
        self.memory.learn_kind("atomic number", "element")
        reverse = frontier.reverse_targets(self.memory, self.stash, self.teacher)
        for target in reverse:
            self.teacher.replies[target.question] = ["Member Six", "**Member Six**",
                                                     "Member Six.", "member six", "UNKNOWN"]
        question = "What is the atomic number of Member Six?"
        self.teacher.replies[question] = ["6"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        path = self.home / "unheld.jsonl"
        with mock.patch.object(frontier, "ask_forward", wraps=frontier.ask_forward) as ask, \
                mock.patch.object(frontier, "comes_back", wraps=frontier.comes_back) as back:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id="unheld", records_path=path)
        ask.assert_called_once()
        self.assertEqual(back.call_count, 2)
        self.assertEqual(result, {"asked": 2, "filed": 1, "queued": 1, "used_up": True})
        self.assertEqual(self.teacher.calls[1][0], [question])
        self.assertEqual(len(self.teacher.calls), 2)
        self.assertEqual(frontier.forward_value(self.memory, "Member Six", "atomic number"), "6")
        rows = ledger.history("source")
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["promoted"])
        self.assertEqual(rows[1]["queued"], {
            "key": "atomic number of member six", "value": "9", "subject": "Member Six",
            "attribute": "atomic number", "forward": "6", "forward_from": "source"})
        self.assertFalse(rows[1]["promoted"])
        recorded = elicit.load_records(path)
        self.assertEqual(len(recorded), 15)
        self.assertEqual([r.question for r in recorded[-5:]], [question] * 5)
        self.assertEqual([t.value for t in frontier.pending(
            self.memory, self.stash, self.teacher, ledger, "next")], ["9"])

    def test_round_trip_missing_forward_answer_and_patched_seams(self):
        target = frontier.ReverseTarget("number", "5", "reverse form", "atomic number", "5")
        records = [elicit.Record("fake", "source", "fake.gguf", 4, "number:5",
                                 target.question, seed, "New Member", "ignored")
                   for seed in elicit.SEEDS]
        path = self.home / "missing.jsonl"
        expected = {"subject": "New Member", "forward": None,
                    "forward_from": "source", "back": False}
        self.assertEqual(frontier.round_trip(
            self.memory, self.stash, self.teacher, "source", [target], records, path),
            {"number:5": expected})
        self.assertEqual(self.teacher.calls, [])
        with mock.patch.object(frontier, "forward_value", return_value="5") as held, \
                mock.patch.object(frontier, "ask_forward") as ask, \
                mock.patch.object(frontier, "comes_back", return_value=True) as back:
            result = frontier.round_trip(
                self.memory, self.stash, self.teacher, "source", [target], records, path)
        held.assert_called_once_with(self.memory, "New Member", "atomic number")
        ask.assert_not_called()
        back.assert_called_once_with("5", "5")
        self.assertEqual(result["number:5"], {**expected, "forward": "5",
                                             "forward_from": "index", "back": True})

    def test_ledger_old_format_and_queued_claim_order_and_deduplication(self):
        for number in range(1, 8):
            self.fact(number)
        self.fact(276)
        path = self.home / "ledger.json"
        old = {"old": [{"question_id": "q0", "promoted": True}]}
        path.write_text(json.dumps(old), encoding="utf-8")
        ledger = sources.SourceLedger(path)
        self.assertEqual(ledger.history("old"), old["old"])
        self.assertEqual(ledger.asked("old"), {"q0"})
        first = {"key": "atomic number of member 276", "value": 276}
        second = {"key": "queued key", "value": 39}
        ledger.record("old", "q1", False, queued=first)
        ledger.record("next", "q2", False, queued=second)
        ledger.record("next", "q3", False, queued=first)
        ledger.record("third", "q4", False, queued=second)
        reopened = sources.SourceLedger(path)
        self.assertEqual(reopened.history("old")[1]["queued"], first)
        self.assertEqual(list(json.loads(path.read_text(encoding="utf-8"))), ["old", "next", "third"])
        self.assertEqual(frontier.verification_queue(self.memory, reopened), [
            ("atomic number of member 276", "276"), ("queued key", "39")])

    def test_integer_share_boundary_and_multiple_attributes(self):
        for number in range(1, 9):
            self.fact(number, "rank", f" {number} ")
            self.fact(number, "position")
        self.fact(20, "rank", "unknown")
        self.fact(21, "rank", "4.5")
        self.assertEqual([g["attribute"] for g in frontier.sequence_gaps(self.memory)],
                         ["position", "rank"])
        self.fact(22, "rank", "-1")
        self.assertEqual([g["attribute"] for g in frontier.sequence_gaps(self.memory)],
                         ["position"])

    def test_kind_is_normalized_persisted_and_reused(self):
        for number in range(1, 7):
            self.fact(number)
        question = self.kind_reply(["Element.", "**element**", "ELEMENT",
                                    "substance", "UNKNOWN"])
        self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                         "element")
        self.assertEqual(self.teacher.calls[0][0], [question])
        self.assertEqual(self.teacher.calls[0][1]["samples"], 5)
        self.assertEqual(self.teacher.calls[0][1]["system"], elicit.SYSTEM)
        self.memory.learn_asking("atomic number", "Member 1", "Number for Member 1?")
        self.memory.save()
        reopened = self.open_memory()
        self.assertEqual(frontier.kind_of(reopened, "atomic number", self.teacher),
                         "element")
        self.assertEqual(len(self.teacher.calls), 1)
        self.assertEqual(reopened._attribute_vocabulary()["atomic number"]["kind"],
                         "element")
        if self.sharded:
            self.assertEqual(reopened.shards.vault.get("index:attributes")
                             ["attributes"]["atomic number"]["kind"], "element")

    def test_kind_requires_majority_and_seed(self):
        for number in range(1, 7):
            self.fact(number)
        self.kind_reply(["element", "element", "substance", "substance", "UNKNOWN"])
        self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertNotIn("kind", self.memory._attribute_vocabulary()["atomic number"])
        with mock.patch.object(frontier, "_seed_questions", return_value={}):
            self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertEqual(len(self.teacher.calls), 1)

    def test_reverse_questions_use_only_seed_kind_and_question_category(self):
        for number in (1, 2, 3, 4, 6):
            self.fact(number)
        self.form(category="custom")
        self.memory.learn_kind("atomic number", "element")
        seed = frontier._seed_questions()
        expected = frontier.ReverseTarget("custom", "5", seed["reverse"].format(
            kind="element", attribute="atomic number", value="5"), "atomic number", "5")
        self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher),
                         [expected])
        self.assertEqual(elicit.question_id(expected), "custom:5")
        with self.assertRaises(FrozenInstanceError):
            expected.value = "6"
        with mock.patch.object(frontier, "_seed_questions", return_value={}):
            self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher), [])
        with mock.patch.object(frontier, "_seed_questions", return_value={
                "reverse": "{value}/{attribute}/{kind}"}):
            self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher)
                             [0].question, "5/atomic number/element")
        self.assertEqual(self.teacher.calls, [])

    def test_file_reverse_preserves_subject_value_forms_and_provenance(self):
        target = frontier.ReverseTarget("number", "5", "seed question", "atomic number", "5")
        records = [elicit.Record("fake", "source", "fake.gguf", 4, "number:5",
                                 target.question, seed, raw, "ignored")
                   for seed, raw in enumerate(["Member Five", "**Member Five**",
                                                "Member Five.", "member five", "UNKNOWN"])]
        with mock.patch.object(sources, "single_source_decide",
                               wraps=sources.single_source_decide) as decide:
            ids = frontier.file_reverse(self.stash, records, [target], 0.97, "run")
        decide.assert_called_once()
        entry = self.stash.get(ids[0])
        self.assertEqual(entry["fields"], {"key": "atomic number of member five",
                         "value": "5", "subject": "Member Five", "attribute": "atomic number"})
        self.assertEqual(entry["claim"], "The atomic number of Member Five is 5.")
        self.assertEqual(entry["provenance"]["question_id"], "number:5")
        self.assertEqual(entry["provenance"]["run_id"], "run")
        self.assertEqual(entry["provenance"]["teachers"], ["fake"])
        self.assertEqual(entry["provenance"]["lineages"], ["source"])
        self.assertEqual(frontier.file_reverse(self.stash, records, [target], 0.97, "run"), [])
        self.form()
        with mock.patch.object(file, "claim_form", return_value="{subject} ranks {value}."), \
                mock.patch.object(file, "key_form", return_value="rank/{subject}"), \
                mock.patch.object(file, "_seed", return_value={}):
            ids = frontier.file_reverse(self.stash, records, [target], 0.97, "learned")
        self.assertEqual(self.stash.get(ids[0])["claim"], "Member Five ranks 5.")
        self.assertEqual(self.stash.get(ids[0])["fields"]["key"], "rank/member five")

    def test_study_round_adds_members_then_symbols_then_exhausts(self):
        for number in range(1, 13):
            if number not in (6, 9):
                self.fact(number)
                self.fact(number, "chemical symbol", f"M{number}")
        self.form()
        self.form("chemical symbol", "symbol")
        self.kind_reply()
        seed = frontier._seed_questions()
        for number in (6, 9):
            self.teacher.replies[seed["reverse"].format(
                kind="element", attribute="atomic number", value=number)] = [f"Member {number}"] * 5
            self.teacher.replies[f"What is the chemical symbol of Member {number}?"] = [f"M{number}"] * 5
            self.teacher.replies[f"What is the atomic number of Member {number}?"] = [str(number)] * 5
        self.assertEqual(targets.completion_targets(self.memory, self.stash), [])
        ledger = sources.SourceLedger(self.home / "ledger.json")
        outcomes = []
        for round_number in range(1, 4):
            with mock.patch.object(sources, "used_up", wraps=sources.used_up) as used_up:
                outcomes.append(frontier.study_round(
                    self.memory, self.stash, self.teacher, ledger, self.teacher.spec.name,
                    confidence=0.99, run_id=f"round-{round_number}",
                    records_path=self.home / f"round-{round_number}.jsonl"))
                used_up.assert_called_once()
            if round_number == 1:
                self.assertEqual([t.subject for t in targets.completion_targets(
                    self.memory, self.stash)], ["Member 6", "Member 9"])
        self.assertEqual([row["asked"] for row in outcomes], [2, 2, 0])
        self.assertEqual([row["filed"] for row in outcomes], [2, 2, 0])
        self.assertEqual([row["used_up"] for row in outcomes], [False, True, True])
        self.assertTrue(all(type(row["used_up"]) is bool for row in outcomes))
        self.assertEqual([row["queued"] for row in outcomes], [0, 0, 0])
        self.assertEqual(len(self.teacher.calls), 4)
        self.assertEqual(len(ledger.history(self.teacher.spec.name)), 4)
        self.assertTrue(all(row["promoted"] for row in ledger.history(self.teacher.spec.name)))
        for number in (6, 9):
            self.assertEqual(self.memory.recall_fact(
                f"chemical symbol of member {number}")["value"], f"M{number}")

    def test_abstentions_are_recorded_and_not_asked_again(self):
        for number in (1, 2, 3, 4, 6):
            self.fact(number)
        self.form()
        self.memory.learn_kind("atomic number", "element")
        question = frontier.reverse_targets(self.memory, self.stash, self.teacher)[0].question
        self.teacher.replies[question] = ["UNKNOWN"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        for index in range(2):
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id=str(index), records_path=self.home / f"{index}.jsonl")
            self.assertEqual(result["asked"], 1 - index)
            self.assertEqual(result["filed"], 0)
        self.assertEqual(ledger.history("source"), [{"question_id": "number:5", "promoted": False}])
        self.assertEqual(len(self.teacher.calls), 1)


class ShardedFrontierTests(MemoryFrontierTests):
    sharded = True


if __name__ == "__main__":
    unittest.main()
