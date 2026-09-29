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
    def test_adopts_requires_four_decisions_and_five_probes(self):
        for decided, asked, expected in ((3, 5, False), (4, 5, True),
                                         (5, 5, True), (4, 4, False),
                                         (3, 3, False), (0, 0, False)):
            with self.subTest(decided=decided, asked=asked):
                self.assertIs(frontier.adopts(decided, asked), expected)

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
        gguf = self.teacher.spec.gguf
        index = {**sources._teacher_index(),
                 self.teacher.spec.name: f"{gguf.name}:{gguf.stat().st_size}"}
        self.enterContext(mock.patch.object(sources, "_teacher_index", return_value=index))
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

    def kind_reply(self, answers=None, *, rows=None, seed=158):
        subjects = sorted({self.memory.recall_fact(key)["subject"]
                           for key in self.memory.fact_keys()
                           if self.memory.recall_fact(key).get("attribute") == "atomic number"})
        members = random.Random(seed).sample(subjects, 3)
        questions = [frontier._seed_questions()["kind"].format(a=member)
                     for member in members]
        if rows is None:
            rows = [answers if answers is not None else ["element"] * elicit.SAMPLES] * 3
        self.teacher.replies.update(zip(questions, rows))
        return questions

    def property_world(self, attribute="atomic number", kind="element",
                       candidate="atomic mass", *, count=8, decided=5):
        for number in range(1, count + 1):
            self.fact(number, attribute)
        self.memory.learn_kind(attribute, kind)
        seeds = frontier._seed_questions()
        question = seeds["property"].format(kind=kind)
        self.teacher.replies[question] = [candidate] * 5
        members = random.Random(161).sample(
            sorted(f"Member {number}" for number in range(1, count + 1)), min(5, count))
        for index, member in enumerate(members):
            self.teacher.replies[seeds["forward"].format(
                attribute=candidate, subject=member)] = ["12.5" if index < decided else "UNKNOWN"] * 5
        return question, members

    def propose(self, ledger=None, run_id="proposal"):
        if ledger is None:
            ledger = sources.SourceLedger(self.home / "ledger.json")
        return frontier.propose(
            self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
            run_id=run_id, records_path=self.home / f"{run_id}.jsonl")

    def test_cluster_normalizes_attributes_and_keeps_recorded_subjects(self):
        self.memory.remember_fact("one", "1", subject="The Íron", attribute="Atomic Number")
        self.memory.remember_fact("two", "2", subject="iron", attribute="atomic-number")
        self.memory.remember_fact("three", "3", subject="Outside", attribute="rank")
        self.memory.remember_fact("unstructured", "4")
        self.memory.remember_fact("no subject", "5", attribute="atomic number")
        self.assertEqual(frontier.cluster(self.memory, " THE ATOMIC NUMBER "), {"The Íron", "iron"})
        self.assertEqual(frontier.cluster(self.memory, "missing"), set())

    def test_learn_property_preserves_entries_and_persists_both_verdicts(self):
        self.fact(1)
        self.memory.learn_kind("atomic number", "element")
        self.memory.learn_kind("atomic mass", "stored kind")
        self.memory.learn_property("Atomic Number", "atomic mass", "adopted")
        self.memory.learn_property("atomic number", "density", "refused")
        self.memory.learn_asking("atomic mass", "Member 1", "Mass for Member 1?")
        self.memory.save()
        vocabulary = self.open_memory()._attribute_vocabulary()
        self.assertEqual(vocabulary["atomic number"]["properties"], {
            "atomic mass": "adopted", "density": "refused"})
        self.assertEqual(vocabulary["atomic number"]["kind"], "element")
        self.assertEqual(vocabulary["atomic mass"]["extends"], "Atomic Number")
        self.assertEqual(vocabulary["atomic mass"]["kind"], "stored kind")
        self.assertNotIn("density", vocabulary)
        if self.sharded:
            self.assertEqual(self.open_memory().shards.vault.get("index:attributes")
                             ["attributes"]["atomic number"]["properties"],
                             vocabulary["atomic number"]["properties"])
        with self.assertRaises(ValueError):
            self.memory.learn_property("atomic number", "density", "maybe")

    def test_propose_one_per_agreeing_kind_sorted_and_held_candidate_dropped(self):
        # Insert out of order; the first normalized attribute supplies the kind.
        self.memory.learn_kind("zulu", "chemical element")
        self.memory.learn_kind("Beta", "country")
        self.memory.learn_kind("Alpha", "element")
        seeds = frontier._seed_questions()
        questions = [seeds["property"].format(kind=kind) for kind in ("element", "country")]
        self.teacher.replies.update({questions[0]: ["the Alpha"] * 5,
                                     questions[1]: ["UNKNOWN"] * 5})
        with mock.patch.object(frontier, "kind_of", wraps=frontier.kind_of) as kinds:
            result = self.propose()
        self.assertEqual([call.args[1] for call in kinds.call_args_list], ["Alpha", "Beta", "zulu"])
        self.assertEqual(result, {"proposed": [], "adopted": [], "refused": [], "asked": 2})
        self.assertEqual([call[0] for call in self.teacher.calls], [[q] for q in questions])
        self.assertEqual(self.stash.entries(), [])

    def test_propose_seed_probes_adoption_filing_and_no_retest(self):
        question, members = self.property_world(decided=4)
        ledger = sources.SourceLedger(self.home / "ledger.json")
        with mock.patch.object(frontier, "cluster", wraps=frontier.cluster) as cluster, \
                mock.patch.object(frontier, "adopts", wraps=frontier.adopts) as adopts, \
                mock.patch.object(sources, "single_source_decide",
                                  wraps=sources.single_source_decide) as decide, \
                mock.patch.object(file, "file_distilled", wraps=file.file_distilled) as filing:
            result = self.propose(ledger)
        self.assertEqual(result, {"proposed": ["atomic mass"], "adopted": ["atomic mass"],
                                  "refused": [], "asked": 6})
        cluster.assert_called_once_with(self.memory, "atomic number")
        adopts.assert_called_once_with(4, 5)
        decide.assert_called_once()
        self.assertEqual(filing.call_args.kwargs, {"min_lineages": 1})
        self.assertEqual([p.subject for p in filing.call_args.args[2]], members)
        self.assertTrue(all(p.category == p.attribute == "atomic mass"
                            for p in filing.call_args.args[2]))
        self.assertEqual(self.teacher.calls[0], ([question], {
            "system": elicit.SYSTEM, "samples": elicit.SAMPLES,
            "temperature": elicit.TEMPERATURE, "top_p": elicit.TOP_P,
            "max_tokens": elicit.MAX_TOKENS, "seeds": elicit.SEEDS}))
        self.assertEqual(self.teacher.calls[1][0], [frontier._seed_questions()["forward"].format(
            attribute="atomic mass", subject=member) for member in members])
        self.assertEqual(len(self.stash.entries()), 4)
        self.assertEqual(ledger.history("source"), [
            {"question_id": f"atomic mass:{member}", "promoted": False} for member in members])
        self.memory.save()
        self.memory = self.open_memory()
        self.assertEqual(self.propose(ledger)["asked"], 0)
        self.assertEqual(len(self.teacher.calls), 2)

    def test_propose_refuses_three_decisions_and_never_files_or_retests(self):
        self.property_world(attribute="capital", kind="country", candidate="population", decided=3)
        with mock.patch.object(file, "file_distilled", wraps=file.file_distilled) as filing:
            result = self.propose()
        filing.assert_not_called()
        self.assertEqual(result, {"proposed": ["population"], "adopted": [],
                                  "refused": ["population"], "asked": 6})
        self.assertEqual(self.memory._attribute_vocabulary()["capital"]["properties"],
                         {"population": "refused"})
        self.memory.learn_kind("language", "country")
        self.memory.save()
        self.memory = self.open_memory()
        self.assertEqual(self.propose()["asked"], 0)
        self.assertEqual(len(self.teacher.calls), 2)
        self.assertEqual(len(sources.SourceLedger(self.home / "ledger.json").history("source")), 5)

    def test_propose_counts_kind_property_and_probe_questions(self):
        for number in range(1, 6):
            self.fact(number)
        self.kind_reply()
        seeds = frontier._seed_questions()
        self.teacher.replies[seeds["property"].format(kind="element")] = ["atomic mass"] * 5
        for number in range(1, 6):
            self.teacher.replies[seeds["forward"].format(
                attribute="atomic mass", subject=f"Member {number}")] = ["12.5"] * 5
        result = self.propose()
        self.assertEqual(result["adopted"], ["atomic mass"])
        self.assertEqual(result["asked"], 9)
        self.assertEqual([len(questions) for questions, _ in self.teacher.calls], [3, 1, 5])
        self.assertEqual(result["asked"], sum(len(qs) for qs, _ in self.teacher.calls))

    def test_no_shared_kind_persists_and_is_not_asked_in_later_proposals_or_rounds(self):
        for number in range(1, 4):
            self.fact(number)
        questions = self.kind_reply(rows=[["person"] * 5, ["book"] * 5, ["novel"] * 5])
        self.assertEqual(self.propose(), {
            "proposed": [], "adopted": [], "refused": [], "asked": 3})
        self.assertIsNone(self.memory._attribute_vocabulary()["atomic number"]["kind"])
        self.memory.save()
        self.memory = self.open_memory()
        self.assertIsNone(self.memory._attribute_vocabulary()["atomic number"]["kind"])
        if self.sharded:
            self.assertIsNone(self.memory.shards.vault.get("index:attributes")
                              ["attributes"]["atomic number"]["kind"])
        self.assertEqual(self.propose()["asked"], 0)
        ledger = sources.SourceLedger(self.home / "ledger.json")
        for index in range(2):
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id=f"no-kind-{index}", records_path=self.home / f"no-kind-{index}.jsonl")
            self.assertEqual(result["asked"], 0)
            self.assertTrue(result["used_up"])
        self.assertEqual([qs for qs, _ in self.teacher.calls], [questions])

    def test_study_with_no_shared_kind_is_used_up_and_extra_round_sends_nothing(self):
        for number in range(1, 4):
            self.fact(number)
        self.kind_reply(rows=[["person"] * 5, ["book"] * 5, ["novel"] * 5])
        ledger = sources.SourceLedger(self.home / "ledger.json")
        for index, expected_asked in enumerate((3, 0)):
            before = sum(len(qs) for qs, _ in self.teacher.calls)
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id=f"exhaust-{index}", records_path=self.home / f"exhaust-{index}.jsonl")
            after = sum(len(qs) for qs, _ in self.teacher.calls)
            self.assertEqual(after - before, expected_asked)
            self.assertEqual(result["asked"], expected_asked)
            self.assertTrue(result["used_up"])

    def test_none_kind_has_no_property_verdict_and_produces_no_reverse_targets(self):
        for number in (1, 2, 3, 4, 6):
            self.fact(number)
        self.form()
        self.memory.learn_kind("atomic number", None)
        self.memory.learn_property("atomic number", "mass", "refused")
        self.assertFalse(frontier._has_property_verdict(self.memory, "element"))
        self.memory.learn_kind("symbol", "element")
        self.memory.learn_property("symbol", "density", "refused")
        self.assertFalse(frontier._has_property_verdict(self.memory, None))
        self.assertTrue(frontier._has_property_verdict(self.memory, "element"))
        self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher), [])
        self.assertEqual(self.teacher.calls, [])

    def test_propose_small_cluster_and_missing_forms(self):
        self.property_world(count=4)
        self.assertEqual(self.propose(), {"proposed": ["atomic mass"], "adopted": [],
                                         "refused": ["atomic mass"], "asked": 5})
        self.assertEqual(self.stash.entries(), [])
        with mock.patch.object(frontier, "_seed_questions", return_value={}):
            self.assertEqual(self.propose()["asked"], 0)
            self.memory.learn_property("atomic number", "mass", "adopted")
            self.assertEqual(frontier.growth_targets(self.memory, self.stash), [])

    def test_propose_and_growth_prefer_learned_form_without_forward_seed(self):
        question, members = self.property_world()
        entry = self.form("atomic mass", "old-category")
        self.stash._entries[entry]["title"] = "Mass <Template> {literal}"
        self.stash.save()
        questions = [f"Mass <{member}> {{literal}}" for member in members]
        self.teacher.replies.update({q: ["12.5"] * 5 for q in questions})
        seeds = {"property": frontier._seed_questions()["property"]}
        with mock.patch.object(frontier, "_seed_questions", return_value=seeds):
            self.assertEqual(self.propose()["adopted"], ["atomic mass"])
            growing = frontier.growth_targets(self.memory, self.stash)
        self.assertEqual(self.teacher.calls[0][0], [question])
        self.assertEqual(self.teacher.calls[1][0], questions)
        self.assertEqual(len(growing), 8)
        self.assertTrue(all(t.category == "atomic mass" and t.question ==
                            f"Mass <{t.subject}> {{literal}}" for t in growing))

    def test_growth_only_members_without_normalized_property(self):
        for number in range(1, 4):
            self.fact(number)
        self.fact(99, "other")
        self.memory.remember_fact("odd key", "12.5", subject="THE MEMBER 2",
                                  attribute="Atomic Mass")
        self.memory.learn_property("atomic number", "atomic mass", "adopted")
        self.memory.learn_property("atomic number", "density", "refused")
        expected = [targets.Target("atomic mass", f"Member {n}",
                    frontier._seed_questions()["forward"].format(
                        attribute="atomic mass", subject=f"Member {n}"), "atomic mass")
                    for n in (1, 3)]
        self.assertEqual(frontier.growth_targets(self.memory, self.stash), expected)

    def test_growth_normalizes_recorded_attribute_name_only_once(self):
        self.memory.remember_fact("one", "1", subject="Member", attribute="The A Rank")
        self.memory.learn_property("The A Rank", "mass", "adopted")
        self.assertEqual([t.subject for t in frontier.growth_targets(self.memory, self.stash)],
                         ["Member"])

    def test_pending_deduplicates_ids_first_occurrence_wins_and_filters_source(self):
        first = targets.Target("mass", "Member", "first", "mass")
        duplicate = targets.Target("mass", "Member", "duplicate", "mass")
        reverse = frontier.ReverseTarget("number", "5", "reverse", "number", "5")
        ledger = sources.SourceLedger(self.home / "ledger.json")
        ledger.record("source", elicit.question_id(reverse), False)
        with mock.patch.object(targets, "completion_targets", return_value=[first, first]), \
                mock.patch.object(frontier, "reverse_targets", return_value=[reverse]), \
                mock.patch.object(frontier, "growth_targets", return_value=[duplicate, duplicate]):
            self.assertEqual(frontier.pending(
                self.memory, self.stash, self.teacher, ledger, "source"), [first])
            self.assertEqual(frontier.pending(
                self.memory, self.stash, self.teacher, ledger, "other"), [first, reverse])

    def test_study_adopts_grows_then_asks_nothing(self):
        self.property_world()
        for number in range(1, 9):
            self.teacher.replies[frontier._seed_questions()["forward"].format(
                attribute="atomic mass", subject=f"Member {number}")] = ["12.5"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        outcomes = []
        for n in range(3):
            with mock.patch.object(sources, "used_up", wraps=sources.used_up) as used_up:
                outcomes.append(frontier.study_round(
                    self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                    run_id=f"growth-{n}", records_path=self.home / f"growth-{n}.jsonl"))
                used_up.assert_called_once()
            if n == 0:
                self.assertEqual(len(frontier.growth_targets(self.memory, self.stash)), 3)
        self.assertEqual([r["asked"] for r in outcomes], [6, 3, 0])
        self.assertEqual([r["filed"] for r in outcomes], [5, 3, 0])
        self.assertEqual([r["used_up"] for r in outcomes], [False, True, True])
        self.assertEqual(outcomes[0]["adopted"], ["atomic mass"])
        self.assertTrue(all(r["queued"] == 0 for r in outcomes))
        self.assertEqual(sum(len(qs) for qs, _ in self.teacher.calls), 9)
        self.assertEqual(len(ledger.history("source")), 8)
        self.assertTrue(all(row["promoted"] for row in ledger.history("source")))
        for number in range(1, 9):
            self.assertEqual(self.memory.held_value(f"Member {number}", "atomic mass"), "12.5")
        self.assertEqual(frontier.pending(self.memory, self.stash, self.teacher, ledger, "source"), [])

    def test_study_probe_ledger_waits_for_custom_approval(self):
        self.property_world()
        ledger = sources.SourceLedger(self.home / "ledger.json")
        approver = mock.Mock()

        def approve_one():
            self.assertEqual(ledger.history("source"), [])
            self.stash._entries[self.stash.entries()[0]["id"]]["status"] = "promoted"

        approver.approve_all.side_effect = approve_one
        result = frontier.study_round(
            self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
            run_id="custom", records_path=self.home / "custom.jsonl", approver=approver)
        self.assertEqual(result["filed"], 5)
        self.assertEqual([row["promoted"] for row in ledger.history("source")],
                         [True, False, False, False, False])

    def test_study_used_up_requires_verdict_for_each_stored_kind(self):
        self.property_world()
        item = targets.Target("number", "missing", "existing question", "atomic number")
        self.teacher.replies[item.question] = ["UNKNOWN"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        with mock.patch.object(frontier, "pending", return_value=[item]), \
                mock.patch.object(frontier, "kind_of", side_effect=AssertionError("No new kinds")), \
                mock.patch.object(sources, "used_up", return_value=True) as used_up:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id="existing", records_path=self.home / "existing.jsonl")
        used_up.assert_called_once()
        self.assertFalse(result["used_up"])

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
        self.memory.learn_property("atomic number", "mass", "refused")
        target = frontier.reverse_targets(self.memory, self.stash, self.teacher)[0]
        self.teacher.replies[target.question] = ["Member 2"] * 5
        ledger = sources.SourceLedger(self.home / "ledger.json")
        with mock.patch.object(frontier, "ask_forward", wraps=frontier.ask_forward) as ask:
            result = frontier.study_round(
                self.memory, self.stash, self.teacher, ledger, "source", confidence=0.99,
                run_id="held", records_path=self.home / "held.jsonl")
        ask.assert_not_called()
        self.assertEqual(result, {"asked": 1, "filed": 0, "queued": 1, "used_up": True,
                                  "proposed": [], "adopted": [], "refused": [],
                                  "checked": 0, "agreed": 0, "contested": 0, "revised": 0})
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
        self.memory.learn_property("atomic number", "mass", "refused")
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
        self.assertEqual(result, {"asked": 2, "filed": 1, "queued": 1, "used_up": True,
                                  "proposed": [], "adopted": [], "refused": [],
                                  "checked": 0, "agreed": 0, "contested": 0, "revised": 0})
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
        questions = self.kind_reply(["Element.", "**element**", "ELEMENT",
                                     "substance", "UNKNOWN"])
        self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                         "element")
        self.assertEqual(self.teacher.calls, [(questions, {
            "system": elicit.SYSTEM, "samples": elicit.SAMPLES,
            "temperature": elicit.TEMPERATURE, "top_p": elicit.TOP_P,
            "max_tokens": elicit.MAX_TOKENS, "seeds": elicit.SEEDS})])
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

    def test_kind_requires_shared_answer_and_seed(self):
        for number in (1, 2, 3, 4, 6):
            self.fact(number)
        self.form()
        with mock.patch.object(frontier, "_seed_questions", return_value={}):
            self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertNotIn("kind", self.memory._attribute_vocabulary()["atomic number"])
        self.kind_reply(rows=[["element"] * 5, ["metal"] * 5, ["element"] * 5])
        self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher), [])
        self.assertIsNone(self.memory._attribute_vocabulary()["atomic number"]["kind"])
        with mock.patch.object(frontier, "_seed_questions", return_value={}):
            self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertEqual(len(self.teacher.calls), 1)

    def test_kind_asks_three_members_in_one_batch_using_seed_and_data(self):
        for number in range(1, 7):
            self.fact(number)
        self.fact(99, "other attribute")
        self.memory.remember_fact("duplicate member", "1", subject="Member 1",
                                  attribute="Atomic Number")
        form = "kind <{a}>"
        members = random.Random(23).sample([f"Member {n}" for n in range(1, 7)], 3)
        questions = [form.format(a=member) for member in members]
        self.teacher.replies.update({q: ["element"] * 5 for q in questions})
        with mock.patch.object(frontier, "_seed_questions", return_value={"kind": form}):
            self.assertEqual(frontier.kind_of(
                self.memory, " ATOMIC NUMBER ", self.teacher, seed=23), "element")
        self.assertEqual(len(self.teacher.calls), 1)
        self.assertEqual(self.teacher.calls[0][0], questions)

    def test_kind_shared_through_agreement(self):
        for number in range(1, 4):
            self.fact(number)
        self.kind_reply(rows=[
            ["element", "chemical element", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
            ["element", "metal", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
            ["element", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"]])
        self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                         "element")

    def test_kind_shared_without_identical_answer_in_every_row(self):
        for number in range(1, 4):
            self.fact(number)
        self.kind_reply(rows=[
            ["chemical element"] * 5, ["element"] * 5, ["element"] * 5])
        self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                         "element")

    def test_kind_ignores_abstentions_and_empty_answers(self):
        for number in range(1, 4):
            self.fact(number)
        self.kind_reply(["**Element.**", "UNKNOWN", "element\nI do not know", "", "!!!"])
        self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                         "element")

    def test_kind_requires_a_position_from_every_member(self):
        for number in range(1, 4):
            self.fact(number)
        self.kind_reply(rows=[
            ["element"] * 5, ["element"] * 5,
            ["UNKNOWN", "", "!!!", "element or metal", "element\nI do not know"]])
        self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertIsNone(self.memory._attribute_vocabulary()["atomic number"]["kind"])

    def test_kind_ranks_agreement_then_exact_count_then_lexically(self):
        for number in range(1, 4):
            self.fact(number)
        cases = [
            ("agreement", ["element", "chemical element", "synthetic element", "metal", "metal"],
             "element"),
            ("exact count", ["chemical element", "chemical element", "element", "element", "element"],
             "element"),
            ("lexical", ["metal", "element", "UNKNOWN", "UNKNOWN", "UNKNOWN"], "element")]
        for rule, answers, expected in cases:
            with self.subTest(rule=rule), mock.patch.object(self.memory, "learn_kind") as learn:
                self.kind_reply(answers)
                self.assertEqual(frontier.kind_of(self.memory, "atomic number", self.teacher),
                                 expected)
                learn.assert_called_once_with("atomic number", expected)

    def test_stored_kind_is_returned_unchanged_without_asking(self):
        with mock.patch.object(self.memory, "_attribute_vocabulary", return_value={
                "atomic number": {"kind": "Stored Kind."}}), \
                mock.patch.object(frontier, "_seed_questions") as seed, \
                mock.patch.object(self.memory, "learn_kind") as learn:
            self.assertEqual(frontier.kind_of(
                self.memory, " ATOMIC NUMBER ", self.teacher), "Stored Kind.")
        seed.assert_not_called()
        learn.assert_not_called()
        self.assertEqual(self.teacher.calls, [])

    def test_kind_requires_three_members(self):
        for number in range(1, 3):
            self.fact(number)
        self.assertIsNone(frontier.kind_of(self.memory, "atomic number", self.teacher))
        self.assertNotIn("kind", self.memory._attribute_vocabulary()["atomic number"])
        self.assertEqual(self.teacher.calls, [])

    def test_kind_rejects_incomplete_reply_matrix(self):
        for number in range(1, 4):
            self.fact(number)
        full = ["element"] * elicit.SAMPLES
        matrices = [[], [full], [full] * 2, [full] * 4]
        for index in range(3):
            for size in (elicit.SAMPLES - 1, elicit.SAMPLES + 1):
                rows = [full] * 3
                rows[index] = ["element"] * size
                matrices.append(rows)
        for rows in matrices:
            with self.subTest(lengths=[len(row) for row in rows]), \
                    mock.patch.object(self.teacher, "ask", return_value=rows), \
                    mock.patch.object(self.memory, "learn_kind") as learn:
                with self.assertRaisesRegex(ValueError, "incomplete sample matrix"):
                    frontier.kind_of(self.memory, "atomic number", self.teacher)
                learn.assert_not_called()

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
        with mock.patch.object(frontier, "kind_of", return_value=None) as kind:
            self.assertEqual(frontier.reverse_targets(self.memory, self.stash, self.teacher), [])
        kind.assert_called_once_with(self.memory, "atomic number", self.teacher)
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
        self.memory.learn_kind("chemical symbol", "element")
        self.memory.learn_property("chemical symbol", "mass", "refused")
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
        self.memory.learn_property("atomic number", "mass", "refused")
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
