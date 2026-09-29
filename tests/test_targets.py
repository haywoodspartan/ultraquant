"""Catalogue-selected distillation targets, using only scratch stores."""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from ultraquant.distill.elicit import question_id
from ultraquant.distill.file import _reject_probes
from ultraquant.distill.targets import Target, completion_targets, question_form
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


class ScratchTargetsTestCase(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="uq_targets_test_")
        self.addCleanup(scratch.cleanup)
        self.home = Path(scratch.name)
        self.memory = SystematicMemory(path=self.home / "memory.json")
        self.stash = ContemporaryStash(self.home / "stash.json")
        self.template_memory = SystematicMemory(self.home / "templates.json")
        self.serial = 0

    def fact(self, subject, attribute):
        self.serial += 1
        # Neither keys nor values reveal the slots being indexed.
        self.memory.remember_fact(
            f"opaque-{self.serial}", "irrelevant", subject=subject,
            attribute=attribute)

    def form(self, subject="seed", attribute="B", category="lookup",
             title="Which mark belongs to seed?", status="promoted",
             legacy=False):
        self.serial += 1
        qid = f"{category}:{subject}"
        provenance = {"run_id": "scratch"}
        if not legacy:
            provenance["question_id"] = qid
        entry_id = self.stash.add_claim(
            f"https://distill.invalid/scratch/{qid}", title,
            "The marker is recorded.", provenance=provenance,
            fields={"key": f"template-{self.serial}", "value": "recorded",
                    "subject": subject, "attribute": attribute})
        if status == "promoted":
            self.stash.promote(entry_id, self.template_memory)
        elif status == "rejected":
            self.stash.reject(entry_id)
        return entry_id


class QuestionFormTests(ScratchTargetsTestCase):
    def test_form_comes_verbatim_from_title_and_category_from_provenance(self):
        self.form(subject="Mixed Case", attribute="opaque slot", category="custom",
                  title="Please name Mixed Case's designation!")
        self.assertEqual(question_form(self.stash, "opaque slot"),
                         ("custom", "Please name {subject}'s designation!"))
        self.assertIsNone(question_form(self.stash, "missing"))

    def test_most_common_pair_wins(self):
        self.form(category="a", title="Alternate seed?")
        for subject in ("one", "two"):
            self.form(subject=subject, category="z",
                      title=f"Identify {subject}!")
        self.assertEqual(question_form(self.stash, "B"),
                         ("z", "Identify {subject}!"))

    def test_ties_use_category_then_form_not_insertion_order(self):
        self.form(category="z", title="A seed?")
        self.form(category="a", title="Z seed?")
        self.form(category="a", title="B seed?")
        self.assertEqual(question_form(self.stash, "B"), ("a", "B {subject}?"))

    def test_only_promoted_entries_with_matching_attribute_count(self):
        self.form(status="staged", category="a")
        self.form(status="rejected", category="a")
        self.form(attribute="other", category="a")
        self.assertIsNone(question_form(self.stash, "B"))
        self.form(category="z")
        self.assertEqual(question_form(self.stash, "B"),
                         ("z", "Which mark belongs to {subject}?"))

    def test_title_must_contain_nonempty_subject_verbatim(self):
        self.form(subject="seed", title="Which mark belongs to SEED?")
        self.form(subject="absent", title="Which mark is this?")
        self.form(subject="", title="Which mark is this?")
        self.assertIsNone(question_form(self.stash, "B"))

    def test_legacy_question_id_uses_shared_provenance_reader(self):
        self.form(category="legacy", legacy=True)
        self.assertEqual(question_form(self.stash, "B"),
                         ("legacy", "Which mark belongs to {subject}?"))

    def test_missing_provenance_cannot_supply_a_category(self):
        entry_id = self.stash.add_claim(
            "https://example.invalid/fact", "Identify seed!",
            "The marker is recorded.",
            fields={"key": "marker", "value": "recorded",
                    "subject": "seed", "attribute": "B"})
        self.stash.promote(entry_id, self.template_memory)
        self.assertIsNone(question_form(self.stash, "B"))


class CompletionTargetTests(ScratchTargetsTestCase):
    def shared(self, count):
        for i in range(count):
            self.fact(f"shared-{i}", "A")
            self.fact(f"shared-{i}", "B")

    def test_gap_at_default_support_and_share_boundaries(self):
        self.form()
        self.shared(5)
        gaps = [f"Gap {i}" for i in range(5)]
        for subject in gaps:
            self.fact(subject, "A")
        self.assertEqual(completion_targets(self.memory, self.stash), [
            Target("lookup", subject, f"Which mark belongs to {subject}?", "B")
            for subject in gaps])
        self.assertEqual(completion_targets(self.memory, self.stash,
                                            min_share=0.5001), [])
        self.assertEqual(completion_targets(self.memory, self.stash,
                                            min_support=6), [])

    def test_support_counts_subjects_not_duplicate_records(self):
        self.form()
        self.shared(4)
        for _ in range(4):
            self.fact("shared-0", "A")
            self.fact("shared-0", "B")
        self.fact("gap", "A")
        self.assertEqual(completion_targets(self.memory, self.stash), [])
        self.assertEqual([t.subject for t in completion_targets(
            self.memory, self.stash, min_support=4)], ["gap"])

    def test_share_is_directed_and_uses_source_population(self):
        self.shared(5)
        self.fact("A gap", "A")
        for i in range(6):
            self.fact(f"B gap {i}", "B")
        self.form(attribute="A")
        self.form(attribute="B")
        self.assertEqual([(t.attribute, t.subject) for t in
                          completion_targets(self.memory, self.stash)],
                         [("B", "A gap")])

    def test_non_cooccurring_attributes_have_no_targets(self):
        for i in range(6):
            self.fact(f"left-{i}", "A")
            self.fact(f"right-{i}", "B")
        self.form(attribute="A")
        self.form(attribute="B")
        self.assertEqual(completion_targets(self.memory, self.stash), [])
        self.assertEqual(completion_targets(self.memory, self.stash,
                                            min_support=1, min_share=0), [])

    def test_missing_question_form_prevents_gap_target(self):
        self.shared(5)
        self.fact("gap", "A")
        self.assertEqual(completion_targets(self.memory, self.stash), [])

    def test_multiple_sources_deduplicate_and_sort_by_attribute_subject(self):
        for attribute in ("D", "B"):
            self.form(attribute=attribute)
        for i in range(5):
            for attribute in ("A", "B", "C", "D"):
                self.fact(f"shared-{i}", attribute)
        for subject in ("Zulu", "Alpha"):
            for attribute in ("A", "C"):
                self.fact(subject, attribute)
        self.assertEqual([(t.attribute, t.subject) for t in
                          completion_targets(self.memory, self.stash)],
                         [("B", "Alpha"), ("B", "Zulu"),
                          ("D", "Alpha"), ("D", "Zulu")])

    def test_unstructured_facts_do_not_supply_slots(self):
        self.form()
        self.shared(5)
        self.memory.remember_fact("A of decoy", "B of decoy")
        self.memory.remember_fact("partial", "A", subject="partial")
        self.memory.remember_fact("other partial", "gap", attribute="A")
        self.assertEqual(completion_targets(self.memory, self.stash), [])

    def test_subjects_are_the_catalogues_normalized_subjects(self):
        self.form()
        for i in range(5):
            self.fact(f"Shared-{i}", "A")
            self.fact(f"shared-{i}", "B")
        self.fact("Gap", "A")
        self.fact("GAP", "A")
        self.assertEqual(completion_targets(self.memory, self.stash), [
            Target("lookup", "GAP", "Which mark belongs to GAP?", "B")])

    def test_braces_in_a_title_are_kept_as_written(self):
        self.form(title="Which {mark} belongs to seed?")
        self.shared(5)
        self.fact("gap", "A")
        self.assertEqual(question_form(self.stash, "B"),
                         ("lookup", "Which {{mark}} belongs to {subject}?"))
        self.assertEqual([t.question for t in completion_targets(
            self.memory, self.stash)], ["Which {mark} belongs to gap?"])

    def test_reopened_sharded_memory_uses_catalogued_records(self):
        self.memory.shards = FactShards(ShardVault(self.home / "vault"))
        self.shared(5)
        self.fact("Mixed Case Subject", "A")
        self.form()
        self.memory.save()
        memory = SystematicMemory(path=self.home / "memory.json")
        memory.shards = FactShards(ShardVault(self.home / "vault"))
        stash = ContemporaryStash(self.home / "stash.json")
        self.assertEqual(completion_targets(memory, stash), [Target(
            "lookup", "Mixed Case Subject",
            "Which mark belongs to Mixed Case Subject?", "B")])


class TargetTests(unittest.TestCase):
    def test_target_is_frozen_has_question_id_and_passes_probe_rejection(self):
        target = Target("number", "iron", "A question", "atomic number")
        self.assertFalse(hasattr(target, "fictitious"))
        self.assertIsNone(_reject_probes([target]))
        self.assertEqual(question_id(target), "number:iron")
        with self.assertRaises(FrozenInstanceError):
            target.subject = "changed"


if __name__ == "__main__":
    unittest.main()
