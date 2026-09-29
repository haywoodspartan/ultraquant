"""Distilled forms come from promoted entries, with data-only fallbacks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ultraquant.distill import elicit as E
from ultraquant.distill import file as F
from ultraquant.interpreter.stash import ContemporaryStash


class ScratchForms(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.stash = ContemporaryStash(Path(scratch.name) / "stash.json")

    def entry(self, subject="Alpha", value="Lux", attribute="motto",
              claim="{subject} declares {value}.", key="motto: {subject}",
              category="custom", status="promoted", legacy=False):
        qid = f"{category}:{subject}"
        provenance = {"run_id": "source"}
        if not legacy:
            provenance["question_id"] = qid
        entry_id = self.stash.add_claim(
            f"https://distill.invalid/source/{qid}", "Which motto?",
            claim.format(subject=subject, value=value), provenance=provenance,
            fields={"key": key.format(subject=subject.lower()), "value": value,
                    "subject": subject, "attribute": attribute})
        # Fixture status is independent of the classification being tested.
        self.stash._entries[entry_id]["status"] = status
        self.stash.save()
        return entry_id

    def file(self, *, category="custom", attribute="motto", subject="Delta",
             value="Fortis"):
        item = SimpleNamespace(category=category, subject=subject,
                               question=f"Which mark belongs to {subject}?")
        if attribute is not None:
            item.attribute = attribute
        records = [E.Record(f"t{n}", f"lineage-{n}", "scratch.gguf", 1,
                            E.question_id(item), item.question, seed, value,
                            E.normalize(E.extract(value)))
                   for n in range(3) for seed in E.SEEDS]
        return F.file_distilled(self.stash, records, [item], 0.99, "target")


class LearnedFormsTests(ScratchForms):
    def test_most_common_and_lexical_ties(self):
        self.entry(claim="Z {subject}: {value}", key="z {subject}", attribute="z")
        self.entry(claim="Z {subject}: {value}", key="z {subject}")
        self.entry(claim="A {subject}: {value}", key="a {subject}")
        self.assertEqual(F.claim_form(self.stash, "motto"), "A {subject}: {value}")
        self.assertEqual(F.key_form(self.stash, "motto"), "a {subject}")
        self.assertEqual(F.category_attribute(self.stash, "custom"), "motto")
        self.entry(subject="Bravo", claim="Z {subject}: {value}", key="z {subject}")
        self.assertEqual(F.claim_form(self.stash, "motto"), "Z {subject}: {value}")
        self.assertEqual(F.key_form(self.stash, "motto"), "z {subject}")

    def test_category_tie_and_legacy_provenance(self):
        self.entry(attribute="z", legacy=True)
        self.entry(attribute="a", legacy=True)
        self.assertEqual(F.category_attribute(self.stash, "custom"), "a")
        self.entry(attribute="z", legacy=True)
        self.assertEqual(F.category_attribute(self.stash, "custom"), "z")
        self.assertIsNone(F.category_attribute(self.stash, "missing"))

    def test_only_promoted_matching_entries_count(self):
        for status in ("staged", "corroborated", "disputed", "rejected"):
            self.entry(status=status)
        self.entry(attribute="other", category="elsewhere")
        self.assertIsNone(F.claim_form(self.stash, "motto"))
        self.assertIsNone(F.key_form(self.stash, "motto"))
        self.assertIsNone(F.category_attribute(self.stash, "custom"))
        self.entry()
        self.assertEqual(F.claim_form(self.stash, "motto"), "{subject} declares {value}.")
        self.assertEqual(F.key_form(self.stash, "motto"), "motto: {subject}")
        self.assertEqual(F.category_attribute(self.stash, "custom"), "motto")

    def test_ambiguous_missing_and_overlapping_slots_are_excluded(self):
        cases = [
            ("Mexico", "Mexico City", "The capital of {subject} is {value}."),
            ("Alpha", "Lux", "{subject}: {value}, {value}"),
            ("Alpha", "Alpha", "{subject}"),
            ("abc", "bcd", "abcd"),
            ("aa", "Lux", "aaa {value}"),
            ("Alpha", "Lux", "Missing {value}"),
            ("Alpha", "Lux", "{subject}: absent"),
            ("", "Lux", "{subject}: {value}"),
            ("Alpha", "", "{subject}: {value}"),
        ]
        for subject, value, claim in cases:
            with self.subTest(subject=subject, value=value, claim=claim):
                self.entry(subject=subject, value=value, claim=claim)
                self.assertIsNone(F.claim_form(self.stash, "motto"))

    def test_key_requires_exactly_one_lowercase_subject(self):
        for key in ("{subject} and {subject}", "missing", "ALPHA"):
            self.entry(key=key)
        self.entry(subject="aa", key="aaa")
        self.assertIsNone(F.key_form(self.stash, "motto"))
        self.entry(subject="Mixed Case", key="mark {subject}")
        self.assertEqual(F.key_form(self.stash, "motto"), "mark {subject}")

    def test_braces_round_trip_and_reversed_slots(self):
        entry_id = self.entry(subject="A{b}", value="L{x}",
                              claim="{{note}} {value} <- {subject} {{subject}}",
                              key="{{note}} {subject} {{value}}")
        entry = self.stash.get(entry_id)
        self.assertEqual(F.claim_form(self.stash, "motto").format(
            subject="A{b}", value="L{x}"), entry["claim"])
        self.assertEqual(F.key_form(self.stash, "motto").format(
            subject="a{b}"), entry["fields"]["key"])

    def test_numeric_value_is_stringified(self):
        self.entry(value=0)
        self.assertEqual(F.claim_form(self.stash, "motto"), "{subject} declares {value}.")


class FilingTests(ScratchForms):
    def test_seed_fallback_preserves_existing_filing(self):
        ids = self.file(category="symbol", attribute=None, subject="Iron", value="Fe")
        entry = self.stash.get(ids[0])
        self.assertEqual(entry["claim"], "The chemical symbol of Iron is Fe.")
        self.assertEqual(entry["fields"], {"key": "chemical symbol of iron",
                         "value": "Fe", "subject": "Iron", "attribute": "chemical symbol"})

    def test_new_attribute_files_as_factual(self):
        ids = self.file(attribute="currency", subject="May Island", value="Lumen")
        entry = self.stash.get(ids[0])
        self.assertEqual(entry["claim"], "The currency of May Island is Lumen.")
        self.assertEqual(entry["classification"], "factual-claim")
        self.assertEqual(entry["fields"]["key"], "currency of may island")

    def test_missing_seed_parts_skip(self):
        seed = F._seed()
        for part in ("claim", "key", "categories"):
            with self.subTest(part=part):
                partial = {k: v for k, v in seed.items() if k != part}
                with mock.patch.object(F, "_seed", return_value=partial):
                    self.assertEqual(self.file(category="symbol", attribute=None), [])
        with mock.patch.object(F, "_seed", return_value={}):
            self.assertEqual(self.file(), [])
        self.assertEqual(self.file(attribute=None), [])
        self.assertEqual(self.stash.entries(), [])

    def test_learned_forms_and_mapping_work_without_seed(self):
        self.entry(claim="By decree, {subject} keeps {value}.",
                   key="The decreed mark {subject}")
        with mock.patch.object(F, "_seed", return_value={}):
            entry = self.stash.get(self.file(attribute=None)[0])
        self.assertEqual(entry["claim"], "By decree, Delta keeps Fortis.")
        self.assertEqual(entry["fields"]["key"], "decreed mark delta")
        self.assertEqual(entry["classification"], "factual-claim")

    def test_explicit_attribute_wins_over_category_mapping(self):
        self.entry(attribute="mapped")
        entry = self.stash.get(self.file(attribute="explicit")[0])
        self.assertEqual(entry["fields"]["attribute"], "explicit")
        self.assertEqual(entry["claim"], "The explicit of Delta is Fortis.")


class ClassificationTests(ScratchForms):
    def classify(self, claim, fields=None):
        entry_id = self.stash.add_claim("https://example.invalid", "", claim,
                                        fields=fields)
        return self.stash.get(entry_id)["classification"]

    def test_without_fields_preserves_regex_and_heuristic(self):
        cases = {
            "The author of A Rather Long Book Title With Many Words is May Alcott.": "factual-claim",
            "This is beautiful.": "opinion",
            "It might rain.": "hedged",
            "Sky is blue.": "factual-claim",
            "Alpha declares Lux.": "unclassified",
        }
        for claim, expected in cases.items():
            with self.subTest(claim=claim):
                self.assertEqual(self.classify(claim), expected)

    def test_complete_fields_and_both_slots_required(self):
        fields = {"key": "mark alpha", "subject": "Alpha", "value": "Lux",
                  "attribute": "mark"}
        self.assertEqual(self.classify("Alpha declares Lux.", fields), "factual-claim")
        for name in fields:
            for blank in (None, ""):
                with self.subTest(name=name, blank=blank):
                    incomplete = {**fields, name: blank}
                    self.assertEqual(self.classify("Alpha declares Lux.", incomplete),
                                     "unclassified")
        for claim in ("Alpha declares Sol.", "Bravo declares Lux."):
            self.assertEqual(self.classify(claim, fields), "unclassified")
        self.assertEqual(self.classify("Alpha declares 0.", {**fields, "value": 0}),
                         "factual-claim")


if __name__ == "__main__":
    unittest.main()
