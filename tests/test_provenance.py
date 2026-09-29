"""Section 11.164: source provenance spoken by catalogue replies."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from ultraquant.distill import corroborate, provenance, sources
from ultraquant.interpreter.thoughts import build_session, run_pipeline


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix="uq_provenance_test_")))
        self.session = build_session(self.root, seed=0)
        self.stash = self.session.stash
        self.ledger = sources.SourceLedger(self.root / "sources.json")
        self.ledger.path.write_text("{}", encoding="utf-8")
        self.enterContext(mock.patch("urllib.request.urlopen",
                                     side_effect=AssertionError("Network forbidden")))
        self.enterContext(mock.patch.object(sources.LMStudioTeacher, "ask",
                                           side_effect=AssertionError("Live teacher forbidden")))

    def claim(self, subject="Gold", value="196.97", teachers=("first",),
              attribute="atomic mass", promoted=True):
        key = f"{attribute} of {subject.lower()}"
        entry = self.stash.add_claim(
            f"https://distill.invalid/{subject}",
            f"What is the {attribute} of {subject}?",
            f"The {attribute} of {subject} is {value}.",
            fields=dict(key=key, value=value, subject=subject, attribute=attribute),
            provenance=dict(teachers=list(teachers)))
        if promoted:
            self.stash.promote(entry, self.session.memory, confidence=.91)
        return key

    def contest(self, key, answer, source="second"):
        self.ledger.record(source, answer, False,
                           queued={"key": key, "contest": answer})

    def ask(self, text):
        return run_pipeline(text, self.session)[0]

    def test_distinct_sorted_teachers_across_matching_promoted_claims(self):
        key = self.claim(teachers=("zeta", "alpha", "alpha"))
        self.claim(value="196.966569", teachers=("beta", "zeta"))
        self.claim(teachers=("staged",), promoted=False)
        self.claim(subject="Silver", teachers=("unrelated",))
        before = deepcopy(self.stash.entries())
        with mock.patch.object(corroborate, "values_agree",
                               wraps=corroborate.values_agree) as agree:
            self.assertEqual(provenance.provenance(self.stash, self.ledger, key, "196.97"),
                             {"teachers": ["alpha", "beta", "zeta"], "contests": []})
        agree.assert_any_call("196.966569", "196.97")
        self.assertEqual(self.stash.entries(), before)

    def test_superseded_value_teachers_are_excluded(self):
        key = self.claim(value="225.87", teachers=("old",))
        self.claim(value="196.97", teachers=("first", "second"))
        self.assertEqual(provenance.provenance(self.stash, self.ledger, key, "196.97"),
                         {"teachers": ["first", "second"], "contests": []})

    def test_alias_claims_count_as_one_source_and_keep_first_name(self):
        key = self.claim(teachers=("command-r-08-2024",))
        self.claim(teachers=("c4ai-command-r-08-2024",))
        found = provenance.provenance(self.stash, self.ledger, key, "196.97")
        self.assertEqual(found, {"teachers": ["command-r-08-2024"], "contests": []})
        self.assertEqual(self.ask("What is the atomic mass of gold?"),
                         "atomic mass of gold is 196.97 (confidence 1.00; one source).")

    def test_stored_identity_and_legacy_name_count_as_one_source(self):
        key = self.claim(teachers=("unlisted",))
        self.stash._entries[self.stash.entries()[-1]["id"]]["provenance"]["teacher_ids"] = [
            sources.identity("command-r-08-2024")]
        self.claim(teachers=("c4ai-command-r-08-2024",))
        self.assertEqual(provenance.provenance(self.stash, self.ledger, key, "196.97"),
                         {"teachers": ["unlisted"], "contests": []})

    def test_contests_preserve_order_deduplicate_and_exclude_settled(self):
        key = self.claim(subject="Molybdenum", value="95.96")
        self.contest(key, "95.95")
        self.contest(key, "95.94", "third")
        self.contest(key, "95.95", "fourth")
        self.contest("unrelated", "999")
        before = self.ledger.path.read_bytes()
        with mock.patch.object(corroborate, "contests",
                               wraps=corroborate.contests) as contests:
            self.assertEqual(provenance.provenance(self.stash, self.ledger, key, "95.96"),
                             {"teachers": ["first"], "contests": ["95.95", "95.94"]})
        contests.assert_called_once_with(self.ledger, key)
        self.assertEqual(self.ledger.path.read_bytes(), before)
        for verdict in ("agreed", "revised"):
            with self.subTest(verdict=verdict):
                self.ledger.path.write_bytes(before)
                self.ledger.record("last", "settled", True,
                                   check={"key": key, "verdict": verdict})
                self.assertEqual(provenance.provenance(
                    self.stash, self.ledger, key, "95.96")["contests"], [])

    def test_absent_ledger_and_missing_provenance(self):
        key = self.claim()
        with mock.patch.object(corroborate, "contests",
                               side_effect=AssertionError("No ledger")):
            self.assertEqual(provenance.provenance(self.stash, None, key, "196.97"),
                             {"teachers": ["first"], "contests": []})
        self.assertEqual(provenance.provenance(self.stash, None, "missing", "1"),
                         {"teachers": [], "contests": []})

    def test_chat_contested_agreement_and_one_teacher(self):
        key = self.claim(subject="Molybdenum", value="95.96")
        self.contest(key, "95.95")
        self.contest(key, "95.94", "third")
        self.claim(teachers=("first", "second"))
        self.claim(subject="Silver", value="107.87")
        for subject, value, note in (
                ("molybdenum", "95.96", "contested: other sources say 95.95, 95.94"),
                ("gold", "196.97", "2 sources agree"),
                ("silver", "107.87", "one source")):
            with self.subTest(subject=subject):
                self.assertEqual(self.ask(f"What is the atomic mass of {subject}?"),
                                 f"atomic mass of {subject} is {value} (confidence 0.91; {note}).")

    def test_user_taught_reply_is_byte_identical(self):
        self.ask("tower height is 300")
        question = "What is the tower height?"
        with mock.patch.object(provenance, "provenance",
                               return_value={"teachers": [], "contests": []}) as found:
            baseline = self.ask(question)
        found.assert_called_once()
        self.assertEqual(self.ask(question), baseline)
        self.assertEqual(baseline, "tower height is 300 (confidence 0.60).")

    def test_chat_by_value_and_multiple_subjects(self):
        self.claim("Gold", "Au", ("first", "second"), "chemical symbol")
        self.claim("Silver", "Ag", ("first",), "chemical symbol")
        self.assertEqual(self.ask("Which element has the chemical symbol Au?"),
                         "Gold: chemical symbol of gold is Au (confidence 0.91; 2 sources agree).")
        self.assertEqual(self.ask("What is the chemical symbol of gold and silver?"),
                         "chemical symbol of gold is Au (confidence 0.91; 2 sources agree); "
                         "chemical symbol of silver is Ag (confidence 0.91; one source).")

    def test_catalogue_exact_reading_and_module_note_seam(self):
        for subject in ("Echo", "Foxtrot"):
            self.claim(subject, "Writer", ("first", "second"), "author")
            self.session.memory.learn_asking("author", subject,
                                             f"Who wrote the novel {subject}?")
        for question, prefix in (
                ("Who wrote Echo?", ""),
                ("Who wrote the strange novel Echo?", "Reading that as 'author of echo': ")):
            with self.subTest(question=question):
                reply, trace = run_pipeline(question, self.session)
                self.assertEqual(reply, prefix + "author of echo is Writer "
                                 "(confidence 0.91; 2 sources agree).")
                self.assertIn("catalogue", str(trace))
        with mock.patch.object(provenance, "note", return_value="; custom") as note:
            self.assertIn("(confidence 0.91; custom)", self.ask("Who wrote Echo?"))
        note.assert_called_once_with({"teachers": ["first", "second"], "contests": []})

    def test_chain_notes_only_final_fact(self):
        self.claim("Nacre", "Iris", ("via",), "velar")
        key = self.claim("Iris", "Quartz", ("first", "second"), "surn")
        question = "What is the surn of the velar of Nacre?"
        self.assertEqual(self.session.memory.catalogue_answer(question)["form"], "chain")
        with mock.patch.object(provenance, "provenance",
                               wraps=provenance.provenance) as found:
            self.assertEqual(self.ask(question),
                             "surn of iris is Quartz (confidence 0.91; 2 sources agree), "
                             "through velar of nacre is Iris.")
        self.assertEqual(found.call_count, 1)
        self.assertEqual(found.call_args.args[2:], (key, "Quartz"))

    def test_chat_without_ledger(self):
        self.claim()
        self.ledger.path.unlink()
        with mock.patch.object(provenance, "provenance",
                               wraps=provenance.provenance) as found:
            self.assertIn("; one source)", self.ask("What is the atomic mass of Gold?"))
        self.assertIsNone(found.call_args.args[1])
        self.assertFalse(self.ledger.path.exists())


class NoteTests(unittest.TestCase):
    def test_every_branch_uses_data_words(self):
        path = Path(provenance.__file__).with_name("data") / "provenance.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        alternate = {"separator": " / ", "contested": "alternatives {answers}",
                     "agree": "supported by {n}", "one": "single"}
        for words in (stored, alternate):
            for teachers, contests, expected in (
                    ([], [], ""),
                    (["a"], [], words["separator"] + words["one"]),
                    (["a", "b"], [], words["separator"] + words["agree"].format(n=2)),
                    (["a", "b", "c"], [], words["separator"] + words["agree"].format(n=3)),
                    ([], ["95.95", "95.94"], words["separator"] + words["contested"].format(
                        answers="95.95, 95.94")),
                    (["a", "b"], ["254"], words["separator"] + words["contested"].format(
                        answers="254"))):
                with self.subTest(words=words, teachers=teachers, contests=contests):
                    with mock.patch.object(Path, "read_text", return_value=json.dumps(words)):
                        self.assertEqual(provenance.note(dict(teachers=teachers, contests=contests)),
                                         expected)


if __name__ == "__main__":
    unittest.main()
