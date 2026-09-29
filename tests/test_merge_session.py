"""tools/merge_session.py: a session staged on a copy, merged into the live library."""

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("merge_session", ROOT / "tools" / "merge_session.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

GUI_CLAIM = "The capital of Testland is Testville."


def _claim(stash, run_id, subject, attribute, value, teachers=("first",), **extra):
    question_id = f"{attribute}:{subject}".lower()
    return stash.add_claim(
        f"https://distill.invalid/{run_id}/{question_id}", f"What is the {attribute} of {subject}?",
        f"The {attribute} of {subject} is {value}.", measured_confidence=0.9,
        provenance={"run_id": run_id, "question_id": question_id, "teachers": list(teachers),
                    "lineages": list(teachers), **extra},
        fields={"key": f"{attribute} of {subject}".lower(), "subject": subject,
                "attribute": attribute, "value": value})


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class MergeSessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="uq_merge_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # The base: the library as the session's copy was taken.
        self.base = self.tmp / "base" / "uq_home"
        self.base.mkdir(parents=True)
        memory, stash, approver = M._open(self.base)
        self.seeded = _claim(stash, "seed-1", "France", "capital", "Paris")
        _claim(stash, "seed-1", "Spain", "capital", "Madrid")
        approver.approve_all()
        memory.save()
        (self.base / "sources.json").write_text(json.dumps({"first": [["q0", True]]}), encoding="utf-8")
        # The session, run on a copy: two rounds of one source, then a second
        # source that corroborates a seeded claim and revises a first-round one.
        self.staging = self.tmp / "staging" / "uq_home"
        shutil.copytree(self.base, self.staging)
        memory, stash, approver = M._open(self.staging)
        _claim(stash, "session-T-first-1", "Chad", "capital", "N'Djamena")
        approver.approve_all()
        _claim(stash, "session-T-first-2", "Mali", "capital", "Timbuktu")
        approver.approve_all()
        stash.add_teacher(self.seeded, "second", "second.gguf:2", prior_teacher_ids=["first"])
        _claim(stash, "session-T-second-1", "Mali", "capital", "Bamako", teachers=("second", "third"),
               settles="capital of mali")
        approver.approve_all()
        memory.learn_kind("capital", "country")
        memory.learn_kind("population", None)     # asked, and no shared kind
        memory.save()
        (self.staging / "sources.json").write_text(json.dumps(
            {"first": [["q0", True], ["q1", True], ["q2", True]], "second": [["q3", True]]}),
            encoding="utf-8")

    def _live(self, name, gui=False):
        live = self.tmp / name / "uq_home"
        shutil.copytree(self.base, live)
        if gui:
            memory, stash, _ = M._open(live)
            memory.remember_fact("favourite colour of the tester", "blue", 0.9,
                                 subject="the tester", attribute="favourite colour")
            memory.remember_fact("capital of spain", "Madrid (corrected in the GUI)", 0.99,
                                 subject="Spain", attribute="capital")
            memory.save()
            stash.add_claim("https://example.org/lookup", "A GUI lookup", GUI_CLAIM,
                            measured_confidence=0.9)
        return live

    def _value(self, root, key):
        memory, _, _ = M._open(root)
        return (memory.recall_fact(key) or {}).get("value")

    def _status(self, root, claim):
        _, stash, _ = M._open(root)
        return [e["status"] for e in stash.entries() if e["claim"] == claim]

    def test_an_unchanged_live_library_is_replaced_whole(self):
        live = self._live("unchanged")
        report = M.merge(self.staging, live, self.base)
        self.assertTrue(report["clean"], report)
        self.assertIn("replaced", report["mode"])
        self.assertEqual(_hashes(live), _hashes(self.staging))
        self.assertEqual(self._value(live, "capital of mali"), "Bamako")

    def test_a_changed_live_library_gets_the_session_replayed(self):
        live = self._live("changed", gui=True)
        report = M.merge(self.staging, live, self.base)
        self.assertTrue(report["clean"], report)
        self.assertIn("replayed", report["mode"])
        self.assertEqual((report["replayed"], report["batches"], report["hidden"]), (3, 3, 1))
        self.assertEqual(report["live changed"], 2)
        # The session's knowledge, with the later round's revision winning.
        self.assertEqual(self._value(live, "capital of chad"), "N'Djamena")
        self.assertEqual(self._value(live, "capital of mali"), "Bamako")
        # The GUI's own changes kept, and its staged claim left for the user.
        self.assertEqual(self._value(live, "favourite colour of the tester"), "blue")
        self.assertEqual(self._value(live, "capital of spain"), "Madrid (corrected in the GUI)")
        self.assertEqual(self._status(live, GUI_CLAIM), ["staged"])
        # The teacher, the kind and the ledger.
        _, stash, _ = M._open(live)
        seeded = next(e for e in stash.entries() if e["claim"] == "The capital of France is Paris.")
        self.assertEqual(seeded["provenance"]["teachers"], ["first", "second"])
        vocabulary = M._open(live)[0]._attribute_vocabulary()
        self.assertEqual(vocabulary["capital"].get("kind"), "country")
        self.assertIn("kind", vocabulary["population"])
        self.assertIsNone(vocabulary["population"]["kind"])
        self.assertEqual(json.loads((live / "sources.json").read_text(encoding="utf-8")),
                         json.loads((self.staging / "sources.json").read_text(encoding="utf-8")))

    def test_merging_twice_changes_nothing_more(self):
        live = self._live("twice", gui=True)
        M.merge(self.staging, live, self.base)
        before = _hashes(live)
        report = M.merge(self.staging, live, self.base)
        self.assertTrue(report["clean"], report)
        self.assertEqual(report["replayed"], 0)
        self.assertEqual(self._value(live, "capital of mali"), "Bamako")
        self.assertEqual(self._status(live, GUI_CLAIM), ["staged"])
        self.assertEqual(set(_hashes(live)), set(before))

    def test_a_ledger_that_moved_on_is_refused(self):
        live = self._live("moved")
        (live / "sources.json").write_text(json.dumps({"first": [["q0", True], ["other", False]]}),
                                           encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            M.merge(self.staging, live, self.base)

    def test_a_skipped_approval_is_reported_lost(self):
        live = self._live("lost", gui=True)
        real = M._open

        def skipping(root):
            memory, stash, approver = real(root)
            if Path(root) == live:
                calls = {"n": 0}
                original = approver.approve_all

                def approve_all():
                    calls["n"] += 1
                    return [] if calls["n"] == 3 else original()
                approver.approve_all = approve_all
            return memory, stash, approver
        with mock.patch.object(M, "_open", skipping):
            report = M.merge(self.staging, live, self.base)
        self.assertFalse(report["clean"])
        self.assertEqual(report["lost"], ["fact: capital of mali"])

    def test_a_kind_not_copied_is_reported_lost(self):
        live = self._live("kind", gui=True)
        real = M.replay

        def forgetful(staging, live_, base):
            report = real(staging, live_, base)
            memory = M._open(live_)[0]
            memory._attribute_vocabulary().get("population", {}).pop("kind", None)
            memory.save()
            return report
        with mock.patch.object(M, "replay", forgetful):
            report = M.merge(self.staging, live, self.base)
        self.assertFalse(report["clean"])
        self.assertEqual(report["lost"], ["kind: population"])

    def test_approving_the_users_own_claim_is_reported_touched(self):
        live = self._live("touched", gui=True)
        with mock.patch.object(M, "_as_the_session_saw_it",
                               lambda stash, hidden: contextlib.nullcontext()):
            report = M.merge(self.staging, live, self.base)
        self.assertFalse(report["clean"])
        self.assertEqual(len(report["touched"]), 1)
        self.assertEqual(self._status(live, GUI_CLAIM), ["promoted"])

    def test_check_writes_nothing(self):
        live = self._live("check", gui=True)
        before = _hashes(live)
        with contextlib.redirect_stdout(io.StringIO()):
            code = M.main([str(self.staging), str(live), "--backup-dir", str(self.tmp / "b"),
                           "--base", str(self.base), "--check"])
        self.assertEqual(code, 0)
        self.assertEqual(_hashes(live), before)
        self.assertFalse((self.tmp / "b").exists())

    def test_a_real_merge_backs_up_first(self):
        live = self._live("real", gui=True)
        before = _hashes(live)
        with contextlib.redirect_stdout(io.StringIO()):
            code = M.main([str(self.staging), str(live), "--backup-dir", str(self.tmp / "b"),
                           "--base", str(self.base)])
        self.assertEqual(code, 0)
        backups = list((self.tmp / "b").glob("uq_home-*-pre-merge"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(_hashes(backups[0]), before)
        self.assertEqual(self._value(live, "capital of mali"), "Bamako")

    def test_an_unclean_check_writes_nothing(self):
        live = self._live("unclean", gui=True)
        before = _hashes(live)
        with mock.patch.object(M, "_as_the_session_saw_it",
                               lambda stash, hidden: contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):
            code = M.main([str(self.staging), str(live), "--backup-dir", str(self.tmp / "b"),
                           "--base", str(self.base)])
        self.assertEqual(code, 1)
        self.assertEqual(_hashes(live), before)
        self.assertFalse((self.tmp / "b").exists())

    def test_the_base_is_found_beside_the_staging_library(self):
        staging = self.tmp / "stage" / "uq_home"
        (self.tmp / "stage" / "backups" / "uq_home-20260929T000000Z").mkdir(parents=True)
        self.assertEqual(M.default_base(staging).name, "uq_home-20260929T000000Z")
        (self.tmp / "stage" / "backups" / "uq_home-20260930T000000Z").mkdir()
        with self.assertRaisesRegex(ValueError, "pass --base"):
            M.default_base(staging)


if __name__ == "__main__":
    unittest.main()
