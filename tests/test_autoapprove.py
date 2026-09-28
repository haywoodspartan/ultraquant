"""Quarantined knowledge approved automatically, and disputable exactly.

11.132: the user said "You may also auto approve all quarantined
knowledge. We can always dispute a claim later." These pins hold what
counts as well-formed, the exact undo, disputes across a restart, the
session never reading the user's setting on its own, and the whole
exam - which runs in a few seconds against a temporary stash and an
in-memory store.
"""

from __future__ import annotations

import copy
import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments import autoapprove_gate as G
from ultraquant.interpreter.autoapprove import AutoApprover
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.systematic import SystematicMemory


class _World(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_aa_test_"))
        self.memory = SystematicMemory(path=None)
        self.stash = ContemporaryStash(self.dir / "stash.json")
        self.journal = self.dir / "approvals.jsonl"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def seed(self, claims):
        data = {"entries": [], "next_id": len(claims) + 1}
        for index, (claim, source) in enumerate(claims, start=1):
            data["entries"].append({
                "id": index, "claim": claim, "classification": "unclassified",
                "status": "staged", "sources": [source], "netloc": source,
                "url": f"https://{source}/{index}", "title": "t",
                "fetched": "2026-09-28T00:00:00+00:00", "notes": ""})
        (self.dir / "stash.json").write_text(json.dumps(data), encoding="utf-8")
        self.stash = ContemporaryStash(self.dir / "stash.json")

    def approver(self):
        return AutoApprover(self.stash, self.memory, self.journal)


class WellFormedTests(_World):

    def test_malformed_claims_are_rejected_with_a_reason(self) -> None:
        self.seed([("What is code: set instructions computer", "lm-studio.invalid"),
                   ("The pixel is a dot end turn token", "lm-studio.invalid"),
                   ("You have mentioned 'code' 6 times and I hold nothing "
                    "about it.", "lm-studio.invalid"),
                   ("The capital of Kenya is Nairobi.", "a.example")])
        approvals = self.approver().approve_all()
        self.assertEqual([a.key for a in approvals], ["capital of kenya"])
        notes = {e["id"]: e["notes"] for e in self.stash.entries()
                 if e["status"] == "rejected"}
        self.assertEqual(sorted(notes), [1, 2, 3])
        self.assertTrue(all(n.startswith("malformed:") for n in notes.values()))

    def test_opinion_and_hedged_claims_wait_for_review(self) -> None:
        self.seed([("The best programming language is Python.", "a.example"),
                   ("The treasure is reportedly buried in Oak Island.",
                    "b.example")])
        self.assertEqual(self.approver().approve_all(), [])
        self.assertEqual({e["status"] for e in self.stash.entries()},
                         {"staged"})


class ExactUndoTests(_World):

    def test_a_revision_and_its_retracted_chain_come_back(self) -> None:
        self.memory.remember_fact("tallest mountain", "K2", confidence=0.6)
        self.memory.consolidate_fact("tallest mountain country", "Pakistan",
                                     0.7, [("tallest mountain", "K2")])
        before = {k: copy.deepcopy(self.memory._fact_record(k))
                  for k in ("tallest mountain", "tallest mountain country")}
        self.seed([("The tallest mountain is Mount Everest.", "a.example")])
        approver = self.approver()
        (approval,) = approver.approve_all()
        self.assertEqual(approval.outcome, "revised")
        self.assertIsNone(self.memory._fact_record("tallest mountain country"))
        approver.dispute("tallest mountain", "K2 was right for this purpose")
        after = {k: self.memory._fact_record(k) for k in before}
        self.assertEqual(after, before)

    def test_a_dispute_after_a_restart(self) -> None:
        self.seed([("The capital of Kenya is Nairobi.", "a.example")])
        self.approver().approve_all()
        fresh = AutoApprover(ContemporaryStash(self.dir / "stash.json"),
                             self.memory, self.journal)
        fresh.dispute("capital of kenya", "checking the undo")
        self.assertIsNone(self.memory._fact_record("capital of kenya"))
        # 11.135: a retried dispute returns the finished one; it never raises
        again = fresh.dispute("capital of kenya", "twice")
        self.assertTrue(again.disputed)
        with self.assertRaises(KeyError):
            fresh.dispute("never approved", "no such approval")

    def test_a_first_version_journal_can_still_be_undone(self) -> None:
        """Claude's review of 11.135: approvals journalled before `after`
        existed - 179 of them in the user's library - must stay disputable."""
        self.memory.remember_fact("chemical symbol of gold", "Au", 0.959)
        legacy = {"event": "approval", "approval_id": "legacy-1",
                  "entry_id": 1, "key": "chemical symbol of gold",
                  "value": "Au", "confidence": 0.959, "sources": ["x"],
                  "time": 1.0, "before": {"chemical symbol of gold": None},
                  "outcome": "new"}
        self.journal.write_text(json.dumps(legacy) + chr(10), encoding="utf-8")
        self.seed([("The chemical symbol of gold is Au.", "distill.invalid")])
        self.approver().dispute("chemical symbol of gold", "legacy undo")
        self.assertIsNone(self.memory._fact_record("chemical symbol of gold"))


class SessionIsolationTests(unittest.TestCase):
    """Claude's review: the session once read the user's settings itself,
    so every gate would have inherited the user's choice."""

    def test_sessions_are_told_never_read(self) -> None:
        from ultraquant.interpreter import thoughts as T
        self.assertIn("auto_approve",
                      inspect.signature(T.build_session).parameters)
        self.assertNotIn("Settings", inspect.getsource(T.Session.__post_init__))
        self.assertIs(inspect.signature(T.build_session)
                      .parameters["auto_approve"].default, False)


class GateTests(unittest.TestCase):

    def test_the_whole_exam(self) -> None:
        report = G.run_gate()
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(len(report.planted), 5)

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Eligible approved", "A dispute undoes exactly",
                       "Disputes survive a restart", "PASSED"):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main()
