"""An undo that respects what came after.

11.135: a dispute restores exactly only when nothing has changed the
fact since; a later correction wins; two approvals of one key are
undone newest first; a stale conclusion stays retracted; approvals and
disputes are transactions that survive a crash. The whole exam runs in
about a second against temporary disk-backed stores.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import undo_gate as G


class UndoGateTests(unittest.TestCase):

    def test_the_whole_exam(self) -> None:
        report = G.run_gate()
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(len(report.planted), 8)

    def test_each_case_on_its_own(self) -> None:
        for name, case in G.CASES.items():
            with self.subTest(case=name):
                self.assertTrue(case())

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Version-aware disputes", "Transactions", "PASSED",
                       "legacy approval"):
            self.assertIn(phrase, doc)




class ReviewSixUndoTests(unittest.TestCase):

    def setUp(self) -> None:
        self.world = G.World()
        self.addCleanup(self.world.cleanup)

    def _revision(self, reinforced=False):
        world = self.world
        world.memory.remember_fact("height of the pylon", "100 metres", 0.6)
        world.memory.consolidate_fact("safety of the pylon", "safe", 0.7,
                                      [("height of the pylon", "100 metres")])
        value = "100" if reinforced else "200"
        world.seed([(f"The height of the pylon is {value} metres.", "a.example")])
        approver = world.approver()
        (approval,) = approver.approve_all()
        return approver, approval

    def _old_format(self, approval, version):
        import json
        from dataclasses import asdict

        row = asdict(approval)
        row.pop("after_states")
        if version == 1:
            row.pop("after")
            row["event"] = "approval"
        else:
            row.update(event="commit", operation="approval",
                       transaction_id=approval.approval_id)
        (self.world.dir / "approvals.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8")

    def test_later_secondary_and_its_new_conclusion_survive_exactly(self) -> None:
        approver, approval = self._revision()
        m = self.world.memory
        m.remember_fact("safety of the pylon", "unsafe", 0.9)
        m.consolidate_fact("access to the pylon", "closed", 0.7,
                           [("safety of the pylon", "unsafe")])
        safety = self.world.record("safety of the pylon")
        access = self.world.record("access to the pylon")
        approver.dispute(approval.key, "old height was right")
        self.assertEqual(self.world.value(approval.key), "100 metres")
        self.assertEqual(self.world.record("safety of the pylon"), safety)
        self.assertEqual(self.world.record("access to the pylon"), access)

    def test_changed_derived_secondary_is_preserved_before_retraction(self) -> None:
        approver, approval = self._revision(reinforced=True)
        m = self.world.memory
        m.confirm_fact("safety of the pylon", 0.9)
        m.consolidate_fact("access to the pylon", "open", 0.7,
                           [("safety of the pylon", "safe")])
        safety = self.world.record("safety of the pylon")
        access = self.world.record("access to the pylon")
        approver.dispute(approval.key, "undo reinforcement")
        self.assertEqual(self.world.record(approval.key), approval.before[approval.key])
        self.assertEqual(self.world.record("safety of the pylon"), safety)
        self.assertEqual(self.world.record("access to the pylon"), access)

    def test_v2_revised_row_restores_absent_secondaries(self) -> None:
        _, approval = self._revision()
        self._old_format(approval, 2)
        fresh = self.world.restart()
        loaded = fresh.approvals()[0]
        self.assertIsNone(loaded.after_states)
        self.assertEqual(loaded.after, approval.after)
        fresh.dispute(approval.key, "v2 revision")
        for key, record in approval.before.items():
            self.assertEqual(self.world.record(key), record)

    def test_v2_reinforcement_row_restores_unchanged_secondaries(self) -> None:
        _, approval = self._revision(reinforced=True)
        self._old_format(approval, 2)
        fresh = self.world.restart()
        fresh.dispute(approval.key, "v2 reinforcement")
        for key, record in approval.before.items():
            self.assertEqual(self.world.record(key), record)

    def test_v2_row_keeps_a_later_secondary(self) -> None:
        _, approval = self._revision()
        self._old_format(approval, 2)
        self.world.memory.remember_fact("safety of the pylon", "unsafe", 0.9)
        later = self.world.record("safety of the pylon")
        self.world.memory.save()
        self.world.restart().dispute(approval.key, "v2 changed secondary")
        self.assertEqual(self.world.record("safety of the pylon"), later)

    def test_v1_untouched_record_loads_and_undoes_exactly(self) -> None:
        world = self.world
        world.seed([("The height of the lamp is 200 metres.", "a.example")])
        (approval,) = world.approver().approve_all()
        self._old_format(approval, 1)
        fresh = world.restart()
        loaded = fresh.approvals()[0]
        self.assertIsNone(loaded.after)
        self.assertIsNone(loaded.after_states)
        self.assertEqual(fresh._dispute_mode(loaded), "exact")
        fresh.dispute(approval.key, "untouched v1")
        self.assertIsNone(world.record(approval.key))

    def test_v1_value_changed_and_back_is_superseded(self) -> None:
        world = self.world
        world.seed([("The height of the lamp is 200 metres.", "a.example")])
        (approval,) = world.approver().approve_all()
        self._old_format(approval, 1)
        world.memory.remember_fact(approval.key, "150 metres", 0.9)
        world.memory.remember_fact(approval.key, "200 metres", 0.9)
        later = world.record(approval.key)
        world.memory.save()
        fresh = world.restart()
        self.assertEqual(fresh._dispute_mode(fresh.approvals()[0]), "superseded")
        fresh.dispute(approval.key, "touched v1")
        self.assertEqual(world.record(approval.key), later)

    def test_every_legacy_identity_condition_matters(self) -> None:
        import copy

        world = self.world
        world.seed([("The height of the lamp is 200 metres.", "a.example")])
        (approval,) = world.approver().approve_all()
        self._old_format(approval, 1)
        fresh = world.approver()
        legacy = fresh.approvals()[0]
        untouched = world.record(approval.key)
        self.assertTrue(fresh._legacy_untouched(legacy, untouched))
        for change in ({"value": "150 metres"}, {"negated": True},
                       {"confidence": 0.9}, {"reinforcements": 1},
                       {"last_seen": "later"}):
            with self.subTest(change=change):
                record = copy.deepcopy(untouched)
                record.update(change)
                world.memory.restore_fact(approval.key, record)
                self.assertEqual(fresh._dispute_mode(legacy), "superseded")
        world.memory.restore_fact(approval.key, None)
        self.assertEqual(fresh._dispute_mode(legacy), "superseded")

    def test_primary_with_changed_premise_is_dropped_and_explained(self) -> None:
        world = self.world
        world.memory.remember_fact("material of the gate", "steel", 0.6)
        world.memory.consolidate_fact("height of the gate", "100 metres", 0.7,
                                      [("material of the gate", "steel")])
        world.seed([("The height of the gate is 200 metres.", "a.example")])
        approver = world.approver()
        (approval,) = approver.approve_all()
        world.memory.remember_fact("material of the gate", "wood", 0.9)
        approver.dispute(approval.key, "incorrect height")
        self.assertIsNone(world.record(approval.key))
        episode = world.memory.recall_episodes(kind="dispute")[0]["content"]
        self.assertEqual(episode["changed_premises"],
                         {approval.key: ["material of the gate"]})
        self.assertIn("material of the gate", episode["note"])

    def test_rollback_restores_losers_and_removes_only_its_episodes(self) -> None:
        from unittest import mock

        for winner_first in (True, False):
            with self.subTest(winner_first=winner_first):
                world = G.World()
                self.addCleanup(world.cleanup)
                rows = [("The capital of Kenya is Nairobi.", "a.example"),
                        ("The capital of Kenya is Mombasa.", "d.example")]
                world.seed(rows if winner_first else rows[::-1])
                winner, loser = (1, 2) if winner_first else (2, 1)
                world.stash._entries[winner]["sources"] = ["a.example", "b.example", "c.example"]
                # Claude's review: a memory holding "Mombasa" made the loser a
                # claim AGREEING with memory, approved first when it came
                # first, so the crash hit the loser's approval, not the
                # winner's. A third value makes both claims rivals of memory.
                world.memory.remember_fact("capital of kenya", "Kisumu", 0.6)
                world.memory.consolidate_fact("destination", "Kisumu", 0.7,
                                              [("capital of kenya", "Kisumu")])
                old_episode = world.memory.remember_episode("observation", {"keep": True})
                world.memory.save()
                approver = world.approver()
                journal = approver._journal

                def no_commit(row):
                    if row["event"] == "commit":
                        raise OSError("injected before commit")
                    journal(row)

                with mock.patch.object(approver, "_journal", no_commit):
                    with self.assertRaisesRegex(OSError, "injected"):
                        approver.approve_all()
                intent = approver._rows()[-1]
                self.assertEqual(world.stash.get(loser)["status"], "rejected")
                self.assertEqual({e["kind"] for e in world.memory.recall_episodes(
                    tags=[f"transaction:{intent['transaction_id']}"])},
                    {"promotion", "revision", "retraction"})
                fresh = world.restart()
                for entry in [intent["entry"], *intent["losers"]]:
                    self.assertEqual(world.stash.get(entry["id"]), entry)
                for key, record in intent["before"].items():
                    self.assertEqual(world.record(key), record)
                self.assertEqual([e["id"] for e in world.memory.recall_episodes()], [old_episode])
                self.assertEqual(list(world.memory._working), [old_episode])
                self.assertEqual(fresh.approvals(), [])
                fresh.recover()
                world.restart()
                self.assertEqual([e["id"] for e in world.memory.working()], [old_episode])

    def test_failure_during_promotion_still_tags_episodes_for_rollback(self) -> None:
        from unittest import mock

        world = self.world
        world.memory.remember_fact("height of the pylon", "100 metres", 0.6)
        world.seed([("The height of the pylon is 200 metres.", "a.example")])
        approver = world.approver()
        promote = world.stash.promote

        def failed(*args, **kwargs):
            promote(*args, **kwargs)
            raise OSError("injected during promotion")

        with mock.patch.object(world.stash, "promote", failed):
            with self.assertRaisesRegex(OSError, "injected"):
                approver.approve_all()
        approver.recover()
        self.assertEqual(world.value("height of the pylon"), "100 metres")
        self.assertEqual(world.memory.recall_episodes(), [])
        self.assertEqual(list(world.memory._working), [])

    def test_forget_episodes_counts_removals_and_keeps_fifo_capacity(self) -> None:
        from ultraquant.memory.systematic import SystematicMemory

        memory = SystematicMemory(working_capacity=2)
        memory.remember_episode("old", {}, tags=["transaction:gone"])
        kept = memory.remember_episode("kept", {}, tags=["transaction:keep"])
        memory.remember_episode("recent", {}, tags=["transaction:gone"])
        next_id = memory._next_id
        self.assertEqual(memory.forget_episodes("transaction:gone"), 2)
        self.assertEqual(memory.forget_episodes("transaction:gone"), 0)
        self.assertEqual([e["id"] for e in memory.recall_episodes()], [kept])
        self.assertEqual(list(memory._working), [kept])
        self.assertEqual(memory._working.maxlen, 2)
        self.assertEqual(memory.remember_episode("new", {}), next_id)


class ClaudeReviewUndoTests(unittest.TestCase):
    """Claude's review of §11.136: truth maintenance after an undo."""

    def setUp(self) -> None:
        self.world = G.World()
        self.addCleanup(self.world.cleanup)

    def _approve_200(self):
        m = self.world.memory
        m.remember_fact("height of the pylon", "100 metres", 0.6)
        m.consolidate_fact("safety of the pylon", "safe", 0.7,
                           [("height of the pylon", "100 metres")])
        self.world.seed([("The height of the pylon is 200 metres.", "a.example")])
        approver = self.world.approver()
        (approval,) = approver.approve_all()
        return approver, approval

    def _stale(self):
        stale = []
        for key in self.world.memory.fact_keys():
            record = self.world.record(key)
            for p_key, p_value in record.get("derived_from", []):
                if str(self.world.value(p_key)) != str(p_value):
                    stale.append(key)
        return stale

    def test_a_later_conclusion_drawn_from_the_disputed_value_goes(self) -> None:
        approver, approval = self._approve_200()
        self.world.memory.consolidate_fact(
            "safety of the pylon", "unsafe", 0.7,
            [("height of the pylon", "200 metres")])
        approver.dispute(approval.key, "the old height was right")
        self.assertEqual(self.world.value(approval.key), "100 metres")
        self.assertIsNone(self.world.record("safety of the pylon"))
        self.assertEqual(self._stale(), [])

    def test_a_conclusion_resting_partly_on_the_disputed_value_goes(self) -> None:
        approver, approval = self._approve_200()
        m = self.world.memory
        m.remember_fact("safety of the pylon", "unsafe", 0.9)
        m.consolidate_fact("access to the pylon", "closed", 0.7,
                           [("safety of the pylon", "unsafe"),
                            ("height of the pylon", "200 metres")])
        approver.dispute(approval.key, "the old height was right")
        self.assertEqual(self.world.value("safety of the pylon"), "unsafe")
        self.assertIsNone(self.world.record("access to the pylon"))
        self.assertEqual(self._stale(), [])

    def test_a_replayed_dispute_restores_what_the_first_attempt_did(self) -> None:
        from unittest import mock
        from ultraquant.interpreter.autoapprove import AutoApprover

        approver, approval = self._approve_200()
        persist = AutoApprover._persist
        calls = []

        def crash_after_saving(self):
            persist(self)
            calls.append(1)
            raise OSError("injected: power cut before the commit")

        with mock.patch.object(AutoApprover, "_persist", crash_after_saving):
            with self.assertRaises(OSError):
                approver.dispute(approval.key, "replayed")
        intent = approver._rows()[-1]
        self.assertEqual(intent["restore"], ["safety of the pylon"])
        self.world.restart()            # recovery completes the dispute
        self.assertEqual(self.world.value(approval.key), "100 metres")
        self.assertEqual(self.world.value("safety of the pylon"), "safe")
        self.assertEqual(self._stale(), [])


if __name__ == "__main__":
    unittest.main()
