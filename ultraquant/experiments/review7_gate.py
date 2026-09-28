"""Batches that commit whole, and an undo that knows its own record. The gate.

GPT-6 Astra's seventh adversarial review, of §11.136, found three high
defects. Claude reproduced each on the committed code:
- **a committed payload could be rewritten in place.** Relocate a loose
  shard inside a batch (pack or attach), then re-add the same payload:
  the write truncated the file the on-disk catalog still referenced, and
  a crash three bytes in made the shard unreadable;
- **a failed batch committed its partial work.** Its ``finally`` wrote
  the catalog while an exception propagated, so x was new and y old. A
  dispute persisted across buckets could land half done, and its replay,
  walking the graph from the disputed key, could not reach a descendant
  whose parent was already gone;
- **legacy identity mistook a later record for the approval's own.** A
  fresh record with the same value and confidence (a later derivation, or
  a re-teaching) passed every check, and the dispute deleted it.
Astra's fourth finding ("No" for invented names beginning "no") is left
to the indexing unit. Its recommended fix hardcodes "nobelium", which is
what the user ruled out.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 82132571...) before any fix existed:

1. **A committed payload is never rewritten**: in Astra's scenario, with
   a torn write injected and a crash before the batch finalizes, the
   on-disk state still reads the shard, and no payload write happened.
2. **A failed batch commits nothing**: after a restart both shards hold
   their old payloads, the vault's own view equals a fresh open, and a
   retry succeeds.
3. **Replay does not depend on the persisted graph**: from Astra's half
   landed state (height restored, safety deleted, access still derived
   from safety) recovery drops access, and nothing rests on a missing or
   changed premise.
4. **Legacy identity**: (a) a record re-derived after a first-format
   approval is superseded, not deleted; (b) so is one re-taught later with
   the same value and confidence; (c) an untouched approval still undoes
   exactly; (d) on a copy of the user's library, all 179 first-format
   approvals remain exact (run when the library is present).
5. **The exam can fail**: P24 (§11.136's reuse rule) breaches 1, P25 (a
   failed batch commits) breaches 2, P26 (replay ignores the journalled
   descendants) breaches 3, and P27 (§11.136's legacy identity) breaches
   4a.
6. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**PASSED** twice on Claude's machine: 6 of 6 cases, 4 of 4 plants, and
all 179 of the user's first-format approvals still exact on a copy.
§11.136's exam needed Amendment A: its fixture stamped a journal line
before the records it created, which the new identity rule rightly
refuses.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.undo_gate import World

__all__ = ["Review7Report", "run_gate"]

LIVE_HOME = Path(__file__).resolve().parents[2] / "uq_home"


def _scratch(prefix):
    return Path(tempfile.mkdtemp(prefix=prefix))


# -- 1-2: the vault ----------------------------------------------------------------

def committed_payload_kept() -> bool:
    """1: relocate a loose shard in a batch, re-add it, crash mid-write."""
    from ultraquant.shards import vault as V
    root = _scratch("uq_r7_reuse_")
    try:
        vault = V.ShardVault(root)
        vault.add_shard("a", "c", {"w": [1, 2, 3]})
        writes = []

        def torn(self, path, data):
            writes.append(Path(path).name)
            Path(path).write_bytes(data[:3])
            raise OSError("injected: power cut three bytes in")

        with vault.batch():
            vault.pack(root / "lib.uql", ["a"], prune_loose=False)
            with mock.patch.object(V.ShardVault, "_write_payload", torn):
                try:
                    vault.add_shard("a", "c", {"w": [1, 2, 3]})
                except OSError:
                    pass
            # A crash here never finalizes the batch: read the disk as is.
            try:
                readable = V.ShardVault(root).get("a") == {"w": [1, 2, 3]}
            except Exception:           # an integrity error is a failure
                readable = False
        return readable and not writes
    finally:
        shutil.rmtree(root, ignore_errors=True)


def failed_batch_commits_nothing() -> bool:
    """2: two shards rewritten in one batch, the second write fails."""
    from ultraquant.shards import vault as V
    root = _scratch("uq_r7_batch_")
    try:
        vault = V.ShardVault(root)
        vault.add_shard("x", "c", {"v": 1})
        vault.add_shard("y", "c", {"v": 1})
        real = V.ShardVault._write_payload
        calls = []

        def second_fails(self, path, data):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected: the second write fails")
            return real(self, path, data)

        with mock.patch.object(V.ShardVault, "_write_payload", second_fails):
            try:
                with vault.batch():
                    vault.add_shard("x", "c", {"v": 2})
                    vault.add_shard("y", "c", {"v": 2})
            except OSError:
                pass
        fresh = V.ShardVault(root)
        old = fresh.get("x") == {"v": 1} and fresh.get("y") == {"v": 1}
        same_view = all(vault.entry(s)["sha256"] == fresh.entry(s)["sha256"]
                        for s in ("x", "y"))
        with vault.batch():
            vault.add_shard("x", "c", {"v": 2})
            vault.add_shard("y", "c", {"v": 2})
        again = V.ShardVault(root)
        retried = again.get("x") == {"v": 2} and again.get("y") == {"v": 2}
        return old and same_view and retried
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- 3-4: the undo -----------------------------------------------------------------

def _world(body):
    world = World()
    try:
        return body(world)
    finally:
        world.cleanup()


def _stale(world) -> list:
    """Keys whose recorded premises no longer hold."""
    stale = []
    for key in world.memory.fact_keys():
        record = world.record(key)
        for p_key, p_value in (record or {}).get("derived_from", []):
            if str(world.value(p_key)) != str(p_value):
                stale.append(key)
    return stale


def replay_independent_of_graph() -> bool:
    """3: recovery from a dispute that landed half done."""
    from ultraquant.interpreter.autoapprove import AutoApprover

    def body(world):
        m = world.memory
        m.remember_fact("height of the pylon", "100 metres", 0.6)
        world.seed([("The height of the pylon is 200 metres.", "a.example")])
        approver = world.approver()
        approver.approve_all()
        m.consolidate_fact("safety of the pylon", "unsafe", 0.7,
                           [("height of the pylon", "200 metres")])
        m.consolidate_fact("access to the pylon", "closed", 0.7,
                           [("safety of the pylon", "unsafe")])

        def half_landed(self, intent):
            # What a half-committed persistence leaves: the height back,
            # safety gone, access still on disk.
            before = intent["approval"]["before"]["height of the pylon"]
            self.memory.restore_fact("height of the pylon", before)
            self.memory.restore_fact("safety of the pylon", None)
            self._persist()
            raise OSError("injected: power cut mid-dispute")

        with mock.patch.object(AutoApprover, "_complete_dispute", half_landed):
            try:
                approver.dispute("height of the pylon", "the old height")
            except OSError:
                pass
        world.restart()                 # recovery replays the dispute
        return (world.value("height of the pylon") == "100 metres"
                and world.record("access to the pylon") is None
                and not _stale(world))
    return _world(body)


def _legacy(world, key, value, confidence, entry_id=1) -> float:
    """A first-format journal line, stamped now, exactly as live."""
    time.sleep(0.005)
    stamp = time.time()
    line = {"event": "approval", "approval_id": f"legacy-{entry_id}",
            "entry_id": entry_id, "key": key, "value": value,
            "confidence": confidence, "sources": ["distill.invalid"],
            "time": stamp, "before": {key: None}, "outcome": "new",
            "disputed": False, "dispute_reason": ""}
    with (world.dir / "approvals.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    time.sleep(0.02)
    return stamp


def legacy_rederived_kept() -> bool:
    """4a: re-derived after the approval, same value and confidence."""
    def body(world):
        world.seed([("The height of the mast is 200 metres.", "distill.invalid")])
        m = world.memory
        m.remember_fact("height of the mast", "200 metres", 0.7)
        _legacy(world, "height of the mast", "200 metres", 0.7)
        m.remember_fact("survey of the mast", "new", 0.9)
        m.consolidate_fact("height of the mast", "200 metres", 0.7,
                           [("survey of the mast", "new")])
        later = world.record("height of the mast")
        world.approver().dispute("height of the mast", "the old approval")
        return world.record("height of the mast") == later
    return _world(body)


def legacy_retaught_kept() -> bool:
    """4b: gone, then re-taught later with the same value and confidence."""
    def body(world):
        world.seed([("The height of the mast is 200 metres.", "distill.invalid")])
        m = world.memory
        m.remember_fact("height of the mast", "200 metres", 0.7)
        _legacy(world, "height of the mast", "200 metres", 0.7)
        m.restore_fact("height of the mast", None)
        m.remember_fact("height of the mast", "200 metres", 0.7)
        later = world.record("height of the mast")
        world.approver().dispute("height of the mast", "the old approval")
        return world.record("height of the mast") == later
    return _world(body)


def legacy_untouched_undone() -> bool:
    """4c: the approval's own record, untouched, still undoes exactly."""
    def body(world):
        world.seed([("The height of the mast is 200 metres.", "distill.invalid")])
        world.memory.remember_fact("height of the mast", "200 metres", 0.7)
        _legacy(world, "height of the mast", "200 metres", 0.7)
        world.approver().dispute("height of the mast", "undo it")
        return world.record("height of the mast") is None
    return _world(body)


def live_legacy_exact() -> bool | None:
    """4d: every first-format approval in a COPY of the user's library."""
    if not (LIVE_HOME / "approvals.jsonl").exists():
        return None
    from ultraquant.interpreter.autoapprove import AutoApprover
    from ultraquant.interpreter.stash import ContemporaryStash
    from ultraquant.memory.factshards import FactShards
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.shards.vault import ShardVault
    root = _scratch("uq_r7_live_") / "uq_home"
    try:
        shutil.copytree(LIVE_HOME, root)
        memory = SystematicMemory(path=root / "memory.json")
        memory.shards = FactShards(ShardVault(root / "vault"))
        approver = AutoApprover(ContemporaryStash(root / "stash.json"), memory,
                                root / "approvals.jsonl")
        legacy = [a for a in approver.approvals() if a.after is None]
        return (len(legacy) == 179 and
                all(approver._dispute_mode(a) == "exact" for a in legacy))
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


CASES = {
    "1 a committed payload is never rewritten": committed_payload_kept,
    "2 a failed batch commits nothing": failed_batch_commits_nothing,
    "3 replay does not depend on the persisted graph": replay_independent_of_graph,
    "4a a re-derived record is kept": legacy_rederived_kept,
    "4b a re-taught record is kept": legacy_retaught_kept,
    "4c an untouched approval undoes": legacy_untouched_undone,
}


def _old_reuse(self, path, digest) -> bool:
    return (path in self._pending_loose_removals and path.exists()
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest)


def _old_identity(self, approval, current) -> bool:
    return (current is not None
            and current["value"] == approval.value
            and not current.get("negated", False)
            and current["confidence"] == approval.confidence
            and current.get("reinforcements", 0) == 0
            and current.get("last_seen") == current.get("first_seen"))


def _plants():
    from ultraquant.interpreter import autoapprove as A
    from ultraquant.shards import vault as V

    def commit_anyway(self, failed):
        if self._dirty:
            self._save_catalog()

    return [
        ("P24 the old reuse rule", "1 a committed payload is never rewritten",
         [mock.patch.object(V.ShardVault, "_reusable", _old_reuse)]),
        ("P25 a failed batch commits", "2 a failed batch commits nothing",
         [mock.patch.object(V.ShardVault, "_finish_batch", commit_anyway)]),
        ("P26 replay ignores the journalled descendants",
         "3 replay does not depend on the persisted graph",
         [mock.patch.object(A.AutoApprover, "_journalled_descendants",
                            lambda self, intent: [])]),
        ("P27 the old legacy identity", "4a a re-derived record is kept",
         [mock.patch.object(A.AutoApprover, "_legacy_untouched",
                            _old_identity)]),
    ]


@dataclass
class Review7Report:
    """The seventh review's fixes, measured.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        live: Criterion 4d on a copy of the user's library, or None when
            the library is not present.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    live: bool | None = None
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> Review7Report:
    report = Review7Report(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.live = live_legacy_exact()
    except Exception as exc:
        report.live = False
        report.errors["4d live library"] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = all(report.cases.values()) and report.live is not False
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        failed = [n for n, ok in report.cases.items() if not ok]
        if report.live is False:
            failed.append("4d the live library")
        report.reason = f"FAIL: {failed}"
    else:
        live = ("all 179 live legacy approvals exact" if report.live
                else "live library not present")
        report.reason = f"PASS: 6 cases; 4 of 4 plants caught; {live}"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "live": report.live,
                      "planted": report.planted, "errors": report.errors},
                     indent=1))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
