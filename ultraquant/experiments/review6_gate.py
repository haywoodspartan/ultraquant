"""Storage that survives a crash, and the sixth review. The gate.

GPT-6 Astra's sixth adversarial review found nine defects, seven of them
high:
- the shard vault overwrote a bucket's payload in place, then published
  the catalog's new hash, so a crash in between made a whole bucket of
  75-120 of the user's facts unreadable;
- undo restored secondary records over later changes;
- Claude's legacy-journal fallback took equal values for identity;
- a second approval pass promoted a rival the first pass had defeated;
- the primary restored record skipped premise validation;
- the nobelium exception made "No" a symbol for every element;
- the filing boundary accepted invented benchmark probes;
- a rollback kept the aborted transaction's history;
- "The Leopard" names two novels.
GPT-6 Astra wrote the fixes; Claude wrote this exam.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 7346c153...) before any fix existed:

1. **A crash cannot lose a bucket**: a failure during the payload write,
   and one after it but before the catalog is published, leave every
   fact of the bucket readable after a restart, including an unrelated
   one.
2. **Undo respects every later change**:
   - (a) on secondary keys;
   - (b) on legacy approvals, which undo exactly only when untouched;
   - (c) a primary record whose premise changed is not restored.
3. **Arbitration holds across passes**, in either id order.
4. **A rollback leaves no history of the aborted change.**
5. **"No" for an invented element is refused**, and nobelium still
   passes.
6. **Invented probes never file.**
7. **The exam can fail**: P14-P21.
8. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

Reported and not gated: whether the vault accepts the next write after
each crash, and how many orphaned payload files that write leaves.

**PASSED** four times on Claude's machine: all 9 cases, and 8 of 8
plants caught. After both crashes the next write read back and left no
orphans. Every earlier gate stayed green. Astra's own run agreed.

**Claude's review caught what this exam did not.** Astra's undo spared
every secondary that had changed, and everything derived from it. So a
conclusion derived from the DISPUTED value ("safety: unsafe", from a
height of 200) outlived the height going back to 100. Now the plan of
what to restore is judged once and journalled, and after the restore
every derivative whose own premises no longer hold is dropped. Tests in
``tests/test_undo.py`` pin both cases and the replay.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.undo_gate import World

__all__ = ["Review6Report", "run_gate"]


# -- 1: storage -----------------------------------------------------------------

def _bucket_keys(shards, n):
    """n keys that land in one bucket, found by the store's own hashing."""
    seen = {}
    index = 0
    while True:
        key = f"height of tower {index}"
        seen.setdefault(shards.bucket_of(key), []).append(key)
        for keys in seen.values():
            if len(keys) >= n:
                return keys[:n]
        index += 1


_AFTERMATH: dict = {}


def _crash_case(stage: str) -> bool:
    """A failure at ``stage`` of a bucket write; a restart must read all."""
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards import vault as V

    root = Path(tempfile.mkdtemp(prefix="uq_vaultcrash_"))
    try:
        shards = FactShards(V.ShardVault(root))
        keys = _bucket_keys(shards, 6)
        for key in keys:
            shards.put(key, {"value": f"{key} = 1", "confidence": 0.5})
        shards.flush()
        shards.put(keys[0], {"value": f"{keys[0]} = 2", "confidence": 0.5})

        def torn(self, path, data):
            Path(path).write_bytes(data[: max(1, len(data) // 2)])
            raise OSError("injected: power cut mid-write")

        def refuse(self, *args, **kwargs):
            raise OSError("injected: power cut before the catalog")

        patch = (mock.patch.object(V.ShardVault, "_write_payload", torn)
                 if stage == "payload"
                 else mock.patch.object(V.ShardVault, "_write_catalog", refuse))
        with patch:
            try:
                shards.flush()
            except OSError:
                pass
        fresh = FactShards(V.ShardVault(root))
        values = [fresh.get(k) for k in keys]
        first_ok = values[0] is not None and values[0]["value"] in (
            f"{keys[0]} = 1", f"{keys[0]} = 2")
        others_ok = all(v is not None and v["value"] == f"{k} = 1"
                        for k, v in zip(keys[1:], values[1:]))
        _aftermath(stage, root, keys)
        return first_ok and others_ok
    except Exception:                   # an integrity error is a failure
        return False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _aftermath(stage, root, keys) -> None:
    """Reported: the next write after the crash, and the orphans it leaves."""
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards import vault as V
    try:
        shards = FactShards(V.ShardVault(root))
        shards.put(keys[0], {"value": f"{keys[0]} = 3", "confidence": 0.5})
        shards.flush()
        reread = FactShards(V.ShardVault(root)).get(keys[0])
        vault = V.ShardVault(root)
        live = {vault._loose_path(s).name for s in
                (e["shard_id"] for e in vault.catalog())
                if vault.entry(s)["location"] == "loose"}
        files = {p.name for p in (root / "loose").iterdir()
                 if re.search(r"\.uqs$", p.name)}
        _AFTERMATH[stage] = {
            "next write reads back": bool(reread)
            and reread["value"] == f"{keys[0]} = 3",
            "orphaned payload files": len(files - live)}
    except Exception as exc:            # reported, never gated
        _AFTERMATH[stage] = {"error": repr(exc)}


# -- 2-4: undo ------------------------------------------------------------------

def _world(body):
    world = World()
    try:
        return body(world)
    finally:
        world.cleanup()


def _seed_sources(world, rows) -> None:
    """World.seed, with several sources per claim."""
    from ultraquant.interpreter.stash import ContemporaryStash
    data = {"entries": [], "next_id": len(rows) + 1}
    for index, (claim, sources) in enumerate(rows, start=1):
        data["entries"].append({
            "id": index, "claim": claim, "classification": "unclassified",
            "status": "staged", "sources": list(sources),
            "netloc": sources[0], "url": f"https://{sources[0]}/{index}",
            "title": "t", "fetched": "2026-09-28T00:00:00+00:00",
            "notes": ""})
    (world.dir / "stash.json").write_text(json.dumps(data), encoding="utf-8")
    world.stash = ContemporaryStash(world.dir / "stash.json")


def secondary_change_survives() -> bool:
    """2a: a later assertion on a secondary key, and its conclusion, stay."""
    def body(world):
        m = world.memory
        m.remember_fact("height of the pylon", "100 metres", 0.6)
        m.consolidate_fact("safety of the pylon", "safe", 0.7,
                           [("height of the pylon", "100 metres")])
        world.seed([("The height of the pylon is 200 metres.", "a.example")])
        approver = world.approver()
        approver.approve_all()
        m.remember_fact("safety of the pylon", "unsafe", 0.9)
        m.consolidate_fact("access to the pylon", "closed", 0.7,
                           [("safety of the pylon", "unsafe")])
        approver.dispute("height of the pylon", "the old height was right")
        return (world.value("height of the pylon") == "100 metres"
                and world.value("safety of the pylon") == "unsafe"
                and world.value("access to the pylon") == "closed")
    return _world(body)


def _legacy(world, entry_id, key, value, confidence) -> None:
    """A first-format journal line, exactly as the user's library holds."""
    line = {"event": "approval", "approval_id": f"legacy-{entry_id}",
            "entry_id": entry_id, "key": key, "value": value,
            "confidence": confidence, "sources": ["distill.invalid"],
            "time": 1790625746.0, "before": {key: None}, "outcome": "new",
            "disputed": False, "dispute_reason": ""}
    with (world.dir / "approvals.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def legacy_identity() -> bool:
    """2b: 200 -> 150 -> 200 is a later change; an untouched one undoes."""
    def body(world):
        world.seed([("The height of the lamp is 200 metres.", "distill.invalid"),
                    ("The height of the bell is 30 metres.", "distill.invalid")])
        m = world.memory
        m.remember_fact("height of the lamp", "200 metres", 0.964)
        _legacy(world, 1, "height of the lamp", "200 metres", 0.964)
        m.remember_fact("height of the lamp", "150 metres", 0.9)
        m.remember_fact("height of the lamp", "200 metres", 0.9)
        m.remember_fact("height of the bell", "30 metres", 0.964)
        _legacy(world, 2, "height of the bell", "30 metres", 0.964)
        approver = world.approver()
        approver.dispute("height of the lamp", "legacy, but changed since")
        approver.dispute("height of the bell", "legacy, untouched")
        return (world.value("height of the lamp") == "200 metres"
                and world.value("height of the bell") is None)
    return _world(body)


def primary_premise_checked() -> bool:
    """2c: the restored primary rests on a premise that no longer holds."""
    def body(world):
        m = world.memory
        m.remember_fact("material of the gate", "steel", 0.6)
        m.consolidate_fact("height of the gate", "100 metres", 0.7,
                           [("material of the gate", "steel")])
        world.seed([("The height of the gate is 200 metres.", "a.example")])
        approver = world.approver()
        approver.approve_all()
        m.remember_fact("material of the gate", "wood", 0.9)
        approver.dispute("height of the gate", "that height was wrong")
        return world.value("height of the gate") is None
    return _world(body)


def _arbitration(winner_first: bool) -> bool:
    def body(world):
        nairobi = ("The capital of Kenya is Nairobi.",
                   ["a.example", "b.example", "c.example"])
        mombasa = ("The capital of Kenya is Mombasa.", ["d.example"])
        rows = [nairobi, mombasa] if winner_first else [mombasa, nairobi]
        _seed_sources(world, rows)
        approver = world.approver()
        approver.approve_all()
        approver.approve_all()
        loser = world.stash.get(2 if winner_first else 1)
        return (world.value("capital of kenya") == "Nairobi"
                and loser["status"] == "rejected")
    return _world(body)


def arbitration_holds() -> bool:
    """3: Nairobi (3 sources) over Mombasa (1), in both id orders."""
    return _arbitration(True) and _arbitration(False)


def rollback_forgets() -> bool:
    """4: a revision rolled back leaves no promotion or revision episode."""
    from ultraquant.interpreter.autoapprove import AutoApprover

    def body(world):
        world.memory.remember_fact("capital of kenya", "Mombasa", 0.6)
        world.seed([("The capital of Kenya is Nairobi.", "a.example")])
        approver = world.approver()
        original = AutoApprover._journal

        def no_commit(self, row):
            if row.get("event") == "commit":
                raise OSError("injected: crash before the commit")
            return original(self, row)

        with mock.patch.object(AutoApprover, "_journal", no_commit):
            try:
                approver.approve_all()
            except OSError:
                pass
        world.restart()                 # recovery rolls the approval back
        left = [e for e in world.memory.recall_episodes(limit=10 ** 6)
                if e["kind"] in ("promotion", "revision", "retraction")
                and "Nairobi" in json.dumps(e["content"])]
        return world.value("capital of kenya") == "Mombasa" and not left
    return _world(body)


# -- 5-6: filter and filing ------------------------------------------------------

def invented_no_refused() -> bool:
    """5: "No" x5 from two lineages: refused for zorbenium, kept for nobelium."""
    from ultraquant.distill import elicit as E
    from ultraquant.experiments import knowledge_bench as K

    def decided(subject):
        item = K.Item("symbol", subject,
                      f"What is the chemical symbol of {subject}?")
        records = [E.Record(t, lineage, "x.gguf", 1, E.question_id(item),
                            item.question, seed, "No",
                            E.normalize(E.extract("No")))
                   for t, lineage in (("A", "L0"), ("B", "L1"))
                   for seed in E.SEEDS]
        return E.decide(records, [item])[E.question_id(item)]
    return decided("zorbenium") is None and decided("nobelium") == "no"


def probes_never_file() -> bool:
    """6: the varnadium probe, which the teachers answered "V", is refused."""
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.experiments import knowledge_bench_hard as H
    from ultraquant.experiments.distill_hard_gate import RECORDS

    def body(world):
        probe = next(i for i in H.FICTITIOUS if i.subject == "varnadium"
                     and i.category == "symbol")
        records = [r for r in E.load_records(RECORDS)
                   if r.question_id == E.question_id(probe)]
        try:
            F.file_distilled(world.stash, records, [probe], 0.964, "11134")
            return False
        except ValueError:
            return world.stash.entries() == []
    return _world(body)


CASES = {
    "1a a crash during the payload write": lambda: _crash_case("payload"),
    "1b a crash before the catalog": lambda: _crash_case("catalog"),
    "2a a later secondary change survives": secondary_change_survives,
    "2b legacy undo only when untouched": legacy_identity,
    "2c a primary with a changed premise stays out": primary_premise_checked,
    "3 arbitration holds across passes": arbitration_holds,
    "4 a rollback leaves no history": rollback_forgets,
    "5 No for an invented element is refused": invented_no_refused,
    "6 invented probes never file": probes_never_file,
}


def _plants():
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.interpreter import autoapprove as A
    from ultraquant.shards import vault as V

    return [
        ("P14 payload written in place", "1a a crash during the payload write",
         [mock.patch.object(V.ShardVault, "_payload_path",
                            lambda self, shard_id, digest:
                            self._loose_path(shard_id))]),
        ("P15 secondaries restored blindly",
         "2a a later secondary change survives",
         [mock.patch.object(A.AutoApprover, "_still_holds",
                            lambda self, key, expected: True)]),
        ("P16 equal values taken for identity",
         "2b legacy undo only when untouched",
         [mock.patch.object(A.AutoApprover, "_legacy_untouched",
                            lambda self, approval, current:
                            current is not None
                            and current.get("value") == approval.value)]),
        ("P17 the primary's premises unchecked",
         "2c a primary with a changed premise stays out",
         [mock.patch.object(A.AutoApprover, "_restorable",
                            lambda self, key, record: True)]),
        ("P18 arbitration not remembered", "3 arbitration holds across passes",
         [mock.patch.object(A.AutoApprover, "_record_loss",
                            lambda self, entry, winner: None)]),
        ("P19 a rollback keeps its episodes", "4 a rollback leaves no history",
         [mock.patch.object(A.AutoApprover, "_forget_transaction_episodes",
                            lambda self, transaction_id: None)]),
        ("P20 No accepted for any element",
         "5 No for an invented element is refused",
         [mock.patch.object(E, "_symbol_no_allowed", lambda subject: True)]),
        ("P21 probes allowed to file", "6 invented probes never file",
         [mock.patch.object(F, "_reject_probes", lambda items: None)]),
    ]


@dataclass
class Review6Report:
    """The sixth review's fixes, measured.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        aftermath: Reported only: the vault's next write after each crash.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    aftermath: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> Review6Report:
    report = Review6Report(passes=False)
    _AFTERMATH.clear()
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.aftermath = dict(_AFTERMATH)
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
    valid = len(report.planted) == 8 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}"
                         f"{'' if plants else ' (no plants built)'}")
    elif not met:
        report.reason = ("FAIL: "
                         f"{[n for n, ok in report.cases.items() if not ok]}")
    else:
        report.reason = "PASS: 9 cases; 8 of 8 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "errors": report.errors, "aftermath": report.aftermath},
                     indent=1))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
