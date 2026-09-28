"""An undo that respects what came after. The dispute gate.

§11.132 made approval automatic and promised that a dispute undoes an
approval exactly. GPT-6 Astra's fifth adversarial review showed what
"exactly" had missed:
- a dispute restored its snapshot whatever had happened since, erasing
  a user's later correction, and two approvals of one key disputed
  oldest-first ended at the wrong value;
- persistence was not atomic;
- derived conclusions came back although their other premises had
  changed;
- re-filing a run re-approved a disputed claim;
- a negation on a later line still manufactured agreement;
- a title containing "is" split at the wrong copula;
- nobelium's "No" was refused as a negation.
Auto-approval stayed on at the user's request while this was fixed, and
the user was told what the undo could not yet do.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 f13065f6...) before any fix existed:

1. **Version-aware disputes**:
   - (a) a user's correction made after an approval survives the
     approval's dispute;
   - (b) two approvals of one key: oldest-first is refused, and
     newest-then-oldest ends where it began;
   - (c) a derived conclusion whose other premise changed is not
     restored.
2. **Transactions**: an approval interrupted before its commit is wholly
   present or wholly rolled back after a restart; an interrupted
   dispute completes; and a retried dispute is idempotent.
3. **Re-filing a disputed run approves nothing.**
4. **A negation on a later line** does not manufacture agreement.
5. **Structured claims** keep their key when a title contains "is".
6. **Nobelium**: "No" is a symbol for a symbol question, and a negation
   everywhere else.
7. **The exam can fail**: P6-P13, each restoring the old behaviour.
8. **§11.132's and §11.133's gates stay PASS, and the suite is green.**
   Checked outside this module.

**PASSED** twice on Claude's machine: all 9 cases, 8 of 8 plants caught.
Astra's two runs agreed. Every other gate stayed green:
- §11.132's auto-approval exam;
- §11.130's replay, still 90/90;
- §11.134's replay, still 101/101, with nobelium now promoted: all 15
  of its samples said "No", and the symbol exception keeps that a
  position.

**Claude's review caught one thing this exam did not.** The user's
library already held 179 approvals, journalled in the first format,
which has no ``after`` snapshot. v2 read any approval without ``after``
as superseded, so disputing one of them would have marked it disputed
and left the fact in place, exactly where "we can always dispute a
claim later" has to work. Now a legacy approval counts as an exact undo
when the key still holds its approved value. A test pins it with a
first-format journal.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["UndoReport", "run_gate"]


class World:
    """A disk-backed memory, stash and journal, so a restart is real."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_undo_"))
        self.open()

    def open(self) -> None:
        from ultraquant.interpreter.stash import ContemporaryStash
        from ultraquant.memory.systematic import SystematicMemory
        self.memory = SystematicMemory(path=self.dir / "memory.json")
        self.stash = ContemporaryStash(self.dir / "stash.json")

    def approver(self):
        from ultraquant.interpreter.autoapprove import AutoApprover
        return AutoApprover(self.stash, self.memory, self.dir / "approvals.jsonl")

    def restart(self):
        """Forget everything in RAM; reopen from disk; a fresh approver."""
        self.open()
        return self.approver()

    def seed(self, claims) -> None:
        data = {"entries": [], "next_id": len(claims) + 1}
        for index, (claim, source) in enumerate(claims, start=1):
            data["entries"].append({
                "id": index, "claim": claim, "classification": "unclassified",
                "status": "staged", "sources": [source], "netloc": source,
                "url": f"https://{source}/{index}", "title": "t",
                "fetched": "2026-09-28T00:00:00+00:00", "notes": ""})
        (self.dir / "stash.json").write_text(json.dumps(data), encoding="utf-8")
        from ultraquant.interpreter.stash import ContemporaryStash
        self.stash = ContemporaryStash(self.dir / "stash.json")

    def value(self, key):
        record = self.memory._fact_record(key)
        return record.get("value") if record else None

    def record(self, key):
        found = self.memory._fact_record(key)
        return copy.deepcopy(found) if found is not None else None

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _world(body):
    world = World()
    try:
        return body(world)
    finally:
        world.cleanup()


# -- 1: version-aware disputes -------------------------------------------------

def correction_survives() -> bool:
    """1a: approve 200 over 100, the user teaches 150, dispute: 150 stays."""
    def body(world):
        world.memory.remember_fact("height of the tower", "100 metres", 0.6)
        world.seed([("The height of the tower is 200 metres.", "a.example")])
        approver = world.approver()
        (approval,) = approver.approve_all()
        world.memory.remember_fact("height of the tower", "150 metres", 0.9)
        approver.dispute("height of the tower", "wrong height")
        episodes = [e for e in world.memory.recall_episodes(limit=100)
                    if "dispute" in str(e.get("kind"))]
        entry = world.stash.get(approval.entry_id)
        return (world.value("height of the tower") == "150 metres"
                and entry["status"] == "rejected" and bool(episodes))
    return _world(body)


def newest_first() -> bool:
    """1b: two approvals of one key; oldest-first refused; LIFO ends at 100."""
    def body(world):
        world.memory.remember_fact("height of the mast", "100 metres", 0.6)
        world.seed([("The height of the mast is 200 metres.", "a.example")])
        approver = world.approver()
        (first,) = approver.approve_all()
        world.stash.add_claim(
            "https://b.example/2", "t", "The height of the mast is 300 metres.")
        (second,) = approver.approve_all()
        refused = False
        try:
            approver.dispute(first.entry_id, "older first")
        except (ValueError, RuntimeError, KeyError):
            refused = True
        unchanged = world.value("height of the mast") == "300 metres"
        approver.dispute(second.entry_id, "newest")
        approver.dispute(first.entry_id, "then the older")
        return (refused and unchanged
                and world.value("height of the mast") == "100 metres")
    return _world(body)


def stale_conclusion_stays_retracted() -> bool:
    """1c: a derived fact whose other premise changed is not restored."""
    def body(world):
        memory = world.memory
        memory.remember_fact("height of the pier", "100 metres", 0.6)
        memory.remember_fact("material of the pier", "steel", 0.6)
        memory.consolidate_fact("pier is safe", "yes", 0.7,
                                [("height of the pier", "100 metres"),
                                 ("material of the pier", "steel")])
        world.seed([("The height of the pier is 200 metres.", "a.example")])
        approver = world.approver()
        approver.approve_all()
        memory.remember_fact("material of the pier", "wood", 0.9)
        approver.dispute("height of the pier", "wrong height")
        return (world.value("height of the pier") == "100 metres"
                and world.record("pier is safe") is None)
    return _world(body)


# -- 2: transactions -----------------------------------------------------------

def interrupted_approval() -> bool:
    """A failure after promotion and before commit; then a restart."""
    from ultraquant.interpreter.autoapprove import AutoApprover

    def body(world):
        world.seed([("The capital of Kenya is Nairobi.", "a.example")])
        approver = world.approver()

        def crash(self, *args, **kwargs):
            raise OSError("injected: disk vanished before the commit")

        with mock.patch.object(AutoApprover, "_persist", crash):
            try:
                approver.approve_all()
            except OSError:
                pass
        fresh = world.restart()
        fresh.recover()
        entry = world.stash.entries()[0]
        present = world.value("capital of kenya") == "Nairobi"
        journal = fresh.approvals()
        committed = [a for a in journal if not a.disputed]
        if present:
            return entry["status"] == "promoted" and len(committed) == 1
        return entry["status"] in ("staged", "corroborated") and not committed
    return _world(body)


def interrupted_dispute() -> bool:
    """A dispute cut off before its commit completes on restart; retry is
    idempotent."""
    from ultraquant.interpreter.autoapprove import AutoApprover

    def body(world):
        world.seed([("The capital of Kenya is Nairobi.", "a.example")])
        approver = world.approver()
        approver.approve_all()

        def crash(self, *args, **kwargs):
            raise OSError("injected: power cut mid-dispute")

        with mock.patch.object(AutoApprover, "_persist", crash):
            try:
                approver.dispute("capital of kenya", "checking recovery")
            except OSError:
                pass
        fresh = world.restart()
        fresh.recover()
        done = world.value("capital of kenya") is None
        try:
            fresh.dispute("capital of kenya", "retry")
            retried = True
        except KeyError:
            retried = False
        return done and retried
    return _world(body)


# -- 3-6 -------------------------------------------------------------------------

def refile_is_idempotent() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.experiments import distill_facts_gate as D
    from ultraquant.experiments import knowledge_bench as K

    def body(world):
        records = E.load_records(D.RECORDS)
        gold = [i for i in K.KNOWN if i.subject == "gold" and i.category == "symbol"]
        mine = [r for r in records if r.question_id == E.question_id(gold[0])]
        approver = world.approver()
        F.file_distilled(world.stash, mine, gold, 0.959, "11130")
        approver.approve_all()
        approver.dispute("chemical symbol of gold", "disputed on purpose")
        again = F.file_distilled(world.stash, mine, gold, 0.959, "11130")
        reapproved = approver.approve_all()
        return not again and not reapproved and world.value(
            "chemical symbol of gold") is None
    return _world(body)


def later_line_negation() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.experiments import knowledge_bench as K
    item = K.Item("symbol", "silver", "What is the chemical symbol of silver?",
                  ("ag",))
    records = [E.Record(t, lineage, "x.gguf", 1, E.question_id(item),
                        item.question, seed, raw, E.normalize(E.extract(raw)))
               for t, lineage, raw in (("A", "L0", "Au"),
                                       ("B", "L1", "Au\nNot Au; Ag."))
               for seed in E.SEEDS]
    return E.decide(records, [item])[E.question_id(item)] is None


def structured_title_with_is() -> bool:
    from ultraquant.distill import file as F  # noqa: F401 (the filer's path)

    def body(world):
        entry_id = world.stash.add_claim(
            "https://distill.invalid/t/author:moon", "Who wrote it?",
            "The author of The Moon Is a Harsh Mistress is Robert Heinlein.",
            measured_confidence=0.959, provenance={"run_id": "t"},
            fields={"key": "author of the moon is a harsh mistress",
                    "value": "Robert Heinlein"})
        world.approver().approve_all()
        return (world.value("author of the moon is a harsh mistress")
                == "Robert Heinlein" and world.value("author of the moon")
                is None and entry_id is not None)
    return _world(body)


def nobelium() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    item = next(i for i in H.KNOWN
                if i.subject == "nobelium" and i.category == "symbol")
    return (E.held(["No"] * 5, category="symbol") == "no"
            and K.is_correct(item, "No")
            and E.held(["No"] * 5, category="capital") is None)


CASES = {
    "1a a correction survives": correction_survives,
    "1b newest first": newest_first,
    "1c a stale conclusion stays retracted": stale_conclusion_stays_retracted,
    "2 an interrupted approval": interrupted_approval,
    "2 an interrupted dispute": interrupted_dispute,
    "3 re-filing approves nothing": refile_is_idempotent,
    "4 a later-line negation": later_line_negation,
    "5 a title containing is": structured_title_with_is,
    "6 nobelium": nobelium,
}


def _plants():
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.interpreter import autoapprove as A
    from ultraquant.interpreter import stash as S

    def first_line_only(text, category=None):
        answer = E.normalize(E.extract(text))
        return (not E.is_abstention(text)
                and not __import__("re").search(
                    r"\b(?:not|no|never|nor|or)\b", answer))

    def category_blind(text, category=None):
        return first_line_only(text)

    return [
        ("P6 snapshot-only disputes", "1a a correction survives",
         [mock.patch.object(A.AutoApprover, "_dispute_mode",
                            lambda self, approval: "exact")]),
        ("P7 no newest-first refusal", "1b newest first",
         [mock.patch.object(A.AutoApprover, "_dispute_mode",
                            lambda self, approval: "exact")]),
        ("P8 derived records restored unchecked",
         "1c a stale conclusion stays retracted",
         [mock.patch.object(A.AutoApprover, "_restorable",
                            lambda self, key, record: True)]),
        ("P9 no commit protocol", "2 an interrupted dispute",
         [mock.patch.object(A.AutoApprover, "recover", lambda self: None)]),
        ("P10 filing not idempotent", "3 re-filing approves nothing",
         [mock.patch.object(F, "_already_filed",
                            lambda stash, run_id, qid: False)]),
        ("P11 first-line negation only", "4 a later-line negation",
         [mock.patch.object(E, "is_position", first_line_only)]),
        ("P12 prose reparsing", "5 a title containing is",
         [mock.patch.object(S.ContemporaryStash, "_fields_of",
                            lambda self, entry: None)]),
        ("P13 the category-blind negation rule", "6 nobelium",
         [mock.patch.object(E, "is_position", category_blind)]),
    ]


@dataclass
class UndoReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> UndoReport:
    import contextlib
    report = UndoReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
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
                         f"{[n for n, c in report.planted.items() if not c]}")
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
                      "errors": report.errors}, indent=1))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
