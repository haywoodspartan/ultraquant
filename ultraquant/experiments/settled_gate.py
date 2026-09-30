"""A kind no source can grow is settled for that source. The gate.

§11.174 failed as frozen: in seeds_gate's world capital gained a dictionary
kind whose property the recorded teacher cannot name; ``propose`` stored
nothing, the used-up judgement wants a property verdict for every learned
kind, and the source re-asked the same property question every round without
ever being used up. The defect predates §11.174 (any kind no teacher can grow
is re-asked every round, bounded only by ``max_rounds``). Here an ask that
names nothing new is recorded in the asking source's ledger, and a kind
counts as settled for a source once that source has been asked about it -
so the next source still gets its own chance.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 55ad9760...) before the implementation existed.
1. **seeds_gate passes again**: its criteria 1-4 hold on its own world.
2. **One unanswered ask, then settled**: in seeds_gate's world the first
   source, studying with the lexicon, is used up within 12 rounds, was asked
   the administrative district's property question exactly once, and its
   ledger holds ``property:administrative district``.
3. **A later source is still asked**: a second source (UNKNOWN to
   everything) then studies until used up (within 12 rounds), asked that
   question exactly once, its ledger holding the same row.
4. **Growth where a property is named is unchanged**: dictkind_gate's four
   criteria hold.
5. **The exam can fail**: P134 (the unanswered ask not recorded) breaches 2;
   P135 (any source's ask settling the kind for all) breaches 3.
6. **Nothing regresses**: checked outside this module.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["SettledReport", "run_gate"]

QUESTION = "Name one measurable property that every administrative district has."
ROW = "property:administrative district"
_RUN: dict = {}


class _Counting:
    """A teacher that remembers every question it was asked."""

    def __init__(self, teacher):
        self.teacher, self.spec, self.asked = teacher, teacher.spec, []

    def ask(self, questions, **kwargs):
        self.asked.extend(questions)
        return self.teacher.ask(questions, **kwargs)


class _Silent:
    """A second source that answers UNKNOWN to everything."""

    def __init__(self, spec):
        self.spec, self.asked = spec, []

    def ask(self, questions, *, samples, **kwargs):
        self.asked.extend(questions)
        return [["UNKNOWN"] * samples for _ in questions]


def _run() -> dict:
    if "run" in _RUN:
        return _RUN["run"]
    seeds_gate = importlib.import_module("ultraquant.experiments.seeds_gate")
    from ultraquant.distill import frontier
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.ownquestions_gate import _decided
    bound = _decided()[3]["wilson_lower"]
    with seeds_gate._world() as (lib, lexicon, teacher, ledger, first):
        results = {}
        for label, source, asker in (("first", first, _Counting(seeds_gate._Answering(teacher))),
                                     ("second", TG.SECOND, _Silent(SG._Second().spec))):
            used = []
            for n in range(1, 13):
                scratch = Path(tempfile.mkdtemp(prefix="uq_settled_"))
                result = frontier.study_round(
                    lib.memory, lib.stash, asker, ledger, source, confidence=bound,
                    run_id=f"settled-{label}-{n}", records_path=scratch / f"{n}.jsonl",
                    approver=lib.approver(), lexicon=lexicon)
                used.append(bool(result["used_up"]))
                if result["used_up"]:
                    break
            results[label] = {"rounds": len(used), "used up": bool(used and used[-1]),
                              "asked the question": asker.asked.count(QUESTION),
                              "ledger row": ROW in ledger.asked(source)}
    _RUN["run"] = results
    return results


# -- criteria -----------------------------------------------------------------------------

def seeds_again() -> bool:
    seeds_gate = importlib.import_module("ultraquant.experiments.seeds_gate")
    outcome = {name: bool(case()) for name, case in seeds_gate.CASES.items()}
    _RUN["seeds"] = outcome
    return len(outcome) == 4 and all(outcome.values())


def settled_after_one() -> bool:
    first = _run()["first"]
    return first["used up"] and first["asked the question"] == 1 and first["ledger row"]


def later_source_asked() -> bool:
    second = _run()["second"]
    return second["used up"] and second["asked the question"] == 1 and second["ledger row"]


def growth_unchanged() -> bool:
    dictkind = importlib.import_module("ultraquant.experiments.dictkind_gate")
    outcome = {name: bool(case()) for name, case in dictkind.CASES.items()}
    _RUN["dictkind"] = outcome
    return len(outcome) == 4 and all(outcome.values())


CASES = {"1 seeds_gate again": seeds_again, "2 settled after one ask": settled_after_one,
         "3 a later source is asked": later_source_asked, "4 growth unchanged": growth_unchanged}


# -- plants --------------------------------------------------------------------------------

def _plants():
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments import third_gate as TG
    real_record = sources.SourceLedger.record

    def forgetful(self, source, question_id, promoted, queued=None, check=None):
        if str(question_id).startswith("property:"):
            return None
        return real_record(self, source, question_id, promoted, queued=queued, check=check)

    def for_everyone(memory, ledger, source, kind):
        if frontier._has_property_verdict(memory, kind):
            return True
        return any(f"property:{kind}" in ledger.asked(name)
                   for name in (TG.FIRST, TG.SECOND, TG.THIRD))

    return [
        ("P134 the unanswered ask not recorded", "2 settled after one ask",
         [mock.patch.object(sources.SourceLedger, "record", forgetful)]),
        ("P135 any source's ask settles the kind", "3 a later source is asked",
         [mock.patch.object(frontier, "kind_settled", for_everyone)]),
    ]


@dataclass
class SettledReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> SettledReport:
    report = SettledReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("seeds", "run", "dictkind")}
    for name, target, patches in _plants():
        _RUN.clear()
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
        finally:
            _RUN.clear()
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = len(report.cases) == 4 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 2 of 2 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
