"""Curiosity about attributes the catalogue knows. The gate.

``inference.missing_premise`` forms a premise from a held bridge's value
plus the question's uncovered words. Measured on the built library
(§11.142's measure, after §11.147):
- there were 69 curiosity hints, all on the form §11.141 left ungated,
  "X was written by whom?";
- "Emma was written by whom?" registered "jane austen written" via
  "author of emma". "written" is neither a catalogued attribute nor an
  attested way of asking for one;
- the 69th hint put §11.145's gate one over its ceiling.
Here a catalogued bridge asks only for an attribute the catalogue knows.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 2b86f608...) before any code, with Amendment A (sha256
3776f598...), also before any code:
1. **No junk**: §11.142's measure shows 0 curiosity hints (was 69).
2. **Curiosity still asks what it should**: with "material of the tower"
   = steel and "conductivity of copper" = high catalogued, "What is the
   conductivity of the tower?" registers "steel conductivity". The
   curiosity gate passes.
3. **Nothing regresses**:
   - §11.145's gate passes again, with §11.142's inside it;
   - the ladder, paraphrase, negation and compound gates keep their
     verdicts at HEAD (ec232d2, measured before any code and recorded
     in ``HEAD_VERDICTS``).
4. **The exam can fail**: P60 (any remainder, as today) breaches 1, and
   P61 (no curiosity from catalogued bridges) breaches 2.
5. **Every earlier gate keeps its HEAD verdict, and the suite is green.**
   Checked outside this module.

**PASSED** on Claude's machine and in Astra's run: 3 of 3 cases, 2 of 2
plants caught. The built library shows 0 curiosity hints (was 69). "steel
conductivity" is still asked, and the curiosity gate passes. §11.145's
gate passes again. The ladder and compound gates keep their HEAD
failures, and the paraphrase and negation gates their passes.

**Criterion 5 FAILED. Found after the commit and corrected during
§11.150.** The §11.139 catalogue gate flipped from PASS to FAIL on its
criterion 4, which pins "steel conductivity" asked through a catalogued
bridge for an uncatalogued attribute. That is exactly what this design
removes. Only the gates named here were run before the commit. Kept, as
a strict improvement.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["AskableReport", "run_gate"]

#: Verdicts at HEAD, measured before any code for this unit.
#: Measured at ec232d2 on a clean worktree: the ladder has failed since
#: f2f4137, and the compound gate fails on its own success (every world
#: +1.000, so nothing varies), like the quantity gate (§11.147).
HEAD_VERDICTS = {"ladder_gate": False, "paraphrase_gate": True,
                 "negation_gate": True, "compound_gate": False}


def no_junk() -> bool:
    from ultraquant.experiments import requests_gate as R
    R._forget(False)                    # measure afresh; keep the build
    measured = R._measured()
    _MEASURED["hints"] = dict(measured["hints"])
    return sum(measured["hints"].values()) == 0


_MEASURED: dict = {}


def still_asks() -> bool:
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    root = Path(tempfile.mkdtemp(prefix="uq_askable_"))
    try:
        session = build_session(root, seed=0)
        memory = session.memory
        memory.remember_fact("material of the tower", "steel", 0.9,
                             subject="the tower", attribute="material")
        memory.remember_fact("conductivity of copper", "high", 0.9,
                             subject="copper", attribute="conductivity")
        run_pipeline("What is the conductivity of the tower?", session)
        asked = [c["premise_key"] for c in session.curiosities]
        tower = asked == ["steel conductivity"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    from ultraquant.experiments import curiosity_gate
    _MEASURED["curiosity_gate"] = bool(curiosity_gate.run_gate().passes)
    return tower and _MEASURED["curiosity_gate"]


def nothing_regresses() -> bool:
    from ultraquant.experiments import review9_gate
    outcomes = {"review9_gate": bool(review9_gate.run_gate().passes)}
    for name in HEAD_VERDICTS:
        module = importlib.import_module(f"ultraquant.experiments.{name}")
        outcomes[name] = bool(module.run_gate().passes)
    _MEASURED["gates"] = outcomes
    return (outcomes["review9_gate"]
            and all(outcomes[name] == verdict
                    for name, verdict in HEAD_VERDICTS.items()))


CASES = {
    "1 no junk": no_junk,
    "2 curiosity still asks what it should": still_asks,
    "3 nothing regresses": nothing_regresses,
}


def _plants():
    from ultraquant.reason import inference as I
    return [
        ("P60 any remainder, as today", "1 no junk",
         [mock.patch.object(I, "_remainder_known", lambda remainder, memory: True)]),
        ("P61 no curiosity from catalogued bridges",
         "2 curiosity still asks what it should",
         [mock.patch.object(I, "_remainder_known", lambda remainder, memory: False)]),
    ]


@dataclass
class AskableReport:
    """Whether catalogued curiosity asks only for known attributes.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: Hints per form, the curiosity gate, and the gates of 3.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> AskableReport:
    from ultraquant.experiments import requests_gate as R
    report = AskableReport(passes=False)
    if any(verdict is None for verdict in HEAD_VERDICTS.values()):
        report.reason = "VOID: HEAD verdicts were not recorded before the run"
        return report
    try:
        for name, case in CASES.items():
            try:
                report.cases[name] = bool(case())
            except Exception as exc:    # a crash is a failure
                report.cases[name] = False
                report.errors[name] = repr(exc)
        report.measured = json.loads(json.dumps(_MEASURED, default=str))
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
            except Exception as exc:    # a crash proves nothing
                report.planted[name] = False
                report.errors[name] = repr(exc)
    finally:
        R._forget(True)
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 3 cases; 2 of 2 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
