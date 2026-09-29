"""Names that look like arithmetic. The gate.

The quantity path (``calculate.evaluate``, §11.78) runs before recall and
the catalogue, and reads operators wherever they appear. Measured on the
user's library (a copy):
- four held subjects contain an operator character, all hyphens:
  Catch-22, Moby-Dick, Nineteen Eighty-Four and Slaughterhouse-Five;
- only Catch-22 misroutes, because "22" is a number. "Who wrote
  Catch-22?" and three other author forms answer "I can't compute that:
  I hold nothing for '...catch'."
- This is the one miss in each author form of §11.142's measure (0.989 =
  88 of 89).
Here the calculator reads the catalogue before its operators: a held
subject's recorded name, named in the text, is one word.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 04054974...) before any code:
1. **Held names are names**: on the built library, four author forms
   answer Joseph Heller for Catch-22, and §11.142's measure shows the three
   gated author forms at 1.0 (was 0.989).
2. **Arithmetic unaffected**:
   - "What is 22 - 7?" is 15, and "What is 10-2?" is 8;
   - "What is the tower height times 3?" is 900 meters over a held
     belief;
   - "What is catch-22 plus 3?" never answers with Joseph Heller.
3. **Nothing regresses**: §11.145's gate (which reruns §11.142's) and
   the arithmetic, quantity, rounding, polar-arithmetic and native
   calculator gates pass.
4. **The exam can fail**: P56 (no catalogue reading) breaches 1, and P57
   (a hardcoded rule: every hyphen between word characters is part of a
   name) breaches 2.
5. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**FAILED criterion 3**, on Claude's machine and in Astra's run. Cases 1
and 2 held, and both plants were caught. There were two causes:
- **The quantity gate fails on HEAD without this change.** Quantity
  arithmetic now scores 1.000 on every seed, and the gate demands a
  nonzero seed sd. The pre-registration listed it without measuring its
  baseline, which was Claude's error.
- **§11.145's gate ends one hint over its ceiling, 69 against 68.**
  "Catch-22 was written by whom?", the form §11.141 left ungated, used
  to be misread as arithmetic ("I can't compute that: I hold nothing for
  'catch'"). It now gets that form's ordinary reply, "Nearest I hold:
  author of catch-22 is Joseph Heller", with the same hint as the other
  68 subjects.

**Kept, as a strict improvement.** Catch-22 is answered in all four
forms, the gated author forms reach 1.0, and arithmetic is unchanged.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["NamesReport", "run_gate"]

FORMS = ("Who wrote Catch-22?", "Who wrote the novel Catch-22?",
         "Who is the author of Catch-22?", "What is the author of Catch-22?")
AUTHOR_FORMS = ("Who wrote {s}?", "Who wrote the novel {s}?", "Who is the author of {s}?")


def _ask(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(text, session)[0]


def held_names_are_names() -> bool:
    from ultraquant.experiments import requests_gate as R
    from ultraquant.interpreter.thoughts import build_session
    R._forget(False)                    # measure afresh; keep the build
    measured = R._measured()
    session = build_session(R._CACHE["lib"].root, seed=0)
    replies = [_ask(session, form) for form in FORMS]
    rates = [R._rate(measured["forms"].get(form, {})) for form in AUTHOR_FORMS]
    return (all("Joseph Heller" in reply and not reply.startswith("I can't")
                for reply in replies)
            and all(rate == 1.0 for rate in rates))


def arithmetic_unaffected() -> bool:
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_names_"))
    try:
        session = build_session(root, seed=0)
        session.memory.remember_fact("tower height", "300 meters", 0.9)
        session.memory.remember_fact("author of catch-22", "Joseph Heller", 0.9,
                                     subject="Catch-22", attribute="author")
        return ("= 15" in _ask(session, "What is 22 - 7?")
                and "= 8" in _ask(session, "What is 10-2?")
                and "= 900 meters" in _ask(session, "What is the tower height times 3?")
                and "Joseph Heller" not in _ask(session, "What is catch-22 plus 3?"))
    finally:
        shutil.rmtree(root, ignore_errors=True)


_GATES = ("review9_gate", "arithmetic_gate", "quantity_gate", "rounding_gate",
          "polararith_gate", "nativecalc_gate")


def nothing_regresses() -> bool:
    outcomes = {}
    for name in _GATES:
        module = importlib.import_module(f"ultraquant.experiments.{name}")
        outcomes[name] = bool(module.run_gate().passes)
    _REGRESSIONS.update(outcomes)
    return all(outcomes.values())


_REGRESSIONS: dict = {}

CASES = {
    "1 held names are names": held_names_are_names,
    "2 arithmetic unaffected": arithmetic_unaffected,
    "3 nothing regresses": nothing_regresses,
}


def _hyphenated(text: str, memory=None) -> list:
    """P57: a hardcoded rule, not the catalogue."""
    return re.findall(r"\w+(?:-\w+)+", text)


def _plants():
    from ultraquant.reason import calculate as C
    return [
        ("P56 no catalogue reading", "1 held names are names",
         [mock.patch.object(C, "_held_names", lambda text, memory=None: [])]),
        ("P57 every hyphen between word characters is a name",
         "2 arithmetic unaffected",
         [mock.patch.object(C, "_held_names", _hyphenated)]),
    ]


@dataclass
class NamesReport:
    """Whether held names that look like arithmetic stay names.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        regressions: Gate -> whether it passed (criterion 3).
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    regressions: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> NamesReport:
    from ultraquant.experiments import requests_gate as R
    report = NamesReport(passes=False)
    try:
        for name, case in CASES.items():
            try:
                report.cases[name] = bool(case())
            except Exception as exc:    # a crash is a failure
                report.cases[name] = False
                report.errors[name] = repr(exc)
        report.regressions = dict(_REGRESSIONS)
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
    print(json.dumps({"cases": report.cases, "regressions": report.regressions,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
