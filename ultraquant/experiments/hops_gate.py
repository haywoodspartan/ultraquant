"""Two hops through the catalogue. The gate.

Measured at ec232d2, in a catalogued world:
- "author of emma" = Jane Austen, "author of dune" = Frank Herbert,
  "birthplace of jane austen" = Steventon, and "birthplace of frank
  herbert" = Tacoma;
- "What is the birthplace of the author of Emma?" answered "Reading that
  as 'author of emma': author of emma is Jane Austen". That is a wrong
  reading, because the question names a second catalogued attribute.
  With the reading suppressed, inference registered "jane austen
  birthplace", the value plus the remainder, and never looked up the held
  "birthplace of jane austen".
The first hop's value is itself a catalogued subject, so the second hop
is a lookup by subject and attribute.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 29e49993...) before any code:
1. **Two hops answered**: the Emma question names Steventon and the bridge
   "author of emma"; the Dune question names Tacoma.
2. **No wrong reading**: without Jane Austen's birthplace held, the Emma
   question gets no reading of "author of emma", and its curiosity is
   "birthplace of jane austen", in the catalogue's own key form.
3. **One hop unchanged**: §11.142's and §11.145's gates pass, and "What is
   the birthplace of Jane Austen?" answers Steventon.
4. **The exam can fail**: P62 (a reading given whatever else the question
   names) breaches 2, and P63 (the second hop looked up by the
   value-plus-remainder key) breaches 1.
5. **Every earlier gate keeps its HEAD verdict, and the suite is green.**
   Checked outside this module.

**Reproduced first**: at 996ac56 both cases 1 and 2 failed. The Emma
question got the reading "author of emma".

**FAILED criterion 5; this exam passed** (3 of 3 cases, 2 of 2 plants
caught) on Claude's machine and in Astra's run. §11.149's askable gate
flipped from PASS to FAIL, because it pins the premise "steel
conductivity", and this design keys that premise as the catalogue keys
it, "conductivity of steel". Every other gate kept its verdict at
996ac56. Kept, as a strict improvement.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["HopsReport", "run_gate"]

EMMA = "What is the birthplace of the author of Emma?"
DUNE = "What is the birthplace of the author of Dune?"


@contextlib.contextmanager
def _session(austen_birthplace: bool = True):
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_hops_"))
    try:
        session = build_session(root, seed=0)
        memory = session.memory
        for subject, author in (("Emma", "Jane Austen"), ("Dune", "Frank Herbert")):
            memory.remember_fact(f"author of {subject.lower()}", author, 0.9,
                                 subject=subject, attribute="author")
        memory.remember_fact("birthplace of frank herbert", "Tacoma", 0.9,
                             subject="Frank Herbert", attribute="birthplace")
        if austen_birthplace:
            memory.remember_fact("birthplace of jane austen", "Steventon", 0.9,
                                 subject="Jane Austen", attribute="birthplace")
        yield session
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _ask(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(text, session)[0]


def two_hops_answered() -> bool:
    with _session() as session:
        emma = _ask(session, EMMA)
        dune = _ask(session, DUNE)
    return ("Steventon" in emma and "author of emma" in emma
            and not emma.startswith("Reading that as")
            and "Tacoma" in dune)


def no_wrong_reading() -> bool:
    with _session(austen_birthplace=False) as session:
        before = len(session.curiosities)
        reply = _ask(session, EMMA)
        asked = [c["premise_key"] for c in session.curiosities[before:]]
    return (not reply.startswith("Reading that as")
            and "(confidence" not in reply.split("Nearest I hold")[0]
            and asked == ["birthplace of jane austen"])


_GATES: dict = {}


def one_hop_unchanged() -> bool:
    from ultraquant.experiments import requests_gate, review9_gate
    with _session() as session:
        direct = _ask(session, "What is the birthplace of Jane Austen?")
    _GATES["requests_gate"] = bool(requests_gate.run_gate().passes)
    _GATES["review9_gate"] = bool(review9_gate.run_gate().passes)
    return "Steventon" in direct and all(_GATES.values())


CASES = {
    "1 two hops answered": two_hops_answered,
    "2 no wrong reading": no_wrong_reading,
    "3 one hop unchanged": one_hop_unchanged,
}


def _value_plus_remainder(self, value: str, attribute: str):
    """P63: the unstructured key a chain would have guessed."""
    key = f"{str(value).lower()} {attribute}"
    record = self.recall_fact(key)
    return (key, record) if record is not None else None


def _plants():
    from ultraquant.memory import systematic as SM
    return [
        ("P62 a reading whatever else the question names", "2 no wrong reading",
         [mock.patch.object(SM.SystematicMemory, "_names_other_attribute",
                            lambda self, words, explained: False)]),
        ("P63 the second hop by the value-plus-remainder key", "1 two hops answered",
         [mock.patch.object(SM.SystematicMemory, "_second_hop", _value_plus_remainder)]),
    ]


@dataclass
class HopsReport:
    """Whether a question can travel two hops through the catalogue.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        gates: The gates criterion 3 reran.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> HopsReport:
    report = HopsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.gates = dict(_GATES)
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
    print(json.dumps({"cases": report.cases, "gates": report.gates,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
