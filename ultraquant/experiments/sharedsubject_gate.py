"""Questions that name no subject, and parts that share one. The gate.

Measured on the user's library (a copy, b6b0597):
- **A question naming no subject asserted an arbitrary one.** 8 of 12
  attribute-only questions did so. "What is the chemical symbol?"
  answered "chemical symbol of fermium is Fm (confidence 0.96)". About
  100 keys cover the question's words, and the keyword fallback asserted
  the first. Covering is not identifying.
- **Two attributes of one subject were 3 of 60 fully right**, and 58
  replies named another subject: "What is the atomic number and chemical
  symbol of gold?" answered with fermium's atomic number.
- **One attribute of two subjects was 6 of 30 fully right.**

**The criteria, written before the run** - frozen in a pre-registration
(sha256 53fcbcc3...) before any code:
1. **No arbitrary subject**: 0 of the 12 attribute-only questions assert
   a fact.
2. **Two attributes of one subject**: at least 57 of 60 fully right,
   with 0 replies naming another subject. "What are the capital and author
   of Kenya?" gives Nairobi.
3. **One attribute of two subjects**: at least 28 of 30 fully right.
4. **Nothing regresses**: a sweep of the change matches the §11.151
   ledger. Checked outside this module.
5. **The exam can fail**: P68 (the first covering key asserted) breaches
   1, P69 (parts answered without the shared subject) breaches 2, and P70
   (only the first named subject answered) breaches 3.
"""

from __future__ import annotations

import contextlib
import json
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import LIVE_HOME

__all__ = ["SharedSubjectReport", "run_gate"]

ATTRIBUTES = ("capital", "author", "atomic number", "chemical symbol")
FORMS = ("What is the {a}?", "Tell me the {a}.", "What is its {a}?")


@contextlib.contextmanager
def _session():
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_shared_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        yield build_session(root, seed=0)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def _ask(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(text, session)[0]


def _held(session) -> dict:
    memory = session.memory
    held = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key)
        if record and record.get("subject") and record.get("attribute"):
            held.setdefault(record["subject"], {})[record["attribute"]] = str(record["value"])
    return held


def _elements(held) -> list:
    both = sorted(x for x, attrs in held.items()
                  if "atomic number" in attrs and "chemical symbol" in attrs)
    return both


_MEASURED: dict = {}


def no_arbitrary_subject() -> bool:
    with _session() as session:
        asserted = 0
        for attribute in ATTRIBUTES:
            for form in FORMS:
                reply = _ask(session, form.format(a=attribute))
                asserted += int("(confidence" in reply and not reply.startswith("Reading")
                                and "Nearest" not in reply)
    _MEASURED["asserted of 12"] = asserted
    return asserted == 0


def two_attributes() -> bool:
    with _session() as session:
        held = _held(session)
        rng = random.Random(153)
        sample = rng.sample(_elements(held), 30)
        right = named_other = 0
        for x in sample:
            for question in (f"What is the atomic number and chemical symbol of {x}?",
                             f"What is the chemical symbol and atomic number of {x}?"):
                reply = _ask(session, question)
                right += int(held[x]["atomic number"] in reply
                             and held[x]["chemical symbol"] in reply)
                named_other += int(any(y != x and len(y) > 3 and y.lower() in reply.lower()
                                       for y in held))
        kenya = _ask(session, "What are the capital and author of Kenya?")
    _MEASURED["two attributes"] = {"right of 60": right, "named another": named_other,
                                   "kenya": kenya[:160]}
    return right >= 57 and named_other == 0 and "Nairobi" in kenya


def two_subjects() -> bool:
    with _session() as session:
        held = _held(session)
        rng = random.Random(153)
        elements = _elements(held)
        rng.sample(elements, 30)                    # the same draws as measured
        pairs = [tuple(rng.sample(elements, 2)) for _ in range(30)]
        right = 0
        for p, q in pairs:
            reply = _ask(session, f"What is the chemical symbol of {p} and {q}?")
            right += int(held[p]["chemical symbol"] in reply
                         and held[q]["chemical symbol"] in reply)
    _MEASURED["two subjects right of 30"] = right
    return right >= 28


CASES = {
    "1 no arbitrary subject": no_arbitrary_subject,
    "2 two attributes of one subject": two_attributes,
    "3 one attribute of two subjects": two_subjects,
}


def _plants():
    from ultraquant.interpreter import thoughts as T
    from ultraquant.memory import systematic as SM
    real_answers = SM.SystematicMemory.catalogue_answers

    def first_only(self, text):
        answers = real_answers(self, text)
        return answers[:1] if answers else answers

    return [
        ("P68 the first covering key asserted", "1 no arbitrary subject",
         [mock.patch.object(T, "_ambiguous_cover", lambda *args, **kwargs: None)]),
        ("P69 parts answered without the shared subject", "2 two attributes of one subject",
         [mock.patch.object(T, "_shared_subject", lambda *args, **kwargs: None)]),
        ("P70 only the first named subject answered", "3 one attribute of two subjects",
         [mock.patch.object(SM.SystematicMemory, "catalogue_answers", first_only)]),
    ]


@dataclass
class SharedSubjectReport:
    """Whether questions keep to the subjects they name.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The batteries' counts.
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


def run_gate() -> SharedSubjectReport:
    report = SharedSubjectReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
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
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 3 cases; 3 of 3 plants caught"
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
