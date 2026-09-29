"""Names that hold "and". The gate.

§11.153 answered compound questions through a shared subject, and its
ledger sweep found the price. Its split cut every " and ", including the
one inside a named subject. "Who wrote Pride and Prejudice?" answered
"none of it is held. Still missing: who wrote pride (unknown); prejudice
(unknown).", the author forms fell from 1.0 to 0.921, and five gates
flipped from PASS to FAIL. The user's library holds 8 such subjects: 7
books, and Newfoundland and Labrador. Here an " and " inside a named
subject is part of the name.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 2bf69e8c...) before any fix:
1. **Names hold together**: the three gated forms of each of the 8
   subjects' attribute name the held value (24 questions).
2. **§11.153's behaviors hold**, measured with whole-word naming:
   - 0 of 12 subject-less assertions;
   - at least 57 of 60 two-attribute answers, with 0 naming another
     subject;
   - at least 28 of 30 two-subject answers;
   - Kenya's capital in the capital-and-author question.
3. **Nothing regresses**: a sweep of the change matches the §11.151
   ledger, the timing gates excepted. Checked outside this module.
4. **The exam can fail**: P77 (splitting inside named subjects) breaches
   1, and P68 (the first covering key asserted) breaches 2.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
from dataclasses import dataclass, field
from unittest import mock

from ultraquant.experiments import sharedsubject_gate as S

__all__ = ["AndNamesReport", "run_gate"]

FORMS = {"author": ("Who wrote the novel {s}?", "Who wrote {s}?", "Who is the author of {s}?"),
         "capital": ("What is the capital of {s}?", "What is {s}'s capital?",
                     "Which city is the capital of {s}?")}


def _names(reply: str, subject: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(subject.lower())}(?![\w])",
                     reply.lower()) is not None


_MEASURED: dict = {}


def names_hold_together() -> bool:
    with S._session() as session:
        memory = session.memory
        held = []
        for key in memory.fact_keys():
            record = memory.recall_fact(key)
            if (record and record.get("subject") and record.get("attribute") in FORMS
                    and " and " in record["subject"].lower()):
                held.append((record["subject"], record["attribute"], str(record["value"])))
        right = 0
        misses = []
        for subject, attribute, value in sorted(held):
            for form in FORMS[attribute]:
                reply = S._ask(session, form.format(s=subject))
                ok = value.lower() in reply.lower() and not reply.startswith(("none", "I don't"))
                right += ok
                if not ok:
                    misses.append((form.format(s=subject), reply[:100]))
    _MEASURED["names"] = {"subjects": len(held), "right": right, "misses": misses[:6]}
    return len(held) == 8 and right == 24


def behaviors_hold() -> bool:
    with S._session() as session:
        asserted = 0
        for attribute in S.ATTRIBUTES:
            for form in S.FORMS:
                reply = S._ask(session, form.format(a=attribute))
                asserted += int("(confidence" in reply and not reply.startswith("Reading")
                                and "Nearest" not in reply)
        held = S._held(session)
        rng = random.Random(153)
        elements = S._elements(held)
        sample = rng.sample(elements, 30)
        two_right = named_other = 0
        for x in sample:
            for question in (f"What is the atomic number and chemical symbol of {x}?",
                             f"What is the chemical symbol and atomic number of {x}?"):
                reply = S._ask(session, question)
                two_right += int(held[x]["atomic number"] in reply
                                 and held[x]["chemical symbol"] in reply)
                named_other += int(any(y != x and len(y) > 3 and _names(reply, y) for y in held))
        pairs = [tuple(rng.sample(elements, 2)) for _ in range(30)]
        pair_right = sum(int(held[p]["chemical symbol"] in (r := S._ask(
            session, f"What is the chemical symbol of {p} and {q}?")) and held[q]["chemical symbol"] in r)
            for p, q in pairs)
        kenya = S._ask(session, "What are the capital and author of Kenya?")
    _MEASURED["behaviors"] = {"asserted of 12": asserted, "two right of 60": two_right,
                              "named another": named_other, "pairs right of 30": pair_right,
                              "kenya": kenya[:120]}
    return (asserted == 0 and two_right >= 57 and named_other == 0
            and pair_right >= 28 and "Nairobi" in kenya)


CASES = {
    "1 names hold together": names_hold_together,
    "2 the behaviors of 11.153 hold": behaviors_hold,
}


def _split_everywhere(memory, text):
    """P77: 11.153's split, cutting every " and "."""
    from ultraquant.interpreter.thoughts import Reason
    return Reason._compound_parts(text, raw=True)


def _plants():
    from ultraquant.interpreter import thoughts as T
    return [
        ("P77 splitting inside named subjects", "1 names hold together",
         [mock.patch.object(T, "_protected_parts", _split_everywhere)]),
        ("P68 the first covering key asserted", "2 the behaviors of 11.153 hold",
         [mock.patch.object(T, "_ambiguous_cover", lambda *args, **kwargs: None)]),
    ]


@dataclass
class AndNamesReport:
    """Whether names that hold "and" stay whole.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: What the batteries counted.
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


def run_gate() -> AndNamesReport:
    report = AndNamesReport(passes=False)
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
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 2 cases; 2 of 2 plants caught"
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
