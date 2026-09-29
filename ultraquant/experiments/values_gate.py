"""Facts found by their value. The gate.

Measured on the user's library (a copy, b6b0597):
- A battery of reverse questions, 20 per attribute, each over a value held
  by exactly one subject, sampled with seed 154:
  - "Which element has the chemical symbol {v}?"
  - "Which element has atomic number {v}?"
  - "Which country has the capital {v}?"
  - "What did {v} write?"
- 0 of 80 were answered. 79 were refused with "I don't hold that:
  nothing it names is in my catalogue", although every value is held.
The catalogue indexed subjects, keys, attributes and derivations, but not
values.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 81bf30b1...) before any code, with Amendment A (sha256 bb1da314...,
also before any code): re-measured on current code, 59 of 80 were refused
(the 20 "write" questions now get the general "I don't hold anything" reply),
and a reply repeating the asked value no longer counts that value as a wrong
subject (Victoria is both a subject and Seychelles' capital); and with
Amendment B (sha256 4187c54e..., before any code): a held value counts only
when the question names it as a value - it names an attribute holding it,
or the value carries an informative token - because "in", "he", "am" and
every integer 1..118 are held values, and "What is in the box?" would
otherwise be answered with indium:
1. **Reverse questions answered**: at least 76 of 80 name the right
   subject, and 0 name a wrong one.
2. **Every holder named**: "What did Jane Austen write?" names every book
   the library holds with Jane Austen as author.
3. **No false refusal**: 0 of the 80 get "nothing it names is in my
   catalogue".
4. **Unknown values still refused**: "Which element has the chemical
   symbol Qx?" asserts nothing.
4b. **Common words are not values** (Amendment B): "What is in the box?",
   "Where did he go?", "Who am I?", "Is it at home?" and "Which planet is
   number 4?" name no element.
5. **Nothing regresses**: a sweep of the change matches the §11.151
   ledger. Checked outside this module.
6. **The exam can fail**: P71 (no value index consulted) breaches 1,
   P72 (only the first holder named) breaches 2, P73 (the
   unknown-subject refusal kept for named values) breaches 3, and P86
   (every held n-gram named as a value, the frozen rule; Amendment B)
   breaches 4b.
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

__all__ = ["ValuesReport", "run_gate"]

FORMS = {"chemical symbol": "Which element has the chemical symbol {v}?",
         "atomic number": "Which element has atomic number {v}?",
         "capital": "Which country has the capital {v}?",
         "author": "What did {v} write?"}


@contextlib.contextmanager
def _session():
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_values_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        yield build_session(root, seed=0)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def _ask(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(text, session)[0]


def _by_attribute(session) -> dict:
    memory = session.memory
    out = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key)
        if record and record.get("subject") and record.get("attribute"):
            out.setdefault(record["attribute"], []).append((record["subject"], str(record["value"])))
    return out


def _battery(session) -> list:
    by_attribute = _by_attribute(session)
    rng = random.Random(154)
    items = []
    for attribute, form in FORMS.items():
        counts = {}
        for _s, value in by_attribute[attribute]:
            counts[value.lower()] = counts.get(value.lower(), 0) + 1
        unique = [(s, v) for s, v in by_attribute[attribute] if counts[v.lower()] == 1]
        for subject, value in rng.sample(unique, min(20, len(unique))):
            items.append((form.format(v=value), subject, value))
    return items


_MEASURED: dict = {}


def _names(reply: str, subject: str) -> bool:
    """Whole-word naming: "actinium" is not named by "protactinium".

    (§11.153's exam counted substrings and flagged four right replies; this
    exam was corrected before its first run.)"""
    import re
    return re.search(rf"(?<![\w]){re.escape(subject.lower())}(?![\w])",
                     reply.lower()) is not None


def _run_battery() -> dict:
    with _session() as session:
        subjects = {s for pairs in _by_attribute(session).values() for s, _v in pairs}
        right = wrong = refused = 0
        for question, subject, value in _battery(session):
            reply = _ask(session, question)
            named = {s for s in subjects if len(s) > 3 and _names(reply, s)}
            # Amendment A: the asked value is not a wrong subject when repeated.
            named = {s for s in named if s.lower() != value.lower()}
            if "nothing it names is in my catalogue" in reply:
                refused += 1
            if _names(reply, subject) and not reply.startswith("I don't hold"):
                right += 1
            if named - {subject}:
                wrong += 1
    return {"right of 80": right, "wrong": wrong, "false refusals": refused}


def reverse_answered() -> bool:
    measured = _run_battery()
    _MEASURED["battery"] = measured
    return measured["right of 80"] >= 76 and measured["wrong"] == 0


def every_holder() -> bool:
    with _session() as session:
        books = sorted(s for s, v in _by_attribute(session)["author"] if v == "Jane Austen")
        reply = _ask(session, "What did Jane Austen write?")
    _MEASURED["jane austen"] = {"books": books, "reply": reply[:300]}
    return len(books) >= 2 and all(_names(reply, book) for book in books)


def no_false_refusal() -> bool:
    measured = _MEASURED.get("battery") or _run_battery()
    return measured["false refusals"] == 0


def unknown_value_refused() -> bool:
    with _session() as session:
        reply = _ask(session, "Which element has the chemical symbol Qx?")
    return "(confidence" not in reply


PROBES = ["What is in the box?", "Where did he go?", "Who am I?", "Is it at home?",
          "Which planet is number 4?"]


def common_words() -> bool:
    """Amendment B: a stopword or a bare number is not a named value."""
    with _session() as session:
        elements = {s for s, _v in _by_attribute(session)["atomic number"]}
        replies = {q: _ask(session, q) for q in PROBES}
    named = {q: sorted(e for e in elements if _names(replies[q], e)) for q in PROBES}
    _MEASURED["common words"] = {q: [replies[q][:120], named[q]] for q in PROBES}
    return not any(named.values())


CASES = {
    "1 reverse questions answered": reverse_answered,
    "2 every holder named": every_holder,
    "3 no false refusal": no_false_refusal,
    "4 unknown values still refused": unknown_value_refused,
    "4b common words are not values": common_words,
}


def _plants():
    from ultraquant.memory import systematic as SM
    real = SM.SystematicMemory.catalogue_by_value

    def first_holder(self, text):
        answer = real(self, text)
        if answer and answer.get("records"):
            answer = dict(answer, records=answer["records"][:1])
        return answer

    real_unheld = SM.SystematicMemory._unheld_subject

    def refuse_values(self, words, known):
        """P73: the committed refusal, blind to held values."""
        from ultraquant.memory.factshards import FactShards
        from ultraquant.shards.router import _informative
        others = {word for word in words - known if _informative(word)}
        if not others:
            return False
        keys = (self.shards.keys_covering(others) if self.shards is not None
                else [key for key in self._facts
                      if others <= set(FactShards.tokens(key))])
        return not any((r := self.recall_fact(k)) is not None and not r.get("subject")
                       for k in keys)

    from ultraquant.memory.factshards import normalize_subject

    def every_ngram(self, value, words):
        """P86: every held n-gram is a value, with all its attributes."""
        attributes = set()
        for key in self.value_keys(value):
            record = self.recall_fact(key)
            if record and record.get("attribute"):
                attributes.add(normalize_subject(record["attribute"]))
        return attributes or None

    return [
        ("P71 no value index consulted", "1 reverse questions answered",
         [mock.patch.object(SM.SystematicMemory, "catalogue_by_value",
                            lambda self, text: None)]),
        ("P72 only the first holder named", "2 every holder named",
         [mock.patch.object(SM.SystematicMemory, "catalogue_by_value", first_holder)]),
        ("P73 the refusal kept for named values", "3 no false refusal",
         [mock.patch.object(SM.SystematicMemory, "catalogue_by_value",
                            lambda self, text: None),
          mock.patch.object(SM.SystematicMemory, "_unheld_subject", refuse_values)]),
        ("P86 every held n-gram named as a value", "4b common words are not values",
         [mock.patch.object(SM.SystematicMemory, "value_attributes", every_ngram)]),
    ]


@dataclass
class ValuesReport:
    """Whether a held value finds its facts.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The battery's counts and the multi-holder reply.
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


def run_gate() -> ValuesReport:
    report = ValuesReport(passes=False)
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
        _MEASURED.clear()
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
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 4 of 4 plants caught"
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
