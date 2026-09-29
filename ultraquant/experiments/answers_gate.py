"""Answers found through the catalogue. The gate (indexing, stage 2).

§11.139 catalogued facts by subject. Chat still answered by word
overlap. Measured on the user's migrated library (a copy), through the
real chat pipeline, over 384 distilled facts in 15 ordinary forms (1,461
questions):
- 1,021 answered (70%). "Who wrote the novel X?", the very question the
  author facts were distilled from, got 0 of 89, as did "Who wrote X?".
  "Which city is the capital of X?" got 1 of 120;
- two wrong answers asserted, both one failure: "What is Australia's
  capital?" said Adelaide (South Australia), and "What is Mexico's
  capital?" said Santa Fe (New Mexico);
- every invented subject got "Nearest I hold: capital of the australian
  capital territory is Canberra";
- 165 replies carried a curiosity hint, and 132 curiosities were queued.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 bfdcff2c...) before any code. Measured on the library rebuilt
from both recorded runs; the same measure on a copy of the user's
library is reported:
1. **Answered**: each form whose words appear in the attribute's name
   or in its facts' distillation questions is answered at least 95% of
   the time (14 of the 15 forms; "X was written by whom?" is reported).
2. **No wrong assertions**: Australia's capital is Canberra and Mexico's
   is Mexico City. Baseline: 2 wrong.
3. **Unknown subjects**: the 30 invented and 20 real-but-unheld capital
   questions get no answer naming another subject's fact, and no
   curiosity.
4. **No junk**: no curiosity hints and nothing queued over the whole set.
5. **Chat facts unchanged**: the native parity gates and the curiosity
   gate stay PASS. Checked outside this module.
6. **The exam can fail**: P40 (no learning of ways of asking) and P41
   (no catalogue answer) breach 1, and P42 (the shortest subject chosen)
   breaches 2.
7. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**FAILED** criteria 1, 3 and 4, twice on Claude's machine and twice in
Astra's runs. Every plant was caught (3 of 3) and every other gate
passed. Kept, as a strict improvement:
- the "Who wrote..." forms went from 0% to 98.9%, and "Which city is
  the capital of X?" from 0.8% to 100%;
- wrong answers went from 2 to 0;
- junk hints went from 165 to 68, and unknown subjects answered with
  another subject's fact from 50 to 18.
The misses:
- "Tell me the capital of X." is classified as chat before any answering
  (57.5%);
- a refusal rule too strict for names sharing words with held subjects
  ("the US state of Kessaway");
- the 68 hints, all from the form criterion 1 exempted: an inconsistency
  in the pre-registration, recorded rather than repaired after the run.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import (LIVE_HOME, REAL_NOT_HELD,
                                                  _built)

__all__ = ["AnswersReport", "run_gate"]

#: Ordinary ways of asking, by attribute. Exam data: the system is never
#: given these; it learns only from the questions its facts came from.
FORMS = {
    "capital": ["What is the capital of {s}?", "What is {s}'s capital?",
                "Which city is the capital of {s}?",
                "Tell me the capital of {s}."],
    "chemical symbol": ["What is the chemical symbol of {s}?",
                        "What is the symbol for {s}?",
                        "What is {s}'s chemical symbol?",
                        "Which symbol does {s} have?"],
    "atomic number": ["What is the atomic number of {s}?",
                      "What is {s}'s atomic number?",
                      "Which atomic number does {s} have?"],
    "author": ["Who wrote the novel {s}?", "Who wrote {s}?",
               "Who is the author of {s}?", "{s} was written by whom?"],
}
#: Reported, not gated: its word ("written") is in no distillation question.
UNGATED = {"{s} was written by whom?"}

_REFUSALS = ("I don't hold", "I have nothing", "I can't")


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


def _session(root: Path):
    from ultraquant.interpreter.thoughts import build_session
    return build_session(root, seed=0)


def measure(root: Path) -> dict:
    """Ask every distilled fact in every form, through the real pipeline."""
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    from ultraquant.interpreter.thoughts import run_pipeline
    session = _session(root)
    memory = session.memory
    facts = [(k, memory._fact_record(k)) for k in memory.fact_keys()]
    facts = [(k, r) for k, r in facts if r and r.get("subject")]
    tally = defaultdict(Counter)
    wrong, hints = [], 0
    for key, record in facts:
        value = _normalize(record["value"])
        for form in FORMS.get(record["attribute"], []):
            question = form.format(s=record["subject"])
            reply, _trace = run_pipeline(question, session)
            answered = (not reply.startswith(_REFUSALS)
                        and value in _normalize(reply))
            tally[form]["answered" if answered else "missed"] += 1
            if not reply.startswith(_REFUSALS) and not answered:
                wrong.append((question, reply[:120]))
            if "If I knew the" in reply:
                hints += 1
    queued = len(session.curiosities)
    unknown = [i.question for i in list(K.FICTITIOUS) + list(H.FICTITIOUS)
               if i.category == "capital"]
    unknown += [f"What is the capital of {s}?" for s in REAL_NOT_HELD]
    before = len(session.curiosities)
    named_other = []
    for question in unknown:
        reply, _trace = run_pipeline(question, session)
        if "Nearest I hold" in reply or not reply.startswith(_REFUSALS):
            named_other.append((question, reply[:120]))
    return {"facts": len(facts),
            "forms": {form: dict(c) for form, c in tally.items()},
            "wrong": wrong, "hints": hints, "queued": queued,
            "unknown asked": len(unknown), "unknown named another": named_other,
            "unknown curiosities": len(session.curiosities) - before}


_CACHE: dict = {}


def _measured() -> dict:
    """The rebuilt library, measured once per context."""
    if "m" not in _CACHE:
        if "lib" not in _CACHE:
            _CACHE["lib"] = _built()
        _CACHE["m"] = measure(_CACHE["lib"].root)
    return _CACHE["m"]


def _forget(rebuild: bool) -> None:
    _CACHE.pop("m", None)
    if rebuild:
        lib = _CACHE.pop("lib", None)
        if lib is not None:
            shutil.rmtree(lib.root, ignore_errors=True)


def _rate(counts) -> float:
    total = counts.get("answered", 0) + counts.get("missed", 0)
    return counts.get("answered", 0) / total if total else 0.0


def answered() -> bool:
    m = _measured()
    gated = [f for forms in FORMS.values() for f in forms if f not in UNGATED]
    return all(_rate(m["forms"].get(form, {})) >= 0.95 for form in gated)


def no_wrong_assertions() -> bool:
    return not _measured()["wrong"]


def unknown_subjects() -> bool:
    m = _measured()
    return (m["unknown asked"] == 50 and not m["unknown named another"]
            and m["unknown curiosities"] == 0)


def no_junk() -> bool:
    m = _measured()
    return m["hints"] == 0 and m["queued"] == 0


CASES = {
    "1 answered": answered,
    "2 no wrong assertions": no_wrong_assertions,
    "3 unknown subjects": unknown_subjects,
    "4 no junk": no_junk,
}


def live_measure() -> dict | None:
    """The same measure on a copy of the user's library (reported)."""
    if not (LIVE_HOME / "stash.json").exists():
        return None
    from ultraquant.experiments.catalogue_gate import Library
    from ultraquant.memory import migrate
    root = Path(tempfile.mkdtemp(prefix="uq_answers_live_")) / "uq_home"
    try:
        shutil.copytree(LIVE_HOME, root)
        lib = Library(root)
        migrate.learn_question_forms(lib.memory, lib.stash)
        lib.memory.save()
        m = measure(root)
        return {"answered": {f: round(_rate(c), 3) for f, c in m["forms"].items()},
                "wrong": len(m["wrong"]), "hints": m["hints"],
                "queued": m["queued"],
                "unknown named another": len(m["unknown named another"])}
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def _plants():
    from ultraquant.memory import factshards as FS
    from ultraquant.memory import systematic as SM
    return [
        ("P40 no learning of ways of asking", "1 answered", True,
         [mock.patch.object(FS.FactShards, "learn_asking",
                            lambda self, attribute, subject, question: None)]),
        ("P41 no catalogue answer", "1 answered", False,
         [mock.patch.object(SM.SystematicMemory, "catalogue_answer",
                            lambda self, text: None)]),
        ("P42 the shortest subject chosen", "2 no wrong assertions", False,
         [mock.patch.object(FS.FactShards, "_choose_subject",
                            lambda self, subjects:
                            min(subjects, key=len) if subjects else None)]),
    ]


@dataclass
class AnswersReport:
    """Whether chat answers through the catalogue.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The rebuilt library's measurements.
        live: The same measure on a copy of the user's library, or None.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    live: dict | None = None
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> AnswersReport:
    report = AnswersReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        m = _measured()
        report.measured = {
            "answered": {f: round(_rate(c), 3) for f, c in m["forms"].items()},
            "wrong": m["wrong"][:10], "hints": m["hints"], "queued": m["queued"],
            "unknown named another": m["unknown named another"][:10]}
    except Exception as exc:
        report.errors["measure"] = repr(exc)
    try:
        report.live = live_measure()
    except Exception as exc:
        report.errors["live copy"] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, rebuild, patches in plants:
        _forget(rebuild)
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
            _forget(rebuild)
    _forget(True)
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = ("FAIL: "
                         f"{[n for n, ok in report.cases.items() if not ok]}")
    else:
        report.reason = "PASS: 4 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "live": report.live, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
