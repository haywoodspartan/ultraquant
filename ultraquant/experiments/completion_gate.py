"""Distillation targets chosen by the catalogue. The gate.

Every distillation so far asked hand-typed benchmark questions, and the
user ruled that the problem is hardcoding instead of indexing. The
catalogue (§11.139) knows each fact's subject and attribute. Measured on
the user's library:
- 62 of the 75 subjects with an atomic number also have a chemical
  symbol, and 62 of the 100 with a symbol also have an atomic number;
- capitals and authors co-occur with nothing;
- 51 gaps: 13 subjects with a number but no symbol, including the
  superheavy elements named in 2016, and 38 with a symbol but no number.
Here the catalogue chooses what to ask, the questions the facts were
distilled from say how, and the same three local teachers and filter as
§11.134 answer, at no cost.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 d388ec97...) before any code or run:
1. **Targets from the catalogue**: exactly the 51 gaps, matching an
   independent recount.
2. **Questions from data**: every asked question is a stored
   distillation question with its subject replaced. No question text is
   in the unit's code.
3. **Precise**: at least 0.95 of promoted answers right against the
   reference periodic table (``data/elements.json``, exam-only).
4. **Enough to matter**: coverage at least 0.30.
5. **Filed and catalogued**: on a copy of the user's library the
   answers file with structure and auto-approve, the catalogue answers
   them, and the 405 earlier facts are unchanged.
6. **The exam can fail**: P47 (every attribute pair counted as
   co-occurring) breaches 1, P48 (a question form not from data)
   breaches 2, and P49 (the least common answer held) breaches 3.
7. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments import distill_facts_gate as F
from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library

__all__ = ["CompletionReport", "run_gate"]

RECORDS = Path(__file__).with_name("records") / "distill_11144.jsonl"
REFERENCE = Path(__file__).with_name("data") / "elements.json"
RUN_ID = "11144"


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


@contextlib.contextmanager
def _live_copy():
    root = Path(tempfile.mkdtemp(prefix="uq_completion_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        yield Library(root)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def _targets(lib):
    from ultraquant.distill import targets
    return targets.completion_targets(lib.memory, lib.stash)


def _recount(lib) -> set:
    """The exam's own count of the gaps, from records alone."""
    holders = defaultdict(set)
    for key in lib.keys():
        record = lib.record(key)
        if record and record.get("subject") and record.get("attribute"):
            holders[record["attribute"]].add(record["subject"])
    gaps = set()
    for a, has_a in holders.items():
        for b, has_b in holders.items():
            both = has_a & has_b
            if a != b and len(both) >= 5 and len(both) / len(has_a) >= 0.5:
                gaps |= {(b, subject) for subject in has_a - has_b}
    return gaps


def _stored_forms(lib) -> dict:
    """attribute -> the stored questions, split around their subject."""
    forms = defaultdict(set)
    for entry in lib.stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        subject, title = fields.get("subject"), entry.get("title") or ""
        if fields.get("attribute") and subject and subject in title:
            forms[fields["attribute"]].add(tuple(title.split(subject, 1)))
    return forms


def _reference() -> dict:
    """normalized element name -> {"chemical symbol", "atomic number"}."""
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    out = {}
    for element in data["elements"]:
        for name in element["names"]:
            out[_normalize(name)] = {"chemical symbol": _normalize(element["symbol"]),
                                     "atomic number": str(element["number"])}
    return out


# -- 1-2: what to ask, and how -----------------------------------------------------

def targets_from_catalogue() -> bool:
    with _live_copy() as lib:
        chosen = {(t.attribute, t.subject) for t in _targets(lib)}
        return len(chosen) == 51 and chosen == _recount(lib)


def questions_from_data() -> bool:
    import re
    from ultraquant.distill import targets as module
    with _live_copy() as lib:
        forms = _stored_forms(lib)
        chosen = _targets(lib)
        from_data = all(
            t.question in {before + t.subject + after
                           for before, after in forms[t.attribute]}
            for t in chosen)
    import ast
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                       ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                  and isinstance(node.body[0].value.value, str)}
    # A question typed into the code, outside documentation, would be a
    # hand-written target: strings ending in "?" that open with a question word.
    typed = [n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and id(n) not in docstrings
             and re.search(r"\b(what|which|who|how)\b.*\?", n.value, re.I)]
    return bool(chosen) and from_data and not typed


# -- the run -------------------------------------------------------------------------

def elicit_targets() -> dict:
    """Serve each local teacher once and record every sample (GPU, no cost)."""
    from ultraquant.distill import elicit as E
    from ultraquant.distill import teachers as T
    with _live_copy() as lib:
        chosen = _targets(lib)
    _ok, cards, lineage = F._independence()
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    for path in (RECORDS, RECORDS.with_suffix(".manifest.json")):
        if path.exists():
            path.unlink()
    seconds = {}
    for (name, gguf), card in zip(F.TEACHERS, cards):
        started = time.monotonic()
        with T.LlamaServerTeacher(T.TeacherSpec(name, gguf),
                                  server_exe=F.SERVER) as teacher:
            E.elicit(teacher, lineage[card.id], chosen, RECORDS)
        seconds[name] = round(time.monotonic() - started, 1)
    return seconds


_CACHE: dict = {}


def _decided() -> tuple:
    """(targets, decisions) from the recorded samples."""
    if "d" not in _CACHE:
        from ultraquant.distill import elicit as E
        with _live_copy() as lib:
            chosen = _targets(lib)
        records = E.load_records(RECORDS)
        _CACHE["d"] = (chosen, records, E.decide(records, chosen))
    return _CACHE["d"]


def _score() -> dict:
    from ultraquant.distill import elicit as E
    chosen, _records, decided = _decided()
    reference = _reference()
    promoted = [(t, decided.get(E.question_id(t))) for t in chosen
                if decided.get(E.question_id(t))]
    right = [t for t, answer in promoted
             if reference.get(_normalize(t.subject), {}).get(t.attribute)
             == _normalize(answer)]
    per = defaultdict(lambda: [0, 0])
    for t in chosen:
        per[t.attribute][1] += 1
    for t, _a in promoted:
        per[t.attribute][0] += 1
    return {"targets": len(chosen), "promoted": len(promoted),
            "right": len(right),
            "precision": len(right) / len(promoted) if promoted else 0.0,
            "coverage": len(promoted) / len(chosen) if chosen else 0.0,
            "confidence": F.wilson_lower(len(right), len(promoted)),
            "per attribute": {a: {"promoted": p, "asked": n}
                              for a, (p, n) in per.items()},
            "wrong": [(t.subject, t.attribute, answer) for t, answer in promoted
                      if t not in right]}


def precise() -> bool:
    return _score()["precision"] >= 0.95


def enough() -> bool:
    return _score()["coverage"] >= 0.30


def filed_and_catalogued() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as FL
    chosen, records, decided = _decided()
    score = _score()
    with _live_copy() as lib:
        before = {key: lib.record(key) for key in lib.keys()}
        FL.file_distilled(lib.stash, records, chosen, score["confidence"], RUN_ID)
        lib.approver().approve_all()
        lib.memory.save()
        lib.open()
        unchanged = all(lib.record(key) == record for key, record in before.items())
        promoted = [t for t in chosen if decided.get(E.question_id(t))]
        structured = all(
            (r := lib.record(key)) is not None and r.get("subject") == t.subject
            and r.get("attribute") == t.attribute
            for t in promoted
            for key in [f"{t.attribute} of {t.subject.lower()}"])
        probe = next((t for t in promoted if t.attribute == "atomic number"), None)
        answered = True
        if probe is not None:
            answer = lib.memory.catalogue_answer(
                f"What is the atomic number of {probe.subject}?")
            answered = (answer is not None and answer.get("form") == "exact"
                        and _normalize(answer["record"]["value"])
                        == _normalize(decided[E.question_id(probe)]))
        return bool(promoted) and unchanged and structured and answered


CASES = {
    "1 targets from the catalogue": targets_from_catalogue,
    "2 questions from data": questions_from_data,
    "3 precise": precise,
    "4 enough to matter": enough,
    "5 filed and catalogued": filed_and_catalogued,
}


def _plants():
    from ultraquant.distill import elicit as E
    from ultraquant.distill import targets as module
    every_pair = functools.partial(module.completion_targets,
                                   min_share=0.0, min_support=0)
    return [
        ("P47 every attribute pair counted", "1 targets from the catalogue",
         [mock.patch.object(module, "completion_targets", every_pair)]),
        ("P48 a question form not from data", "2 questions from data",
         [mock.patch.object(module, "question_form",
                            lambda stash, attribute: ("x", "Tell me about {subject}."))]),
        ("P49 the least common answer held", "3 precise",
         [mock.patch.object(E, "held", F._plant_least_common)]),
    ]


@dataclass
class CompletionReport:
    """Distillation whose targets and questions came from the index.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        score: Precision, coverage, confidence, per attribute, wrong ones.
        planted: Plant -> whether it broke its case.
        seconds: Teacher wall time, when the run elicited.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(elicit: bool = True) -> CompletionReport:
    report = CompletionReport(passes=False)
    if elicit:
        report.seconds = elicit_targets()
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.score = _score()
    except Exception as exc:
        report.errors["score"] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        _CACHE.clear()
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
            _CACHE.clear()
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    s = report.score
    summary = (f"{s.get('right')}/{s.get('promoted')} right of {s.get('targets')} "
               f"targets (precision {s.get('precision', 0):.3f}, coverage "
               f"{s.get('coverage', 0):.2f}, confidence {s.get('confidence', 0):.3f})")
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}; {summary}")
    elif not met:
        report.reason = ("FAIL: "
                         f"{[n for n, ok in report.cases.items() if not ok]}; {summary}")
    else:
        report.reason = f"PASS: 5 cases; 3 of 3 plants caught; {summary}"
    return report


def main() -> int:
    import sys
    report = run_gate(elicit="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "score": report.score,
                      "planted": report.planted, "seconds": report.seconds,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
