"""Its own questions, one source at a time. The gate.

The user, 2026-09-29: "We need it to ask its own questions", and "use
Command-R as the sole source then we do another source and then another
after 1 model is used up."

Measured before any code:
- Command-R is loaded in the user's LM Studio. A sample takes 0.3 to
  0.7 s, and LM Studio does not honour per-request seeds.
- 24 held facts asked of it with the distillation prompt were 24 of 24
  self-consistent and right.
- The distillation filter promotes only on two or more lineages, so a
  single source promoted nothing.
Here the frontier is what the catalogue proposes (§11.144's gaps), the
questions are the library's stored forms, one source answers, and its
confidence is measured on facts the library already holds.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 ba7d3c2c...) before any code, with Amendment A (sha256
fda2816b...), also before any code:
1. **Its own questions**: every frontier and calibration question is a
   stored form with its subject slotted, and no question text is in code.
2. **Calibrated**: the precision on 40 held facts and its Wilson lower
   bound are reported, and every filed claim carries that bound.
3. **Precise**: at least 0.95 of promoted answers right against the
   reference periodic table (``data/elements.json``, exam-only).
4. **Enough**: coverage (promoted over frontier) at least 0.30.
5. **Filed and catalogued**: on a copy of the user's library the answers
   file and auto-approve, the catalogue answers one, and the 405 earlier
   facts are unchanged.
6. **Used up, and said**: after the frontier, a second pass asks
   Command-R nothing and reports it used up. A source that only answers
   UNKNOWN is declared used up within 20 asks.
7. **The exam can fail**: P74 (a typed question) breaches 1, P75 (answers
   filed for the wrong target, Amendment A) breaches 3, and P76 (a source
   never used up) breaches 6.
8. **Nothing regresses**: a sweep of the change matches the §11.151
   ledger. Checked outside this module.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library

__all__ = ["OwnQuestionsReport", "run_gate"]

HERE = Path(__file__).with_name("records")
RECORDS = HERE / "distill_11155.jsonl"
CALIBRATION = HERE / "distill_11155_calibration.jsonl"
LEDGER = HERE / "sources_11155.json"
REFERENCE = Path(__file__).with_name("data") / "elements.json"
SOURCE = "c4ai-command-r-08-2024"
LINEAGE = "lineage-command-r"
RUN_ID = "11155"


def _gguf() -> Path:
    from ultraquant.experiments import distill_facts_gate as F
    return next(path for name, path in F.TEACHERS if name == "command-r-08-2024")


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


@contextlib.contextmanager
def _live_copy():
    root = Path(tempfile.mkdtemp(prefix="uq_ownq_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        yield Library(root)
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def _frontier(lib):
    from ultraquant.distill import targets
    return targets.completion_targets(lib.memory, lib.stash)


def _calibration(lib):
    from ultraquant.distill import sources
    return sources.calibration_items(lib.memory, lib.stash, k=40, seed=155)


def _reference() -> dict:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    out = {}
    for element in data["elements"]:
        for name in element["names"]:
            out[_normalize(name)] = {"chemical symbol": _normalize(element["symbol"]),
                                     "atomic number": str(element["number"])}
    return out


# -- the run: the only step that calls the source --------------------------------------

def ask_source() -> dict:
    """Calibrate, then ask the frontier, recording every sample and the ledger."""
    from ultraquant.distill import elicit as E
    from ultraquant.distill import sources
    teacher = sources.LMStudioTeacher(SOURCE, _gguf())
    with _live_copy() as lib:
        frontier, pairs = _frontier(lib), _calibration(lib)
    HERE.mkdir(parents=True, exist_ok=True)
    for path in (RECORDS, CALIBRATION, LEDGER,
                 RECORDS.with_suffix(".manifest.json"),
                 CALIBRATION.with_suffix(".manifest.json")):
        if path.exists():
            path.unlink()
    started = time.monotonic()
    E.elicit(teacher, LINEAGE, [t for t, _v in pairs], CALIBRATION)
    records = E.elicit(teacher, LINEAGE, frontier, RECORDS)
    decided = sources.single_source_decide(records, frontier)
    ledger = sources.SourceLedger(LEDGER)
    for target in frontier:
        ledger.record(SOURCE, E.question_id(target), decided.get(E.question_id(target)) is not None)
    return {"seconds": round(time.monotonic() - started, 1)}


_CACHE: dict = {}


def _decided() -> tuple:
    if "d" not in _CACHE:
        from ultraquant.distill import elicit as E
        from ultraquant.distill import sources
        with _live_copy() as lib:
            frontier, pairs = _frontier(lib), _calibration(lib)
        records = E.load_records(RECORDS)
        calibration = sources.calibrate(E.load_records(CALIBRATION), pairs)
        _CACHE["d"] = (frontier, pairs, records, calibration,
                       sources.single_source_decide(records, frontier))
    return _CACHE["d"]


# -- 1: its own questions ----------------------------------------------------------------

def _split_forms(stash) -> dict:
    forms = defaultdict(set)
    for entry in stash.entries(status="promoted"):
        fields = entry.get("fields") or {}
        subject, title = fields.get("subject"), entry.get("title") or ""
        if fields.get("attribute") and subject and subject in title:
            forms[fields["attribute"]].add(tuple(title.split(subject, 1)))
    return forms


def _typed_questions(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(node.body[0].value) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs
            and re.search(r"\b(what|which|who|how)\b.*\?", n.value, re.I)]


def own_questions() -> bool:
    from ultraquant.distill import sources, targets
    with _live_copy() as lib:
        forms = _split_forms(lib.stash)
        frontier, pairs = _frontier(lib), _calibration(lib)
    asked = list(frontier) + [t for t, _v in pairs]
    from_data = all(t.question in {b + t.subject + a for b, a in forms[t.attribute]}
                    for t in asked)
    typed = (_typed_questions(Path(sources.__file__))
             + _typed_questions(Path(targets.__file__)))
    return bool(frontier) and len(pairs) == 40 and from_data and not typed


# -- 2-4: calibrated, precise, enough -----------------------------------------------------

def _score() -> dict:
    from ultraquant.distill import elicit as E
    frontier, _pairs, _records, calibration, decided = _decided()
    reference = _reference()
    promoted = [(t, decided.get(E.question_id(t))) for t in frontier
                if decided.get(E.question_id(t))]
    right = [t for t, answer in promoted
             if reference.get(_normalize(t.subject), {}).get(t.attribute) == _normalize(answer)]
    per = defaultdict(lambda: [0, 0])
    for t in frontier:
        per[t.attribute][1] += 1
    for t, _a in promoted:
        per[t.attribute][0] += 1
    return {"frontier": len(frontier), "promoted": len(promoted), "right": len(right),
            "precision": len(right) / len(promoted) if promoted else 0.0,
            "coverage": len(promoted) / len(frontier) if frontier else 0.0,
            "calibration": calibration,
            "per attribute": {a: {"promoted": p, "asked": n} for a, (p, n) in per.items()},
            "wrong": [(t.subject, t.attribute, a) for t, a in promoted if t not in right]}


def calibrated() -> bool:
    calibration = _score()["calibration"]
    return (calibration["asked"] == 40 and calibration["promoted"] > 0
            and 0.0 < calibration["wilson_lower"] <= 1.0)


def precise() -> bool:
    return _score()["precision"] >= 0.95


def enough() -> bool:
    return _score()["coverage"] >= 0.30


# -- 5: filed and catalogued ---------------------------------------------------------------

def filed_and_catalogued() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as FL
    frontier, _pairs, records, calibration, decided = _decided()
    bound = calibration["wilson_lower"]
    with _live_copy() as lib:
        before = {key: lib.record(key) for key in lib.keys()}
        ids = FL.file_distilled(lib.stash, records, frontier, bound, RUN_ID,
                                min_lineages=1)
        filed = [e for e in lib.stash.entries() if e["id"] in set(ids)]
        confidences = all(e["measured_confidence"] == bound for e in filed)
        single = all(e["provenance"]["lineages"] == [LINEAGE] for e in filed)
        lib.approver().approve_all()
        lib.memory.save()
        lib.open()
        unchanged = all(lib.record(key) == record for key, record in before.items())
        probe = next((t for t in frontier if decided.get(E.question_id(t))), None)
        answered = False
        if probe is not None:
            answer = lib.memory.catalogue_answer(probe.question)
            answered = (answer is not None and answer.get("form") == "exact"
                        and _normalize(answer["record"]["value"])
                        == _normalize(decided[E.question_id(probe)]))
        _CACHE["filed"] = {"filed": len(filed), "confidence": bound, "probe": probe and probe.question}
        return bool(filed) and confidences and single and unchanged and answered


# -- 6: used up ----------------------------------------------------------------------------

def used_up_and_said() -> bool:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import sources
    frontier = _decided()[0]
    ids = [E.question_id(t) for t in frontier]
    ledger = sources.SourceLedger(LEDGER)
    pending = [q for q in ids if q not in ledger.asked(SOURCE)]
    command_r = (len(ledger.history(SOURCE)) == len(ids) and not pending
                 and sources.used_up(ledger, SOURCE, ids))
    scratch = Path(tempfile.mkdtemp(prefix="uq_ownq_ledger_"))
    try:
        mute = sources.SourceLedger(scratch / "ledger.json")
        declared_at = None
        for n, qid in enumerate(ids, start=1):     # a source that only says UNKNOWN
            if sources.used_up(mute, "mute", ids):
                declared_at = n - 1
                break
            mute.record("mute", qid, False)
        silent = declared_at is not None and declared_at <= 20
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return command_r and silent


CASES = {
    "1 its own questions": own_questions,
    "2 calibrated": calibrated,
    "3 precise": precise,
    "4 enough to matter": enough,
    "5 filed and catalogued": filed_and_catalogued,
    "6 used up, and said": used_up_and_said,
}


def _plants():
    from ultraquant.distill import sources, targets
    real = sources.single_source_decide

    def rotated(records, items):
        decided = real(records, items)
        keys = list(decided)
        values = [decided[k] for k in keys]
        return dict(zip(keys, values[1:] + values[:1]))

    return [
        ("P74 a question typed in code", "1 its own questions",
         [mock.patch.object(targets, "question_form",
                            lambda stash, attribute: ("x", "Tell me about {subject}?"))]),
        ("P75 answers filed for the wrong target", "3 precise",
         [mock.patch.object(sources, "single_source_decide", rotated)]),
        ("P76 a source never used up", "6 used up, and said",
         [mock.patch.object(sources, "used_up", lambda *args, **kwargs: False)]),
    ]


@dataclass
class OwnQuestionsReport:
    """Whether the system asks its own questions of one source, and stops.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        score: Precision, coverage, calibration, per attribute, wrong ones.
        filed: What filing on a copy produced.
        planted: Plant -> whether it broke its case.
        seconds: The source's wall time, when the run asked it.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    filed: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(ask: bool = True) -> OwnQuestionsReport:
    report = OwnQuestionsReport(passes=False)
    if ask:
        report.seconds = ask_source()
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
    report.filed = dict(_CACHE.get("filed", {}))
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
    summary = (f"{s.get('right')}/{s.get('promoted')} right of {s.get('frontier')} "
               f"(precision {s.get('precision', 0):.3f}, coverage {s.get('coverage', 0):.2f}; "
               f"calibration {s.get('calibration', {}).get('right')}/"
               f"{s.get('calibration', {}).get('promoted')}, bound "
               f"{s.get('calibration', {}).get('wilson_lower', 0):.3f})")
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}; {summary}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}; {summary}"
    else:
        report.reason = f"PASS: 6 cases; 3 of 3 plants caught; {summary}"
    return report


def main() -> int:
    import sys
    report = run_gate(ask="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "score": report.score, "filed": report.filed,
                      "planted": report.planted, "seconds": report.seconds,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
