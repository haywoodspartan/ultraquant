"""Questions from the shape of what it knows. The gate.

The user wants the system to ask its own questions, with Command-R as the
sole source until it is used up. Measured on copies of the user's library:
- **The shape once §11.155's answers are filed.** With the user's two
  corrections, the library holds 113 atomic numbers spanning 1..118, with
  exactly 21, 23, 37, 39 and 40 missing. Every element then holds both
  attributes, so the co-occurrence gaps are exhausted.
- **The shape without the corrections.** One wrong single-source value
  (bohrium = 276) stretched the range to 1..276 and buried the 5 real
  gaps under 157 false ones.
Here the system reads the gaps from the dense part of an integer range,
asks for the missing members, and queues outliers for the next source.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 bd620088...) before any code, with Amendment A (sha256 1ad66fc4...), also before any code:
1. **Its own questions**: every question comes from an index gap plus seed
   meta-forms plus the source's kind label, and no question text is in
   code.
2. **The gaps found**: the atomic-number window is 1..118, and the reverse
   questions are exactly those for 21, 23, 37, 39 and 40.
3. **Precise**: at least 4 of the 5 filed reverse answers name the right
   element, and the new members' symbols are at least 0.95 right, against
   the reference periodic table (exam-only).
4. **The loop closes**: round 2 asks the new members' symbols, and round 3
   asks nothing and the source is used up.
5. **Outliers are not believed**: with bohrium = 276, the window stays
   1..118, 276 is queued for the next source, and no question beyond 118
   is asked.
6. **The exam can fail**: P79 (the raw minimum-to-maximum range) breaches
   5, P80 (reverse answers filed with the value shifted by one) breaches 3,
   and P81 (the source never declared used up; Amendment A, sha256
   1ad66fc4..., before any code) breaches 4.
7. **Nothing regresses**: a sweep of the change matches the §11.151
   ledger. Checked outside this module.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library

__all__ = ["ShapeReport", "run_gate"]

HERE = Path(__file__).with_name("records")
ROUNDS = [HERE / f"distill_11158_round{n}.jsonl" for n in (1, 2, 3)]
LEDGER = HERE / "sources_11158.json"
KIND_RECORDS = HERE / "distill_11158_kinds.json"
REFERENCE = Path(__file__).with_name("data") / "elements.json"
SOURCE = "c4ai-command-r-08-2024"
RUN_ID = "11158"
MISSING = [21, 23, 37, 39, 40]


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


def _reference() -> tuple[dict, dict]:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    by_number = {e["number"]: {_normalize(n) for n in e["names"]} for e in data["elements"]}
    symbol = {}
    for e in data["elements"]:
        for n in e["names"]:
            symbol[_normalize(n)] = _normalize(e["symbol"])
    return by_number, symbol


CORRECTIONS = {"bohrium": "107", "curium": "96"}   # the user's testimony, 2026-09-29


def _filed_11155(lib) -> None:
    """§11.155's answers, filed as the user chose, unless the library holds them.

    The user chose "File all, dispute the 2": the 51 single-source answers
    file at the calibrated bound, then bohrium's and curium's atomic numbers
    are disputed and the user's corrections filed. While the user's GUI holds
    the live library open, the exam builds that state on its own copy.
    Written after Astra's implementation and before any run; the criteria
    already name this world ("a copy of the user's library after §11.155's
    filing, with the corrections").
    """
    from ultraquant.distill import file as FL
    from ultraquant.distill.targets import question_form
    from ultraquant.experiments import ownquestions_gate as G
    from ultraquant.interpreter.stash import _claim_provenance
    if any((_claim_provenance(e) or ("",))[0] == G.RUN_ID for e in lib.stash.entries()):
        return
    frontier, _pairs, records, calibration, _decided = G._decided()
    FL.file_distilled(lib.stash, records, frontier, calibration["wilson_lower"],
                      G.RUN_ID, min_lineages=1)
    approver = lib.approver()
    approver.approve_all()
    attribute = "atomic number"
    category, asking = question_form(lib.stash, attribute)
    claim = FL.claim_form(lib.stash, attribute) or FL._seed()["claim"]
    key_form = FL.key_form(lib.stash, attribute) or FL._seed()["key"]
    for subject, value in CORRECTIONS.items():
        slots = dict(subject=subject, attribute=attribute, value=value)
        key = key_form.format(**slots).lower()
        approver.dispute(key, f"user correction 2026-09-29: {value}")
        lib.stash.add_claim(
            f"user:correction/2026-09-29/{category}:{subject}", asking.format(subject=subject),
            claim.format(**slots), measured_confidence=0.9,
            provenance={"run_id": "user-2026-09-29", "question_id": f"{category}:{subject}",
                        "source": "the user's correction of a single-source answer"},
            fields={"key": key, **slots})
    approver.approve_all()
    lib.memory.save()
    lib.open()


@contextlib.contextmanager
def _live_copy():
    root = Path(tempfile.mkdtemp(prefix="uq_shape_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        lib = Library(root)
        _filed_11155(lib)
        yield lib
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


class _Replay:
    """A teacher answering from recorded samples: the run's own replies."""

    def __init__(self, recorded: dict, name: str = SOURCE):
        from ultraquant.distill.teachers import TeacherSpec
        from ultraquant.experiments.ownquestions_gate import _gguf
        self.spec = TeacherSpec(name, _gguf())
        self.recorded = recorded

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        return [list(self.recorded[q])[:samples] for q in questions]


def _teacher(live: bool):
    from ultraquant.distill import sources
    from ultraquant.experiments.ownquestions_gate import _gguf
    if live:
        return sources.LMStudioTeacher(SOURCE, _gguf())
    recorded = {}
    for path in ROUNDS:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                recorded.setdefault(row["question"], []).append(row["raw"])
    if KIND_RECORDS.exists():
        for question, raws in json.loads(KIND_RECORDS.read_text(encoding="utf-8")).items():
            recorded[question] = raws
    return _Replay(recorded)


class _Recording:
    """Wraps the live teacher to keep the kind question's raw samples."""

    def __init__(self, inner):
        self.inner, self.spec, self.kinds = inner, inner.spec, {}

    def ask(self, questions, **kwargs):
        replies = self.inner.ask(questions, **kwargs)
        for q, row in zip(questions, replies):
            self.kinds[q] = row
        return replies


_RUN: dict = {}


def _study(live: bool) -> dict:
    """Three rounds on a copy of the user's library; returns what happened."""
    if "study" in _RUN:
        return _RUN["study"]
    from ultraquant.distill import frontier, sources
    from ultraquant.interpreter.autoapprove import AutoApprover
    teacher = _teacher(live)
    if live:
        teacher = _Recording(teacher)
        for path in ROUNDS + [LEDGER, KIND_RECORDS]:
            if path.exists():
                path.unlink()
    rounds = []
    with _live_copy() as lib:
        before = {k: lib.record(k) for k in lib.keys()}
        gaps = frontier.sequence_gaps(lib.memory)
        ledger = sources.SourceLedger(LEDGER if live else Path(tempfile.mkdtemp()) / "ledger.json")
        from ultraquant.experiments.ownquestions_gate import _decided
        bound = _decided()[3]["wilson_lower"]
        asked_questions = []
        from ultraquant.distill import elicit as E
        for n, path in enumerate(ROUNDS, start=1):
            # The reverse questions this round asks: a gap the ledger has
            # already asked of the source is not asked again (fixed before
            # any run; the first draft also counted those).
            asked = ledger.asked(SOURCE)
            targets = [t for t in frontier.reverse_targets(lib.memory, lib.stash, teacher)
                       if E.question_id(t) not in asked]
            outcome = frontier.study_round(
                lib.memory, lib.stash, teacher, ledger, SOURCE, confidence=bound,
                run_id=f"{RUN_ID}-{n}", records_path=path if live else Path(tempfile.mkdtemp()) / "r.jsonl",
                approver=lib.approver())
            rounds.append({"round": n, **outcome})
            asked_questions += [t.question for t in targets]
        filed = {}
        for key in lib.keys():
            record = lib.record(key)
            if key not in before and record and record.get("attribute"):
                filed[key] = record
        if live and hasattr(teacher, "kinds"):
            kinds = {q: r for q, r in teacher.kinds.items() if "kind of thing" in q.lower()}
            KIND_RECORDS.write_text(json.dumps(kinds, ensure_ascii=False, indent=1), encoding="utf-8")
    _RUN["study"] = {"gaps": gaps, "rounds": rounds, "filed": filed,
                     "questions": asked_questions}
    return _RUN["study"]


def own_questions() -> bool:
    from ultraquant.distill import frontier
    study = _study(live=False)
    seeds = json.loads((Path(frontier.__file__).with_name("data") / "seed_questions.json")
                       .read_text(encoding="utf-8"))
    reverse_shape = re.escape(seeds["reverse"]).replace(r"\{kind\}", ".+").replace(
        r"\{attribute\}", ".+").replace(r"\{value\}", r"\d+")
    from_seed = all(re.fullmatch(reverse_shape, q) for q in study["questions"] if q)
    tree = ast.parse(Path(frontier.__file__).read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr)
            and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    typed = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and id(n) not in docs
             and re.search(r"\b(what|which|who|how)\b.*\?", n.value, re.I)]
    return bool(study["questions"]) and from_seed and not typed


def gaps_found() -> bool:
    study = _study(live=False)
    atomic = next((g for g in study["gaps"] if g["attribute"] == "atomic number"), None)
    asked = sorted(int(re.search(r"(\d+)", q).group(1)) for q in study["questions"])
    return (atomic is not None and tuple(atomic["window"]) == (1, 118)
            and atomic["missing"] == MISSING and asked == MISSING)


def _precision() -> dict:
    study = _study(live=False)
    by_number, symbol = _reference()
    reverse_right = reverse_total = sym_right = sym_total = 0
    for key, record in study["filed"].items():
        subject = _normalize(record["subject"])
        if record["attribute"] == "atomic number" and str(record["value"]).isdigit():
            reverse_total += 1
            reverse_right += int(subject in by_number.get(int(record["value"]), set()))
        elif record["attribute"] == "chemical symbol":
            sym_total += 1
            sym_right += int(symbol.get(subject) == _normalize(record["value"]))
    return {"reverse right": reverse_right, "reverse filed": reverse_total,
            "symbols right": sym_right, "symbols filed": sym_total}


def precise() -> bool:
    p = _precision()
    _RUN["precision"] = p
    return (p["reverse right"] >= 4 and p["symbols filed"] > 0
            and p["symbols right"] / p["symbols filed"] >= 0.95)


def loop_closes() -> bool:
    rounds = _study(live=False)["rounds"]
    return (len(rounds) == 3 and rounds[0]["asked"] == len(MISSING)
            and rounds[1]["asked"] > 0 and rounds[2]["asked"] == 0
            and rounds[2]["used_up"])


def outliers_not_believed() -> bool:
    from ultraquant.distill import frontier
    with _live_copy() as lib:
        lib.memory.remember_fact("atomic number of bohrium", "276", 0.9,
                                 subject="bohrium", attribute="atomic number")
        gaps = frontier.sequence_gaps(lib.memory)
        queue = frontier.verification_queue(lib.memory)
    atomic = next((g for g in gaps if g["attribute"] == "atomic number"), None)
    return (atomic is not None and tuple(atomic["window"]) == (1, 118)
            and all(n <= 118 for n in atomic["missing"])
            and ("atomic number of bohrium", "276") in [(k, str(v)) for k, v in queue])


CASES = {
    "1 its own questions": own_questions,
    "2 the gaps found": gaps_found,
    "3 precise": precise,
    "4 the loop closes": loop_closes,
    "5 outliers are not believed": outliers_not_believed,
}


def _plants():
    from ultraquant.distill import frontier, sources
    real_file = frontier.file_reverse

    def shifted(stash, records, targets, confidence, run_id):
        moved = [type(t)(t.category, t.subject, t.question, t.attribute, str(int(t.value) + 1))
                 for t in targets]
        return real_file(stash, records, moved, confidence, run_id)

    return [
        ("P79 the raw minimum-to-maximum range", "5 outliers are not believed",
         [mock.patch.object(frontier, "dense_window",
                            lambda values: (min(values), max(values)) if values else None)]),
        ("P80 reverse answers filed with the value shifted", "3 precise",
         [mock.patch.object(frontier, "file_reverse", shifted)]),
        ("P81 the source never declared used up", "4 the loop closes",
         [mock.patch.object(sources, "used_up", lambda *args, **kwargs: False)]),
    ]


@dataclass
class ShapeReport:
    """Whether the system asks for what the shape of its knowledge is missing.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        study: Gaps, rounds, filed facts.
        precision: Reverse and symbol answers against the reference.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    study: dict = field(default_factory=dict)
    precision: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(live: bool = True) -> ShapeReport:
    report = ShapeReport(passes=False)
    if live:
        _study(live=True)
        _RUN.clear()
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    study = _RUN.get("study", {})
    report.study = {"gaps": study.get("gaps"), "rounds": study.get("rounds"),
                    "filed": {k: (v.get("subject"), v.get("value"))
                              for k, v in (study.get("filed") or {}).items()}}
    report.precision = dict(_RUN.get("precision", {}))
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
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
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    import sys
    report = run_gate(live="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "study": report.study,
                      "precision": report.precision, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
