"""A second source. The gate.

The user: "use Command-R as the sole source then we do another source and
then another after 1 model is used up." After §11.161 Command-R is used up
on a copy of the user's library, and its answers for a new property were
about 92% right. Here the next source - qwen/qwen3.8-27b in the user's LM
Studio - checks every single-source claim another source made, asking the
attribute's forward question. Agreement adds its lineage; a contrary answer
does not revise the held value but queues it, with both answers, for the
source after.

Measured before any code: Qwen needs the request option
``"reasoning_effort": "none"`` (it reasons away the 24-token budget
otherwise); calibration 40/40; on Command-R's 51 answers it agrees on 48
(bohrium: 276 vs 109, truth 107; curium: 61 vs 96; terbium: 65 vs 69);
on atomic masses it is 83/83 right, and where both decided, the 66 they
agree on are all right.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 d77682d7...) before any code. The world: a copy of the user's
library after §11.155's filing with the corrections, then Command-R's study
replayed from its recordings until used up; then Qwen's rounds until used
up, asked once and recorded, scored on replay. Reference (exam-only):
atomic numbers, symbols, standard atomic weights at the coarser precision.
1. **Agreement is precise**: of scored claims both sources agree on, at
   least 0.99 are right.
2. **Contests find the first source's errors**: of Command-R's scored
   claims that are wrong and that Qwen decided, at least 0.9 are contested.
3. **Right answers are rarely contested**: of scored claims where both
   answers are right, at most 0.03 are contested.
4. **Nothing is revised by one contrary source**: every fact held before
   Qwen's rounds holds the same value after them.
5. **Contests wait for the next source**: every contested claim is in the
   verification queue with its held value.
6. **Options are data**: no "reasoning_effort" string constant in the
   unit's code; Qwen's requests carry it, Command-R's do not.
7. **The exam can fail**: P93 (every claim agreed without the check)
   breaches 1; P94 (agreement by string only) breaches 3; P95 (a contrary
   answer revises the held value) breaches 4; P96 (contests not queued)
   breaches 5.
8. **Nothing regresses**: a sweep matches the §11.151 ledger. Checked
   outside this module.
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

__all__ = ["SecondReport", "run_gate"]

HERE = Path(__file__).with_name("records")
SINK = HERE / "distill_11162.jsonl"
WEIGHTS = Path(__file__).with_name("data") / "atomic_weights.json"
FIRST = "c4ai-command-r-08-2024"
SECOND = "qwen/qwen3.8-27b"
QWEN_GGUF = Path(r"J:\Models\lmstudio-community\Qwen3.8-27B-GGUF\Qwen3.8-27B-Q4_K_M.gguf")
RUN_ID = "11162"
MAX_ROUNDS = 8

_MODE = {"live": False}
_RUN: dict = {}


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


class _Second:
    """Qwen: this unit's recordings first; the live source only for what none holds."""

    def __init__(self):
        from ultraquant.distill.teachers import TeacherSpec
        from ultraquant.experiments import roundtrip_gate as R
        self.spec = TeacherSpec(SECOND, QWEN_GGUF)
        self.recorded, self.asked, self._live = R._load(SINK), [], None

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        self.asked += list(questions)
        missing = [q for q in dict.fromkeys(questions) if q not in self.recorded]
        if missing:
            if not _MODE["live"]:
                raise KeyError(f"not recorded: {missing[:3]}")
            if self._live is None:
                from ultraquant.distill import sources
                self._live = sources.LMStudioTeacher(SECOND, QWEN_GGUF)
            replies = self._live.ask(missing, system=system, samples=samples,
                                     temperature=temperature, top_p=top_p,
                                     max_tokens=max_tokens, seeds=seeds)
            SINK.parent.mkdir(parents=True, exist_ok=True)
            with SINK.open("a", encoding="utf-8") as handle:
                for question, row in zip(missing, replies):
                    self.recorded[question] = list(row)
                    for raw in row:
                        handle.write(json.dumps({"question": question, "raw": raw},
                                                ensure_ascii=False) + "\n")
        return [list(self.recorded[q])[:samples] for q in questions]


def _calibration_bound(lib, teacher) -> float:
    """The second source's own bound on §11.155's 40 held facts.

    The pairs are §11.155's own (drawn from the user's library before any
    source filed into it), not a fresh draw from the studied world, whose
    facts include the first source's claims (fixed before any run).
    """
    from ultraquant.distill import elicit as E
    from ultraquant.distill import sources
    from ultraquant.experiments.ownquestions_gate import _decided
    pairs = _decided()[1]
    scratch = Path(tempfile.mkdtemp(prefix="uq_second_cal_"))
    try:
        records = E.elicit(teacher, SECOND, [t for t, _v in pairs], scratch / "cal.jsonl")
        return sources.calibrate(records, pairs)["wilson_lower"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _world() -> dict:
    if "W" in _RUN:
        return _RUN["W"]
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    first, second = GG._Teacher(), _Second()
    bound = _decided()[3]["wilson_lower"]
    scratch = Path(tempfile.mkdtemp(prefix="uq_second_"))
    live_before = GG._MODE["live"]
    GG._MODE["live"] = False            # Command-R is replayed only: it is not loaded
    try:
        with _live_copy() as lib:
            ledger = sources.SourceLedger(scratch / "ledger.json")
            for n in range(1, MAX_ROUNDS + 1):
                out = frontier.study_round(
                    lib.memory, lib.stash, first, ledger, FIRST, confidence=bound,
                    run_id=f"{RUN_ID}-first-{n}", records_path=scratch / f"first{n}.jsonl",
                    approver=lib.approver())
                if out["used_up"]:
                    break
            before = {k: (lib.record(k) or {}).get("value") for k in lib.keys()}
            second_bound = _calibration_bound(lib, second)
            rounds = []
            for n in range(1, MAX_ROUNDS + 1):
                rounds.append(frontier.study_round(
                    lib.memory, lib.stash, second, ledger, SECOND, confidence=second_bound,
                    run_id=f"{RUN_ID}-second-{n}", records_path=scratch / f"second{n}.jsonl",
                    approver=lib.approver()))
                if rounds[-1]["used_up"]:
                    break
            after = {k: (lib.record(k) or {}).get("value") for k in before}
            _RUN["W"] = {
                "before": before, "after": after, "rounds": rounds,
                "second bound": second_bound,
                "checks": [row["check"] for row in ledger.history(SECOND) if row.get("check")],
                "queue": [(k, str(v)) for k, v in frontier.verification_queue(lib.memory, ledger)],
            }
    finally:
        GG._MODE["live"] = live_before
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN["W"]


def _reference():
    from ultraquant.experiments import roundtrip_gate as R
    by_number, symbol = R._reference()
    name_to_z = {name: z for z, names in by_number.items() for name in names}
    weights = {e["number"]: e for e in json.loads(WEIGHTS.read_text(encoding="utf-8"))["elements"]}
    return name_to_z, symbol, weights


def _right(subject, attribute, value, ref) -> bool | None:
    """Right against the exam-only reference; None when the reference does not score it."""
    from ultraquant.experiments import growth_gate as GG
    name_to_z, symbol, weights = ref
    z = name_to_z.get(_normalize(subject))
    if z is None or value is None:
        return None
    attribute = _normalize(attribute)
    if attribute == "atomic number":
        number = GG._numbers(value)
        return number is not None and number == str(z)
    if attribute == "chemical symbol":
        return symbol.get(_normalize(subject)) == _normalize(value)
    if attribute == "atomic mass" and not weights[z]["mass_number"]:
        number = GG._numbers(value)
        return number is not None and GG._agrees(number, weights[z]["weight"])
    return None


def _scored() -> list:
    w = _world()
    ref = _reference()
    out = []
    for check in w["checks"]:
        held = _right(check["subject"], check["attribute"], check["value"], ref)
        answer = _right(check["subject"], check["attribute"], check.get("answer"), ref)
        out.append({**check, "held right": held, "answer right": answer})
    return out


def agreement_precise() -> bool:
    agreed = [c for c in _scored() if c["verdict"] == "agreed" and c["held right"] is not None]
    right = sum(bool(c["held right"]) for c in agreed)
    _RUN["agreed"] = f"{right}/{len(agreed)}"
    return bool(agreed) and right / len(agreed) >= 0.99


def contests_find_errors() -> bool:
    wrong = [c for c in _scored() if c["held right"] is False and c["verdict"] in ("agreed", "contested")]
    contested = sum(c["verdict"] == "contested" for c in wrong)
    _RUN["first's errors contested"] = f"{contested}/{len(wrong)}"
    return bool(wrong) and contested / len(wrong) >= 0.9


def right_rarely_contested() -> bool:
    both = [c for c in _scored() if c["held right"] is True and c["answer right"] is True]
    contested = [c for c in both if c["verdict"] == "contested"]
    _RUN["right contested"] = f"{len(contested)}/{len(both)}"
    return bool(both) and len(contested) / len(both) <= 0.03


def nothing_revised() -> bool:
    w = _world()
    changed = [k for k, v in w["before"].items() if w["after"].get(k) != v]
    _RUN["revised"] = changed[:5]
    return not changed


def contests_wait() -> bool:
    w = _world()
    contested = [c for c in w["checks"] if c["verdict"] == "contested"]
    missing = [c["key"] for c in contested if (c["key"], str(c["value"])) not in w["queue"]]
    _RUN["contested"] = len(contested)
    return bool(contested) and not missing


def options_are_data() -> bool:
    from ultraquant.distill import frontier, sources
    modules = [sources, frontier]
    with contextlib.suppress(ImportError):
        from ultraquant.distill import corroborate
        modules.append(corroborate)
    typed = []
    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        typed += [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and n.value == "reasoning_effort"]
    from ultraquant.experiments.ownquestions_gate import _gguf
    qwen = sources.LMStudioTeacher(SECOND, QWEN_GGUF).options
    command_r = sources.LMStudioTeacher(FIRST, _gguf()).options
    return not typed and qwen.get("reasoning_effort") == "none" and not command_r


CASES = {
    "1 agreement is precise": agreement_precise,
    "2 contests find the first source's errors": contests_find_errors,
    "3 right answers are rarely contested": right_rarely_contested,
    "4 nothing is revised by one contrary source": nothing_revised,
    "5 contests wait for the next source": contests_wait,
    "6 options are data": options_are_data,
}


def _plants():
    from ultraquant.distill import corroborate
    from ultraquant.distill import elicit as E

    def revise(memory, check):
        """P95: the contrary answer replaces the held value."""
        memory.remember_fact(check["key"], check["answer"], 0.9,
                             subject=check["subject"], attribute=check["attribute"])
        return {"key": check["key"], "value": check["value"], "subject": check["subject"],
                "attribute": check["attribute"], "contest": check["answer"]}

    return [
        ("P93 every claim agreed without the check", "1 agreement is precise",
         [mock.patch.object(corroborate, "values_agree", lambda a, b: True)]),
        ("P94 agreement by string only", "3 right answers are rarely contested",
         [mock.patch.object(corroborate, "values_agree",
                            lambda a, b: E.agree(E.normalize(str(a)), E.normalize(str(b))))]),
        ("P95 a contrary answer revises the held value", "4 nothing is revised by one contrary source",
         [mock.patch.object(corroborate, "contest", revise)]),
        ("P96 contests not queued", "5 contests wait for the next source",
         [mock.patch.object(corroborate, "contest", lambda memory, check: None)]),
    ]


@dataclass
class SecondReport:
    """Whether a second source corroborates, contests and never overrules alone.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: Agreement precision, errors contested, right answers contested.
        rounds: Qwen's rounds.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def _evaluate(report: SecondReport | None) -> None:
    for name, case in CASES.items():
        try:
            outcome = bool(case())
        except Exception as exc:        # a crash is a failure
            outcome = False
            if report is not None:
                report.errors[name] = repr(exc)
        if report is not None:
            report.cases[name] = outcome
    if report is not None:
        w = _RUN.get("W") or {}
        report.measured = {k: _RUN.get(k) for k in (
            "agreed", "first's errors contested", "right contested", "revised", "contested")}
        report.measured["second bound"] = w.get("second bound")
        report.measured["contested claims"] = [
            (c["subject"], c["attribute"], c["value"], c.get("answer"))
            for c in w.get("checks") or [] if c["verdict"] == "contested"][:20]
        report.rounds = w.get("rounds") or []
    for name, target, patches in _plants():
        _RUN.clear()
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            if report is not None:
                report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            if report is not None:
                report.planted[name] = False
                report.errors[name] = repr(exc)
        finally:
            _RUN.clear()


def run_gate(live: bool = True) -> SecondReport:
    if live:
        if SINK.exists():
            SINK.unlink()
        _MODE["live"] = True
        try:
            _RUN.clear()
            _evaluate(None)             # the live pass: record what no recording holds
        finally:
            _MODE["live"] = False
            _RUN.clear()
    report = SecondReport(passes=False)
    _evaluate(report)                   # scored on replay only
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = len(report.cases) == 6 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 6 cases; 4 of 4 plants caught"
    return report


def main() -> int:
    import sys
    report = run_gate(live="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "rounds": report.rounds, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
