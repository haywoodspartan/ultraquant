"""A third source settles what two disputed. The gate.

After §11.162, 14 held claims were contested: Command-R said one thing,
Qwen another, and the held value waited for a third source. Here
cydonia-v1.3-magnum-v4-22b (Mistral Small 22B, a third lineage) checks
them: agreeing with the held value settles it there; agreeing with the
queued contrary answer gives that answer two lineages against one, and
the held value is revised to it; agreeing with neither contests it again.
A held value is never revised on one source's word.

Measured before any code: calibration 40/40; atomic masses 72 of 76
decided right. On the contests it sides with Command-R on terbium,
dubnium, cadmium, germanium, selenium and sulfur, with Qwen on osmium,
copernicium and flerovium, with neither on molybdenum (95.94), and is
undecided on holmium, bohrium, moscovium and seaborgium. Two of three
models learned the superseded atomic weights of germanium and selenium.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 6fae1b74...) before any code. The world: a copy of the user's
library after §11.155's filing with the corrections, Command-R's study and
Qwen's round replayed from their recordings, then Cydonia's rounds until
used up (asked once, recorded, scored on replay). Reference (exam-only):
standard atomic weights at the coarser precision, mass numbers exactly,
atomic numbers, symbols.
1. **Two against one, never one**: every changed held value was revised
   by a claim naming two teachers whose answers agree on the new value; no
   held value changed otherwise.
2. **Settlement beats holding**: across settled contests the reference
   scores, settled values are right strictly more often than the held
   values were.
3. **Settled contests leave the queue**: the verification queue holds no
   settled contest and still holds every unsettled one.
4. **The rest is checked**: every held single-source claim of another
   teacher has a Cydonia check, Qwen's atomic masses included.
5. **The exam can fail**: P97 (revise whenever the new source contests,
   even alone) breaches 1; P98 (settled contests stay queued) breaches 3;
   P99 (the contrary answer's second lineage ignored) breaches 2.
6. **Nothing regresses**: a sweep matches the §11.151 ledger. Checked
   outside this module.
Reported, not a criterion: settled contests where the majority is wrong by
the current reference (expected: germanium, selenium).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["ThirdReport", "run_gate"]

HERE = Path(__file__).with_name("records")
SINK = HERE / "distill_11163.jsonl"
FIRST = "c4ai-command-r-08-2024"
SECOND = "qwen/qwen3.8-27b"
THIRD = "cydonia-v1.3-magnum-v4-22b"
THIRD_GGUF = Path(r"J:\Models\knifeayumu\Cydonia-v1.3-Magnum-v4-22B-GGUF\Cydonia-v1.3-Magnum-v4-22B-Q6_K.gguf")
RUN_ID = "11163"
MAX_ROUNDS = 8

_MODE = {"live": False}
_RUN: dict = {}


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


class _Third:
    """Cydonia: this unit's recordings first; the live source only for what none holds."""

    def __init__(self):
        from ultraquant.distill.teachers import TeacherSpec
        from ultraquant.experiments import roundtrip_gate as R
        self.spec = TeacherSpec(THIRD, THIRD_GGUF)
        self.recorded, self.asked, self._live = R._load(SINK), [], None

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        self.asked += list(questions)
        missing = [q for q in dict.fromkeys(questions) if q not in self.recorded]
        if missing:
            if not _MODE["live"]:
                raise KeyError(f"not recorded: {missing[:3]}")
            if self._live is None:
                from ultraquant.distill import sources
                self._live = sources.LMStudioTeacher(THIRD, THIRD_GGUF)
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


def _bound(teacher, name) -> float:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import sources
    from ultraquant.experiments.ownquestions_gate import _decided
    pairs = _decided()[1]
    scratch = Path(tempfile.mkdtemp(prefix="uq_third_cal_"))
    try:
        records = E.elicit(teacher, name, [t for t, _v in pairs], scratch / "cal.jsonl")
        return sources.calibrate(records, pairs)["wilson_lower"]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _world() -> dict:
    if "W" in _RUN:
        return _RUN["W"]
    from ultraquant.distill import corroborate, frontier, sources
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    from ultraquant.interpreter.stash import _claim_provenance
    first, second, third = GG._Teacher(), SG._Second(), _Third()
    scratch = Path(tempfile.mkdtemp(prefix="uq_third_"))
    modes = (GG._MODE["live"], SG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = False     # the first two are replayed only
    try:
        with _live_copy() as lib:
            ledger = sources.SourceLedger(scratch / "ledger.json")
            plan = [(FIRST, first, _decided()[3]["wilson_lower"]),
                    (SECOND, second, _bound(second, SECOND))]
            for name, teacher, bound in plan:
                for n in range(1, MAX_ROUNDS + 1):
                    out = frontier.study_round(
                        lib.memory, lib.stash, teacher, ledger, name, confidence=bound,
                        run_id=f"{RUN_ID}-{name}-{n}", records_path=scratch / f"{n}.jsonl",
                        approver=lib.approver())
                    if out["used_up"]:
                        break
            before = {k: (lib.record(k) or {}).get("value") for k in lib.keys()}
            contests_before = [dict(row["queued"], source=source)
                               for source, rows in ledger._read().items() for row in rows
                               if (row.get("queued") or {}).get("contest") is not None]
            third_bound = _bound(third, THIRD)
            rounds = []
            for n in range(1, MAX_ROUNDS + 1):
                rounds.append(frontier.study_round(
                    lib.memory, lib.stash, third, ledger, THIRD, confidence=third_bound,
                    run_id=f"{RUN_ID}-third-{n}", records_path=scratch / f"third{n}.jsonl",
                    approver=lib.approver()))
                if rounds[-1]["used_up"]:
                    break
            after = {k: (lib.record(k) or {}).get("value") for k in before}
            revisions = []
            for entry in lib.stash.entries(status="promoted"):
                provenance = entry.get("provenance") or {}
                if str(provenance.get("run_id", "")).startswith(f"{RUN_ID}-third"):
                    fields = entry.get("fields") or {}
                    revisions.append({"key": fields.get("key"), "value": fields.get("value"),
                                      "teachers": list(provenance.get("teachers") or [])})
            unchecked = [c["key"] for c in corroborate.claims_to_check(lib.memory, lib.stash, THIRD)
                         if frontier.elicit.question_id(corroborate._forward_target(lib.stash, c))
                         not in ledger.asked(THIRD)]
            _RUN["W"] = {
                "before": before, "after": after, "rounds": rounds, "third bound": third_bound,
                "contests before": contests_before, "revisions": revisions,
                "checks": [row["check"] for row in ledger.history(THIRD) if row.get("check")],
                "queue": [(k, str(v)) for k, v in frontier.verification_queue(lib.memory, ledger)],
                "unchecked": unchecked,
                "qwen masses": sum(1 for e in lib.stash.entries(status="promoted")
                                   if (e.get("provenance") or {}).get("teachers", [None])[0] == SECOND),
            }
    finally:
        GG._MODE["live"], SG._MODE["live"] = modes
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN["W"]


def _right(subject, attribute, value) -> bool | None:
    """Right against the exam-only reference; mass numbers exactly; None when unscored."""
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    name_to_z, symbol, weights = SG._reference()
    z = name_to_z.get(_normalize(subject))
    if z is None or value is None:
        return None
    if _normalize(attribute) == "atomic mass" and weights[z]["mass_number"]:
        number = GG._numbers(value)
        return number is not None and number == weights[z]["weight"]
    return SG._right(subject, attribute, value, SG._reference())


def two_against_one() -> bool:
    from ultraquant.distill import corroborate
    w = _world()
    changed = [k for k, v in w["before"].items() if w["after"].get(k) != v]
    answers = {c["key"]: c.get("answer") for c in w["checks"]}
    contrary = {}
    for c in w["contests before"]:
        contrary.setdefault(c["key"], []).append((c["source"], c["contest"]))
    bad = []
    for key in changed:
        new = w["after"][key]
        claim = next((r for r in w["revisions"] if r["key"] == key
                      and corroborate.values_agree(r["value"], new)), None)
        supported = claim is not None and len(set(claim["teachers"])) == 2 and (
            answers.get(key) is not None and corroborate.values_agree(answers[key], new)
            and any(corroborate.values_agree(answer, new) for _s, answer in contrary.get(key, [])))
        if not supported:
            bad.append((key, w["before"][key], new))
    _RUN["changed"] = [(k, w["before"][k], w["after"][k]) for k in changed]
    _RUN["unsupported"] = bad
    return not bad


def _settled(w) -> list:
    keys = {c["key"] for c in w["contests before"]}
    return [c for c in w["checks"] if c["key"] in keys and c["verdict"] in ("agreed", "revised")]


def settlement_beats_holding() -> bool:
    w = _world()
    held_right = settled_right = scored = 0
    majority_wrong = []
    for check in _settled(w):
        before = _right(check["subject"], check["attribute"], w["before"].get(check["key"]))
        after = _right(check["subject"], check["attribute"], w["after"].get(check["key"]))
        if before is None or after is None:
            continue
        scored += 1
        held_right += int(before)
        settled_right += int(after)
        if not after:
            majority_wrong.append((check["subject"], check["attribute"], w["after"].get(check["key"])))
    _RUN["settlement"] = {"scored": scored, "held right": held_right, "settled right": settled_right,
                          "majority wrong": majority_wrong}
    return scored > 0 and settled_right > held_right


def settled_leave_queue() -> bool:
    w = _world()
    settled = {c["key"] for c in _settled(w)}
    unsettled = {c["key"] for c in w["contests before"]} - settled
    queued = {k for k, _v in w["queue"]}
    _RUN["queue"] = {"settled still queued": sorted(settled & queued),
                     "unsettled missing": sorted(unsettled - queued)}
    return bool(settled) and not (settled & queued) and unsettled <= queued


def rest_checked() -> bool:
    w = _world()
    return not w["unchecked"] and w["qwen masses"] > 0


CASES = {
    "1 two against one, never one": two_against_one,
    "2 settlement beats holding": settlement_beats_holding,
    "3 settled contests leave the queue": settled_leave_queue,
    "4 the rest is checked": rest_checked,
}


def _plants():
    from ultraquant.distill import corroborate

    def alone(answer, contests):
        """P97: the new source's own answer, as if a second lineage held it."""
        return {"key": None, "contest": answer, "source": "the new source alone"}

    return [
        ("P97 revise whenever the new source contests, even alone", "1 two against one, never one",
         [mock.patch.object(corroborate, "second_lineage", alone)]),
        ("P98 settled contests stay queued", "3 settled contests leave the queue",
         [mock.patch.object(corroborate, "settled", lambda ledger: set())]),
        ("P99 the contrary answer's second lineage ignored", "2 settlement beats holding",
         [mock.patch.object(corroborate, "second_lineage", lambda answer, contests: None)]),
    ]


@dataclass
class ThirdReport:
    """Whether a third source settles contests two against one, and only so.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: Changes, settlement scores, queue state, majority-wrong cases.
        rounds: Cydonia's rounds.
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


def _evaluate(report: ThirdReport | None) -> None:
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
        report.measured = {k: _RUN.get(k) for k in ("changed", "unsupported", "settlement", "queue")}
        report.measured["third bound"] = w.get("third bound")
        report.measured["verdicts"] = [(c["subject"], c["attribute"], c["value"], c.get("answer"), c["verdict"])
                                       for c in w.get("checks") or [] if c["verdict"] != "agreed"][:25]
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


def run_gate(live: bool = True) -> ThirdReport:
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
    report = ThirdReport(passes=False)
    _evaluate(report)                   # scored on replay only
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = len(report.cases) == 4 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 3 of 3 plants caught"
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
