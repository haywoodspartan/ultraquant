"""Answers that come back. The gate.

§11.158 asked its own reverse questions ("Which elements has atomic number
39?") and FAILED: Command-R answered "potassium" 4 of 5 times, and the
single-source claim revised potassium's held atomic number (19 -> 39). The
next round found a gap at 19, asked it, and revised potassium back. Yttrium
stayed missing, and ``used_up`` was declared after every round because it
was judged on the frontier just asked.

Here a reverse answer is believed only when it comes back: the subject's
forward value - from the index when the subject holds the attribute, else
from the same source, asked in the library's own form - must equal the
asked value. What does not come back is queued for the next source, and the
gap stays open for it. Used up is judged on the frontier regenerated after
filing.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 4641589c...) before any code. The worlds are copies of the user's
library after §11.155's filing with the user's corrections: A replays
§11.158's recorded answers; B is a fresh run against Command-R; C is A with
39 answered "Zirconium" 5 of 5. Questions no recording holds are asked of
Command-R once and recorded; every case is scored on replay.
1. **Nothing held is revised by one source's answer**: in A and B every fact
   held before the study holds the same value after it; in A, "potassium"
   for 39 is queued with the forward value 19 from the index, not filed.
2. **An unheld answer must come back**: in C, zirconium is filed for 40 and
   not for 39, and "Zirconium" for 39 is queued with the source's forward
   value 40; no question text is in the unit's code.
3. **Precise and enough**: in B, every filed reverse answer names the right
   element (reference periodic table, exam-only), at least 4 of the 5 gaps
   are filed, and the new members' symbols are at least 0.95 right.
4. **The loop closes on its own signal**: in B, the source is not used up
   after round 1; rounds continue until it is; then nothing in the
   regenerated frontier is unasked of it, and one more round asks nothing.
5. **The rest waits for the next source**: in A, once Command-R is used up,
   a source never asked is offered the reverse question for 39, and the
   verification queue holds ("atomic number of potassium", "39").
6. **The exam can fail**: P82 (every decided reverse answer comes back)
   breaches 1; P83 (an unheld subject comes back without being asked)
   breaches 2; P84 (a source declared used up once it has answered a round)
   breaches 4; P85 (a gap asked of any source is closed for all) breaches 5.
7. **Nothing regresses**: a sweep of the change matches the §11.151 ledger.
   Checked outside this module.
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

__all__ = ["RoundTripReport", "run_gate"]

HERE = Path(__file__).with_name("records")
SHAPE_ROUNDS = [HERE / f"distill_11158_round{n}.jsonl" for n in (1, 2, 3)]
SHAPE_KINDS = HERE / "distill_11158_kinds.json"
LIVE_B = HERE / "distill_11159_b.jsonl"     # every sample world B heard
LIVE_AC = HERE / "distill_11159_ac.jsonl"   # samples A and C needed beyond §11.158's
REFERENCE = Path(__file__).with_name("data") / "elements.json"
SOURCE = "c4ai-command-r-08-2024"
RUN_ID = "11159"
NEXT = "a source never asked"
MAX_ROUNDS = 6

_MODE = {"live": False}     # the live pass lets teachers ask what no recording holds
_RUN: dict = {}


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


def _load(path: Path) -> dict:
    recorded: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                recorded.setdefault(row["question"], []).append(row["raw"])
    return recorded


def _shape_recorded() -> dict:
    recorded: dict = {}
    for path in SHAPE_ROUNDS:
        for question, raws in _load(path).items():
            recorded.setdefault(question, []).extend(raws)
    if SHAPE_KINDS.exists():
        recorded.update(json.loads(SHAPE_KINDS.read_text(encoding="utf-8")))
    return recorded


def _reverse_question(number: int) -> str:
    """§11.158's own reverse question for a number, as it was asked."""
    for line in SHAPE_ROUNDS[0].read_text(encoding="utf-8").splitlines():
        question = json.loads(line)["question"]
        if re.search(rf"\b{number}\b", question):
            return question
    raise KeyError(number)


def _gguf():
    from ultraquant.experiments.ownquestions_gate import _gguf as gguf
    return gguf()


class _Teacher:
    """Answers from recordings; asks the live source only what none holds, and keeps it."""

    def __init__(self, recorded: dict, sink: Path):
        from ultraquant.distill.teachers import TeacherSpec
        self.spec = TeacherSpec(SOURCE, _gguf())
        self.recorded, self.sink, self._live = recorded, sink, None

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        missing = [q for q in dict.fromkeys(questions) if q not in self.recorded]
        if missing:
            if not _MODE["live"]:
                raise KeyError(f"not recorded: {missing}")
            if self._live is None:
                from ultraquant.distill import sources
                self._live = sources.LMStudioTeacher(SOURCE, _gguf())
            replies = self._live.ask(missing, system=system, samples=samples,
                                     temperature=temperature, top_p=top_p,
                                     max_tokens=max_tokens, seeds=seeds)
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as handle:
                for question, row in zip(missing, replies):
                    self.recorded[question] = list(row)
                    for raw in row:
                        handle.write(json.dumps({"question": question, "raw": raw},
                                                ensure_ascii=False) + "\n")
        return [list(self.recorded[q])[:samples] for q in questions]


def _teacher(world: str) -> _Teacher:
    if world == "B":
        return _Teacher(_load(LIVE_B), LIVE_B)
    recorded = {**_shape_recorded(), **_load(LIVE_AC)}
    if world == "C":
        recorded[_reverse_question(39)] = ["Zirconium"] * 5
    return _Teacher(recorded, LIVE_AC)


def _world(world: str) -> dict:
    """One study on a filed copy, run until the source is used up, plus one round."""
    if world in _RUN:
        return _RUN[world]
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    from ultraquant.interpreter.stash import _claim_provenance
    teacher = _teacher(world)
    bound = _decided()[3]["wilson_lower"]
    scratch = Path(tempfile.mkdtemp(prefix="uq_rt_"))
    run = f"{RUN_ID}{world}-"
    try:
        with _live_copy() as lib:
            before = {k: (lib.record(k) or {}).get("value") for k in lib.keys()}
            ledger = sources.SourceLedger(scratch / "ledger.json")
            rounds = []
            for n in range(1, MAX_ROUNDS + 1):
                rounds.append(frontier.study_round(
                    lib.memory, lib.stash, teacher, ledger, SOURCE, confidence=bound,
                    run_id=f"{run}{n}", records_path=scratch / f"round{n}.jsonl",
                    approver=lib.approver()))
                if rounds[-1]["used_up"]:
                    break
            unasked = [t for t in frontier.pending(lib.memory, lib.stash, teacher, ledger, SOURCE)]
            extra = frontier.study_round(
                lib.memory, lib.stash, teacher, ledger, SOURCE, confidence=bound,
                run_id=f"{run}extra", records_path=scratch / "extra.jsonl",
                approver=lib.approver())
            after = {k: (lib.record(k) or {}).get("value") for k in before}
            claims = []
            for entry in lib.stash.entries():
                provenance = _claim_provenance(entry) or ("",)
                fields = entry.get("fields") or {}
                if str(provenance[0]).startswith(run) and fields.get("attribute"):
                    claims.append({"subject": fields.get("subject"), "value": str(fields.get("value")),
                                   "attribute": fields.get("attribute"), "status": entry["status"]})
            _RUN[world] = {
                "before": before, "after": after, "rounds": rounds, "extra": extra,
                "unasked": [t.question for t in unasked], "claims": claims,
                "ledger": ledger.history(SOURCE),
                "queue": [(k, str(v)) for k, v in frontier.verification_queue(lib.memory, ledger)],
                "next": [t for t in frontier.pending(lib.memory, lib.stash, teacher, ledger, NEXT)],
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN[world]


def _queued(world: dict, key: str) -> dict | None:
    for row in world["ledger"]:
        queued = row.get("queued") or {}
        if queued.get("key") == key:
            return {**queued, "promoted": row.get("promoted")}
    return None


def _unchanged(world: dict) -> bool:
    return all(world["after"].get(k) == v for k, v in world["before"].items())


def nothing_revised() -> bool:
    a, b = _world("A"), _world("B")
    potassium = _queued(a, "atomic number of potassium")
    filed = [c for c in a["claims"] if _normalize(c["subject"]) == "potassium"]
    return (_unchanged(a) and _unchanged(b) and potassium is not None
            and potassium["promoted"] is False and str(potassium.get("value")) == "39"
            and _normalize(potassium.get("forward")) == "19"
            and potassium.get("forward_from") == "index" and not filed)


def _typed_questions() -> list:
    from ultraquant.distill import frontier, sources
    typed = []
    for module in (frontier, sources):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        typed += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                  and isinstance(n.value, str) and id(n) not in docs
                  and re.search(r"\b(what|which|who|how)\b.*\?", n.value, re.I)]
    return typed


def must_come_back() -> bool:
    c = _world("C")
    zirconium = sorted(x["value"] for x in c["claims"]
                       if _normalize(x["subject"]) == "zirconium"
                       and x["attribute"] == "atomic number")
    queued = _queued(c, "atomic number of zirconium")
    return (zirconium == ["40"] and queued is not None and queued["promoted"] is False
            and str(queued.get("value")) == "39" and _normalize(queued.get("forward")) == "40"
            and queued.get("forward_from") == "source" and not _typed_questions())


def _precision() -> dict:
    b = _world("B")
    by_number, symbol = _reference()
    reverse = [c for c in b["claims"] if c["attribute"] == "atomic number"]
    symbols = [c for c in b["claims"] if c["attribute"] == "chemical symbol"]
    return {
        "reverse filed": len(reverse),
        "reverse right": sum(_normalize(c["subject"]) in by_number.get(int(c["value"]), set())
                             for c in reverse if c["value"].isdigit()),
        "gaps filed": len({c["value"] for c in reverse}),
        "symbols filed": len(symbols),
        "symbols right": sum(symbol.get(_normalize(c["subject"])) == _normalize(c["value"])
                             for c in symbols),
    }


def precise() -> bool:
    p = _precision()
    _RUN["precision"] = p
    return (p["reverse filed"] > 0 and p["reverse right"] == p["reverse filed"]
            and p["gaps filed"] >= 4 and p["symbols filed"] > 0
            and p["symbols right"] / p["symbols filed"] >= 0.95)


def loop_closes() -> bool:
    b = _world("B")
    rounds = b["rounds"]
    return (len(rounds) >= 2 and rounds[0]["used_up"] is False
            and rounds[-1]["used_up"] is True and not b["unasked"]
            and b["extra"]["asked"] == 0)


def waits_for_next() -> bool:
    a = _world("A")
    return (a["rounds"][-1]["used_up"] is True
            and any(getattr(t, "value", None) == "39" for t in a["next"])
            and ("atomic number of potassium", "39") in a["queue"])


CASES = {
    "1 nothing held is revised": nothing_revised,
    "2 an unheld answer must come back": must_come_back,
    "3 precise and enough": precise,
    "4 the loop closes on its own signal": loop_closes,
    "5 the rest waits for the next source": waits_for_next,
}


def _plants():
    from ultraquant.distill import elicit as E
    from ultraquant.distill import frontier, sources

    def asked_anywhere(self, source):
        return {row["question_id"] for rows in self._read().values() for row in rows}

    return [
        ("P82 every decided reverse answer comes back", "1 nothing held is revised",
         [mock.patch.object(frontier, "comes_back", lambda forward, value: True)]),
        ("P83 an unheld subject comes back without being asked", "2 an unheld answer must come back",
         [mock.patch.object(frontier, "ask_forward", lambda *args, **kwargs: {}),
          mock.patch.object(frontier, "comes_back",
                            lambda forward, value: forward is None
                            or E.normalize(str(forward)) == E.normalize(str(value)))]),
        ("P84 a source used up once it has answered a round", "4 the loop closes on its own signal",
         [mock.patch.object(sources, "used_up",
                            lambda ledger, source, frontier_ids, **kwargs: bool(ledger.asked(source)))]),
        ("P85 a gap asked of any source is closed for all", "5 the rest waits for the next source",
         [mock.patch.object(sources.SourceLedger, "asked", asked_anywhere)]),
    ]


@dataclass
class RoundTripReport:
    """Whether reverse answers are believed only when they come back.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        worlds: Per world: rounds, claims filed, queue, what the next source is offered.
        precision: World B's filed answers against the reference.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    worlds: dict = field(default_factory=dict)
    precision: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def _evaluate(report: RoundTripReport | None) -> None:
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
        report.precision = dict(_RUN.get("precision", {}))
        for world in ("A", "B", "C"):
            w = _RUN.get(world) or {}
            report.worlds[world] = {
                "rounds": w.get("rounds"), "extra": w.get("extra"), "claims": w.get("claims"),
                "queue": w.get("queue"), "unasked": w.get("unasked"),
                "next": [getattr(t, "question", str(t)) for t in w.get("next") or []],
                "queued": [row for row in w.get("ledger") or [] if row.get("queued")]}
    plants = _plants()
    for name, target, patches in plants:
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


def run_gate(live: bool = True) -> RoundTripReport:
    if live:
        for path in (LIVE_B, LIVE_AC):
            if path.exists():
                path.unlink()
        _MODE["live"] = True
        try:
            _RUN.clear()
            _evaluate(None)             # the live pass: record what no recording holds
        finally:
            _MODE["live"] = False
            _RUN.clear()
    report = RoundTripReport(passes=False)
    _evaluate(report)                   # scored on replay only
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = len(report.cases) == 5 and all(report.cases.values())
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
    import sys
    report = run_gate(live="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "precision": report.precision,
                      "worlds": report.worlds, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
