"""The kind, asked one member at a time. The gate.

§11.158 learned a cluster's kind by asking about three members together.
Command-R answered "Elements", 5 of 5, and the reverse question read "Which
elements has atomic number 39?". It was answered "potassium" 4 of 5, in
§11.158's run, in §11.159's, and again when re-measured. Asked "Which element
has atomic number 39?", Command-R says yttrium 5 of 5, and in the singular
every gap was answered right. So the plural label lost yttrium, and Command-R
was declared used up with a gap it can answer.

Here the kind form asks about one member ("What kind of thing is {a}?").
It is asked of three members, and the kind is the category every member's
answers share, judged by ``elicit.agree`` (which already lets "element" and
"chemical element" agree by head noun).

**The criteria, written before the run** - frozen in a pre-registration
(sha256 133841fc...) before any code. The worlds are copies of the user's
library after §11.155's filing with the corrections. Every world answers
first from §11.158's and §11.159's recorded samples; what no recording
holds is asked of Command-R once and recorded; every case is scored on
replay. B is the study as it runs; C is B with 39 answered "Zirconium" 5 of 5.
1. **Every gap answered, in the singular**: in B the learned kind agrees
   with "element", every reverse question carries it, all 5 gaps are filed
   with the right element (39 as yttrium), the new members' symbols are at
   least 0.95 right, every fact held before holds after, and nothing is
   queued for verification.
2. **The round trip still guards**: in C, zirconium is filed for 40 only,
   and the 39 claim is queued with the source's forward value 40.
3. **The loop closes on its own signal**: in B, not used up after round 1,
   used up after round 2, and one more round asks nothing.
4. **No question text** in the unit's code.
5. **The exam can fail**: P87 (the kind asked of three members together,
   §11.158's form) breaches 1; P88 (every decided reverse answer comes
   back) breaches 2; P89 (a source used up once it has answered a round)
   breaches 3.
6. **Nothing regresses**: a sweep matches the §11.151 ledger; roundtrip_gate
   is expected to become superseded by design. Checked outside this module.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["KindReport", "run_gate"]

HERE = Path(__file__).with_name("records")
SINK = HERE / "distill_11160.jsonl"
SOURCE = "c4ai-command-r-08-2024"
RUN_ID = "11160"
MISSING = [21, 23, 37, 39, 40]
MAX_ROUNDS = 6
#: §11.158's kind form, for P87 only (exam-side; the unit's code holds none).
PLURAL_KIND = "What kind of thing are {a}, {b} and {c}?"

_MODE = {"live": False}
_RUN: dict = {}


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


def _recorded() -> dict:
    """§11.158's and §11.159's samples, then this unit's own."""
    from ultraquant.experiments import roundtrip_gate as R
    recorded = R._shape_recorded()
    for path in (R.LIVE_B, R.LIVE_AC, SINK):
        for question, raws in R._load(path).items():
            recorded.setdefault(question, list(raws))
    return recorded


def _reverse_shape() -> str:
    from ultraquant.distill import frontier
    form = frontier._seed_questions()["reverse"]
    return (re.escape(form).replace(r"\{kind\}", "(?P<kind>.+)")
            .replace(r"\{attribute\}", ".+").replace(r"\{value\}", r"(?P<value>\d+)"))


class _Teacher:
    """Recordings first; the live source only for what none holds; a planted answer in C."""

    def __init__(self, world: str):
        from ultraquant.distill.teachers import TeacherSpec
        from ultraquant.experiments.ownquestions_gate import _gguf
        self.spec = TeacherSpec(SOURCE, _gguf())
        self.recorded, self.world, self.asked, self._live = _recorded(), world, [], None
        self._gguf = _gguf

    def _planted(self, question):
        if self.world != "C":
            return None
        match = re.fullmatch(_reverse_shape(), question)
        if match and match.group("value") == "39":
            return ["Zirconium"] * 5
        return None

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        self.asked += list(questions)
        missing = [q for q in dict.fromkeys(questions)
                   if self._planted(q) is None and q not in self.recorded]
        if missing:
            if not _MODE["live"]:
                raise KeyError(f"not recorded: {missing}")
            if self._live is None:
                from ultraquant.distill import sources
                self._live = sources.LMStudioTeacher(SOURCE, self._gguf())
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
        return [list(self._planted(q) or self.recorded[q])[:samples] for q in questions]


def _world(world: str) -> dict:
    if world in _RUN:
        return _RUN[world]
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    from ultraquant.interpreter.stash import _claim_provenance
    from ultraquant.memory.factshards import normalize_subject
    teacher = _Teacher(world)
    bound = _decided()[3]["wilson_lower"]
    scratch = Path(tempfile.mkdtemp(prefix="uq_kind_"))
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
            kind = lib.memory._attribute_vocabulary().get(
                normalize_subject("atomic number"), {}).get("kind")
            _RUN[world] = {
                "before": before, "after": after, "rounds": rounds, "extra": extra,
                "claims": claims, "kind": kind,
                "reverse": [q for q in teacher.asked if re.fullmatch(_reverse_shape(), q)],
                "ledger": ledger.history(SOURCE),
                "queue": [(k, str(v)) for k, v in frontier.verification_queue(lib.memory, ledger)],
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN[world]


def _reference() -> tuple[dict, dict]:
    from ultraquant.experiments import roundtrip_gate as R
    return R._reference()


def every_gap_in_the_singular() -> bool:
    from ultraquant.distill import elicit as E
    b = _world("B")
    by_number, symbol = _reference()
    kind = b["kind"] or ""
    reverse_claims = [c for c in b["claims"] if c["attribute"] == "atomic number"]
    right = {int(c["value"]) for c in reverse_claims if c["value"].isdigit()
             and _normalize(c["subject"]) in by_number.get(int(c["value"]), set())}
    symbols = [c for c in b["claims"] if c["attribute"] == "chemical symbol"]
    symbols_right = sum(symbol.get(_normalize(c["subject"])) == _normalize(c["value"])
                        for c in symbols)
    kinds_in_questions = {re.fullmatch(_reverse_shape(), q).group("kind") for q in b["reverse"]}
    _RUN["measured"] = {"kind": kind, "gaps right": sorted(right),
                        "reverse filed": len(reverse_claims),
                        "symbols": f"{symbols_right}/{len(symbols)}",
                        "kinds in questions": sorted(kinds_in_questions)}
    return (bool(kind) and E.agree(E.normalize(kind), "element")
            and kinds_in_questions == {kind}
            and sorted(right) == MISSING and len(reverse_claims) == len(MISSING)
            and symbols and symbols_right / len(symbols) >= 0.95
            and all(b["after"].get(k) == v for k, v in b["before"].items())
            and not b["queue"])


def _queued(world: dict, key: str) -> dict | None:
    for row in world["ledger"]:
        queued = row.get("queued") or {}
        if queued.get("key") == key:
            return {**queued, "promoted": row.get("promoted")}
    return None


def round_trip_guards() -> bool:
    c = _world("C")
    zirconium = sorted(x["value"] for x in c["claims"]
                       if _normalize(x["subject"]) == "zirconium" and x["attribute"] == "atomic number")
    queued = _queued(c, "atomic number of zirconium")
    return (zirconium == ["40"] and queued is not None and queued["promoted"] is False
            and str(queued.get("value")) == "39" and _normalize(queued.get("forward")) == "40"
            and queued.get("forward_from") == "source")


def loop_closes() -> bool:
    b = _world("B")
    rounds = b["rounds"]
    return (len(rounds) == 2 and rounds[0]["used_up"] is False
            and rounds[1]["used_up"] is True and b["extra"]["asked"] == 0)


def no_question_text() -> bool:
    from ultraquant.experiments import roundtrip_gate as R
    return not R._typed_questions()


CASES = {
    "1 every gap answered, in the singular": every_gap_in_the_singular,
    "2 the round trip still guards": round_trip_guards,
    "3 the loop closes on its own signal": loop_closes,
    "4 no question text": no_question_text,
}


def _plants():
    from ultraquant.distill import elicit as E
    from ultraquant.distill import frontier, sources
    from ultraquant.memory.factshards import normalize_subject

    def plural_kind(memory, attribute, teacher, *, seed=158):
        """§11.158's kind_of: three members asked together."""
        normalized = normalize_subject(attribute)
        kind = memory._attribute_vocabulary().get(normalized, {}).get("kind")
        if kind:
            return kind
        subjects = set()
        for key in memory.fact_keys():
            record = memory.recall_fact(key) or {}
            if normalize_subject(record.get("attribute") or "") == normalized and record.get("subject"):
                subjects.add(record["subject"])
        if len(subjects) < 3:
            return None
        a, b, c = random.Random(seed).sample(sorted(subjects), 3)
        replies = teacher.ask([PLURAL_KIND.format(a=a, b=b, c=c)], system=E.SYSTEM,
                              samples=E.SAMPLES, temperature=E.TEMPERATURE, top_p=E.TOP_P,
                              max_tokens=E.MAX_TOKENS, seeds=E.SEEDS)
        kind = E.held(replies[0])
        if kind is not None:
            memory.learn_kind(attribute, kind)
        return kind

    return [
        ("P87 the kind asked of three members together", "1 every gap answered, in the singular",
         [mock.patch.object(frontier, "kind_of", plural_kind)]),
        ("P88 every decided reverse answer comes back", "2 the round trip still guards",
         [mock.patch.object(frontier, "comes_back", lambda forward, value: True)]),
        ("P89 a source used up once it has answered a round", "3 the loop closes on its own signal",
         [mock.patch.object(sources, "used_up",
                            lambda ledger, source, frontier_ids, **kwargs: bool(ledger.asked(source)))]),
    ]


@dataclass
class KindReport:
    """Whether the kind is learned so that the reverse questions can be answered.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The learned kind, the gaps filed right, the symbols.
        worlds: Per world: rounds, claims, reverse questions asked, queue.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    worlds: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def _evaluate(report: KindReport | None) -> None:
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
        report.measured = dict(_RUN.get("measured", {}))
        for world in ("B", "C"):
            w = _RUN.get(world) or {}
            report.worlds[world] = {"kind": w.get("kind"), "rounds": w.get("rounds"),
                                    "extra": w.get("extra"), "reverse": w.get("reverse"),
                                    "claims": w.get("claims"), "queue": w.get("queue")}
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


def run_gate(live: bool = True) -> KindReport:
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
    report = KindReport(passes=False)
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
                      "worlds": report.worlds, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
