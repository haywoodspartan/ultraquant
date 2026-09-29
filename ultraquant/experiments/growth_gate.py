"""Properties it does not hold yet. The gate.

After §11.160, on a copy of the user's library, Command-R was used up: every
co-occurrence and dense-window gap closed, nothing queued. To go on asking
its own questions the system proposes properties it does not hold - "Name
one measurable property that every {kind} has." - and adopts only those the
source can answer: the seed forward form, asked of five members of the
cluster, must be decided for at least four. An adopted property is then
asked of every member of its cluster.

Measured before any code (Command-R, 5 samples): element -> "atomic mass"
(4/5, twice), country -> "population" (4/5), novel -> "word count" (5/5);
atomic mass was decided for 5 of 5 elements, population for 0 of 5
countries, word count for 0 of 5 novels.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 b7839884...) before any code, with Amendment A (sha256 744322e7...,
after Astra's first implementation, before any run): "one more round asks
nothing" is read from the teacher's own log, because the round's count left
out the kind questions a growth step asks. The world is a copy of the user's
library after §11.155's filing with the corrections; the study runs until
the source is used up, then one more round. Recordings of §11.158-§11.160
answer first; what none holds is asked of Command-R once and recorded;
every case is scored on replay.
1. **Its own new questions**: every property question carries a learned
   kind in the seed form; every adoption and growth question is the seed
   forward form or the property's learned form; no question text in code.
2. **Only what it can answer is adopted**: every adopted property had at
   least 4 of 5 probes decided, every refused one fewer.
3. **Precise and broad**: filed atomic masses equal the reference at the
   coarser precision for at least 0.95 of the elements with a standard
   atomic weight; at least 0.8 of the 118 elements hold it after the
   study; every fact held before holds after.
4. **Growth stays in its cluster**: every growth question names a subject
   holding the naming attribute.
5. **The loop closes on its own signal**: not used up after the round that
   adopts; used up after the growth round; nothing pending then, one more
   round asks nothing, and no refused property is asked again.
6. **The exam can fail**: P90 (every proposal adopted without the probe)
   breaches 2; P91 (growth asked of every catalogued subject) breaches 4;
   P92 (a source used up once it has answered a round) breaches 5.
7. **Nothing regresses**: a sweep matches the §11.151 ledger; kind_gate is
   expected to become superseded by design. Checked outside this module.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from unittest import mock

__all__ = ["GrowthReport", "run_gate"]

HERE = Path(__file__).with_name("records")
SINK = HERE / "distill_11161.jsonl"
WEIGHTS = Path(__file__).with_name("data") / "atomic_weights.json"
SOURCE = "c4ai-command-r-08-2024"
RUN_ID = "11161"
MAX_ROUNDS = 8
SCORED = "atomic mass"

_MODE = {"live": False}
_RUN: dict = {}


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


def _recorded() -> dict:
    from ultraquant.experiments import kind_gate as KG
    from ultraquant.experiments import roundtrip_gate as R
    recorded = KG._recorded()
    for question, raws in R._load(SINK).items():
        recorded.setdefault(question, list(raws))
    return recorded


def _shape(form: str, **slots) -> str:
    """A seed form as a regex; each named slot becomes a named group."""
    pattern = re.escape(form)
    for name in re.findall(r"\{(\w+)\}", form):
        pattern = pattern.replace(re.escape("{" + name + "}"), slots.get(name, f"(?P<{name}>.+?)"), 1)
    return pattern


class _Teacher:
    """Recordings first; the live source only for what none holds."""

    def __init__(self):
        from ultraquant.distill.teachers import TeacherSpec
        from ultraquant.experiments.ownquestions_gate import _gguf
        self.spec = TeacherSpec(SOURCE, _gguf())
        self.recorded, self.asked, self._live, self._gguf = _recorded(), [], None, _gguf

    def ask(self, questions, *, system, samples, temperature, top_p, max_tokens, seeds):
        self.asked += list(questions)
        missing = [q for q in dict.fromkeys(questions) if q not in self.recorded]
        if missing:
            if not _MODE["live"]:
                raise KeyError(f"not recorded: {missing[:3]} (+{len(missing) - 3})"
                               if len(missing) > 3 else f"not recorded: {missing}")
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
        return [list(self.recorded[q])[:samples] for q in questions]


def _world() -> dict:
    if "B" in _RUN:
        return _RUN["B"]
    from ultraquant.distill import frontier, sources, targets
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    from ultraquant.interpreter.stash import _claim_provenance
    from ultraquant.memory.factshards import normalize_subject
    teacher = _Teacher()
    bound = _decided()[3]["wilson_lower"]
    scratch = Path(tempfile.mkdtemp(prefix="uq_growth_"))
    run = f"{RUN_ID}-"
    try:
        with _live_copy() as lib:
            before = {k: (lib.record(k) or {}).get("value") for k in lib.keys()}
            ledger = sources.SourceLedger(scratch / "ledger.json")
            rounds, marks = [], []
            for n in range(1, MAX_ROUNDS + 1):
                marks.append(len(teacher.asked))
                rounds.append(frontier.study_round(
                    lib.memory, lib.stash, teacher, ledger, SOURCE, confidence=bound,
                    run_id=f"{run}{n}", records_path=scratch / f"round{n}.jsonl",
                    approver=lib.approver()))
                if rounds[-1]["used_up"]:
                    break
            pending = frontier.pending(lib.memory, lib.stash, teacher, ledger, SOURCE)
            marks.append(len(teacher.asked))
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
                                   "attribute": fields.get("attribute")})
            vocabulary = lib.memory._attribute_vocabulary()
            holders: dict = {}
            for key in lib.keys():
                record = lib.record(key) or {}
                if record.get("subject") and record.get("attribute"):
                    holders.setdefault(normalize_subject(record["attribute"]), set()).add(
                        normalize_subject(record["subject"]))
            learned = {}
            for attribute in holders:
                form = targets.question_form(lib.stash, vocabulary.get(attribute, {}).get("name", attribute))
                if form is not None:
                    learned[attribute] = form[1]
            _RUN["B"] = {
                "before": before, "after": after, "rounds": rounds, "extra": extra,
                "pending": [t.question for t in pending], "claims": claims,
                "asked": list(teacher.asked), "marks": marks,
                "vocabulary": json.loads(json.dumps(vocabulary, default=list)),
                "holders": {a: sorted(s) for a, s in holders.items()}, "learned": learned,
                "samples": {q: teacher.recorded.get(q) for q in dict.fromkeys(teacher.asked)},
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN["B"]


def _verdicts(world) -> dict:
    """property -> (verdict, naming attribute), read from the attribute index."""
    out = {}
    for attribute, item in world["vocabulary"].items():
        for prop, verdict in (item.get("properties") or {}).items():
            out[prop] = (verdict, attribute)
    return out


def _forward_questions(world, prop) -> list:
    from ultraquant.distill import frontier
    seeds = frontier._seed_questions()
    forms = [seeds["forward"].replace("{attribute}", prop)]
    learned = world["learned"].get(prop)
    if learned:
        forms.append(learned)
    shapes = [_shape(form) for form in dict.fromkeys(forms)]
    out = []
    for q in dict.fromkeys(world["asked"]):
        # Once filed, the learned form can equal the seed form: list each question once.
        match = next((m for shape in shapes if (m := re.fullmatch(shape, q))), None)
        if match:
            out.append((q, match.group("subject")))
    return out


def own_new_questions() -> bool:
    from ultraquant.distill import frontier
    from ultraquant.experiments import roundtrip_gate as R
    w = _world()
    seeds = frontier._seed_questions()
    kinds = {item.get("kind") for item in w["vocabulary"].values() if item.get("kind")}
    allowed = [_shape(seeds["kind"]), _shape(seeds["reverse"]), _shape(seeds["forward"])]
    allowed += [_shape(form) for form in w["learned"].values()]
    property_shape = _shape(seeds["property"])
    ok = True
    strays = []
    for question in dict.fromkeys(w["asked"]):
        match = re.fullmatch(property_shape, question)
        if match:
            ok &= match.group("kind") in kinds
            continue
        if not any(re.fullmatch(shape, question) for shape in allowed):
            strays.append(question)
    _RUN["strays"] = strays[:5]
    return ok and not strays and not R._typed_questions() and any(
        re.fullmatch(property_shape, q) for q in w["asked"])


def _decided(samples) -> bool:
    """Judged on the first five samples - the ones the teacher replays."""
    from ultraquant.distill import elicit as E
    return samples is not None and E.held(list(samples)[:5]) is not None


def only_answerable() -> bool:
    w = _world()
    verdicts = _verdicts(w)
    report = {}
    ok = bool(verdicts)
    for prop, (verdict, _attribute) in verdicts.items():
        probes = [q for q, _s in _forward_questions(w, prop)][:5]
        decided = sum(_decided(w["samples"].get(q)) for q in probes)
        report[prop] = {"verdict": verdict, "probes": len(probes), "decided": decided}
        ok &= len(probes) == 5 and (verdict == "adopted") == (decided >= 4)
    _RUN["verdicts"] = report
    return ok


def _numbers(text):
    match = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return match.group(0) if match else None


def _agrees(filed: str, reference: str) -> bool:
    """Equal at the coarser of the two precisions (12 for 12.011)."""
    places = min(len(x.split(".")[1]) if "." in x else 0 for x in (filed, reference))
    quantum = Decimal(1).scaleb(-places)
    return (Decimal(filed).quantize(quantum, ROUND_HALF_UP)
            == Decimal(reference).quantize(quantum, ROUND_HALF_UP))


def precise_and_broad() -> bool:
    w = _world()
    from ultraquant.experiments import roundtrip_gate as R
    by_number, _symbol = R._reference()
    name_to_z = {name: z for z, names in by_number.items() for name in names}
    weights = {e["number"]: e for e in json.loads(WEIGHTS.read_text(encoding="utf-8"))["elements"]}
    scored = right = 0
    wrong = []
    for claim in w["claims"]:
        if _normalize(claim["attribute"]) != SCORED:
            continue
        z = name_to_z.get(_normalize(claim["subject"]))
        if z is None or weights[z]["mass_number"]:
            continue
        scored += 1
        number = _numbers(claim["value"])
        if number is not None and _agrees(number, weights[z]["weight"]):
            right += 1
        else:
            wrong.append((claim["subject"], claim["value"], weights[z]["weight"]))
    holding = len(w["holders"].get(SCORED, ()))
    unchanged = all(w["after"].get(k) == v for k, v in w["before"].items())
    _RUN["precision"] = {"scored": scored, "right": right, "wrong": wrong[:10],
                         "elements holding": holding, "unchanged": unchanged}
    return (scored > 0 and right / scored >= 0.95 and holding >= 0.8 * 118 and unchanged)


def growth_in_cluster() -> bool:
    w = _world()
    from ultraquant.memory.factshards import normalize_subject
    ok = False
    outside = []
    for prop, (verdict, attribute) in _verdicts(w).items():
        if verdict != "adopted":
            continue
        members = set(w["holders"].get(attribute, ()))
        asked = _forward_questions(w, prop)
        ok = ok or bool(asked)
        outside += [q for q, subject in asked if normalize_subject(subject) not in members]
    _RUN["outside"] = outside[:5]
    return ok and not outside


def loop_closes() -> bool:
    w = _world()
    rounds = w["rounds"]
    adopting = [i for i, r in enumerate(rounds) if r.get("adopted")]
    refused = [p for p, (v, _a) in _verdicts(w).items() if v != "adopted"]
    asked_again = [p for p in refused if len(_forward_questions(w, p)) > 5]
    # Amendment A: what the teacher actually heard during the extra round.
    extra_heard = len(w["asked"]) - w["marks"][-1]
    _RUN["extra heard"] = extra_heard
    return (len(adopting) == 1 and rounds[adopting[0]]["used_up"] is False
            and len(rounds) > adopting[0] + 1 and rounds[-1]["used_up"] is True
            and not w["pending"] and w["extra"]["asked"] == 0 and extra_heard == 0
            and not asked_again)


CASES = {
    "1 its own new questions": own_new_questions,
    "2 only what it can answer is adopted": only_answerable,
    "3 precise and broad": precise_and_broad,
    "4 growth stays in its cluster": growth_in_cluster,
    "5 the loop closes on its own signal": loop_closes,
}


def _plants():
    from ultraquant.distill import frontier, sources

    def everyone(memory, attribute):
        """P91: every catalogued subject, whatever it holds."""
        from ultraquant.memory.factshards import normalize_subject
        out = set()
        for key in memory.fact_keys():
            record = memory.recall_fact(key) or {}
            if record.get("subject") and record.get("attribute"):
                out.add(record["subject"])
        return out

    return [
        ("P90 every proposal adopted without the probe", "2 only what it can answer is adopted",
         [mock.patch.object(frontier, "adopts", lambda decided, asked: True)]),
        ("P91 growth asked of every catalogued subject", "4 growth stays in its cluster",
         [mock.patch.object(frontier, "cluster", everyone)]),
        ("P92 a source used up once it has answered a round", "5 the loop closes on its own signal",
         [mock.patch.object(sources, "used_up",
                            lambda ledger, source, frontier_ids, **kwargs: bool(ledger.asked(source)))]),
    ]


@dataclass
class GrowthReport:
    """Whether new properties are proposed, adopted only when answerable, and grown.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: Verdicts, precision, strays, growth outside the cluster.
        rounds: The study's rounds.
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


def _evaluate(report: GrowthReport | None) -> None:
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
        report.measured = {k: _RUN.get(k) for k in ("verdicts", "precision", "strays", "outside",
                                                    "extra heard")}
        report.rounds = (_RUN.get("B") or {}).get("rounds") or []
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


def run_gate(live: bool = True) -> GrowthReport:
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
    report = GrowthReport(passes=False)
    _evaluate(report)                   # scored on replay only
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = len(report.cases) == 5 and all(report.cases.values())
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
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "rounds": report.rounds, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
