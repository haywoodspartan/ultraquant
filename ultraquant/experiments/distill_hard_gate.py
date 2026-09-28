"""The filter on harder facts. The hard-tier distillation gate.

§11.130 measured the distillation filter on facts all three local
teachers knew, so it never had to choose, and coverage and precision
sat at their ceiling. This tier (``knowledge_bench_hard``) keeps every
answer beyond dispute but drops fame:
- US-state, Canadian and Australian capitals;
- heavy-element atomic numbers;
- rare-element symbols;
- authors of less famous novels.
That makes 211 known answers and 40 invented subjects. Where teachers
disagree, the filter's real trade-off between coverage and precision
shows, and that is the evidence needed before a massive teacher is paid
for. The same three teachers and the same filter as §11.133 are used,
unchanged.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 444dab75...) before the tier had ever been run:

1. **Precise**: at least 0.95 of the held-out promoted answers are
   right.
2. **Enough to matter**: held-out coverage is at least 0.30.
3. **Silent about what does not exist**: at most 1 of the 40 invented
   subjects is promoted.
4. **The confidence is honest**: the calibration half's Wilson 95%
   lower bound is at most the held-out precision.
5. **The exam can fail**: P1 (abstentions as answers) and P2 (any
   single sample) must breach 3, and P3 (the least common answer) must
   breach 1.
6. **On the record**: every sample exactly once, with the manifest.

Reported and not gated: per-category numbers, the wrong promotions, the
trade-off under stricter filters, and what the negation rule costs.

**PASSED**, at no cost, in 6.1 minutes of the 4090:

| | result |
|---|---|
| held-out promoted | 101 of 105 (coverage 0.96) |
| held-out right | **101 of 101** (precision 1.000) |
| invented promoted | **1 of 40** (the limit was 1) |
| confidence (calibration Wilson 95% lower bound) | **0.964** |
| planted defects caught | 3 of 3 |

Per category, held out: capitals 34 of 35 promoted, heavy-element
atomic numbers 23 of 24, rare symbols 25 of 25, authors 19 of 21. Every
one was right.

**The one invented promotion is the benchmark's fault, again.**
"Varnadium" is one letter from vanadium, and the teachers answered "V".
§11.130's "Mordavia" had sat one letter from Mordovia. Invented names
checked by eye have now failed twice. The next tier checks them by edit
distance against real names instead.

**Stricter filters, the same records re-scored.** Holding at 4 of 5
gives coverage 0.98 at precision 1.000. Holding at 5 of 5 gives 0.92 at
1.000. Requiring all three families gives 0.95 at 1.000, and zero
invented promotions. On this tier stricter filters cost coverage and
buy nothing, except that the three-family rule also stops the one
near-name slip.

**What the negation rule cost**: all 15 of nobelium's samples said
"No", and the rule refused every one. §11.135, in progress, fixes that
for symbol questions.

**The plain reading.** These local teachers know more than this tier
asked. Even "hard" facts barely made them disagree, so the filter's
trade-off is still mostly unmeasured, and the questions that would
measure it are ones the local teachers get wrong. Those are also where
a massive teacher would earn its cost.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments import distill_facts_gate as F
from ultraquant.experiments import knowledge_bench as K
from ultraquant.experiments import knowledge_bench_hard as H

__all__ = ["HardReport", "run_gate"]

RECORDS = Path(__file__).with_name("records") / "distill_11134.jsonl"


def _items() -> list:
    return list(H.KNOWN) + list(H.FICTITIOUS)


def _key(item) -> str:
    return f"{item.category}:{item.subject}"


def _known_score(items, decided) -> dict:
    promoted = [(i, decided.get(_key(i))) for i in items
                if decided.get(_key(i))]
    right = sum(K.is_correct(i, a) for i, a in promoted)
    return {"asked": len(items), "promoted": len(promoted), "right": right,
            "coverage": len(promoted) / len(items) if items else 0.0,
            "precision": right / len(promoted) if promoted else 0.0,
            "wrong": [(i.category, i.subject, a) for i, a in promoted
                      if not K.is_correct(i, a)]}


def _score(decided) -> dict:
    calibration = _known_score(K.split(H.KNOWN, 0), decided)
    held_out = _known_score(K.split(H.KNOWN, 1), decided)
    invented = [(i.category, i.subject, decided[_key(i)])
                for i in H.FICTITIOUS if decided.get(_key(i))]
    lower = F.wilson_lower(calibration["right"], calibration["promoted"])
    return {"calibration": calibration, "held_out": held_out,
            "invented_promoted": invented, "confidence": lower,
            "c1": held_out["promoted"] > 0 and held_out["precision"] >= 0.95,
            "c2": held_out["coverage"] >= 0.30,
            "c3": len(invented) <= 1,
            "c4": lower <= held_out["precision"]}


def _decide(records, **strict):
    """decide() under the filter, or a stricter variant for the report."""
    from ultraquant.distill import elicit as E
    if not strict:
        return {k: v for k, v in E.decide(records, _items()).items()}
    min_count = strict.get("min_count", 3)
    min_lineages = strict.get("min_lineages", 2)
    original_held, original_promote = E.held, E.promote
    with mock.patch.object(E, "held",
                           lambda answers, min_count_=3:
                           original_held(answers, min_count)), \
            mock.patch.object(E, "promote",
                              lambda held, min_lineages_=2:
                              original_promote(held, min_lineages)):
        return E.decide(records, _items())


def _elicit() -> dict:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import teachers as T
    _ok, cards, lineage = F._independence()
    seconds = {}
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    for path in (RECORDS, RECORDS.with_suffix(".manifest.json")):
        if path.exists():
            path.unlink()
    for (name, gguf), card in zip(F.TEACHERS, cards):
        started = time.monotonic()
        with T.LlamaServerTeacher(T.TeacherSpec(name, gguf),
                                  server_exe=F.SERVER) as teacher:
            E.elicit(teacher, lineage[card.id], _items(), RECORDS)
        seconds[name] = time.monotonic() - started
    return seconds


@dataclass
class HardReport:
    """What the filter does when the teachers do not all know.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        score: Criteria 1-4, with both halves, the confidence, and every
            wrong or invented promotion.
        per_category: Held-out coverage and precision per category.
        tradeoff: The same records under stricter filters (reported).
        negation_cost: What the negation rule withheld (reported).
        planted: Plant -> whether it breached its criterion.
        records: Samples on file; complete: every one exactly once.
        seconds: Wall time per teacher.
        reason: Plain-language verdict.
    """

    passes: bool
    score: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    tradeoff: dict = field(default_factory=dict)
    negation_cost: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    records: int = 0
    complete: bool = False
    seconds: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(elicit: bool = True) -> HardReport:
    from ultraquant.distill import elicit as E
    report = HardReport(passes=False)
    if elicit:
        report.seconds = _elicit()
    records = E.load_records(RECORDS)
    report.records = len(records)
    expected = {(name, E.question_id(i), seed) for name, _p in F.TEACHERS
                for i in _items() for seed in E.SEEDS}
    triples = [(r.teacher, r.question_id, r.seed) for r in records]
    report.complete = (set(triples) == expected
                       and len(triples) == len(expected)
                       and RECORDS.with_suffix(".manifest.json").exists())

    decided = _decide(records)
    report.score = _score(decided)
    for category in ("capital", "number", "symbol", "author"):
        subset = [i for i in K.split(H.KNOWN, 1) if i.category == category]
        report.per_category[category] = _known_score(subset, decided)
    for label, strict in (("hold 4 of 5", {"min_count": 4}),
                          ("hold 5 of 5", {"min_count": 5}),
                          ("all 3 families", {"min_lineages": 3})):
        variant = _score(_decide(records, **strict))
        report.tradeoff[label] = {
            "held_out_coverage": variant["held_out"]["coverage"],
            "held_out_precision": variant["held_out"]["precision"],
            "invented": len(variant["invented_promoted"])}
    nobelium = [r for r in records if r.question_id == "symbol:nobelium"]
    report.negation_cost = {
        "nobelium samples saying 'No'": sum(
            E.normalize(E.extract(r.raw)) == "no" for r in nobelium),
        "promoted": bool(decided.get("symbol:nobelium"))}

    plants = (("P1 abstentions counted as answers", "c3",
               mock.patch.object(E, "is_abstention", lambda text: False)),
              ("P2 any single sample promoted", "c3",
               mock.patch.object(E, "decide", F._plant_single_sample)),
              ("P3 the least common answer promoted", "c1",
               mock.patch.object(E, "held", F._plant_least_common)))
    for name, criterion, patch in plants:
        try:
            with patch:
                planted = _decide(records)
            report.planted[name] = _score(planted)[criterion] is False
        except Exception:               # a crash proves nothing
            report.planted[name] = False

    valid = all(report.planted.values()) and len(report.planted) == 3
    met = [report.score[c] for c in ("c1", "c2", "c3", "c4")]
    report.passes = valid and all(met) and report.complete
    held = report.score["held_out"]
    summary = (f"held-out {held['right']}/{held['promoted']} right "
               f"({held['precision']:.3f}) at coverage {held['coverage']:.2f}; "
               f"invented {len(report.score['invented_promoted'])}/40; "
               f"confidence {report.score['confidence']:.3f}")
    if not valid:
        report.reason = f"VOID: {report.planted}; {summary}"
    elif report.passes:
        report.reason = f"PASS: {summary}"
    else:
        failed = [c for c in ("c1", "c2", "c3", "c4") if not report.score[c]]
        report.reason = (f"FAIL: {failed}{'' if report.complete else ' + c6'}; "
                         f"{summary}")
    return report


def main() -> int:
    import sys
    result = run_gate(elicit="--rescore" not in sys.argv)
    print(result.reason)
    print(json.dumps({"per_category": result.per_category,
                      "tradeoff": result.tradeoff,
                      "negation_cost": result.negation_cost,
                      "planted": result.planted, "seconds": result.seconds,
                      "wrong_held_out": result.score["held_out"]["wrong"],
                      "wrong_calibration": result.score["calibration"]["wrong"],
                      "invented_promoted": result.score["invented_promoted"]},
                     indent=1, default=str))
    return 0 if result.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
