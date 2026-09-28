"""Facts distilled from local teachers, and what they are worth. The gate.

The user wants "distilling information out of a massive model" into
the library. Rented GPUs come later, on demand, behind §11.127-§11.129's
runner. This unit proves the method at no cost first: three local
teachers from three model families, one at a time on the RTX 4090. Each
teacher's server is started when needed and stopped afterwards. GPT-6
Astra wrote the teachers, the elicitation and the filter
(``ultraquant/distill``); Claude wrote this exam and its benchmark
(``knowledge_bench.py``).

**What distillation may claim here.** The stash never counts a model as
a corroborating source, because nothing can establish what a model was
trained on, and this unit does not change that. It measures something
else instead: how often an answer that passes a fixed filter is right,
on questions whose answers are known. That measured rate, and nothing
more, is the confidence a distilled fact may carry.

**The filter, fixed before any run** - frozen in a pre-registration
(sha256 ad7098fd...) before any distillation code existed:
- 5 samples per teacher per question at temperature 0.7.
- A teacher *holds* an answer when 3 of its 5 samples give it, and
  abstentions are never positions.
- Two held answers agree when equal, or when the shorter one's words
  end the longer one ("tolstoy", "leo tolstoy").
- An answer is *promoted* when at least 2 independent lineages hold
  agreeing answers and none holds a different one.
- The filter never sees the benchmark's answers.

**The criteria, written before the run:**

1. **Precise on the held-out half**: coverage >= 0.40 (36 of 90) and
   precision >= 0.95.
2. **Silent about what does not exist**: at most 2 of the 80 invented
   subjects promoted.
3. **The confidence is honest**: the calibration half's Wilson 95% lower
   bound is at most the held-out precision.
4. **Three independent voices**: three groups under
   ``llmls.independent_groups``, from the GGUFs' own metadata.
5. **The exam can fail**: P1 (abstentions counted as answers) and P2
   (any single sample promoted) must breach 2, and P3 (the least common
   answer promoted) must breach 1. Each re-scores the same recorded
   samples, or the gate is VOID.
6. **Everything is on the record**: every raw sample is saved, and the
   verdict recomputes from the file alone.
7. **The full suite is green.** Checked outside this module.

Amendment A was recorded before any elicitation:
- Criterion 6 counts DISTINCT samples, as Astra asked.
- ``held`` judges whole replies. Claude's review found it saw only the
  first line, where a refusal on a later line would have become a
  position.

**PASSED**, at no cost, in 6.3 minutes of the 4090's time (Command-R
69 s, Qwen3.8-27B 242 s, Cydonia 69 s, each server started and stopped
in turn):

| | result |
|---|---|
| held-out known answers promoted | **90 of 90** (coverage 1.00) |
| held-out promoted answers right | **90 of 90** (precision 1.000) |
| invented subjects promoted | **0 of 80** |
| confidence (calibration half, Wilson 95% lower bound) | **0.959** |
| planted defects caught | 3 of 3 |
| raw samples on record | 3,900, every one distinct |

The teachers abstained on 88-94% of their samples about invented
subjects. The other 99 were hallucinations, and no two families held
the same one. Qwen answered "Saransk" for "Mordavia" four times.
Saransk is the capital of Mordovia, a real Russian republic, so the
invented name sat one letter away from a real one, and the answer was a
reasonable reading of a typo. The benchmark's next tier checks its
inventions against real names.

**What this does not show, said plainly.** The known answers were
chosen to be beyond dispute, and that made them easy. All three teachers
knew all of them, so coverage and precision are both at their ceiling.
The 0.959 is limited by 90 samples, not by any error. How the filter
trades coverage for precision on obscure facts, where teachers really
disagree, is untested. That is the next measurement, and the one that
matters before a massive teacher is paid for.

**§11.133: re-scored after the fourth review** (a post-hoc re-scoring,
stated as such). Astra found four ways the filter or the scorer could
be fooled, and Claude reproduced each with synthetic samples:
- "Ag, not Au" agreed with "Au" by the suffix rule, and was scored
  right;
- a second teacher from a dissenting family erased that family's
  dissent;
- "N/A" counted as an answer;
- the replay needed the model files.
None of those shapes occurs anywhere in the 3,900 recorded samples.
Under the corrected filter the verdict is unchanged: held-out 90/90
right at coverage 1.00, invented 0/80, confidence 0.959. All seven
plants are caught, including P4-P7, which restore each old behaviour in
every layer it lived in. The replay now runs from the records and a
manifest alone.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments import knowledge_bench as K

__all__ = ["TEACHERS", "DistillReport", "run_gate", "wilson_lower"]

SERVER = Path(r"C:\Users\Stephen Hawking\.lmstudio\extensions\backends"
              r"\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.46.0"
              r"\llama-server.exe")
TEACHERS = (
    ("command-r-08-2024", Path(r"J:\Models\bartowski\c4ai-command-r-08-2024-GGUF"
                               r"\c4ai-command-r-08-2024-Q4_K_S.gguf")),
    ("qwen3.8-27b", Path(r"J:\Models\lmstudio-community\Qwen3.8-27B-GGUF"
                         r"\Qwen3.8-27B-Q4_K_M.gguf")),
    ("cydonia-22b", Path(r"J:\Models\knifeayumu\Cydonia-v1.3-Magnum-v4-22B-GGUF"
                         r"\Cydonia-v1.3-Magnum-v4-22B-Q6_K.gguf")),
)
RECORDS = Path(__file__).with_name("records") / "distill_11130.jsonl"


def wilson_lower(successes: int, trials: int, z: float = 1.959964) -> float:
    """The Wilson score interval's lower bound (95% by default)."""
    if trials <= 0:
        return 0.0
    p = successes / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, (centre - margin) / (1 + z * z / trials))


def _items() -> list:
    return list(K.KNOWN) + list(K.FICTITIOUS)


def _key(item) -> str:
    return f"{item.category}:{item.subject}"


def _score_known(items, decided) -> dict:
    promoted = [(i, decided[_key(i)]) for i in items
                if decided.get(_key(i)) is not None]
    right = sum(K.is_correct(i, a) for i, a in promoted)
    return {"asked": len(items), "promoted": len(promoted), "right": right,
            "coverage": len(promoted) / len(items) if items else 0.0,
            "precision": right / len(promoted) if promoted else 0.0,
            "wrong": [(i.subject, i.category, a) for i, a in promoted
                      if not K.is_correct(i, a)]}


def _score(decided) -> dict:
    """Criteria 1-3 on one set of decisions."""
    calibration = _score_known(K.split(K.KNOWN, 0), decided)
    held_out = _score_known(K.split(K.KNOWN, 1), decided)
    invented = [(i, decided[_key(i)]) for i in K.FICTITIOUS
                if decided.get(_key(i)) is not None]
    lower = wilson_lower(calibration["right"], calibration["promoted"])
    return {
        "calibration": calibration, "held_out": held_out,
        "invented_promoted": [(i.subject, i.category, a) for i, a in invented],
        "confidence": lower,
        "c1": (held_out["promoted"] >= 36 and held_out["precision"] >= 0.95),
        "c2": len(invented) <= 2,
        "c3": lower <= held_out["precision"],
    }


def _independence(records_path: Path = RECORDS, replay: bool = False) -> tuple:
    """Three groups or not, from the GGUFs - or, on replay, from the
    manifest recorded beside the samples (§11.133: replay needs no model
    files)."""
    from ultraquant.distill import teachers as T
    from ultraquant.interpreter.llmls import independent_groups
    if replay:
        cards = T.cards_from_manifest(T.load_manifest(records_path))
    else:
        cards = [T.gguf_card(path) for _name, path in TEACHERS]
    groups = independent_groups(cards)
    lineage = {}
    for index, group in enumerate(groups):
        for card in group:
            lineage[card.id] = f"lineage-{index}"
    return len(groups) == 3, cards, lineage


def _elicit(records_path: Path, lineage: dict, cards) -> dict:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import teachers as T
    seconds = {}
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if records_path.exists():
        records_path.unlink()
    for (name, path), card in zip(TEACHERS, cards):
        started = time.monotonic()
        with T.LlamaServerTeacher(T.TeacherSpec(name, path),
                                  server_exe=SERVER) as teacher:
            E.elicit(teacher, lineage[card.id], _items(), records_path)
        seconds[name] = time.monotonic() - started
    return seconds


def _plant_single_sample(records, items):
    """P2: the first non-abstaining sample of any teacher is promoted."""
    from ultraquant.distill import elicit as E
    first = {}
    for record in records:
        if record.question_id not in first and not E.is_abstention(record.raw):
            first[record.question_id] = record.answer
    return {E.question_id(item): first.get(E.question_id(item))
            for item in items}


def _plant_least_common(answers, min_count=3, **_ignored):
    """P3: the rarest non-abstaining answer, whatever its count."""
    from ultraquant.distill import elicit as E
    counts = Counter(E.normalize(a) for a in answers
                     if not E.is_abstention(a) and E.normalize(a))
    if not counts:
        return None
    return min(counts, key=lambda k: (counts[k], k))


# -- §11.133: the review's counterexamples, as regression cases ---------------

def _synthetic(answers_by_teacher: dict, lineage_of: dict) -> list:
    """Five identical samples per teacher for one invented question."""
    from ultraquant.distill import elicit as E
    return [E.Record(teacher, lineage_of[teacher], "x.gguf", 1,
                     "capital:Testland", "What is the capital of Testland?",
                     seed, raw, E.normalize(E.extract(raw)))
            for teacher, raw in answers_by_teacher.items()
            for seed in E.SEEDS]


def review4_cases() -> dict:
    from ultraquant.distill import elicit as E
    item = K.Item("capital", "Testland", "What is the capital of Testland?")
    qid = E.question_id(item)

    def promoted(answers, lineages):
        return E.decide(_synthetic(answers, lineages), [item]).get(qid)

    gold = next(i for i in K.KNOWN
                if i.subject == "gold" and i.category == "symbol")
    return {
        "a negated contradiction is not promoted": promoted(
            {"A": "Au", "B": "Ag, not Au", "C": "UNKNOWN"},
            {"A": "L0", "B": "L1", "C": "L2"}) is None,
        "a family's dissent survives a second teacher": promoted(
            {"A": "Lyon", "B": "Lyon", "C1": "Paris", "C2": "Lyon"},
            {"A": "L0", "B": "L1", "C1": "L2", "C2": "L2"}) is None,
        "N/A is not an answer": promoted(
            {"A": "N/A", "B": "N/A", "C": "UNKNOWN"},
            {"A": "L0", "B": "L1", "C": "L2"}) is None,
        "refusal forms abstain": all(E.is_abstention(t) for t in (
            "N/A", "n/a", "Not applicable", "No capital", "None known",
            "No answer", "Not available")),
        "real answers do not": not any(E.is_abstention(t) for t in (
            "Unknownium", "Norway", "Oregon")),
        "the scorer rejects negations and alternatives":
            not K.is_correct(gold, "Ag, not Au")
            and not K.is_correct(gold, "Au or Ag")
            and K.is_correct(gold, "Au"),
    }


def replay_without_models(records_path: Path = RECORDS) -> bool:
    """Criterion 2: with every GGUF read failing, the replay still passes."""
    from ultraquant.distill import teachers as T

    def gone(path):
        raise FileNotFoundError(f"model file withheld: {path}")

    with mock.patch.object(T, "gguf_card", gone):
        try:
            return run_gate(records_path, elicit=False, _nested=True).passes
        except Exception:               # a crash is a failure
            return False


def _old_is_position(text, category=None):
    """P4: before §11.133, anything not an abstention was a position."""
    from ultraquant.distill import elicit as E
    return not E.is_abstention(text)


def _old_agree(a, b):
    """P4: the suffix rule with no guard against negations."""
    if a == b:
        return True
    shorter, longer = sorted((a.split(), b.split()), key=len)
    return bool(shorter) and longer[-len(shorter):] == shorter


def _old_family(helds: dict):
    """P5: a family in conflict collapsed to None, like an abstention."""
    from ultraquant.distill import elicit as E
    return E.promote(helds, min_lineages=1)


_OLD_REFUSAL = __import__("re").compile(
    r"\b(?:unknown|do not know|don t know|dont know|not sure|unsure|"
    r"not known|does not exist|doesn t exist|doesnt exist|not exist|"
    r"no such|there is no|there are no|fictional|fictitious|imaginary|"
    r"made up|invented|hypothetical|not a real|not real|not an actual|"
    r"no known|no record|no information|not aware|not familiar|"
    r"cannot answer|can t answer|cannot determine|unable to|"
    r"not recognized|not a recognized|not a known|no country|"
    r"no element|no novel|cannot provide|can t provide)\b")


def _old_is_abstention(text):
    """P6: §11.130's refusal list, which missed N/A and its kind."""
    from ultraquant.distill import elicit as E
    answer = E.normalize(E.extract(text))
    return not answer or answer == "none" or bool(_OLD_REFUSAL.search(answer))


def review4_plants() -> dict:
    """Each plant restores the old behaviour in every layer it lived in."""
    from ultraquant.distill import elicit as E
    from ultraquant.distill import teachers as T
    plants = {
        "P4 the old suffix rule": ("a negated contradiction is not promoted",
            [mock.patch.object(E, "is_position", _old_is_position),
             mock.patch.object(E, "agree", _old_agree)]),
        "P5 the old family collapse": (
            "a family's dissent survives a second teacher",
            [mock.patch.object(E, "family_position", _old_family)]),
        "P6 the old abstention list": ("N/A is not an answer",
            [mock.patch.object(E, "is_abstention", _old_is_abstention)]),
    }
    caught = {}
    for name, (target, patches) in plants.items():
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                caught[name] = review4_cases()[target] is False
        except Exception:               # a crash proves nothing: not caught
            caught[name] = False
    # P7: replay that ignores the manifest and reads the GGUFs regardless
    with mock.patch.object(T, "cards_from_manifest",
                           lambda manifest: [T.gguf_card(p)
                                             for _n, p in TEACHERS]):
        caught["P7 independence read from the GGUFs"] = (
            replay_without_models() is False)
    return caught


@dataclass
class DistillReport:
    """What three local teachers know, filtered and measured.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        score: Criteria 1-3 on the real filter: halves, confidence, the
            invented subjects promoted, every wrong promotion.
        independent: Criterion 4.
        planted: Plant -> whether it breached its criterion.
        per_category: Held-out coverage and precision per category.
        abstention: Per teacher, the share of samples that abstained on
            invented subjects.
        seconds: Wall time per teacher.
        records: How many raw samples are on file.
        reason: Plain-language verdict.
    """

    passes: bool
    score: dict = field(default_factory=dict)
    independent: bool = False
    planted: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    abstention: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    records: int = 0
    review4: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(records_path: Path = RECORDS, elicit: bool = True,
             _nested: bool = False) -> DistillReport:
    """Elicit (unless ``elicit`` is False), decide, score, then plant.

    ``_nested`` is the replay-without-models check calling back in: it
    runs criteria 1-6 only, so the check does not recurse.
    """
    from ultraquant.distill import elicit as E
    report = DistillReport(passes=False)
    report.independent, cards, lineage = _independence(records_path,
                                                        replay=not elicit)
    if elicit:
        report.seconds = _elicit(records_path, lineage, cards)
    records = E.load_records(records_path)
    report.records = len(records)
    items = _items()
    decided = E.decide(records, items)
    report.score = _score({_key(i): decided.get(E.question_id(i))
                           for i in items})

    for category in ("symbol", "capital", "author", "number"):
        subset = [i for i in K.split(K.KNOWN, 1) if i.category == category]
        report.per_category[category] = _score_known(
            subset, {_key(i): decided.get(E.question_id(i)) for i in subset})
    invented_ids = {E.question_id(i) for i in K.FICTITIOUS}
    for teacher in sorted({r.teacher for r in records}):
        mine = [r for r in records
                if r.teacher == teacher and r.question_id in invented_ids]
        report.abstention[teacher] = (
            sum(E.is_abstention(r.raw) for r in mine) / len(mine)
            if mine else 0.0)

    plants = (
        ("P1 abstentions counted as answers", "c2",
         lambda: mock.patch.object(E, "is_abstention", lambda text: False)),
        ("P2 any single sample promoted", "c2",
         lambda: mock.patch.object(E, "decide", _plant_single_sample)),
        ("P3 the least common answer promoted", "c1",
         lambda: mock.patch.object(E, "held", _plant_least_common)),
    )
    for name, criterion, context in plants:
        try:
            with context():
                planted = E.decide(records, items)
            score = _score({_key(i): planted.get(E.question_id(i))
                            for i in items})
            report.planted[name] = score[criterion] is False
        except Exception as exc:        # a crash proves nothing: not caught
            report.planted[name] = False
            report.per_category.setdefault("plant errors", {})[name] = repr(exc)

    if not _nested:
        cases = review4_cases()
        report.review4 = {"cases": cases,
                          "replay without models": replay_without_models(
                              records_path),
                          "plants": review4_plants()}
        report.planted.update(report.review4["plants"])
    valid = bool(report.planted) and all(report.planted.values())
    met = {c: report.score[c] for c in ("c1", "c2", "c3")}
    if not _nested:
        met["r4 cases"] = all(report.review4["cases"].values())
        met["r4 replay"] = report.review4["replay without models"]
    met["c4"] = report.independent
    # Amendment A: every (teacher, question, seed) exactly once, so a
    # duplicate cannot stand in for a missing sample.
    triples = [(r.teacher, r.question_id, r.seed) for r in records]
    expected = {(name, E.question_id(i), seed) for name, _path in TEACHERS
                for i in items for seed in E.SEEDS}
    met["c6"] = set(triples) == expected and len(triples) == len(expected)
    report.passes = valid and all(met.values())
    held_out = report.score["held_out"]
    summary = (f"held-out {held_out['right']}/{held_out['promoted']} right "
               f"({held_out['precision']:.3f}) at coverage "
               f"{held_out['coverage']:.2f}; invented promoted "
               f"{len(report.score['invented_promoted'])}/80; confidence "
               f"{report.score['confidence']:.3f}")
    if not valid:
        missed = [n for n, caught in report.planted.items() if not caught]
        report.reason = f"VOID: planted defects not caught: {missed}; {summary}"
    elif report.passes:
        report.reason = f"PASS: {summary}"
    else:
        failed = [c for c, ok in met.items() if not ok]
        report.reason = f"FAIL: {failed}; {summary}"
    return report


def main() -> int:
    import sys
    result = run_gate(elicit="--rescore" not in sys.argv)
    print(result.reason)
    print(json.dumps({"per_category": result.per_category,
                      "abstention": result.abstention,
                      "seconds": result.seconds,
                      "planted": result.planted,
                      "wrong_held_out": result.score["held_out"]["wrong"],
                      "invented_promoted": result.score["invented_promoted"]},
                     indent=1, default=str)[:6000])
    return 0 if result.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
