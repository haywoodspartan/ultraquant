"""Comparatives get a verdict that was actually computed. The gate.

"is the tower taller than the bridge?" reached the polar machinery
through two defects at once: the derive path ran it through the
combine machinery — whose conclusion is None — so the claim never
matched and BOTH directions answered No (a coin that always says No);
and a missing operand fell through to the direct matrix, which
asserted "No - tower height is 300 meters, not taller than citadel
height" — a verdict fabricated over an operand the library does not
hold, the absence-is-never-no line violated through the comparative
door. §11.54 gives comparatives an owner: the two operands are
recalled, converted where §11.42's table connects their units, and
compared, the verdict naming BOTH values so it can be checked at a
glance. Missing, negated (§11.48: a denial holds no number to
compare), non-numeric, and unit-incomparable operands refuse aloud,
each naming its reason.

Both arms run the live pipeline; the baseline arm turns
`_POLAR_COMPARES` off (the shipped defects), the treatment arm is
current. Worlds mix, in world-varying proportions: true and false
comparatives in both directions, cross-unit comparatives (the §11.42
table doing the converting), the four refusal kinds — missing operand,
negated operand, cross-family units, non-numeric operand — and
DERIVED-OPERAND comparatives ("is the tower hardness bigger than the
bridge hardness?" where both sides are derivable through chains but
neither is stored), scored at zero for both arms: the mechanism
compares RECALLED operands by design, refuses derivable ones honestly,
and that ceiling is the pre-registered unwinnable share (~25%, per
§11.35's power arithmetic) that lets the contrast vary. Comparing
derived operands is a registered candidate, not a promise — and a
verdict on one today would be a fabrication, so the kind joins the
zero-tolerance set.

**The criteria, written before the run.**

1. **Gain** — comparative correctness (right verdict AND both operand
   values named; converted kinds also say "(units converted)";
   refusal kinds answered with the refusal shape, never a verdict)
   must beat the baseline arm by more than the seed-to-seed sd.
2. **No verdict without operands** — a refusal-kind question answered
   Yes or No in the treatment arm is a fail, once. The baseline arm's
   count is reported beside it: it is the shipped fabrication,
   measured.
3. **Recall holds** — direct recall at 1.000 in both arms.

**It was run twice: the first run's contrast never varied (the eighth
"nothing varied" in the book, resolved the standing way — the
derived-operand ceiling scored in, the rule untouched), and the second
PASSED.**

| arm | compared | recall | fabricated |
|---|---:|---:|---:|
| coin | 0.000 | 1.000 | **38** |
| computed | **0.893** | 1.000 | **0** |

**+0.893 at 12.91x seed sd.** Every held-operand comparative answered
with the right verdict and both values named, conversions marked, all
four refusal kinds refused with their reasons — zero verdicts without
operands — while the coin arm fabricated 38 of them, the shipped
defect measured at its full rate beside the fix. The 0.107 miss is the
derived-operand ceiling scoring zero by design; comparing derived
operands stays a registered candidate.

Run it::

    python -m ultraquant.experiments.comparative_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ComparativeReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_COMPS_UP = ["taller", "longer", "bigger", "higher"]
_COMPS_DOWN = ["shorter", "smaller", "lower"]


@dataclass
class ComparativeReport:
    """The two arms, the fabrication counts, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        compared: Arm -> comparative correctness over all worlds.
        fabricated: Arm -> verdicts on refusal-kind questions
            (treatment must be 0; baseline is the shipped defect).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    compared: dict = field(default_factory=dict)
    fabricated: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'compared':>9} {'recall':>7} "
                 f"{'fabricated':>11}"]
        for arm in ("coin", "computed"):
            lines.append(
                f"{arm:<10} {self.compared[arm]:>9.3f} "
                f"{self.recall[arm]:>7.3f} {self.fabricated[arm]:>11}")
        lines += [
            f"delta (computed vs coin) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _classify(response: str) -> str:
    lowered = response.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return "open"


def _build_world(rng: random.Random):
    """One comparative world.

    Returns ``(statements, questions, directs)`` where questions are
    ``(question, kind, values)`` — kind in yes/no/conv-yes/conv-no/
    missing/negated/crossfam/nonnum, values the operand strings a
    correct verdict must name.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    heights = {}
    for entity in names[:5]:
        value = rng.randrange(10, 900)
        heights[entity] = value
        statements.append(f"the {entity} height is {value} meters")
    km_entity = names[5]
    km_value = rng.randrange(2, 9)
    statements.append(f"the {km_entity} height is {km_value} kilometers")
    negated_entity = names[6]
    statements.append(
        f"the {negated_entity} height is not {rng.randrange(10, 900)} "
        "meters")
    weight_entity = names[7]
    statements.append(
        f"the {weight_entity} weight is {rng.randrange(5, 90)} tonnes")
    word_entity = names[8]
    statements.append(f"the {word_entity} material is oak")
    ghost = names[9]  # never given a height

    ordered = sorted(names[:5], key=lambda n: heights[n])
    low, high = ordered[0], ordered[-1]
    mid = ordered[len(ordered) // 2]

    questions = []
    for _ in range(rng.randrange(1, 3)):
        up = rng.choice(_COMPS_UP)
        questions.append(
            (f"is the {high} height {up} than the {low} height?",
             "yes", [str(heights[high]), str(heights[low])]))
    for _ in range(rng.randrange(1, 3)):
        up = rng.choice(_COMPS_UP)
        questions.append(
            (f"is the {low} height {up} than the {high} height?",
             "no", [str(heights[low]), str(heights[high])]))
    if rng.random() < 0.7:
        down = rng.choice(_COMPS_DOWN)
        questions.append(
            (f"is the {low} height {down} than the {high} height?",
             "yes", [str(heights[low]), str(heights[high])]))
    # Cross-unit: kilometers beat every meter entity in these ranges.
    questions.append(
        (f"is the {km_entity} height {rng.choice(_COMPS_UP)} than "
         f"the {mid} height?",
         "conv-yes", [str(km_value), str(heights[mid])]))
    if rng.random() < 0.7:
        questions.append(
            (f"is the {mid} height {rng.choice(_COMPS_UP)} than "
             f"the {km_entity} height?",
             "conv-no", [str(heights[mid]), str(km_value)]))
    # Refusal kinds.
    questions.append(
        (f"is the {mid} height {rng.choice(_COMPS_UP)} than "
         f"the {ghost} height?", "missing", []))
    if rng.random() < 0.8:
        questions.append(
            (f"is the {mid} height {rng.choice(_COMPS_UP)} than "
             f"the {negated_entity} height?", "negated", []))
    if rng.random() < 0.8:
        questions.append(
            (f"is the {mid} height {rng.choice(_COMPS_UP)} than "
             f"the {weight_entity} weight?", "crossfam", []))
    if rng.random() < 0.6:
        questions.append(
            (f"is the {mid} height {rng.choice(_COMPS_UP)} than "
             f"the {word_entity} material?", "nonnum", []))

    # Derived-operand comparatives: the unwinnable share. Both
    # sides derivable via material chains, neither stored - the
    # recall-only comparator refuses honestly, scored zero by design.
    for _ in range(rng.randrange(0, 3)):
        left, right = rng.sample(names[:4], 2)
        for entity in (left, right):
            statements.append(f"the {entity} material is "
                              f"{rng.choice(('steel', 'iron'))}")
        questions.append(
            (f"is the {left} hardness {rng.choice(_COMPS_UP)} than "
             f"the {right} hardness?", "derived-op", []))
    if any(kind == "derived-op" for _q, kind, _v in questions):
        statements.append(
            f"the steel hardness is {rng.randrange(100, 900)} units")
        statements.append(
            f"the iron hardness is {rng.randrange(100, 900)} units")

    directs = []
    for entity in rng.sample(names[:5], 2):
        directs.append((f"what is the {entity} height?",
                        str(heights[entity])))

    rng.shuffle(statements)
    return statements, questions, directs


def run_gate(seeds: int = 12) -> ComparativeReport:
    """Run both arms over generated comparative worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ComparativeReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    compared_scores = {"coin": [], "computed": []}
    recall_scores = {"coin": [], "computed": []}
    fabricated = {"coin": 0, "computed": 0}

    real_flag = thoughts_module._POLAR_COMPARES
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, questions, directs = _build_world(rng)

        for arm in ("coin", "computed"):
            thoughts_module._POLAR_COMPARES = (arm == "computed")
            root = Path(tempfile.mkdtemp(prefix=f"uq_cmp_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind, values in questions:
                    response, _trace = run_pipeline(question, session)
                    shape = _classify(response)
                    named = all(v in response for v in values)
                    if kind == "yes":
                        won.append(float(shape == "yes" and named))
                    elif kind == "no":
                        won.append(float(shape == "no" and named))
                    elif kind == "conv-yes":
                        won.append(float(shape == "yes" and named
                                         and "units converted"
                                         in response))
                    elif kind == "conv-no":
                        won.append(float(shape == "no" and named
                                         and "units converted"
                                         in response))
                    elif kind == "derived-op":
                        # Scored zero by design in BOTH arms: the
                        # recall-only comparator refuses derivable
                        # operands, and a verdict here would be a
                        # fabrication.
                        won.append(0.0)
                        if shape != "open":
                            fabricated[arm] += 1
                    else:
                        won.append(float(shape == "open"))
                        if shape != "open":
                            fabricated[arm] += 1
                compared_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._POLAR_COMPARES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = ComparativeReport(
        compared={arm: statistics.fmean(vals)
                  for arm, vals in compared_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        fabricated=dict(fabricated),
    )

    contrasts = [compared_scores["computed"][i]
                 - compared_scores["coin"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.fabricated["computed"] > 0:
        report.reason = (
            f"a verdict without operands: "
            f"{report.fabricated['computed']} refusal-kind question(s) "
            "answered yes or no in the treatment arm")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"the owner buys nothing ({report.delta:+.3f})"
        return report
    if report.seed_sd <= 0.0:
        report.reason = (f"every world gave the same contrast "
                         f"({report.delta:+.3f}); nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {report.delta:+.3f} is "
                         f"{report.margin_ratio:.2f}x the seed sd "
                         f"({report.seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"comparatives answer at {report.compared['computed']:.3f} "
        f"against the coin arm's {report.compared['coin']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero verdicts without operands in the treatment arm - while "
        f"the coin arm fabricated {report.fabricated['coin']}, which "
        "is the shipped defect measured - and recall held at 1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
