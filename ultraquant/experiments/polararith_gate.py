"""Asking whether the arithmetic holds. The polar-arithmetic gate.

The system could compute "2 + 2" and could compare two beliefs, and
could do nothing at all with the question that joins them::

    > is 2 + 2 = 5?
      I don't hold anything on that yet.
    > is 3 * 4 greater than 10?
      I can't compare those: I hold nothing for '3 * 4'.

The second refusal is the honest shape of the gap: '3 * 4' really is
not a key, and saying so is true and useless, because the question
carries its own left-hand side and the reader that evaluates it was
already in the building. §11.83 lets a comparison side be an
EXPRESSION, and teaches the branch equality - "=", "equals", "equal
to" - which pivots exactly as "taller than" does, so both sides go
through the same machinery: units, conversions, derived operands and
all.

Everything composes because nothing was special-cased. "is the tower
height * 2 greater than 500 meters?" multiplies a belief (§11.78),
keeps its unit, compares against a written quantity (§11.82), and
answers Yes - and each of those was built for its own reason,
separately, before this rung existed.

One shape had to be fixed on the way in, and it was this rung's own
doing: §11.78's reader was taking polar questions, so "is 2 + 2
equal to 4?" refused with "I hold nothing for 'is'" - a refusal
about the wrong word entirely. A polar lead is §11.83's question and
the reader declines it.

Both arms run the live pipeline; the baseline arm turns
`_EXPRESSION_OPERANDS` off, leaving §11.82's comparisons in place, so
the contrast is exactly this rung. Worlds mix, in world-varying
proportions: literal equalities true and false, equalities against a
belief, comparisons with an expression on the left, comparisons
mixing a belief-expression with a written quantity, comparisons in
§11.82's shape (which must not move), plain polar questions (which
must not move), and an expression naming a belief nobody holds
(which must refuse in both arms).

**The criteria, written before the run.**

1. **Gain** - verdict correctness against the gate's own arithmetic
   must beat the baseline arm by more than the seed-to-seed sd.
2. **No wrong verdict** - a Yes or No differing from ground truth,
   in the treatment arm, is a fail, once.
3. **An unreadable side is never a verdict** - an expression over an
   unheld belief given a Yes or No in the treatment arm is a fail,
   once. The absence-is-never-no line, arriving in arithmetic.
4. **§11.82's comparisons do not move** - belief-against-quantity
   answers are identical in both arms.
5. **Plain polar questions do not move.**

**It was run once, and PASSED - on the claim and on all four
lines.**

| arm | correct |
|---|---:|
| unheard | 0.341 |
| compared | **1.000** |

**+0.659 at 8.52x seed sd, zero wrong verdicts, zero unreadable
sides answered, zero comparisons moved, zero plain polar answers
moved.**

What the run shows best is composition. Nothing in this rung knows
about units, chains, or written quantities, and yet:

    > is the tower height * 2 greater than 500 meters?
      Yes - tower height * 2 is 600 meters, 500 meters
      (confidence 0.60)

That answer multiplies a belief and keeps its unit (§11.78),
compares against a quantity nobody stored (§11.82), and reads the
whole left side as an expression (here) - three rungs built
separately, for separate reasons, meeting in one sentence because
each was built as a rule rather than a case.

The equality voice is deliberate too. "Yes - 4, 4" is a machine
agreeing with itself; "Yes - 2 + 2 is 4" and "No - 2 + 2 is 4, not
5" are the shapes the polar family has always used, so the newest
question sounds like the oldest one.

Run it::

    python -m ultraquant.experiments.polararith_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["PolarArithReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_FAMILIES = [("height", "meters"), ("length", "meters"),
             ("weight", "kilograms")]


@dataclass
class PolarArithReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> verdict correctness over all worlds.
        wrong: Verdicts differing from ground truth.
        unheld_answered: Unreadable sides given a verdict.
        compare_moved: §11.82 comparisons that differ across arms.
        polar_moved: Plain polar answers that differ across arms.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    unheld_answered: int = 0
    compare_moved: int = 0
    polar_moved: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected)."""
    names = []
    seen = set()
    while len(names) < 5:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    attr, unit = rng.choice(_FAMILIES)

    a = names[0]
    va = rng.randrange(150, 900, 25)
    statements = [f"the {a} {attr} is {va} {unit}",
                  f"the {names[1]} material is iron"]
    cases = []

    # Literal equalities, true and false, in world-varying number.
    for _ in range(rng.choice((2, 3))):
        left = rng.randint(2, 40)
        right = rng.randint(2, 40)
        claimed = (left + right if rng.random() < 0.5
                   else left + right + rng.randint(1, 5))
        cases.append((f"is {left} + {right} = {claimed}?", "verdict",
                      "Yes" if claimed == left + right else "No"))

    # An equality against a belief.
    if rng.random() < 0.8:
        claimed = va if rng.random() < 0.5 else va + 25
        cases.append((f"is the {a} {attr} equal to {claimed} {unit}?",
                      "verdict", "Yes" if claimed == va else "No"))

    # A comparison with an expression on the left.
    for _ in range(rng.choice((1, 2))):
        left = rng.randint(2, 12)
        right = rng.randint(2, 12)
        against = rng.randint(5, 150)
        cases.append((f"is {left} * {right} greater than {against}?",
                      "verdict",
                      "Yes" if left * right > against else "No"))

    # A belief-expression against a written quantity.
    if rng.random() < 0.8:
        scale = rng.randint(2, 5)
        against = rng.randrange(100, 3000, 50)
        cases.append((f"is the {a} {attr} * {scale} greater than "
                      f"{against} {unit}?", "verdict",
                      "Yes" if va * scale > against else "No"))

    # An expression over a belief nobody holds: refuses in both arms.
    cases.append((f"is the {names[2]} {attr} * 2 greater than 5 "
                  f"{unit}?", "unheld", None))

    # §11.82's shape, which must not move.
    against = rng.randrange(100, 900, 25)
    cases.append((f"is the {a} {attr} greater than {against} {unit}?",
                  "compare", "Yes" if va > against else "No"))

    # Plain polar questions, which must not move.
    cases.append((f"is the {names[1]} material iron?", "polar", None))
    cases.append((f"is the {names[1]} material steel?", "polar", None))

    rng.shuffle(cases)
    return statements, cases


def run_gate(seeds: int = 12) -> PolarArithReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    arms = {"unheard": False, "compared": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    unheld_answered = 0
    compare_answers: dict = {}
    polar_answers: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(13800 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_polar_"))
            old = thoughts._EXPRESSION_OPERANDS
            thoughts._EXPRESSION_OPERANDS = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                scored = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    verdict = ("Yes" if response.startswith("Yes")
                               else "No" if response.startswith("No")
                               else "")
                    if kind == "verdict":
                        scored += 1
                        if verdict == expected:
                            hits += 1
                        elif verdict and flag:
                            wrong += 1
                    elif kind == "unheld":
                        scored += 1
                        if not verdict:
                            hits += 1
                        elif flag:
                            unheld_answered += 1
                    elif kind == "compare":
                        scored += 1
                        if verdict == expected:
                            hits += 1
                        elif verdict and flag:
                            wrong += 1
                        compare_answers.setdefault(
                            (seed, question), {})[arm] = response
                    else:
                        polar_answers.setdefault((seed, question),
                                                 {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                thoughts._EXPRESSION_OPERANDS = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    compare_moved = sum(1 for by_arm in compare_answers.values()
                        if len(set(by_arm.values())) > 1)
    polar_moved = sum(1 for by_arm in polar_answers.values()
                      if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["compared"],
                                       per_seed["unheard"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and unheld_answered == 0 and compare_moved == 0
              and polar_moved == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - polar arithmetic at "
        f"{correct['compared']:.3f} against {correct['unheard']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {wrong} wrong "
        f"verdicts, {unheld_answered} unreadable sides answered, "
        f"{compare_moved} comparisons moved, {polar_moved} plain "
        f"polar answers moved")
    return PolarArithReport(
        passes=passes, correct=correct, wrong=wrong,
        unheld_answered=unheld_answered, compare_moved=compare_moved,
        polar_moved=polar_moved, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"unheard      {report.correct['unheard']:.3f}")
    print(f"compared     {report.correct['compared']:.3f}")
    print(f"wrong verdicts: {report.wrong}")
    print(f"unreadable sides answered: {report.unheld_answered}")
    print(f"comparisons moved: {report.compare_moved}")
    print(f"plain polar answers moved: {report.polar_moved}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
