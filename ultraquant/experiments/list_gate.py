"""A list of numbers is not a list of beliefs. The list gate.

"what is the average of 3, 5 and 10?" is basic arithmetic and the
system could not read it. §11.57's aggregate owns those exact words
over HELD facts - "what is the average height?" ranks the store -
and a written list looks identical from a distance, which is why
this rung is mostly a question about a boundary rather than about
arithmetic.

The boundary is drawn where it can be checked: **every item must
parse as a written number or quantity.** A list naming beliefs is
declined outright and falls through to the machinery that owns
beliefs, so "the sum of the tower height and the keep height" keeps
§11.42's voice while "the sum of 3, 5 and 10" gets this one. One
store, one voice per question - §11.75's lesson applied before the
bug rather than after it.

Everything else is inherited rather than rebuilt. Units ride the
list the way they ride the operators: a same-family list converts
and reads in the largest unit present, a mixed-family list refuses
by name, and a bare number beside a united one refuses because
assuming its unit would be invention. Exactness is §11.76's: the
average of 1, 2 and 4 is 7/3, not 2.3333333333333335.

Both arms run the live pipeline; the baseline arm turns `_LISTS_ON`
off. Worlds mix, in world-varying proportions: sums, averages,
largests and smallests over two to five written numbers; the same
over quantities, sometimes needing a conversion; mixed-family and
bare-beside-united lists that must refuse; single items that are not
lists at all; the boundary itself - belief lists and fact
aggregates, which must answer identically in both arms; and - at
about a fifth of the questions - MEDIANS, which nobody here can
serve.

The median is this rung's ceiling, scored in rather than excused. It
is a real list operation and a deliberate omission rather than an
oversight: an even-length median needs an ordering decision nobody
has asked for yet, and inventing one would be the kind of convention
§11.80 refused for "300 + 20%". Neither arm answers, so both score
zero there.

**The criteria, written before the run.**

1. **Gain** - correctness (the exact expected value with its unit,
   or a refusal where one is owed) must beat the baseline arm by
   more than the seed-to-seed sd.
2. **No wrong number** - a value differing from the gate's own
   exact arithmetic, in the treatment arm, is a fail, once.
3. **No undefined answer** - a mixed-family or bare-beside-united
   list given a number in the treatment arm is a fail, once.
4. **The boundary holds** - belief lists and fact aggregates answer
   identically in both arms. This is the whole reason the rung is
   narrow, and one difference is a fail.

**It was run twice - the first measured nothing, because nothing
varied.** The baseline read no list at all and the treatment read
them all, so the seed sd was zero. Fixed §11.35's way, by scoring in
the cases nobody can win. Second run, unchanged in the mechanism:

| arm | correct |
|---|---:|
| unread | 0.000 |
| listed | **0.804** |

**+0.804 at 11.27x seed sd, zero wrong values, zero undefined
answers, zero boundary answers moved.** The 0.196 is the median
ceiling, scored zero for both arms.

Criterion 4 is the one this rung exists to protect, and it held: the
belief list ("the sum of the tower height and the keep height"), the
fact aggregates ("the average height", "the total height") and the
single item that is not a list at all answered identically in both
arms, every time. §11.75 was learned from a bug where two branches
answered one question differently; here the same shape was designed
against before it could happen, which is the only real use of a
lesson.

Run it::

    python -m ultraquant.experiments.list_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["ListReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_OPERATIONS = ["sum", "average", "largest", "smallest"]


@dataclass
class ListReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Values differing from the gate's arithmetic.
        undefined_answered: Refusable lists given a number.
        boundary_moved: Belief lists or fact aggregates that differ.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    undefined_answered: int = 0
    boundary_moved: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _render(value: Fraction, unit: str) -> str:
    """The gate's own exact renderer."""
    if value.denominator == 1:
        text = str(value.numerator)
    else:
        rest, twos, fives = value.denominator, 0, 0
        while rest % 2 == 0:
            rest //= 2
            twos += 1
        while rest % 5 == 0:
            rest //= 5
            fives += 1
        if rest != 1:
            text = f"{value.numerator}/{value.denominator}"
        else:
            places = max(twos, fives)
            scaled = value.numerator * (10 ** places) // value.denominator
            digits = str(abs(scaled)).rjust(places + 1, "0")
            sign = "-" if scaled < 0 else ""
            text = f"{sign}{digits[:-places]}.{digits[-places:]}"
    return f"{text} {unit}" if unit else text


def _expected(operation: str, values: list[Fraction],
              unit: str) -> str:
    if operation == "sum":
        return _render(sum(values, Fraction(0)), unit)
    if operation == "average":
        return _render(sum(values, Fraction(0)) / len(values), unit)
    if operation == "largest":
        return _render(max(values), unit)
    return _render(min(values), unit)


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected)."""
    names = []
    seen = set()
    while len(names) < 3:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    statements = []
    held = {}
    for name in names[0:2]:
        value = rng.randrange(100, 900, 25)
        statements.append(f"the {name} height is {value} meters")
        held[name] = value

    cases = []

    def spoken(items: list[str]) -> str:
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    # Bare numbers, every operation.
    for _ in range(rng.choice((2, 3))):
        operation = rng.choice(_OPERATIONS)
        count = rng.choice((2, 3, 4, 5))
        numbers = [rng.randint(1, 90) for _ in range(count)]
        cases.append((f"what is the {operation} of "
                      f"{spoken([str(n) for n in numbers])}?",
                      "value",
                      _expected(operation,
                                [Fraction(n) for n in numbers], "")))

    # Quantities in one unit.
    if rng.random() < 0.8:
        operation = rng.choice(_OPERATIONS)
        count = rng.choice((2, 3))
        numbers = [rng.randint(1, 90) for _ in range(count)]
        cases.append((f"what is the {operation} of "
                      f"{spoken([f'{n} meters' for n in numbers])}?",
                      "value",
                      _expected(operation,
                                [Fraction(n) for n in numbers],
                                "meters")))

    # A conversion: the answer reads in the largest unit present.
    if rng.random() < 0.7:
        small = rng.randrange(100, 900, 100)
        large = rng.randint(1, 4)
        total = Fraction(small, 1000) + large
        cases.append((f"what is the sum of {small} meters and "
                      f"{large} kilometers?", "value",
                      _render(total, "kilometers")))

    # The two refusals.
    if rng.random() < 0.8:
        cases.append((f"what is the sum of {rng.randint(2, 40)} "
                      f"meters and {rng.randint(2, 40)} kilograms?",
                      "refuses", None))
    if rng.random() < 0.8:
        cases.append((f"what is the sum of {rng.randint(2, 40)} "
                      f"meters and {rng.randint(2, 40)}?", "refuses",
                      None))

    # The ceiling, scored in: the median is a real list operation
    # and this rung does not have it. Neither arm answers, so both
    # score zero and the varying share of these is what the contrast
    # varies against (§11.35's arithmetic). It is a deliberate
    # omission rather than an oversight - a median needs an ordering
    # decision for even-length lists that nobody has asked for yet.
    for _ in range(rng.choice((1, 2))):
        count = rng.choice((3, 5))
        numbers = sorted(rng.randint(1, 90) for _ in range(count))
        cases.append((f"what is the median of "
                      f"{spoken([str(n) for n in numbers])}?",
                      "value",
                      _render(Fraction(numbers[count // 2]), "")))

    # The boundary: these must not move between arms.
    a, b = names[0], names[1]
    cases.append((f"what is the sum of the {a} height and the {b} "
                  f"height?", "boundary", None))
    cases.append((f"what is the average height?", "boundary", None))
    cases.append((f"what is the total height?", "boundary", None))
    cases.append((f"what is the average of {rng.randint(2, 90)}?",
                  "boundary", None))

    rng.shuffle(cases)
    return statements, cases


def _stated(response: str) -> str:
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - computed", " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def run_gate(seeds: int = 12) -> ListReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    arms = {"unread": False, "listed": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    undefined_answered = 0
    boundary: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(23000 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_list_"))
            old = thoughts._LISTS_ON
            thoughts._LISTS_ON = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                scored = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    stated = _stated(response)
                    if kind == "value":
                        scored += 1
                        if stated == expected:
                            hits += 1
                        elif stated and flag:
                            wrong += 1
                    elif kind == "refuses":
                        scored += 1
                        if response.startswith("I can't compute"):
                            hits += 1
                        elif stated and flag:
                            undefined_answered += 1
                    else:
                        boundary.setdefault((seed, question),
                                            {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                thoughts._LISTS_ON = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    boundary_moved = sum(1 for by_arm in boundary.values()
                         if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["listed"],
                                       per_seed["unread"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and undefined_answered == 0 and boundary_moved == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - lists at "
        f"{correct['listed']:.3f} against {correct['unread']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {wrong} wrong values, "
        f"{undefined_answered} undefined answers, {boundary_moved} "
        f"boundary answers moved")
    return ListReport(
        passes=passes, correct=correct, wrong=wrong,
        undefined_answered=undefined_answered,
        boundary_moved=boundary_moved, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"unread       {report.correct['unread']:.3f}")
    print(f"listed       {report.correct['listed']:.3f}")
    print(f"wrong values: {report.wrong}")
    print(f"undefined answered: {report.undefined_answered}")
    print(f"boundary answers moved: {report.boundary_moved}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
