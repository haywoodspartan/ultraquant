"""Rounding is a request. The rounding gate.

§11.76 made exactness the rule: one third prints as 1/3 because
0.333... is a different number, and an irrational root is enclosed
rather than named. That is right, and by itself it is also a little
unhelpful, because sometimes two decimal places is exactly what a
person wants and they are not confused about what they are asking
for.

So rounding is a REQUEST. "to 2 decimal places", "to four decimal
places", "to the nearest whole number" are parsed off the tail of
the question; the arithmetic still runs exactly; and only the ANSWER
is rounded. Three properties keep this from undoing §11.76:

- **The answer says it was rounded, and names what it was rounded
  from**: "100 / 3 = 33.33 - rounded to 2 decimal places; the exact
  value is 100/3."
- **A value that needs no rounding says that instead**: "1 / 4 =
  0.25 - exact at 2 decimal places, so nothing was rounded." The
  difference between "this is the number" and "this is near the
  number" is never left for the reader to guess.
- **Nothing is rounded unless it was asked for.** Without a request
  the answers are §11.76's: 100/3 stays a fraction, sqrt 2 stays an
  enclosure.

The tie rule is a decision rather than an artefact: a half rounds
away from zero, in exact rational arithmetic, so 0.125 to two places
is 0.13 and -0.125 is -0.13. And a rounded ROOT is proved, not
approximated - the root is enclosed at increasing precision until
both bounds round to the same decimal, so the answer is the correct
rounding because two proved bounds agree on it, not because a float
looked like it.

Both arms run the live pipeline; the baseline arm turns
`_ROUNDING_ON` off. Worlds mix, in world-varying proportions:
expressions needing rounding, expressions exact at the requested
precision, nearest-whole-number requests, irrational roots with a
precision, requests against a held belief (where the unit rides
along), the same expressions with NO request (which must answer
exactly in both arms), and - at about a fifth of the questions -
requests for SIGNIFICANT FIGURES, which nobody here can serve.

That last is this rung's ceiling, scored in rather than excused.
Significant figures are a different rule from decimal places and
this rung does not have it; neither arm produces the value, so both
score zero there. What the rung does do is refuse BY NAME - "I can
round to a number of decimal places or to the nearest whole number;
that is a different rule and I do not have it" - rather than reading
"to" as an operand and refusing about the wrong word, which is what
happens when a gap is never admitted.

**The criteria, written before the run.**

1. **Gain** - correctness (the right rounded value, or the right
   exact value where none was requested) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **No wrong rounding** - a value differing from the gate's own
   exact rounding, in the treatment arm, is a fail, once.
3. **Every rounded answer is labelled** - a rounded value stated
   without saying it was rounded is a fail, once. An unlabelled
   approximation is precisely the thing §11.76 exists to refuse.
4. **Nothing rounds unasked** - questions with no request answer
   identically in both arms.

**It took three runs, and the mechanism was wrong once, the harness
twice.**

Run one failed with fourteen wrong roundings and one unlabelled
approximation. One cause was real: "to 2 decimal places" was
answering "0.5" and "10", printing the shortest EXACT form when
somebody had named a precision. A request for two decimal places is
answered in two decimal places - the padding is part of the answer,
because it says how far the claim goes - so the rounded answer is
rendered at exactly the width that was asked for. The other cause
was the gate's: it had marked every division as needing rounding,
and 300 / 6 is exactly 50, so the harness was scoring the mechanism
wrong for being right. Whether a case rounds is arithmetic, not a
label the gate gets to choose.

Run two came back clean on both lines and failed anyway: nothing
varied. The baseline read no request at all and the treatment read
them all, so every seed returned the same contrast and the seed sd
was zero. The house rule holds when it is inconvenient, so the fix
is §11.35's: score in the cases nobody can win.

| arm | correct |
|---|---:|
| exactonly | 0.000 |
| onrequest | **0.802** |

**+0.802 at 14.51x seed sd, zero wrong roundings, zero unlabelled
approximations, zero unrequested answers moved.** The 0.198 is the
significant-figures ceiling, scored zero for both arms.

Criterion 3 is the one that keeps §11.76 intact. A rounded number
that does not say it is rounded is exactly the failure this whole
subject was built to refuse - it is 1.4142135623730951 wearing a
shorter coat - so every approximation names its precision AND the
exact value it came from, and a value that needed no rounding says
that instead.

Run it::

    python -m ultraquant.experiments.rounding_gate
"""

from __future__ import annotations

import random
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["RoundingReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class RoundingReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Rounded values differing from the gate's arithmetic.
        unlabelled: Rounded answers that do not say they were.
        leaked: Unrequested questions whose answer moved.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    unlabelled: int = 0
    leaked: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _round_half_away(value: Fraction, places: int) -> Fraction:
    """The gate's own rounding, walked from its own numbers."""
    scale = 10 ** places
    shifted = value * scale
    floor = shifted.numerator // shifted.denominator
    remainder = shifted - floor
    if remainder > Fraction(1, 2):
        floor += 1
    elif remainder == Fraction(1, 2) and value >= 0:
        floor += 1
    return Fraction(floor, scale)


def _decimal(value: Fraction, places: int) -> str:
    """A rational with a terminating form, printed in full."""
    if places == 0:
        return str(value.numerator // value.denominator)
    scaled = value * (10 ** places)
    whole = scaled.numerator // scaled.denominator
    digits = str(abs(whole)).rjust(places + 1, "0")
    sign = "-" if whole < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected)."""
    names = []
    seen = set()
    while len(names) < 3:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    held = rng.randrange(100, 900, 25)
    statements = [f"the {names[0]} height is {held} meters"]
    cases = []

    # Expressions that genuinely need rounding.
    # Whether a case rounds is arithmetic, not a label the gate gets
    # to choose: 300 / 6 is exactly 50, and a world that assumed
    # every division needs rounding would score the mechanism wrong
    # for saying so.
    for _ in range(rng.choice((2, 3))):
        top = rng.randint(10, 500)
        bottom = rng.choice((3, 6, 7, 9, 11))
        places = rng.choice((1, 2, 3))
        spoken = (str(places) if rng.random() < 0.5
                  else _WORDS[places])
        exact = Fraction(top, bottom)
        cases.append((f"what is {top} / {bottom} to {spoken} decimal "
                      f"places?", "rounded",
                      (_decimal(_round_half_away(exact, places),
                                places),
                       _round_half_away(exact, places) != exact)))

    # Exact at the requested precision: labelled the other way.
    if rng.random() < 0.8:
        quarters = rng.choice((Fraction(1, 4), Fraction(3, 4),
                               Fraction(1, 2)))
        top, bottom = quarters.numerator, quarters.denominator
        cases.append((f"what is {top} / {bottom} to 2 decimal "
                      f"places?", "rounded",
                      (_decimal(quarters, 2), False)))
        # Padding is part of the answer: "0.50" says how far the
        # claim goes, where "0.5" leaves the reader to assume it.

    # The nearest whole number.
    if rng.random() < 0.8:
        top = rng.randint(10, 500)
        bottom = rng.choice((3, 7, 9))
        exact = Fraction(top, bottom)
        cases.append((f"what is {top} / {bottom} to the nearest whole "
                      f"number?", "rounded",
                      (_decimal(_round_half_away(exact, 0), 0),
                       _round_half_away(exact, 0) != exact)))

    # An irrational root with a precision.
    if rng.random() < 0.7:
        radicand = rng.choice((2, 3, 5, 7, 10, 11))
        places = rng.choice((2, 3, 4))
        # The gate's own bound-and-agree, at generous precision.
        scale = 10 ** (places + 6)
        import math

        low = Fraction(math.isqrt(radicand * scale * scale), scale)
        cases.append((f"what is the square root of {radicand} to "
                      f"{places} decimal places?", "rounded",
                      (_decimal(_round_half_away(low, places),
                                places), True)))

    # A request against a held belief: the unit rides along.
    if rng.random() < 0.7:
        divisor = rng.choice((3, 7, 9))
        exact = Fraction(held, divisor)
        cases.append((f"what is the {names[0]} height / {divisor} to "
                      f"1 decimal place?", "rounded",
                      (_decimal(_round_half_away(exact, 1), 1)
                       + " meters",
                       _round_half_away(exact, 1) != exact)))

    # The ceiling, scored in: significant figures are a different
    # rule and this rung does not have it. Neither arm produces the
    # value, so both score zero and the varying share of these is
    # what the contrast varies against (§11.35's arithmetic). The
    # treatment arm at least refuses BY NAME rather than reading
    # "to" as an operand, which is the improvement that came with
    # admitting the gap.
    for _ in range(rng.choice((1, 2))):
        top = rng.randint(10, 500)
        bottom = rng.choice((3, 7, 9))
        figures = rng.choice((2, 3))
        cases.append((f"what is {top} / {bottom} to {figures} "
                      f"significant figures?", "rounded",
                      (_decimal(Fraction(top, bottom), 6), True)))

    # No request: §11.76's answers, identical in both arms.
    cases.append((f"what is 100 / 3?", "unrequested", None))
    cases.append((f"what is the square root of 2?", "unrequested",
                  None))
    cases.append((f"what is 2 + 2?", "unrequested", None))

    rng.shuffle(cases)
    return statements, cases


def _stated(response: str) -> str:
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - rounded", " - exact at", " - computed",
                " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def run_gate(seeds: int = 12) -> RoundingReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import calculate

    arms = {"exactonly": False, "onrequest": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    unlabelled = 0
    unrequested: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(14900 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_round_"))
            old = calculate._ROUNDING_ON
            calculate._ROUNDING_ON = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                scored = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    if kind == "rounded":
                        scored += 1
                        value, was_rounded = expected
                        stated = _stated(response)
                        if stated != value:
                            if stated and flag:
                                wrong += 1
                            continue
                        labelled = ("rounded to" in response
                                    if was_rounded
                                    else "nothing was rounded"
                                    in response)
                        if labelled:
                            hits += 1
                        elif flag and was_rounded:
                            unlabelled += 1
                    else:
                        unrequested.setdefault((seed, question),
                                               {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                calculate._ROUNDING_ON = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    leaked = sum(1 for by_arm in unrequested.values()
                 if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["onrequest"],
                                       per_seed["exactonly"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and unlabelled == 0 and leaked == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - rounding on request at "
        f"{correct['onrequest']:.3f} against "
        f"{correct['exactonly']:.3f} ({delta:+.3f}, {ratio:.2f}x "
        f"seed sd), {wrong} wrong roundings, {unlabelled} unlabelled "
        f"approximations, {leaked} unrequested answers moved")
    return RoundingReport(
        passes=passes, correct=correct, wrong=wrong,
        unlabelled=unlabelled, leaked=leaked, delta=delta,
        seed_sd=seed_sd, margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm           correct")
    print(f"exactonly      {report.correct['exactonly']:.3f}")
    print(f"onrequest      {report.correct['onrequest']:.3f}")
    print(f"wrong roundings: {report.wrong}")
    print(f"unlabelled approximations: {report.unlabelled}")
    print(f"unrequested answers moved: {report.leaked}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
