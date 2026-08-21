"""A root that has no value still has a place. The power gate.

Powers and roots are where "say exactly what is true" stops being a
style and starts costing something. A whole power is exact and easy:
2^10 is 1024, 2^-3 is one eighth, and the sign sits outside the
power so -2^2 is -4. A root is not. **The square root of two has no
exact value** - not as a decimal, not as a fraction - and the
standard option answers it anyway, with 1.4142135623730951, a number
whose square is 2.0000000000000004.

This rung refuses to name it and says where it is instead::

    > what is the square root of 2?
      sqrt 2 is between 1.414213562 and 1.414213563 - that value is
      irrational, so those are proved bounds, not the number.

The bounds are PROVED, by integer arithmetic and then verified as
rationals: the low bound raised to the degree is checked to be at
most the radicand, and the next step up is checked to exceed it. No
float touches the computation, and the interval is a fact rather
than a rendering choice. A root that IS rational is given exactly -
sqrt 16 is 4, sqrt 0.25 is 0.5, cbrt -8 is -2 - so the enclosure is
reserved for the numbers that need it.

Three shapes refuse, each for a reason worth stating rather than
papering over: a negative radicand under an even root ("no real
number raised to the power 2 gives -4"), zero to the power zero ("a
convention, not a value - different fields choose differently, and I
will not choose for you"), and an exponent past a thousand ("names a
number with more digits than an answer has"). Units refuse too, and
by the same rule §11.78 already held: raising meters to a power
would name a unit no definition covers.

Both arms run the live pipeline; the baseline arm turns
`_EXACT_RATIONALS` off, so it is the standard implementation
honestly built - same grammar, same refusals, `**` and `math.sqrt`
in binary floating point - and not an absence.

**The criteria, written before the run.**

1. **Gain** - correctness (the exact expected value, or an enclosure
   for an irrational one) must beat the baseline arm by more than
   the seed-to-seed sd.
2. **No wrong number** - a stated value differing from the gate's
   own integer-arithmetic expectation, in the treatment arm, is a
   fail, once.
3. **An irrational root is never named** - a single value given for
   an irrational root in the treatment arm is a fail, once. This is
   the rung's whole claim.
4. **Every enclosure is true** - each bound is checked by the gate
   with exact rational arithmetic: low**degree <= radicand <
   high**degree. One false bracket is a fail, once, and would be
   worse than the float answer it replaces.
5. **The refusals hold** - negative radicands, zero to the zero, and
   oversized exponents refuse in the treatment arm.

**It was run once, and PASSED - on the claim and on all four
lines.**

| arm | correct |
|---|---:|
| floats | 0.702 |
| exact | **1.000** |

**+0.298 at 3.56x seed sd, zero wrong values, zero irrationals
named, zero false bounds, zero refusals missed.** Criterion 4 is the
one that keeps this rung from being a worse answer dressed as a
better one: the gate checked every enclosure itself, in exact
rational arithmetic, and every bracket held. A wrong bracket would
have been a firmer lie than the float.

The two arms, on the same two questions:

    [floats] sqrt 2 = 1.4142135623730951
    [exact]  sqrt 2 is between 1.414213562 and 1.414213563 - that
             value is irrational, so those are proved bounds, not
             the number.

    [floats] 0.1 ^ 2 = 0.010000000000000002
    [exact]  0.1 ^ 2 = 0.01

1.4142135623730951 squared is 2.0000000000000004, so the float arm's
answer is not the square root of two; it is a number near it, given
to seventeen digits with no indication that sixteen of them are a
promise the arithmetic cannot keep. The enclosure is shorter, and it
is true.

One thing the run surfaced about the standard option, kept in the
baseline rather than papered over: Python answers `(-4) ** 0.5` with
a COMPLEX number. The float arm was raising that all the way to the
renderer until the domain error was handled, which is a fair
description of what naive float roots do to a question nobody asked
in the complex plane.

Run it::

    python -m ultraquant.experiments.power_gate
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

__all__ = ["PowerReport", "run_gate"]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PowerReport:
    """The two arms, the four integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Stated values differing from expectation.
        named_irrationals: Irrational roots given a single value.
        false_bounds: Enclosures that do not contain the root.
        refusals_missed: Refusal shapes given an answer.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    named_irrationals: int = 0
    false_bounds: int = 0
    refusals_missed: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _render(value: Fraction) -> str:
    """The gate's own exact renderer."""
    if value.denominator == 1:
        return str(value.numerator)
    rest, twos, fives = value.denominator, 0, 0
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = value.numerator * (10 ** places) // value.denominator
    digits = str(abs(scaled)).rjust(places + 1, "0")
    sign = "-" if scaled < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _is_perfect(number: int, degree: int) -> bool:
    root = round(number ** (1.0 / degree))
    for candidate in (root - 1, root, root + 1):
        if candidate >= 0 and candidate ** degree == number:
            return True
    return False


def _build_world(rng: random.Random):
    """Cases: (question, kind, expected-or-degree-and-radicand)."""
    cases = []

    # Whole powers, in world-varying number.
    for _ in range(rng.choice((2, 3))):
        base = rng.randint(2, 12)
        exponent = rng.randint(2, 6)
        cases.append((f"what is {base} ^ {exponent}?", "value",
                      _render(Fraction(base) ** exponent)))

    # A negative exponent - exact as a fraction or a decimal.
    if rng.random() < 0.8:
        base = rng.choice((2, 4, 5, 8, 10))
        exponent = rng.randint(1, 3)
        cases.append((f"what is {base} ^ -{exponent}?", "value",
                      _render(Fraction(1, base ** exponent))))

    # The spoken forms, which must read as the symbols.
    if rng.random() < 0.7:
        base = rng.randint(2, 20)
        cases.append((f"what is {base} squared?", "value",
                      _render(Fraction(base * base))))

    # A rational root, given exactly.
    for _ in range(rng.choice((1, 2))):
        root = rng.randint(2, 20)
        cases.append((f"what is the square root of {root * root}?",
                      "value", _render(Fraction(root))))

    # An irrational root - the claim.
    for _ in range(rng.choice((2, 3))):
        radicand = rng.randint(2, 200)
        while _is_perfect(radicand, 2):
            radicand = rng.randint(2, 200)
        cases.append((f"what is the square root of {radicand}?",
                      "enclosed", (2, Fraction(radicand))))
    if rng.random() < 0.6:
        radicand = rng.randint(2, 100)
        while _is_perfect(radicand, 3):
            radicand = rng.randint(2, 100)
        cases.append((f"what is the cube root of {radicand}?",
                      "enclosed", (3, Fraction(radicand))))

    # The refusals.
    for question in (
            f"what is the square root of -{rng.randint(2, 50)}?",
            "what is 0 ^ 0?",
            f"what is 2 ^ {rng.randint(1001, 9000)}?"):
        if rng.random() < 0.8:
            cases.append((question, "refuses", None))

    rng.shuffle(cases)
    return cases


def _stated(response: str) -> str:
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - computed", " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def _bounds(response: str):
    if " is between " not in response:
        return None
    tail = response.split(" is between ", 1)[1]
    numbers = _NUMBER_RE.findall(tail)
    if len(numbers) < 2:
        return None
    return Fraction(numbers[0]), Fraction(numbers[1])


def run_gate(seeds: int = 12) -> PowerReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import calculate

    arms = {"floats": False, "exact": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    named_irrationals = 0
    false_bounds = 0
    refusals_missed = 0

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(11600 + seed)
            cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_power_"))
            old = calculate._EXACT_RATIONALS
            calculate._EXACT_RATIONALS = flag
            try:
                session = build_session(root, seed=seed)
                hits = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    stated = _stated(response)
                    if kind == "value":
                        if stated == expected:
                            hits += 1
                        elif stated and flag:
                            wrong += 1
                    elif kind == "enclosed":
                        degree, radicand = expected
                        pair = _bounds(response)
                        if pair is None:
                            if flag and stated:
                                named_irrationals += 1
                            continue
                        low, high = pair
                        if (low ** degree <= radicand
                                < high ** degree):
                            hits += 1
                        else:
                            false_bounds += 1
                    else:
                        if response.startswith("I can't compute"):
                            hits += 1
                        elif flag:
                            refusals_missed += 1
                scores.append(hits / len(cases))
            finally:
                calculate._EXACT_RATIONALS = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    contrasts = [t - b for t, b in zip(per_seed["exact"],
                                       per_seed["floats"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and named_irrationals == 0 and false_bounds == 0
              and refusals_missed == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - exact powers and roots at "
        f"{correct['exact']:.3f} against the float arm's "
        f"{correct['floats']:.3f} ({delta:+.3f}, {ratio:.2f}x seed "
        f"sd), {wrong} wrong values, {named_irrationals} irrationals "
        f"named, {false_bounds} false bounds, {refusals_missed} "
        f"refusals missed")
    return PowerReport(
        passes=passes, correct=correct, wrong=wrong,
        named_irrationals=named_irrationals, false_bounds=false_bounds,
        refusals_missed=refusals_missed, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"floats       {report.correct['floats']:.3f}")
    print(f"exact        {report.correct['exact']:.3f}")
    print(f"wrong values: {report.wrong}")
    print(f"irrationals named: {report.named_irrationals}")
    print(f"false bounds: {report.false_bounds}")
    print(f"refusals missed: {report.refusals_missed}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
