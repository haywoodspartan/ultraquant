"""The arithmetic capstone: every math form at density.

The §11.76-§11.84 arc built arithmetic one rung at a time - exact
evaluation, quantities and beliefs as operands, reached operands,
percentages, powers and enclosed roots, comparisons, polar
arithmetic, rounding on request - and every one of those gates ran
at a small world, which is where mechanisms are debuggable. §11.37's
lesson governs what has to happen next: **found is not believed, and
clean-room success means nothing until density has its say.** This
is that run, in §11.74's capstone tradition - one colliding world of
thousands of facts, every arithmetic form asked against it, floors
and controls and prices.

Two things are measured and one is priced. Measured: correctness
floors per form, and FABRICATION at absolute zero across every
refusal family the arc built - an operand nobody holds, an operand
that is a denial, an operand that holds no number, a unit multiplied
by a unit, a division by zero, a percentage of nothing, a relation
no vocabulary covers, and zero to the power zero. Priced: latency
per form, reported and not gated.

The floors are 1.000, and deliberately so. Arithmetic is
deterministic: an expression that evaluates correctly in a
twenty-fact world evaluates correctly in a ten-thousand-fact one,
because nothing about the store enters the computation. What density
CAN break is the part that touches the store - resolving a belief
operand, reaching one through a chain, ranking a comparison - so a
miss here is a mechanism bug rather than a density effect, and the
floors say so by leaving no room.

**The criteria, written before the run.**

1. **Fabrication is zero at scale.** Every refusal family must be
   refused every time. One number where a refusal is owed is a fail.
2. **The floors hold at 1.000** for every form: literal arithmetic,
   percentages, powers, exact roots, quantity arithmetic, reached
   operands, comparisons, polar arithmetic, and rounding.
3. **Every enclosure is true.** An irrational root's bounds are
   checked by the gate in exact rational arithmetic; one false
   bracket is a fail, and would be worse than the float it replaced.
4. **The price is named.** Mean per-question latency reported for
   every form. Reported, never gated.

SKIPS nothing and contrasts nothing: this is a floors-and-controls
gate in the §11.41 containment tradition - the arms already ran, one
gate at a time.

**It was run once, and PASSED - every floor, every control, on the
first run.**

At 5,818 facts:

| form | floor | ms/query |
|---|---:|---:|
| literal | 1.000 | 38.9 |
| percent | 1.000 | 14.9 |
| power | 1.000 | 5.7 |
| root exact | 1.000 | 3.6 |
| root enclosed | 1.000 | 3.2 |
| quantity | 1.000 | 3.5 |
| reached | 1.000 | 50.5 |
| comparison | 1.000 | 2.4 |
| polar arithmetic | 1.000 | 2.0 |
| rounding | 1.000 | 2.2 |
| controls | - | 30.6 |

**Ten forms at 1.000, zero fabrications across all eight refusal
families, zero false bounds.** The arc holds at density: an operand
nobody holds, an operand that is a denial, an operand that holds no
number, a unit times a unit, a division by zero, a percentage of
nothing, a relation no vocabulary covers, and zero to the power zero
were asked eighty times between them and answered with a number
zero times.

**The price line found something, which is what price lines are
for.** Literal arithmetic - two additions and a multiplication over
written numbers, touching no belief at all - cost 38.9 ms, which is
absurd on its face. It was chased down rather than reported:

    calculate.evaluate alone :    0.016 ms
    memory.find_facts alone  :    0.190 ms
    whole pipeline           :   26.517 ms

and then per stage, at 4,000 facts:

    Recall                 28.213 ms
    Learn                   3.063 ms
    Reason                  0.180 ms
    Route / Perceive / Respond  under 0.03 ms each

So the arithmetic is a rounding error in its own cost, retrieval is
a rounding error too, and **the Recall stage is ~90% of the price of
a question it cannot possibly help with** - "what is 12 + 3 * 4?"
names no belief, and Recall runs before any branch decides whether
the question needs recall. That is a measured opportunity, not a
change: it is registered here with its number, in the §11.45
tradition where the frontier clock earned its 3.2x only after parity
was proved.

Run it::

    python -m ultraquant.experiments.mathdensity_gate
"""

from __future__ import annotations

import random
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["MathDensityReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great", "grand", "little", "upper",
             "lower", "inner", "outer", "far", "near", "first",
             "second"]
_MIDDLES = ["stone", "iron", "river", "hill", "market", "abbey",
            "castle", "harbour", "garden", "forest", "valley",
            "meadow", "bridge", "chapel", "manor", "orchard",
            "spring", "summit", "vault", "arch"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome",
          "mill", "keep", "wall", "hall", "arch", "pier", "vault",
          "court", "lodge", "barn", "shed", "stair", "well"]
_ATTRIBUTES = ["height", "length", "span", "reach", "rise"]
_KINDS = ["spirekind", "vaultkind", "archkind", "slabkind",
          "pierkind", "domekind", "beamkind", "trusskind"]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class MathDensityReport:
    """Floors, controls and prices for one dense arithmetic session.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        facts: Facts actually stored.
        floors: Form -> correctness.
        prices: Form -> mean ms per question.
        fabrications: Refusal family -> numbers given where a
            refusal was owed (all must be zero).
        false_bounds: Enclosures not containing their root.
        reason: Plain-language verdict.
    """

    passes: bool
    facts: int = 0
    floors: dict = field(default_factory=dict)
    prices: dict = field(default_factory=dict)
    fabrications: dict = field(default_factory=dict)
    false_bounds: int = 0
    reason: str = ""


def _entities(rng: random.Random, count: int) -> list[str]:
    """Colliding three-word names, in the §11.37 style."""
    names = []
    seen = set()
    guard = 0
    while len(names) < count and guard < count * 40:
        guard += 1
        name = (f"{rng.choice(_PREFIXES)} {rng.choice(_MIDDLES)} "
                f"{rng.choice(_NOUNS)}")
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _stated(response: str) -> str:
    """The value an answer states, or "" if it states none."""
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - rounded", " - exact at", " - computed",
                " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def _asserted(response: str) -> bool:
    """Whether the answer gives a number or a verdict at all."""
    return bool(_stated(response)) or response.startswith(("Yes", "No"))


def run_gate(n_facts: int = 6000, seed: int = 0) -> MathDensityReport:
    """Build one dense world and ask it every arithmetic form."""
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    rng = random.Random(seed)
    root = Path(tempfile.mkdtemp(prefix="uq_mathdense_"))
    report = MathDensityReport(passes=False)
    try:
        session = build_session(root, seed=seed)
        memory = session.memory

        # --- the filler mass: colliding names with united values.
        names = _entities(rng, max(60, n_facts // 4))
        heights: dict = {}
        stored = 0
        for entity in names:
            for attr in _ATTRIBUTES:
                value = rng.randrange(25, 900, 25)
                memory.remember_fact(f"{entity} {attr}",
                                     f"{value} meters", confidence=0.6)
                if attr == "height":
                    heights[entity] = value
                stored += 1
                if stored >= n_facts - 200:
                    break
            if stored >= n_facts - 200:
                break

        # --- chains: an entity whose height is only reachable.
        reached = {}
        for kind in _KINDS:
            value = rng.randrange(100, 900, 25)
            memory.remember_fact(f"{kind} height", f"{value} meters",
                                 confidence=0.6)
            stored += 1
            holder = _entities(rng, 1)[0]
            while holder in heights:
                holder = _entities(rng, 1)[0]
            memory.remember_fact(f"{holder} kind", kind,
                                 confidence=0.6)
            stored += 1
            reached[holder] = value

        # --- the decoy families every refusal must survive.
        denial_entity = _entities(rng, 1)[0]
        memory.remember_fact(f"{denial_entity} height", "40 meters",
                             confidence=0.6, negated=True)
        wordy_entity = _entities(rng, 1)[0]
        memory.remember_fact(f"{wordy_entity} height", "oak",
                             confidence=0.6)
        ghost = "phantom unlikely spire"
        stored += 2
        report.facts = stored

        forms: dict = {}
        prices: dict = {}
        fabrications = {name: 0 for name in
                        ("unheld operand", "denial operand",
                         "non-numeric operand", "unit times unit",
                         "division by zero", "dangling percent",
                         "unknown relation", "zero to the zero")}
        false_bounds = 0

        def ask(question: str):
            start = time.perf_counter()
            response, _t = run_pipeline(question, session)
            return response, (time.perf_counter() - start) * 1000.0

        def score(form: str, questions, check):
            hits = 0
            spent = []
            for question, expected in questions:
                response, ms = ask(question)
                spent.append(ms)
                if check(response, expected):
                    hits += 1
            forms[form] = hits / len(questions)
            prices[form] = statistics.mean(spent)

        held = [name for name in heights if " " in name][:400]
        sample = rng.sample(held, 20)
        reached_sample = list(reached.items())

        # 1. literal arithmetic, precedence included
        literal = []
        for _ in range(20):
            a, b, c = (rng.randint(2, 40), rng.randint(2, 40),
                       rng.randint(2, 12))
            literal.append((f"what is {a} + {b} * {c}?",
                            str(a + b * c)))
        score("literal", literal,
              lambda r, e: _stated(r) == e)

        # 2. percentages
        percents = []
        for _ in range(20):
            pct = rng.randrange(5, 95, 5)
            amount = rng.randrange(20, 900, 20)
            percents.append((f"what is {pct}% of {amount}?",
                             _render(Fraction(pct * amount, 100))))
        score("percent", percents, lambda r, e: _stated(r) == e)

        # 3. powers
        powers = []
        for _ in range(20):
            base, exponent = rng.randint(2, 12), rng.randint(2, 6)
            powers.append((f"what is {base} ^ {exponent}?",
                           str(base ** exponent)))
        score("power", powers, lambda r, e: _stated(r) == e)

        # 4. exact roots
        roots = []
        for _ in range(20):
            root_of = rng.randint(2, 40)
            roots.append((f"what is the square root of "
                          f"{root_of * root_of}?", str(root_of)))
        score("root exact", roots, lambda r, e: _stated(r) == e)

        # 5. enclosed roots - and every bracket checked
        enclosed = []
        for _ in range(20):
            radicand = rng.randint(2, 400)
            while _perfect(radicand):
                radicand = rng.randint(2, 400)
            enclosed.append((f"what is the square root of "
                             f"{radicand}?", radicand))

        def _bounded(response: str, radicand: int) -> bool:
            nonlocal false_bounds
            if " is between " not in response:
                return False
            numbers = _NUMBER_RE.findall(
                response.split(" is between ", 1)[1])[:2]
            if len(numbers) < 2:
                return False
            low, high = Fraction(numbers[0]), Fraction(numbers[1])
            if not (low ** 2 <= radicand < high ** 2):
                false_bounds += 1
                return False
            return True

        score("root enclosed", enclosed, _bounded)

        # 6. quantity arithmetic over held beliefs
        quantities = []
        for entity in sample:
            scale = rng.randint(2, 9)
            quantities.append((f"what is the {entity} height * "
                               f"{scale}?",
                               f"{heights[entity] * scale} meters"))
        score("quantity", quantities, lambda r, e: _stated(r) == e)

        # 7. reached operands
        reaching = []
        for holder, value in reached_sample:
            scale = rng.randint(2, 9)
            reaching.append((f"what is the {holder} height * "
                             f"{scale}?", f"{value * scale} meters"))
        score("reached", reaching, lambda r, e: _stated(r) == e)

        # 8. comparisons against a written quantity
        comparisons = []
        for entity in sample:
            against = rng.randrange(25, 900, 25)
            truth = ("Yes" if heights[entity] > against else "No")
            comparisons.append((f"is the {entity} height greater "
                                f"than {against} meters?", truth))
        score("comparison", comparisons,
              lambda r, e: r.startswith(e))

        # 9. polar arithmetic
        polar = []
        for _ in range(20):
            a, b = rng.randint(2, 60), rng.randint(2, 60)
            claimed = (a + b if rng.random() < 0.5
                       else a + b + rng.randint(1, 4))
            polar.append((f"is {a} + {b} = {claimed}?",
                          "Yes" if claimed == a + b else "No"))
        score("polar arithmetic", polar,
              lambda r, e: r.startswith(e))

        # 10. rounding on request
        rounding = []
        for _ in range(20):
            top = rng.randint(10, 900)
            bottom = rng.choice((3, 6, 7, 9, 11))
            exact = Fraction(top, bottom)
            rounding.append((f"what is {top} / {bottom} to 2 decimal "
                             f"places?", _fixed(_round2(exact), 2)))
        score("rounding", rounding, lambda r, e: _stated(r) == e)

        # --- the controls: every refusal family, every time.
        controls = [
            ("unheld operand",
             [f"what is the {ghost} height * 2?" for _ in range(10)]),
            ("denial operand",
             [f"what is the {denial_entity} height * {n}?"
              for n in range(2, 12)]),
            ("non-numeric operand",
             [f"what is the {wordy_entity} height * {n}?"
              for n in range(2, 12)]),
            ("unit times unit",
             [f"what is {n} meters * {n + 1} meters?"
              for n in range(2, 12)]),
            ("division by zero",
             [f"what is {n} / 0?" for n in range(2, 12)]),
            ("dangling percent",
             [f"what is {n * 10} + 20%?" for n in range(2, 12)]),
            ("unknown relation",
             [f"is the {sample[0]} height wider than {n} meters?"
              for n in range(2, 12)]),
            ("zero to the zero",
             ["what is 0 ^ 0?" for _ in range(10)]),
        ]
        control_prices = []
        for family, questions in controls:
            for question in questions:
                response, ms = ask(question)
                control_prices.append(ms)
                if _asserted(response):
                    fabrications[family] += 1
        prices["controls"] = statistics.mean(control_prices)

        report.floors = forms
        report.prices = prices
        report.fabrications = fabrications
        report.false_bounds = false_bounds
        floors_hold = all(value >= 1.0 for value in forms.values())
        clean = all(count == 0 for count in fabrications.values())
        report.passes = floors_hold and clean and false_bounds == 0
        report.reason = (
            f"{'PASS' if report.passes else 'FAIL'} - "
            f"{report.facts} facts; "
            f"{sum(1 for v in forms.values() if v >= 1.0)}/"
            f"{len(forms)} forms at 1.000; "
            f"{sum(fabrications.values())} fabrications across "
            f"{len(fabrications)} refusal families; "
            f"{false_bounds} false bounds")
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _perfect(number: int) -> bool:
    import math

    return math.isqrt(number) ** 2 == number


def _round2(value: Fraction) -> Fraction:
    shifted = value * 100
    floor = shifted.numerator // shifted.denominator
    remainder = shifted - floor
    if remainder > Fraction(1, 2):
        floor += 1
    elif remainder == Fraction(1, 2) and value >= 0:
        floor += 1
    return Fraction(floor, 100)


def _fixed(value: Fraction, places: int) -> str:
    scaled = value * (10 ** places)
    whole = scaled.numerator // scaled.denominator
    digits = str(abs(whole)).rjust(places + 1, "0")
    sign = "-" if value < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _render(value: Fraction) -> str:
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
    return _fixed(value, max(twos, fives))


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"world: {report.facts} facts")
    print()
    print("form                floor    ms/query")
    for form, floor in report.floors.items():
        print(f"{form:<18} {floor:.3f}   "
              f"{report.prices[form]:8.2f}")
    print(f"{'controls':<18}   -      "
          f"{report.prices['controls']:8.2f}")
    print()
    for family, count in report.fabrications.items():
        print(f"fabrications, {family:<22} {count}")
    print(f"false bounds: {report.false_bounds}")
    print()
    print(report.reason)
