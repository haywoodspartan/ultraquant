"""Arithmetic over what is believed. The quantity gate.

§11.76 could evaluate what the speaker wrote. It could not use what
the system knows: "what is the tower height times 3?" answered
"tower height is 300 meters" - a recall of one operand offered as
the answer to an expression, which is §11.29's coverage failure
wearing an operator. The question's tokens were covered; the
question was not.

§11.78 makes an operand any of three things - a written number, a
written quantity ("300 meters"), or a held belief ("the tower
height") - and lets units ride the operators:

- **addition and subtraction** need one unit, reached through §11.42's
  definitions, and the result reads in the LARGER of the two, which
  is the rule the combine path already answers by. A quantity beside
  a plain number refuses: assuming its unit would be invention.
- **multiplication** takes at most one unit. Meters times meters
  would name a unit no definition covers, so it refuses rather than
  invent square meters.
- **division** of two quantities in one family gives a plain RATIO,
  which is the useful and honest answer ("twice as long"); a plain
  number over a quantity refuses for the same reason multiplication
  does.

Conversions apply the definitions EXACTLY. §11.42's table stores its
factors as floats, and a float is the wrong shape for a definition:
0.3048 is exactly what a foot is in meters, but the double nearest
0.3048 is not. The factors are read back through their decimal
spelling, which restores them.

A result resting on a belief is a DERIVATION, not a calculation, so
it is marked inferred rather than computed, names its premises, and
carries the least confident of them.

Placement is the safety argument. §11.76's literal reader is
provably safe anywhere - an expression of nothing but numbers cannot
be another branch's question - but this one reads WORDS, and words
are what every other branch is made of. So it runs after every
word-reading question form and before recall: late enough to steal
nothing, early enough that an expression is never answered by one of
its own operands. A run where nothing resolves is not arithmetic at
all and falls through untouched ("what is the plus size?" reads
"plus" as an operator and must not refuse).

Both arms run the live pipeline; the baseline arm turns
`_QUANTITY_ARITHMETIC` off, leaving §11.76's literal reader in
place, so the contrast is exactly this rung and not the whole
subject. Worlds mix, in world-varying proportions: belief operands,
written quantities, two-belief sums across units, ratios, the four
refusal kinds (cross-family, unit times unit, unheld operand, denial
and non-numeric operands), ordinary questions that must not move,
and - at about a fifth of the questions - cases NOBODY can answer.

Those last are this rung's own ceiling, scored in rather than
excused. §11.55 derives a comparative operand the store does not
hold outright ("is the west tower taller than the south tower?"
reaches through `west tower kind is spirekind`); this rung does not.
An arithmetic operand must be held, so "what is the west tower
height * 2?" refuses by naming the key it looked for - honest, and
not the answer. Both arms score zero there, which is what gives the
contrast something to vary against (§11.35's arithmetic) and puts
the next rung on the record instead of in a promise.

**The criteria, written before the run.**

1. **Gain** - answer correctness (the exact expected value AND its
   unit, or a refusal where one is owed) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **No wrong number** - a computed value that differs from the
   gate's independently-walked expectation, in the treatment arm, is
   a fail, once.
3. **No undefined answer** - a refusal case given a number in the
   treatment arm is a fail, once. Absence of a unit definition is
   not a licence to invent one.
4. **One matrix, one voice** - for a two-belief sum, "X + Y" and
   "the sum of X and Y" must agree on both the number and the unit.
   §11.75's lesson as a measured line: a store with two voices for
   one arithmetic has a bug in one of them.
5. **The branch steals nothing** - ordinary questions answer
   identically in both arms.

**It took two runs, and the first was the harness's fault.**

Run one failed criterion 3 with two undefined answers - and the
mechanism was right both times. The worlds drew their "cross-family"
refusal cases by picking a different ATTRIBUTE, and "height" and
"length" are both measured in meters, so two of those cases were
same-family sums that correctly computed while the gate scored them
as inventions. An attribute is not a family; the table carries a
physical-family tag now. Run two, unchanged in the mechanism:

| arm | correct |
|---|---:|
| literals | 0.000 |
| quantities | **0.800** |

**+0.800 at 18.46x seed sd, zero wrong values, zero undefined
answers, zero voice splits against the combine path, zero ordinary
questions moved.** The 0.200 is exactly the derive ceiling, scored
zero for both arms. Two lines are worth naming beyond the headline.

**One matrix, one voice.** For every two-belief sum the gate asked
the same arithmetic twice - "X + Y" through this rung and "the sum
of X and Y" through §11.42's combine path - and required agreement
on the number AND the unit. Zero splits. §11.75 was caught because
two branches answered one question differently; this line makes that
class of bug measurable instead of noticeable.

**The refusals name what is missing, not that something is.** Across
the refusal kinds the answers read "no definition I hold connects
meters and tonnes", "multiplying meters by meters would name a unit
no definition I hold covers", "I hold only a denial for 'gate
height'", "'mill material' holds no number (oak)", and "I hold
nothing for 'north gate height'" - five different sentences for five
different absences, none of them a number.

Run it::

    python -m ultraquant.experiments.quantity_gate
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

__all__ = ["QuantityReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]

#: (attribute, small unit, large unit, factor, physical family) - the
#: gate's own table, so an expectation never comes from the code
#: under test. The family tag is load-bearing and was learned the
#: hard way: the first run drew its "cross-family" refusal cases by
#: picking a different ATTRIBUTE, and "height" and "length" are both
#: measured in meters - so two of those cases were same-family sums
#: that correctly computed, and the gate scored the mechanism wrong
#: for being right. An attribute is not a family.
_FAMILIES = [
    ("height", "meters", "kilometers", 1000, "length"),
    ("length", "meters", "kilometers", 1000, "length"),
    ("weight", "kilograms", "tonnes", 1000, "mass"),
    ("span", "seconds", "minutes", 60, "time"),
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class QuantityReport:
    """The two arms, the four integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Computed values that differ from expectation.
        undefined_answered: Refusal cases given a number.
        voice_split: Two-belief sums where "X + Y" and "the sum of X
            and Y" disagreed on value or unit.
        stolen: Ordinary questions whose answer moved between arms.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    undefined_answered: int = 0
    voice_split: int = 0
    stolen: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _render(value: Fraction, unit: str) -> str:
    """The gate's own exact renderer, walked from its own numbers."""
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


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected-or-None)."""
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    attr, small, large, factor, family = rng.choice(_FAMILIES)
    other = rng.choice([f for f in _FAMILIES if f[4] != family])
    statements = []
    cases = []

    a, b, c = names[0], names[1], names[2]
    va, vb = rng.sample(range(100, 900), 2)
    vc = rng.randint(2, 9)
    statements.append(f"the {a} {attr} is {va} {small}")
    statements.append(f"the {b} {attr} is {vb} {small}")
    statements.append(f"the {c} {attr} is {vc} {large}")

    # A belief scaled by a written number.
    scale = rng.randint(2, 9)
    cases.append((f"what is the {a} {attr} times {scale}?", "value",
                  _render(Fraction(va * scale), small)))

    # Two beliefs in one unit; also the one-voice check.
    cases.append((f"what is the {a} {attr} + the {b} {attr}?", "value",
                  _render(Fraction(va + vb), small)))
    cases.append((f"what is the {a} {attr} + the {b} {attr}?", "voice",
                  f"what is the sum of the {a} {attr} and the {b} "
                  f"{attr}?"))

    # Two beliefs across units - the result reads in the larger.
    if rng.random() < 0.8:
        total = Fraction(va, factor) + vc
        cases.append((f"what is the {a} {attr} + the {c} {attr}?",
                      "value", _render(total, large)))

    # A ratio: quantity over quantity is a plain number.
    if rng.random() < 0.7:
        cases.append((f"what is the {c} {attr} / the {a} {attr}?",
                      "value", _render(Fraction(vc * factor, va), "")))

    # Written quantities, no belief involved.
    if rng.random() < 0.8:
        written = rng.randint(100, 900)
        cases.append((f"what is {written} {small} + {vc} {large}?",
                      "value",
                      _render(Fraction(written, factor) + vc, large)))

    # The refusal kinds, in world-varying number.
    odd_attr, odd_unit = other[0], other[1]
    statements.append(f"the {names[3]} {odd_attr} is "
                      f"{rng.randint(10, 90)} {odd_unit}")
    statements.append(f"the {names[4]} material is oak")
    statements.append(f"the {names[5]} {attr} is not {vb} {small}")
    refusals = [
        f"what is the {a} {attr} + the {names[3]} {odd_attr}?",
        f"what is {va} {small} * {vb} {small}?",
        f"what is the {names[6]} {attr} * 2?",
        f"what is the {names[4]} material * 2?",
        f"what is the {names[5]} {attr} * 2?",
    ]
    for question in rng.sample(refusals, rng.choice((2, 3))):
        cases.append((question, "refuses", None))

    # The ceiling, scored in: an operand reachable only through a
    # chain. §11.55 derives one for a comparative; this rung does
    # not, so neither arm answers these and their varying share is
    # what the contrast varies against.
    kind_name = f"{rng.choice(('spire', 'vault', 'arch'))}kind"
    chained = names[7]
    chain_value = rng.randint(100, 900)
    statements.append(f"the {chained} kind is {kind_name}")
    statements.append(f"{kind_name} {attr} is {chain_value} {small}")
    for _ in range(rng.choice((1, 2))):
        multiplier = rng.randint(2, 9)
        cases.append((f"what is the {chained} {attr} * {multiplier}?",
                      "value",
                      _render(Fraction(chain_value * multiplier),
                              small)))

    # Ordinary questions: must not move between arms.
    for question in (f"what is the {a} {attr}?",
                     f"what is the {names[4]} material?",
                     f"which is the tallest?"):
        cases.append((question, "ordinary", None))

    rng.shuffle(cases)
    return statements, cases


def _asserted_value(response: str) -> str:
    """The value the answer states, or "" if it states none."""
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - computed", " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def run_gate(seeds: int = 12) -> QuantityReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    arms = {"literals": False, "quantities": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    undefined_answered = 0
    voice_split = 0
    ordinary: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(7300 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_quant_"))
            old = thoughts._QUANTITY_ARITHMETIC
            thoughts._QUANTITY_ARITHMETIC = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                scored = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    refused = response.startswith("I can't compute")
                    if kind == "value":
                        scored += 1
                        stated = _asserted_value(response)
                        if stated == expected:
                            hits += 1
                        elif stated and flag:
                            wrong += 1
                    elif kind == "refuses":
                        scored += 1
                        if refused:
                            hits += 1
                        elif flag and _asserted_value(response):
                            undefined_answered += 1
                    elif kind == "voice":
                        if not flag:
                            continue
                        other, _t2 = run_pipeline(expected, session)
                        mine = _asserted_value(response)
                        theirs = _asserted_value(other)
                        if not mine or not theirs:
                            voice_split += 1
                        elif (_numbers(mine) != _numbers(theirs)
                              or mine.split()[-1] != theirs.split()[-1]):
                            voice_split += 1
                    else:
                        ordinary.setdefault((seed, question),
                                            {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                thoughts._QUANTITY_ARITHMETIC = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    stolen = sum(1 for by_arm in ordinary.values()
                 if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["quantities"],
                                       per_seed["literals"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and undefined_answered == 0 and voice_split == 0
              and stolen == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - quantity arithmetic at "
        f"{correct['quantities']:.3f} against {correct['literals']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {wrong} wrong values, "
        f"{undefined_answered} undefined answers, {voice_split} voice "
        f"splits against the combine path, {stolen} ordinary "
        f"questions moved")
    return QuantityReport(
        passes=passes, correct=correct, wrong=wrong,
        undefined_answered=undefined_answered, voice_split=voice_split,
        stolen=stolen, delta=delta, seed_sd=seed_sd, margin_ratio=ratio,
        reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm           correct")
    print(f"literals       {report.correct['literals']:.3f}")
    print(f"quantities     {report.correct['quantities']:.3f}")
    print(f"wrong values: {report.wrong}")
    print(f"undefined answered: {report.undefined_answered}")
    print(f"voice splits vs the combine path: {report.voice_split}")
    print(f"ordinary questions moved: {report.stolen}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
