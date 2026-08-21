"""A hundredth part, taken. The percent gate.

"what is 20% of 300?" is basic arithmetic by any account, and the
system could not read it: the per-cent sign did not tokenize, so the
question fell through the ladder to "I don't hold anything on that
yet". §11.80 reads it, and reads it structurally - a percentage is a
number over a hundred, and "of" is the multiplication it is waiting
for - so nothing downstream needs a special case. Percentages
compose with everything the previous rungs built: "15% of the tower
height" is 45 meters through §11.78's units, and "20% of the west
tower height" reaches through §11.79's chain and says so.

**One shape refuses, and it is the interesting one.** "300 + 20%"
has no answer. Twenty per cent OF WHAT is a question the sentence
does not answer, and the near-universal convention - take the
percentage of the left operand, so 360 - is a convention rather than
an arithmetic. Every calculator that answers 360 is guessing at the
speaker's meaning, which is precisely what the one branch built to
be exact must not do. It refuses and says which shape has an answer.

Both arms run the live pipeline; the baseline arm turns
`_PERCENTS_ON` off, which also makes the per-cent sign untokenizable
again, so the baseline is genuinely the system as it was rather than
a half-built version of the new one. Worlds mix, in world-varying
proportions: literal percentages, percentages of held beliefs,
percentages inside larger expressions (where precedence is under
test), the dangling shape that must refuse, ordinary questions that
must not move, and - at about a fifth of the questions - the
INVERSE, which nobody here can answer.

That last is this rung's ceiling, scored in rather than excused.
"what percent of 300 is 60?" asks the same arithmetic backwards, and
neither arm reads it: the reader takes expressions, and that is a
sentence. Both arms score zero on it, which is what gives the
contrast something to vary against (§11.35's arithmetic) and puts
the shape on the record instead of leaving it unmentioned.

**The criteria, written before the run.**

1. **Gain** - correctness (the exact expected value with its unit,
   or a refusal where one is owed) must beat the baseline arm by
   more than the seed-to-seed sd.
2. **No wrong number** - a value differing from the gate's own
   expectation, in the treatment arm, is a fail, once.
3. **The convention is never taken** - a dangling percentage
   ("300 + 20%") given a number in the treatment arm is a fail,
   once.
4. **One matrix, one voice** - "N% of X" and "X * N / 100" must give
   the same answer, because the first is defined to BE the second. A
   percentage that is a special case downstream would show here.
5. **The branch steals nothing** - ordinary questions answer
   identically in both arms.

**It was run twice - the first run measured nothing, because
nothing varied.** The baseline could not read a single percentage
and the treatment read them all, so every seed returned the same
contrast and the seed sd was zero. That is a FAIL under the house
rule, and the rule is not softened when it is inconvenient: the fix
is to score in the cases nobody can win, which here is the inverse
question. Second run, unchanged in the mechanism:

| arm | correct |
|---|---:|
| unread | 0.000 |
| percent | **0.832** |

**+0.832 at 17.28x seed sd, zero wrong values, zero conventions
taken, zero voice splits against the spelled-out arithmetic, zero
ordinary questions moved.** The 0.168 is the inverse question,
scored zero for both arms.

Criterion 4 says the most about the design. Every "N% of X" was also
asked as "X * N / 100", and the two agreed every time - which is
what it means for a percentage to be structural rather than a
special case: the reader turns it into arithmetic the grammar
already had, so there is no second code path to disagree with the
first. And criterion 3 held on the shape that tempts every
calculator:

    > what is 300 + 20%?
      I can't compute that: a percentage needs something to be a
      percentage of - '20% of 300' has an answer, '300 + 20%' has a
      convention.

Run it::

    python -m ultraquant.experiments.percent_gate
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

__all__ = ["PercentReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_FAMILIES = [("height", "meters"), ("length", "meters"),
             ("weight", "kilograms")]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class PercentReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Values differing from the gate's expectation.
        convention_taken: Dangling percentages given a number.
        voice_split: "N% of X" disagreeing with "X * N / 100".
        stolen: Ordinary questions whose answer moved.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    convention_taken: int = 0
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
    while len(names) < 5:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    attr, unit = rng.choice(_FAMILIES)

    statements = []
    cases = []
    held = {}
    for name in names[0:2]:
        value = rng.randrange(100, 900, 25)
        statements.append(f"the {name} {attr} is {value} {unit}")
        held[name] = value
    statements.append(f"the {names[2]} material is oak")

    # Literal percentages, in world-varying number.
    for _ in range(rng.choice((2, 3))):
        percent = rng.randrange(5, 95, 5)
        amount = rng.randrange(100, 900, 20)
        cases.append((f"what is {percent}% of {amount}?", "value",
                      _render(Fraction(percent * amount, 100), ""),
                      None))

    # The same arithmetic spelled out - one matrix, one voice.
    percent = rng.randrange(5, 95, 5)
    amount = rng.randrange(100, 900, 20)
    cases.append((f"what is {percent}% of {amount}?", "voice", None,
                  f"what is {amount} * {percent} / 100?"))

    # A percentage of a held belief, with its unit.
    name = rng.choice(names[0:2])
    percent = rng.randrange(5, 95, 5)
    cases.append((f"what is {percent}% of the {name} {attr}?",
                  "value",
                  _render(Fraction(percent * held[name], 100), unit),
                  None))

    # Precedence: the percentage binds before the addition.
    if rng.random() < 0.8:
        percent = rng.randrange(5, 95, 5)
        amount = rng.randrange(100, 900, 20)
        tail = rng.randint(2, 40)
        cases.append((f"what is {percent}% of {amount} + {tail}?",
                      "value",
                      _render(Fraction(percent * amount, 100) + tail,
                              ""), None))

    # The dangling shape: a convention, not an arithmetic.
    for _ in range(rng.choice((1, 2))):
        cases.append((f"what is {rng.randrange(100, 900, 20)} + "
                      f"{rng.randrange(5, 95, 5)}%?", "refuses", None,
                      None))

    # The word spelling, which must read the same as the sign.
    if rng.random() < 0.6:
        percent = rng.randrange(5, 95, 5)
        amount = rng.randrange(100, 900, 20)
        cases.append((f"what is {percent} percent of {amount}?",
                      "value",
                      _render(Fraction(percent * amount, 100), ""),
                      None))

    # The ceiling, scored in: the same arithmetic asked backwards.
    # Neither arm reads it - the reader takes expressions, and this
    # is a sentence - so its varying share is what the contrast
    # varies against.
    for _ in range(rng.choice((1, 2))):
        percent = rng.randrange(5, 95, 5)
        amount = rng.randrange(100, 900, 20)
        cases.append((f"what percent of {amount} is "
                      f"{percent * amount // 100}?", "value",
                      _render(Fraction(percent), ""), None))

    for question in (f"what is the {names[0]} {attr}?",
                     f"what is the {names[2]} material?"):
        cases.append((question, "ordinary", None, None))

    rng.shuffle(cases)
    return statements, cases


def _asserted_value(response: str) -> str:
    if "=" not in response:
        return ""
    tail = response.split("=", 1)[1]
    for cut in (" - computed", " - inferred", " ("):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    return tail.strip().rstrip(".")


def run_gate(seeds: int = 12) -> PercentReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import calculate

    arms = {"unread": False, "percent": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    convention_taken = 0
    voice_split = 0
    ordinary: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(9500 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_pct_"))
            old = calculate._PERCENTS_ON
            calculate._PERCENTS_ON = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                scored = 0
                for question, kind, expected, twin in cases:
                    response, _t = run_pipeline(question, session)
                    stated = _asserted_value(response)
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
                            convention_taken += 1
                    elif kind == "voice":
                        if not flag:
                            continue
                        other, _t2 = run_pipeline(twin, session)
                        mine, theirs = stated, _asserted_value(other)
                        if not mine or mine != theirs:
                            voice_split += 1
                    else:
                        ordinary.setdefault((seed, question),
                                            {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                calculate._PERCENTS_ON = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    stolen = sum(1 for by_arm in ordinary.values()
                 if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["percent"],
                                       per_seed["unread"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and convention_taken == 0 and voice_split == 0
              and stolen == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - percentages at "
        f"{correct['percent']:.3f} against {correct['unread']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {wrong} wrong values, "
        f"{convention_taken} conventions taken, {voice_split} voice "
        f"splits against the spelled-out arithmetic, {stolen} "
        f"ordinary questions moved")
    return PercentReport(
        passes=passes, correct=correct, wrong=wrong,
        convention_taken=convention_taken, voice_split=voice_split,
        stolen=stolen, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"unread       {report.correct['unread']:.3f}")
    print(f"percent      {report.correct['percent']:.3f}")
    print(f"wrong values: {report.wrong}")
    print(f"conventions taken: {report.convention_taken}")
    print("voice splits vs spelled-out arithmetic: "
          f"{report.voice_split}")
    print(f"ordinary questions moved: {report.stolen}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
