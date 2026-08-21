"""A coin that always said No. The comparison gate.

Found by asking the shipped system a question in the words people
actually use::

    stored: tower height 300 meters, keep height 100 meters

    > is the tower height greater than 200 meters?
      No - tower height is 300 meters, not greater than 200 meters
    > is the tower height greater than 500 meters?
      No - tower height is 300 meters, not greater than 500 meters
    > is the tower height less than 500 meters?
      No - tower height is 300 meters, not less than 500 meters

Every one No, in every direction, and three of those six probes were
wrong. §11.54 owns comparatives and had ended exactly this failure
for the words in its vocabulary - "taller than" answered correctly
throughout the same probe - but "greater", "more" and "less" were
not in it, so the questions fell past it to the polar branch, which
compared the CLAIM to the stored VALUE as strings. "300 meters" is
never the string "greater than 200 meters", so the answer was always
No. §11.54's coin had not been removed; it had been narrowed to the
words nobody had thought of yet.

The fix is two things, and the second is the one that lasts:

1. The missing words join the vocabulary - greater, more, less,
   fewer - so the comparisons people actually write are compared.
2. **A comparison the comparative branch declined is REFUSED, not
   answered.** If "than" is in the claim and §11.54 did not take the
   question, the relation is one this system cannot compare by, and
   it says so: "I can't tell: I don't know how to compare by
   'wider'. I hold that tower height is 300 meters." That line
   generalises past any vocabulary list, which the first fix alone
   never could.

A third thing came free once comparisons were being compared: the
right-hand side may be a WRITTEN quantity rather than a key. "is the
tower height greater than 200 meters?" needs no stored `200 meters`
- the number is in the question - and §11.78's reader already knows
how to read one, converting units where a definition connects them.

Both arms run the live pipeline; the baseline arm turns
`_COMPARISON_GUARD` off AND takes the four words back out of the
vocabulary, so it is the system exactly as it shipped. Worlds mix,
in world-varying proportions: a belief against a written quantity in
both directions, a belief against a belief, comparisons needing a
unit conversion, comparisons in §11.54's OLD vocabulary (which must
not move), an unknown comparative word (which must refuse), an
unknown unit (which must refuse in both arms), and plain polar
questions (which must not move).

**The criteria, written before the run.**

1. **Gain** - verdict correctness against the gate's own arithmetic
   must beat the baseline arm by more than the seed-to-seed sd.
2. **No wrong verdict** - a Yes or No differing from the gate's
   ground truth, in the treatment arm, is a fail, once.
3. **An unknown relation is never a verdict** - "wider than" given a
   Yes or No in the treatment arm is a fail, once. This is the line
   that outlives the vocabulary.
4. **The old vocabulary does not move** - "taller than" comparisons
   answer identically in both arms.
5. **Plain polar questions do not move** - "is the tower material
   iron?" answers identically in both arms.

**It was run once, and PASSED - on the claim and on all four
lines.**

| arm | correct |
|---|---:|
| coin | 0.383 |
| compared | **1.000** |

**+0.617 at 3.23x seed sd, zero wrong verdicts, zero unknown
relations answered, zero old-vocabulary answers moved, zero plain
polar answers moved.**

The baseline's 0.383 is worth reading properly, because it is not a
score - it is the shape of the bug. A branch that always says No is
right exactly as often as the truth happens to be No, so the number
measures the worlds, not the system. That is what makes this failure
mode dangerous rather than merely wrong: it looks like a system with
opinions, it is right in nearly two cases out of five, and nothing
about a single answer distinguishes a computed No from a No that was
never computed at all.

Criterion 3 is the line that outlives any vocabulary. Adding four
words fixes four words; refusing every "than" the comparative branch
declined fixes the ones nobody has thought of yet:

    > is the tower height wider than the keep height?
      I can't tell: I don't know how to compare by 'wider'. I hold
      that tower height is 300 meters.

And criterion 4 says the fix was additive: every comparison that
already worked answered identically in both arms, so nothing was
traded for the repair.

Run it::

    python -m ultraquant.experiments.comparison_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["ComparisonReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]

#: (attribute, small unit, large unit, factor).
_FAMILIES = [("height", "meters", "kilometers", 1000),
             ("length", "meters", "kilometers", 1000),
             ("weight", "kilograms", "tonnes", 1000)]

#: The words §11.82 added, and the direction each means.
_NEW_WORDS = {"greater": "larger", "more": "larger",
              "less": "smaller", "fewer": "smaller"}

#: §11.54's vocabulary, which must answer identically in both arms.
_OLD_WORDS = {"taller": "larger", "heavier": "larger",
              "shorter": "smaller", "lighter": "smaller"}

#: Words nothing can compare by - a refusal, never a verdict.
_UNKNOWN_WORDS = ["wider", "denser", "prettier"]


@dataclass
class ComparisonReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> verdict correctness over all worlds.
        wrong: Verdicts differing from the gate's ground truth.
        unknown_answered: Unknown relations given a verdict.
        old_moved: §11.54-vocabulary answers that differ across arms.
        polar_moved: Plain polar answers that differ across arms.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    unknown_answered: int = 0
    old_moved: int = 0
    polar_moved: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected)."""
    names = []
    seen = set()
    while len(names) < 4:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    attr, small, large, factor = rng.choice(_FAMILIES)

    a, b = names[0], names[1]
    va = rng.randrange(150, 900, 25)
    vb = rng.randrange(150, 900, 25)
    while vb == va:
        vb = rng.randrange(150, 900, 25)
    statements = [f"the {a} {attr} is {va} {small}",
                  f"the {b} {attr} is {vb} {small}",
                  f"the {names[2]} material is iron"]
    cases = []

    def verdict(left: Fraction, word: str, right: Fraction) -> str:
        direction = {**_NEW_WORDS, **_OLD_WORDS}[word]
        if left == right:
            return "No"
        bigger = left > right
        return "Yes" if (bigger == (direction == "larger")) else "No"

    # A belief against a written quantity, both directions.
    for _ in range(rng.choice((2, 3))):
        word = rng.choice(list(_NEW_WORDS))
        against = rng.randrange(100, 950, 25)
        cases.append((f"is the {a} {attr} {word} than {against} "
                      f"{small}?", "verdict",
                      verdict(Fraction(va), word, Fraction(against))))

    # A conversion: the written side in the larger unit.
    if rng.random() < 0.8:
        word = rng.choice(list(_NEW_WORDS))
        against = rng.randint(1, 3)
        cases.append((f"is the {a} {attr} {word} than {against} "
                      f"{large}?", "verdict",
                      verdict(Fraction(va), word,
                              Fraction(against * factor))))

    # A belief against a belief, in the new words.
    if rng.random() < 0.8:
        word = rng.choice(list(_NEW_WORDS))
        cases.append((f"is the {a} {attr} {word} than the {b} "
                      f"{attr}?", "verdict",
                      verdict(Fraction(va), word, Fraction(vb))))

    # §11.54's vocabulary: identical in both arms.
    word = rng.choice(list(_OLD_WORDS))
    cases.append((f"is the {a} {attr} {word} than the {b} {attr}?",
                  "old", verdict(Fraction(va), word, Fraction(vb))))

    # An unknown relation: a refusal, never a verdict.
    for _ in range(rng.choice((1, 2))):
        cases.append((f"is the {a} {attr} "
                      f"{rng.choice(_UNKNOWN_WORDS)} than the {b} "
                      f"{attr}?", "unknown", None))

    # An unknown unit: refused in both arms.
    if rng.random() < 0.6:
        cases.append((f"is the {a} {attr} greater than "
                      f"{rng.randint(10, 90)} florins?", "refuses",
                      None))

    # Plain polar questions: identical in both arms.
    cases.append((f"is the {names[2]} material iron?", "polar", None))
    cases.append((f"is the {names[2]} material steel?", "polar", None))

    rng.shuffle(cases)
    return statements, cases


def run_gate(seeds: int = 12) -> ComparisonReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import inference

    arms = {"coin": False, "compared": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    unknown_answered = 0
    old_answers: dict = {}
    polar_answers: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(12700 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_cmp_"))
            old_flag = thoughts._COMPARISON_GUARD
            taken = {}
            thoughts._COMPARISON_GUARD = flag
            if not flag:
                # The baseline is the system as it shipped: without
                # the guard AND without the words §11.82 added.
                for word in _NEW_WORDS:
                    taken[word] = inference._COMBINE_WORDS.pop(word)
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
                    elif kind == "old":
                        scored += 1
                        if verdict == expected:
                            hits += 1
                        elif verdict and flag:
                            wrong += 1
                        old_answers.setdefault((seed, question),
                                               {})[arm] = response
                    elif kind == "unknown":
                        scored += 1
                        if not verdict:
                            hits += 1
                        elif flag:
                            unknown_answered += 1
                    elif kind == "refuses":
                        scored += 1
                        if not verdict:
                            hits += 1
                    else:
                        polar_answers.setdefault((seed, question),
                                                 {})[arm] = response
                scores.append(hits / scored if scored else 1.0)
            finally:
                thoughts._COMPARISON_GUARD = old_flag
                inference._COMBINE_WORDS.update(taken)
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    old_moved = sum(1 for by_arm in old_answers.values()
                    if len(set(by_arm.values())) > 1)
    polar_moved = sum(1 for by_arm in polar_answers.values()
                      if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["compared"],
                                       per_seed["coin"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and unknown_answered == 0 and old_moved == 0
              and polar_moved == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - comparisons at "
        f"{correct['compared']:.3f} against the coin's "
        f"{correct['coin']:.3f} ({delta:+.3f}, {ratio:.2f}x seed sd), "
        f"{wrong} wrong verdicts, {unknown_answered} unknown "
        f"relations answered, {old_moved} old-vocabulary answers "
        f"moved, {polar_moved} plain polar answers moved")
    return ComparisonReport(
        passes=passes, correct=correct, wrong=wrong,
        unknown_answered=unknown_answered, old_moved=old_moved,
        polar_moved=polar_moved, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"coin         {report.correct['coin']:.3f}")
    print(f"compared     {report.correct['compared']:.3f}")
    print(f"wrong verdicts: {report.wrong}")
    print(f"unknown relations answered: {report.unknown_answered}")
    print(f"old-vocabulary answers moved: {report.old_moved}")
    print(f"plain polar answers moved: {report.polar_moved}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
