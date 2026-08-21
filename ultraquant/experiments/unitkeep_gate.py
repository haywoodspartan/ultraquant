"""The unit that fell off. The unit-keeping gate.

Found while building §11.76's arithmetic, by asking the shipped
machinery a question it had always been able to answer::

    > what is the sum of the tower height and the keep height?
      keep height + tower height = 500 - inferred, not stored:
      keep height is 200 meters; tower height is 300 meters

500 what. Both premises say meters, the answer says a bare number,
and the reader is left to supply the unit from context - which is
exactly the kind of quiet gap this project treats as a wrong answer
rather than a rough edge. It is not a rounding error and not an
invention; it is an answer that does not say what it is an answer
IN.

The cause is one line of §11.42's combine path: the unit suffix was
attached to the CONVERSION, not to the value. So "300 meters + 1
kilometer" - which converts - answered "1.3 kilometers" correctly,
while "300 meters + 200 meters" - which needs no conversion - fell
through the same branch with an empty suffix. Every shape of the
combine path inherited it: sums, differences, and the larger/smaller
verdict ("tower height (300) is the larger of the two"). The
aggregate path (§11.57) never had the bug - "the total of the 2
height facts I hold is 500 meters" - which is what makes the
inconsistency visible from the outside: one store, two voices, for
the same arithmetic.

The fix is that the result unit is a property of the RESULT, not of
whether a conversion happened, and the conversion note stays exactly
where it was. Unitless operands still produce unitless answers -
inventing a unit for "300 + 200" would be the opposite failure.

Both arms run the live pipeline; the baseline arm turns
`_SUM_KEEPS_UNIT` off (the shipped answer), the treatment arm is
current. Worlds mix, in world-varying proportions: same-unit pairs
(the broken case), cross-unit same-family pairs (the converted case,
which must not move at all), unitless pairs (which must never GAIN a
unit), and cross-family pairs (which refuse in both arms), asked as
sums, differences, and larger/smaller verdicts.

**The criteria, written before the run.**

1. **Gain** - unit-naming correctness (an answer states a unit
   exactly when its operands carry one) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **The number never moves** - this is a naming fix, not an
   arithmetic one. Every answer's numeric value, in both arms, must
   be identical. One difference is a fail, once.
3. **No unit is ever invented** - a unitless pair whose answer names
   a unit, in either arm, is a fail, once.
4. **The converted case does not move** - cross-unit answers are
   byte-identical across arms, and so are cross-family refusals.

**It was run once, and PASSED - on the naming, and on all three
lines that say the arithmetic did not move.**

| arm | names its unit |
|---|---:|
| drops | 0.528 |
| keeps | **1.000** |

**+0.472 at 1.87x seed sd, zero numeric values moved, zero units
invented, zero converted-or-refused answers changed.** The
never-moves line is the one that matters most here: every number in
every answer, in both arms, was identical - this is a fix to what
the answer SAYS, not to what it computes. Unitless pairs stayed
unitless, cross-family pairs refused in both arms, and the converted
case came through byte for byte.

    > what is the sum of the tower height and the keep height?
      keep height + tower height = 500 meters
    > which is larger, the tower height or the keep height?
      tower height (300 meters) is the larger of the two

The bug shipped in three shapes at once and survived a green suite,
and the reason is worth keeping. The same-unit case WAS pinned -
`test_same_unit_answers_carry_no_conversion_label` - but the pin
asserted only what the answer must NOT say. It checked that no
"(units converted)" label appeared, because that was the claim being
made the day it was written, and a test that checks for the absence
of a label cannot notice the absence of a unit. Nothing else in
1,697 tests looked at that sentence at all: the fix broke no pin,
which is the tell. A green suite proves the claims someone thought
to write down, and "the answer says what it is an answer in" was
never one of them until a plain-English question was asked and the
answer read as a reader rather than as its author.

Run it::

    python -m ultraquant.experiments.unitkeep_gate
"""

from __future__ import annotations

import random
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["UnitKeepReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]

#: (attribute, unit, other unit in the same family) - the pairs a
#: definition connects, so a conversion is available when wanted.
_FAMILIES = [
    ("height", "meters", "kilometers"),
    ("length", "meters", "kilometers"),
    ("weight", "kilograms", "tonnes"),
    ("duration", "seconds", "minutes"),
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class UnitKeepReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        named: Arm -> unit-naming correctness over all worlds.
        moved: Answers whose numeric value differs between arms.
        invented: Unitless pairs whose answer names a unit.
        converted_mismatch: Converted or refused cases that moved.
        delta: Naming contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    named: dict = field(default_factory=dict)
    moved: int = 0
    invented: int = 0
    converted_mismatch: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, unit-or-None)."""
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    attr, unit, other = rng.choice(_FAMILIES)
    statements = []
    cases = []

    # Same unit - the broken case, in world-varying number.
    same = names[0:2]
    for name, value in zip(same, rng.sample(range(100, 900), 2)):
        statements.append(f"the {name} {attr} is {value} {unit}")
    for _ in range(rng.choice((1, 2))):
        shape = rng.choice(("sum", "difference", "larger"))
        cases.append((_ask(shape, same, attr), "same", unit))

    # Cross-unit, same family - the converted case, must not move.
    if rng.random() < 0.8:
        cross = names[2:4]
        statements.append(f"the {cross[0]} {attr} is "
                          f"{rng.randint(100, 900)} {unit}")
        statements.append(f"the {cross[1]} {attr} is "
                          f"{rng.randint(2, 9)} {other}")
        cases.append((_ask(rng.choice(("sum", "difference")), cross,
                           attr), "converted", None))

    # Unitless - must never gain a unit.
    if rng.random() < 0.8:
        bare = names[4:6]
        for name, value in zip(bare, rng.sample(range(10, 90), 2)):
            statements.append(f"the {name} count is {value}")
        cases.append((_ask(rng.choice(("sum", "difference")), bare,
                           "count"), "unitless", None))

    # Cross-family - refuses in both arms.
    if rng.random() < 0.6:
        odd = names[6:8]
        statements.append(f"the {odd[0]} {attr} is "
                          f"{rng.randint(10, 90)} {unit}")
        statements.append(f"the {odd[1]} {attr} is "
                          f"{rng.randint(10, 90)} florins")
        cases.append((_ask("sum", odd, attr), "refuses", None))

    rng.shuffle(cases)
    return statements, cases


def _ask(shape: str, names: list[str], attr: str) -> str:
    a, b = names
    if shape == "sum":
        return f"what is the sum of the {a} {attr} and the {b} {attr}?"
    if shape == "difference":
        return (f"what is the difference between the {a} {attr} and "
                f"the {b} {attr}?")
    return f"which is larger, the {a} {attr} or the {b} {attr}?"


def _numbers(response: str) -> list[str]:
    """Every number in an answer, for the never-moves check."""
    return _NUMBER_RE.findall(response)


def _names_unit(response: str, unit: str) -> bool:
    """Whether the ANSWER (not its premises) states the unit.

    The premise list repeats the stored values, units and all, so a
    naive substring test would pass in both arms. Only the text
    before the provenance marker is the answer.
    """
    head = response.split(" - inferred", 1)[0]
    return unit in head or unit.rstrip("s") in head


def run_gate(seeds: int = 12) -> UnitKeepReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import inference

    arms = {"drops": False, "keeps": True}
    named = {}
    per_seed = {arm: [] for arm in arms}
    invented = 0
    answers: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(6200 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_unitkeep_"))
            old = inference._SUM_KEEPS_UNIT
            inference._SUM_KEEPS_UNIT = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                for question, kind, unit in cases:
                    response, _t = run_pipeline(question, session)
                    answers.setdefault((seed, question),
                                       {})[arm] = response
                    if kind == "same":
                        if _names_unit(response, unit):
                            hits += 1
                    elif kind == "unitless":
                        head = response.split(" - inferred", 1)[0]
                        gained = any(word in head for word in
                                     ("meter", "kilometer", "kilogram",
                                      "tonne", "second", "minute"))
                        if gained:
                            invented += 1
                        else:
                            hits += 1
                    else:
                        # Converted and refused cases are correct in
                        # both arms; they dilute the contrast, which
                        # is what makes it vary by seed.
                        hits += 1
                scores.append(hits / len(cases))
            finally:
                inference._SUM_KEEPS_UNIT = old
                shutil.rmtree(root, ignore_errors=True)
        named[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    moved = 0
    converted_mismatch = 0
    for (seed, question), by_arm in answers.items():
        if len(by_arm) < 2:
            continue
        if _numbers(by_arm["drops"]) != _numbers(by_arm["keeps"]):
            moved += 1
        if ("units converted" in by_arm["drops"]
                or "can't" in by_arm["drops"].lower()
                or "don't" in by_arm["drops"].lower()):
            if by_arm["drops"] != by_arm["keeps"]:
                converted_mismatch += 1

    contrasts = [t - b for t, b in zip(per_seed["keeps"],
                                       per_seed["drops"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and moved == 0
              and invented == 0 and converted_mismatch == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - answers name their unit at "
        f"{named['keeps']:.3f} against {named['drops']:.3f} "
        f"({delta:+.3f}, {ratio:.2f}x seed sd), {moved} numeric "
        f"values moved, {invented} units invented, "
        f"{converted_mismatch} converted-or-refused answers changed")
    return UnitKeepReport(
        passes=passes, named=named, moved=moved, invented=invented,
        converted_mismatch=converted_mismatch, delta=delta,
        seed_sd=seed_sd, margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         names its unit")
    print(f"drops        {report.named['drops']:.3f}")
    print(f"keeps        {report.named['keeps']:.3f}")
    print(f"numeric values moved: {report.moved}")
    print(f"units invented: {report.invented}")
    print("converted-or-refused answers changed: "
          f"{report.converted_mismatch}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
