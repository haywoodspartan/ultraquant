"""An operand the store can reach. The derived-operand gate.

§11.78 closed with a ceiling on the record: an arithmetic operand
had to be HELD outright, while §11.55's comparative operand could be
DERIVED. So one store answered two questions about the same fact two
different ways::

    > is the west tower taller than the south tower?
      Yes - west tower height is 900 meters (derived via spirekind),
      south tower height is 450 meters
    > what is the west tower height * 2?
      I can't compute that: I hold nothing for 'west tower height'

Both sentences are about `west tower height`. One reaches it through
`west tower kind is spirekind`; the other says it does not exist.
That is §11.75's failure again - one matrix, two voices - and the
fix is the same shape: the arithmetic operand goes through the chain
machinery under the SAME three guards §11.55 uses, no
modifier-rescued subject, no derived denial, no empty conclusion.

A derived operand is marked where it is used, in the same words the
comparative uses ("derived via spirekind"), and its diluted
confidence joins the minimum like any other premise - a number
reached through two facts is not as good as a number that was
stated, and the answer says so.

Both arms run the live pipeline; the baseline arm turns
`_DERIVE_OPERANDS` off, leaving §11.78's held-outright rule. Worlds
mix, in world-varying proportions: chained operands (reachable only
with the rung), directly held operands (both arms), and three
refusal kinds that must refuse in BOTH arms - a chain reaching a
denial, a chain reaching a non-number, and an operand nothing
reaches at all.

**The criteria, written before the run.**

1. **Gain** - answer correctness (the exact expected value with its
   unit, or a refusal where one is owed) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **No wrong number** - a computed value differing from the gate's
   own expectation, in the treatment arm, is a fail, once.
3. **A derived denial is never a number** - the chain that reaches
   "is not 20 meters" must refuse in both arms. One number there is
   a fail, once, and it is the §11.48 line arriving in arithmetic.
4. **One matrix, one voice** - the value the arithmetic uses for a
   derived operand must equal the value the chain machinery gives
   for that operand asked on its own. This is the line the whole
   rung exists to hold.
5. **Directly held operands do not move** - their answers are
   byte-identical across arms.

**It was run once, and PASSED - on the claim and on all four
lines.**

| arm | correct |
|---|---:|
| held | 0.662 |
| reached | **1.000** |

**+0.338 at 3.64x seed sd, zero wrong values, zero derived denials
given a number, zero voice splits against the chain machinery, zero
held operands moved.**

Criterion 4 is the one this rung exists for, and it is the one worth
reading twice: for every derived operand the gate asked the store
the same thing two ways - once inside the arithmetic and once on its
own - and required the same number. Zero splits. The ceiling §11.78
recorded is closed, and closed in the direction that removes a
disagreement rather than adding a capability:

    > what is the west tower height * 2?
      west tower height * 2 = 1800 meters - inferred, not stored:
      west tower height is 900 meters (derived via spirekind)

The refusals kept their shape too. A chain that reaches a denial
says so in the §11.48 words - "I can only derive a denial for 'old
gate height' (old gate height is not 20 meters), and a denial holds
no number" - rather than reporting a number or a bare absence, which
are the two ways this could have gone wrong.

Run it::

    python -m ultraquant.experiments.derivedoperand_gate
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

__all__ = ["DerivedReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_KINDS = ["spirekind", "vaultkind", "archkind", "slabkind", "pierkind",
          "domekind"]
_FAMILIES = [("height", "meters"), ("length", "meters"),
             ("weight", "kilograms")]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class DerivedReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> correctness over all worlds.
        wrong: Computed values differing from expectation.
        denial_numbers: Derived denials given a number.
        voice_split: Derived operands whose arithmetic value differs
            from the chain machinery's own answer.
        direct_moved: Directly held operands whose answer moved.
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: int = 0
    denial_numbers: int = 0
    voice_split: int = 0
    direct_moved: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _build_world(rng: random.Random):
    """Statements and cases: (question, kind, expected-or-None)."""
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    kinds = rng.sample(_KINDS, 3)
    attr, unit = rng.choice(_FAMILIES)

    statements = []
    cases = []

    # Chained operands: reachable only through the kind fact.
    for slot in range(rng.choice((1, 2, 3))):
        name, kind = names[slot], kinds[0]
        value = rng.randint(100, 900)
        if slot == 0:
            statements.append(f"{kind} {attr} is {value} {unit}")
        else:
            kind = f"{kinds[0]}{slot}"
            statements.append(f"{kind} {attr} is {value} {unit}")
        statements.append(f"the {name} kind is {kind}")
        multiplier = rng.randint(2, 9)
        cases.append((f"what is the {name} {attr} * {multiplier}?",
                      "derived", f"{value * multiplier} {unit}",
                      f"what is the {name} {attr}?"))

    # Directly held operands: both arms, and they must not move.
    for slot in (3, 4):
        if rng.random() < 0.8:
            value = rng.randint(100, 900)
            statements.append(f"the {names[slot]} {attr} is "
                              f"{value} {unit}")
            multiplier = rng.randint(2, 9)
            cases.append((f"what is the {names[slot]} {attr} * "
                          f"{multiplier}?", "direct",
                          f"{value * multiplier} {unit}", None))

    # A chain that reaches a denial - refuses in both arms.
    if rng.random() < 0.8:
        statements.append(f"{kinds[1]} {attr} is not "
                          f"{rng.randint(10, 90)} {unit}")
        statements.append(f"the {names[5]} kind is {kinds[1]}")
        cases.append((f"what is the {names[5]} {attr} * 2?", "denial",
                      None, None))

    # A chain that reaches a non-number - refuses in both arms.
    if rng.random() < 0.7:
        statements.append(f"{kinds[2]} {attr} is oak")
        statements.append(f"the {names[6]} kind is {kinds[2]}")
        cases.append((f"what is the {names[6]} {attr} * 2?",
                      "nonnumeric", None, None))

    # Nothing reaches it at all - refuses in both arms.
    cases.append((f"what is the {names[7]} {attr} * 2?",
                  "unreachable", None, None))

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


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def run_gate(seeds: int = 12) -> DerivedReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import calculate

    arms = {"held": False, "reached": True}
    correct = {}
    per_seed = {arm: [] for arm in arms}
    wrong = 0
    denial_numbers = 0
    voice_split = 0
    direct: dict = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(8400 + seed)
            statements, cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_deriv_"))
            old = calculate._DERIVE_OPERANDS
            calculate._DERIVE_OPERANDS = flag
            try:
                session = build_session(root, seed=seed)
                for line in statements:
                    run_pipeline(line, session)
                hits = 0
                for question, kind, expected, alone in cases:
                    response, _t = run_pipeline(question, session)
                    refused = response.startswith("I can't compute")
                    stated = _asserted_value(response)
                    if kind in ("derived", "direct"):
                        if stated == expected:
                            hits += 1
                        elif stated and flag:
                            wrong += 1
                        if kind == "direct":
                            direct.setdefault((seed, question),
                                              {})[arm] = response
                        elif flag and alone is not None:
                            # One matrix, one voice: the arithmetic's
                            # operand and the chain machinery's own
                            # answer must be the same number.
                            chain, _t2 = run_pipeline(alone, session)
                            chain_numbers = _numbers(chain)
                            trail = response.split("inferred, not "
                                                   "stored:", 1)[-1]
                            if not chain_numbers:
                                voice_split += 1
                            elif chain_numbers[0] not in _numbers(trail):
                                voice_split += 1
                    else:
                        if refused:
                            hits += 1
                        elif stated:
                            if kind == "denial":
                                denial_numbers += 1
                            elif flag:
                                wrong += 1
                scores.append(hits / len(cases))
            finally:
                calculate._DERIVE_OPERANDS = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    direct_moved = sum(1 for by_arm in direct.values()
                       if len(set(by_arm.values())) > 1)

    contrasts = [t - b for t, b in zip(per_seed["reached"],
                                       per_seed["held"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd and wrong == 0
              and denial_numbers == 0 and voice_split == 0
              and direct_moved == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - reached operands at "
        f"{correct['reached']:.3f} against the held-only arm's "
        f"{correct['held']:.3f} ({delta:+.3f}, {ratio:.2f}x seed sd), "
        f"{wrong} wrong values, {denial_numbers} derived denials "
        f"given a number, {voice_split} voice splits against the "
        f"chain machinery, {direct_moved} held operands moved")
    return DerivedReport(
        passes=passes, correct=correct, wrong=wrong,
        denial_numbers=denial_numbers, voice_split=voice_split,
        direct_moved=direct_moved, delta=delta, seed_sd=seed_sd,
        margin_ratio=ratio, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"held         {report.correct['held']:.3f}")
    print(f"reached      {report.correct['reached']:.3f}")
    print(f"wrong values: {report.wrong}")
    print(f"derived denials given a number: {report.denial_numbers}")
    print(f"voice splits vs the chain machinery: {report.voice_split}")
    print(f"held operands moved: {report.direct_moved}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
