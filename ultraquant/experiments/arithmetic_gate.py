"""Arithmetic that is exactly right. The arithmetic gate.

The system could combine two STORED numbers since §11.42 but could
not add two: "what is 2 + 2?" fell through the whole question ladder
to "I don't hold anything on that yet" - honest, and useless. §11.76
reads and evaluates arithmetic expressions, with precedence and
parentheses structural rather than special-cased.

The interesting question is not whether it can add. It is WHAT KIND
of number it gives back. The standard option - the one every
language hands you - is binary floating point, and binary floating
point cannot hold a tenth: it answers "what is 0.1 + 0.2?" with
0.30000000000000004, a number that is not the answer. So the
baseline arm here is not a strawman or an absence. **It is the
standard implementation, honestly built**: the same reader, the same
grammar, the same refusals, evaluating in floats. The treatment arm
evaluates the identical expressions over exact rationals.

Worlds are random expression TREES, in world-varying proportions:
integer and decimal leaves, all four operators, depths one to three,
minimal parentheses inserted only where the intended grouping needs
them (so precedence is genuinely under test). **Ground truth comes
from the tree, not from the parser** - the gate walks its own tree
with ``Fraction`` and never asks the mechanism what it thinks - so
the measurement checks the tokenizer, the grammar and the arithmetic
at once. Three case kinds are scored:

- **normal**: correct iff the answer parses back to EXACTLY the
  tree's value.
- **divzero**: an expression whose divisor is exactly zero ("10 / 0",
  "5 / (3 - 3)") - correct iff refused aloud, in either arm.
- **driftzero**: an expression whose divisor is zero only if you can
  hold tenths ("1 / (0.1 + 0.2 - 0.3)") - correct iff refused. This
  is where a rounding error stops being a wrong digit and becomes a
  wrong BRANCH: the exact arm refuses an undefined division, and the
  float arm divides by 5.5e-17 and reports a number near ten
  quadrillion.

**The criteria, written before the run.**

1. **Gain** - answer correctness (exact match against the tree) must
   beat the baseline arm by more than the seed-to-seed sd.
2. **The exact arm is never wrong** - a normal case whose answer
   differs from the tree's value, at all, in the treatment arm is a
   fail, once. This is arithmetic; there is no acceptable margin.
3. **Undefined stays undefined** - a divzero or driftzero case
   ANSWERED (rather than refused) in the treatment arm is a fail,
   once. The baseline's count is reported beside it as the measured
   corruption, not as a fail.
4. **The branch steals nothing** - ordinary questions ("what is the
   tower material?") and stored-fact questions answer identically in
   both arms, and identically to a session with no math at all.

**It was run once, and PASSED - perfectly on the claim, and the
standard option was wrong in two different ways.**

| arm | correct |
|---|---:|
| floats | 0.443 |
| exact | **1.000** |

**+0.557 at 3.69x seed sd, zero wrong answers, zero undefined
divisions answered** - while the float arm printed a number that was
not the answer **45 times** and answered **8 undefined divisions**.
The two failures are different in kind and both are in the record:

    [floats] what is 0.1 + 0.2?
             0.1 + 0.2 = 0.30000000000000004
    [exact]  what is 0.1 + 0.2?
             0.1 + 0.2 = 0.3

    [floats] what is 7 / (0.1 + 0.2 - 0.3)?
             7 / (0.1 + 0.2 - 0.3) = 126100789566373888
    [exact]  what is 7 / (0.1 + 0.2 - 0.3)?
             I can't compute that: division by zero is undefined

The first is a wrong digit. The second is a wrong BRANCH: the
divisor is zero, the float arm reaches a residue near 1e-17 instead,
and reports a hundred and twenty-six quadrillion for an expression
that has no value at all. A system that says "absence is never no"
about beliefs cannot say 1.26e17 about a division by zero.

Every ordinary question answered identically in both arms AND in a
control session with the branch switched off entirely: the reader
takes only what is nothing but numbers, so the ladder below it is
untouched.

Run it::

    python -m ultraquant.experiments.arithmetic_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

__all__ = ["ArithmeticReport", "run_gate"]

_OPS = ["+", "-", "*", "/"]
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


@dataclass
class ArithmeticReport:
    """The two arms, the three integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        correct: Arm -> answer correctness over all worlds.
        wrong: Arm -> answers that differ from the tree's value.
        undefined_answered: Arm -> undefined divisions given a number.
        stolen: Ordinary questions whose answer moved (must be 0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool
    correct: dict = field(default_factory=dict)
    wrong: dict = field(default_factory=dict)
    undefined_answered: dict = field(default_factory=dict)
    stolen: int = 0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""


def _leaf(rng: random.Random) -> Fraction:
    """An integer, or a decimal with one or two places."""
    if rng.random() < 0.5:
        return Fraction(rng.randint(1, 20))
    places = rng.choice((1, 2))
    return Fraction(rng.randint(1, 40), 10 ** places)


def _tree(rng: random.Random, depth: int):
    """``(op, left, right)`` or a Fraction leaf; never divides by 0."""
    if depth <= 0:
        return _leaf(rng)
    op = rng.choice(_OPS)
    left = _tree(rng, depth - 1)
    right = _tree(rng, depth - 1)
    if op == "/" and _value(right) == 0:
        op = "*"
    return (op, left, right)


def _value(node) -> Fraction:
    """Ground truth, walked from the tree - never from the parser."""
    if isinstance(node, Fraction):
        return node
    op, left, right = node
    a, b = _value(left), _value(right)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


def _show_leaf(value: Fraction) -> str:
    """A leaf as written: an integer, or one or two decimal places."""
    if value.denominator == 1:
        return str(value.numerator)
    places = 1 if (value * 10).denominator == 1 else 2
    scaled = int(value * 10 ** places)
    digits = str(scaled).rjust(places + 1, "0")
    return f"{digits[:-places]}.{digits[-places:]}"


def _render(node, parent_op: str | None = None,
            side: str = "left") -> str:
    """The tree as a string, parenthesised only where needed.

    Minimal parentheses are the point: if the gate parenthesised
    everything, the parser's precedence would never be tested.
    """
    if isinstance(node, Fraction):
        return _show_leaf(node)
    op, left, right = node
    text = (f"{_render(left, op, 'left')} {op} "
            f"{_render(right, op, 'right')}")
    if parent_op is None:
        return text
    mine, theirs = _PRECEDENCE[op], _PRECEDENCE[parent_op]
    needs = mine < theirs or (mine == theirs and side == "right"
                              and parent_op in ("-", "/"))
    return f"({text})" if needs else text


def _divzero_case(rng: random.Random) -> tuple[str, str]:
    """An expression whose divisor is exactly zero."""
    top = rng.randint(2, 30)
    if rng.random() < 0.5:
        return f"what is {top} / 0?", "divzero"
    same = rng.randint(2, 20)
    return f"what is {top} / ({same} - {same})?", "divzero"


def _driftzero_case(rng: random.Random) -> tuple[str, str]:
    """A divisor that is zero only if tenths can be held exactly.

    Every one of these is a true division by zero. A float evaluator
    reaches a residue near 1e-17 instead and reports a number in the
    quadrillions - the rounding error becoming a wrong branch, not a
    wrong digit.
    """
    top = rng.randint(2, 30)
    a, b = rng.choice(((1, 2), (2, 4), (3, 6), (1, 7)))
    return (f"what is {top} / (0.{a} + 0.{b} - 0.{a + b})?",
            "driftzero")


def _build_world(rng: random.Random):
    """Question plan for one world: normal, divzero, driftzero."""
    cases = []
    for _ in range(rng.choice((5, 6, 7))):
        node = _tree(rng, rng.choice((1, 2, 3)))
        cases.append((f"what is {_render(node)}?", "normal",
                      _value(node)))
    for _ in range(rng.choice((1, 2))):
        question, kind = _divzero_case(rng)
        cases.append((question, kind, None))
    if rng.random() < 0.7:
        question, kind = _driftzero_case(rng)
        cases.append((question, kind, None))
    rng.shuffle(cases)
    return cases


def _parse_answer(response: str) -> Fraction | None:
    """Read the number back out of the spoken answer."""
    if "=" not in response:
        return None
    tail = response.split("=", 1)[1].strip()
    for cut in (" (exact", " - computed"):
        if cut in tail:
            tail = tail.split(cut, 1)[0]
    tail = tail.strip().rstrip(".")
    try:
        if "/" in tail:
            top, bottom = tail.split("/", 1)
            return Fraction(int(top), int(bottom))
        return Fraction(tail)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(float(tail))
        except (ValueError, OverflowError):
            return None


def run_gate(seeds: int = 12) -> ArithmeticReport:
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)
    from ultraquant.reason import calculate

    arms = {"floats": False, "exact": True}
    correct = {}
    wrong = {arm: 0 for arm in arms}
    undefined_answered = {arm: 0 for arm in arms}
    per_seed = {arm: [] for arm in arms}
    ordinary = {}

    for arm, flag in arms.items():
        scores = []
        for seed in range(seeds):
            rng = random.Random(5100 + seed)
            cases = _build_world(rng)
            root = Path(tempfile.mkdtemp(prefix="uq_arith_"))
            old = calculate._EXACT_RATIONALS
            calculate._EXACT_RATIONALS = flag
            try:
                session = build_session(root, seed=seed)
                run_pipeline("the tower material is iron", session)
                run_pipeline("the tower height is 300 meters", session)
                hits = 0
                for question, kind, expected in cases:
                    response, _t = run_pipeline(question, session)
                    refused = response.startswith("I can't compute")
                    if kind == "normal":
                        got = _parse_answer(response)
                        if got is not None and got == expected:
                            hits += 1
                        else:
                            wrong[arm] += 1
                    else:
                        if refused:
                            hits += 1
                        else:
                            undefined_answered[arm] += 1
                for question in ("what is the tower material?",
                                 "what is the tower height?",
                                 "what is the tower architect?"):
                    response, _t = run_pipeline(question, session)
                    ordinary.setdefault((seed, question),
                                        {})[arm] = response
                scores.append(hits / len(cases))
            finally:
                calculate._EXACT_RATIONALS = old
                shutil.rmtree(root, ignore_errors=True)
        correct[arm] = statistics.mean(scores)
        per_seed[arm] = scores

    # The no-math control: ordinary questions must answer the same
    # in a session where the branch never runs at all.
    from ultraquant.interpreter import thoughts

    control = {}
    old_branch = thoughts._ARITHMETIC_ON
    thoughts._ARITHMETIC_ON = False
    try:
        for seed in range(seeds):
            root = Path(tempfile.mkdtemp(prefix="uq_arith_ctl_"))
            try:
                session = build_session(root, seed=seed)
                run_pipeline("the tower material is iron", session)
                run_pipeline("the tower height is 300 meters", session)
                for question in ("what is the tower material?",
                                 "what is the tower height?",
                                 "what is the tower architect?"):
                    response, _t = run_pipeline(question, session)
                    control[(seed, question)] = response
            finally:
                shutil.rmtree(root, ignore_errors=True)
    finally:
        thoughts._ARITHMETIC_ON = old_branch

    stolen = 0
    for case, responses in ordinary.items():
        voices = set(responses.values()) | {control[case]}
        if len(voices) > 1:
            stolen += 1

    contrasts = [t - b for t, b in zip(per_seed["exact"],
                                       per_seed["floats"])]
    delta = statistics.mean(contrasts)
    seed_sd = statistics.stdev(contrasts)
    ratio = (delta / seed_sd) if seed_sd else float("inf")

    passes = (seed_sd > 0.0 and delta > seed_sd
              and wrong["exact"] == 0
              and undefined_answered["exact"] == 0
              and stolen == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - exact arithmetic at "
        f"{correct['exact']:.3f} against the float arm's "
        f"{correct['floats']:.3f} ({delta:+.3f}, {ratio:.2f}x seed "
        f"sd), {wrong['exact']} wrong answers, "
        f"{undefined_answered['exact']} undefined divisions answered "
        f"- while the float arm was wrong {wrong['floats']} times "
        f"and answered {undefined_answered['floats']} undefined "
        f"divisions - and {stolen} ordinary questions moved")
    return ArithmeticReport(
        passes=passes, correct=correct, wrong=wrong,
        undefined_answered=undefined_answered, stolen=stolen,
        delta=delta, seed_sd=seed_sd, margin_ratio=ratio,
        reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print("arm         correct")
    print(f"floats       {report.correct['floats']:.3f}")
    print(f"exact        {report.correct['exact']:.3f}")
    print(f"wrong answers: exact {report.wrong['exact']}, "
          f"floats {report.wrong['floats']}")
    print("undefined divisions answered: exact "
          f"{report.undefined_answered['exact']}, floats "
          f"{report.undefined_answered['floats']}")
    print(f"ordinary questions moved: {report.stolen}")
    print(f"delta {report.delta:+.3f}   seed sd {report.seed_sd:.3f}"
          f"   ratio {report.margin_ratio:.2f}x")
    print(report.reason)
