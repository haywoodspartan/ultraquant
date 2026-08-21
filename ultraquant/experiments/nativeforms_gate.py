"""A claim about a set, twice. The native enumerated-family gate.

Superlatives and aggregates stand on the same move - enumerate the
WHOLE store for facts whose key ends in an attribute - and carry the
same honesty contract: scope named, exclusions named BY NAME, ties
named rather than broken, and denials never counted in arithmetic,
because a denial names no number and a denied 900 meters must not
move a mean by a millimetre.

Those trailing clauses are the reason this gate compares sentences
rather than winners. "The west keep height is the tallest of the 3
height facts I hold: 2 kilometers (confidence 0.60). Excluded: barn
height (no number). 1 denial(s) not counted - a denial names no
number." is one answer, and a port that got the winner right and the
scope, the exclusions or the denial accounting wrong would be
answering a different question with the same number.

Two details were ported deliberately rather than improved. The
ranking multiplies by the FLOAT unit factors, because the Python
tier does - using the exact rationals there could order two facts
differently at the last bit, and be wrong for a reason nobody could
see. And the combined value prints through %g, which is what
Python's format(x, 'g') is, so 11/12 of a kilometre reads
"0.916667" in both tiers without either being asked to.

Worlds mix, in world-varying proportions: families of two to six
facts; mixed units inside one family (which convert) and outside it
(which are excluded by name); non-numeric values (excluded); denials
(counted separately, never averaged); ties for the winner; empty
families; polysemous superlatives that must refuse to guess an
attribute; and count questions, which tell the whole truth by
reporting denials beside the beliefs.

**The criteria, written before the run.**

1. **Parity is total** - every answer identical, including the
   scope clause, the exclusion list and the denial accounting. One
   mismatch is a fail.
2. **Refusal is symmetric** - where Python declines to own a
   question, the native tier declines too. A form that claims more
   questions than Python is as wrong as one that claims fewer,
   because the ladder below it depends on what falls through.
3. **The hard cases are present and reported** - ties, mixed units,
   exclusions, denials, empty families and polysemous words each
   appear, with counts.

**It took two runs, and the difference the first one found is the
best thing in this unit.**

Run one: 540 set-claims, **two answers differed**, both like this:

    what is the average weight?
      python: The average of the 6 weight facts I hold is 250.463
              tonnes (units converted) (confidence 0.60).
      native: The average of the 6 weight facts I hold is 250.462
              tonnes (units converted) (confidence 0.60).

Same facts, same order, same unit conversions, same formula - and a
different last digit. The cause is not arithmetic but the SUM:
**CPython's `sum()` over floats has carried a Neumaier compensation
term since 3.12**, so summing that family gives 1502.775 where a
plain left-to-right accumulation gives 1502.7749999999999, and %g
prints those as 250.463 and 250.462.

The native tier sums the way the oracle sums now. That is the right
fix and the alternative was not: widening the comparison to "close
enough" would have turned this gate into one that cannot see the
next real difference either. A tier that disagrees about a mean by
one digit in the last place has not reproduced the semantics, and it
would have been found eventually by a person rather than by a gate.

Run two, unchanged except for the summation:

| | |
|---|---:|
| set-claims put to both tiers | 540 |
| answers that differed | **0** |
| owned only by Python | 0 |
| owned only by the native tier | 0 |

Hard cases present, across sixty worlds: empty families and
polysemous words in all 60, mixed units in 45, denials in 26,
non-numeric values in 24, ties in 21, uncomparable units in 16.

Run it::

    python -m ultraquant.experiments.nativeforms_gate
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FormsParityReport", "run_gate", "probe_path"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_chat.exe"


def probe_path() -> Path:
    """Where the built pipeline lives, so tests can skip without it."""
    return _PROBE


@dataclass
class FormsParityReport:
    """Parity over set-claims, and which hard cases were present.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        asked: Questions put to both tiers.
        mismatches: Answers that differed.
        owned_only_by_native: Questions the native tier claimed and
            Python let fall through.
        owned_only_by_python: The reverse.
        features: Hard case -> how many worlds contained it.
        first_mismatch: The first disagreement, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    asked: int = 0
    mismatches: int = 0
    owned_only_by_native: int = 0
    owned_only_by_python: int = 0
    features: dict = field(default_factory=dict)
    first_mismatch: str = ""
    reason: str = ""


_ENTITIES = ["north tower", "south tower", "west keep", "east mill",
             "old barn", "royal hall", "grand arch", "little spire"]
_ATTRS = ["height", "weight", "length"]
_UNITS = {"height": ["meters", "kilometers"],
          "weight": ["kilograms", "tonnes"],
          "length": ["meters", "feet"]}


def _world(rng: random.Random):
    """Statements and questions, with the hard cases tagged."""
    attribute = rng.choice(_ATTRS)
    units = _UNITS[attribute]
    statements: list[str] = []
    features: set[str] = set()

    count = rng.choice((2, 3, 4, 5, 6))
    values = []
    for index in range(count):
        entity = _ENTITIES[index % len(_ENTITIES)]
        unit = units[0]
        if rng.random() < 0.3:
            unit = units[1]
            features.add("mixed units")
        amount = rng.randrange(25, 900, 25)
        values.append(amount)
        statements.append(f"the {entity} {attribute} is {amount} {unit}")
    if len(values) >= 2 and rng.random() < 0.3:
        # A tie for the winner, which must be named rather than broken.
        statements.append(f"the grand arch {attribute} is "
                          f"{max(values)} {units[0]}")
        features.add("ties")
    if rng.random() < 0.4:
        statements.append(f"the little spire {attribute} is tall")
        features.add("non-numeric")
    if rng.random() < 0.4:
        statements.append(f"the old barn {attribute} is not "
                          f"{rng.randrange(25, 900, 25)} {units[0]}")
        features.add("denials")
    if rng.random() < 0.3:
        statements.append(f"the royal hall {attribute} is "
                          f"{rng.randint(2, 90)} florins")
        features.add("uncomparable units")

    questions = [
        f"which {attribute} is the tallest?",
        f"which {attribute} is the shortest?",
        f"what is the total {attribute}?",
        f"what is the average {attribute}?",
        f"how many {attribute} facts do you hold?",
        "which is the tallest?",
        "which is the biggest?",
    ]
    empty = rng.choice([a for a in _ATTRS if a != attribute])
    questions.append(f"what is the total {empty}?")
    questions.append(f"which {empty} is the tallest?")
    features.add("empty families")
    features.add("polysemous words")
    rng.shuffle(questions)
    return statements, questions, features


def _python_answers(statements, questions):
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_formsparity_"))
    try:
        session = build_session(root, seed=0)
        for line in statements:
            run_pipeline(line, session)
        return [run_pipeline(question, session)[0]
                for question in questions]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _native_answers(statements, questions):
    lines = list(statements) + list(questions)
    got = subprocess.run([str(_PROBE), "--plain"],
                         input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    answers = got.stdout.splitlines()
    return answers[len(statements):]


def _unowned(answer: str) -> bool:
    """Whether the answer is the ladder's fall-through, not a form's."""
    return answer.startswith("I don't hold anything on that yet")


def run_gate(worlds: int = 60, seed: int = 0) -> FormsParityReport:
    """Build the same families twice and compare every set-claim."""
    if not _PROBE.exists():
        return FormsParityReport(
            passes=False,
            reason="FAIL - the native pipeline is not built; run "
                   "native/uq/build.ps1")
    asked = 0
    mismatches = 0
    native_only = 0
    python_only = 0
    features: dict = {}
    first = ""
    for index in range(worlds):
        rng = random.Random(61000 + seed * 100 + index)
        statements, questions, present = _world(rng)
        for feature in present:
            features[feature] = features.get(feature, 0) + 1
        want = _python_answers(statements, questions)
        have = _native_answers(statements, questions)
        for step, question in enumerate(questions):
            asked += 1
            expected = want[step]
            actual = have[step] if step < len(have) else "(no answer)"
            if expected == actual:
                continue
            mismatches += 1
            if _unowned(expected) and not _unowned(actual):
                native_only += 1
            elif _unowned(actual) and not _unowned(expected):
                python_only += 1
            if not first:
                first = (f"world {index} {question!r}\n"
                         f"    python: {expected}\n"
                         f"    native: {actual}")
    passes = mismatches == 0
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {asked} set-claims put to "
        f"both tiers across {worlds} worlds, {mismatches} answers "
        f"differed ({python_only} owned only by Python, "
        f"{native_only} only by the native tier)"
        + (f"\nfirst:\n{first}" if first else ""))
    return FormsParityReport(passes=passes, asked=asked,
                             mismatches=mismatches,
                             owned_only_by_native=native_only,
                             owned_only_by_python=python_only,
                             features=features, first_mismatch=first,
                             reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"set-claims put to both tiers: {report.asked}")
    print(f"answers that differed: {report.mismatches}")
    print(f"  owned only by Python: {report.owned_only_by_python}")
    print(f"  owned only by the native tier: "
          f"{report.owned_only_by_native}")
    print()
    for feature, count in sorted(report.features.items()):
        print(f"  {feature:<20} in {count:3d} worlds")
    print()
    print(report.reason)
