"""The same sentence, from a different language. The native calc gate.

§11.88 gave the native tier integers that agree with Python's. This
gate asks the harder question: does the whole arithmetic READER
agree - the tokeniser, the grammar, the unit table, the rounding
rules, the refusal sentences, and the exact spelling of every
answer?

Parity is checked as a RECORD, not as a value. Two tiers that both
answer "4" have proved very little; these have to agree on the
expression echo ("3 + 4 * 5", with the precedence visible), on the
rendered value ("0.3", not 0.30000000000000004, and "1/3" rather
than a decimal that would be a different number), on whether a
rounding happened and what the exact value was before it, on the
bounds of an irrational root, and on the WORDING of every refusal.
The refusal wording is not decoration in this system - §11.80's
"'20% of 300' has an answer, '300 + 20%' has a convention" is the
product - so a native tier that refuses correctly in different words
has not reproduced the semantics.

The hardest line is the quiet one. **A native tier that answers MORE
than Python is exactly as wrong as one that answers less.** "what is
the tower height?" is not arithmetic; "what is 5?" is not a
calculation; "hello there" is not a question for this reader. Each
must come back as "not my question" from both tiers, because the
whole ladder above depends on this branch declining what is not its
own.

Both tiers are asked the same generated corpus: literal expressions
with precedence and parentheses, word operators, signs, percentages
(including the dangling shape that must refuse), powers (including
zero to the zero and an exponent past the cap), roots (exact,
irrational, negative radicand, cube), rounding requests in digits
and words and to the nearest whole number, requests for significant
figures that must be refused by name, lists over written numbers and
quantities, unit mismatches, division by zero, malformed
expressions, and plain English that is not arithmetic at all.

**The criteria, written before the run.**

1. **Parity is total** - every record identical, field for field.
   One mismatch is a fail; there is no margin to be within.
2. **Refusals match by wording** - a refusal whose sentence differs
   counts as a mismatch, not as an agreement about refusing.
3. **Not-my-question is symmetric** - every input Python declines,
   the native tier declines, and the reverse. Counted separately so
   the failure is legible.
4. **Every form is exercised** - the corpus reports how many cases
   fell into each family, so a green run cannot come from a corpus
   that quietly stopped covering something.

**It took two runs, and the first one caught exactly the kind of
thing this gate exists for.**

Run one: 3,000 questions, **11 records differed** - all of them the
same shape, and none of them a wrong number:

    [list] 'list the maximum of 22 meters and 51 kilograms'
        python: refuse|no definition I hold connects meters and kilograms
        native: refuse|no definition I hold connects kilograms and meters

Both tiers refused. Both refused for the right reason. They named
the two units in opposite ORDER, because the Python tier computes
the running total before it branches on the operation - so on a
mixed-unit list the refusal names the units in the order the failing
ADDITION met them, not the order a comparison would. The native port
had skipped straight to the comparison for largest and smallest,
which is the same arithmetic and a different sentence.

A gate that compared values would have called that a pass. A gate
that compared "did both refuse" would have called it a pass. It is
not a pass: the wording of a refusal is what this system hands the
person asking, and two tiers that hand over different sentences have
not reproduced each other. The port was made faithful to the
ordering and run two came back clean.

| | |
|---|---:|
| questions asked of both tiers | 3,000 |
| records that differed | **0** |
| answered only by Python | 0 |
| answered only by the native tier | 0 |

Coverage, reported so a green run cannot come from a corpus that
quietly stopped testing something: literal 506, rounding 430, list
415, root 393, not-arithmetic 376, power 338, percent 291, spoken
operator 251. Three further seeds at 2,000 questions each: zero
differences.

The symmetry line held in both directions, which matters more than
it looks: 376 inputs that are not arithmetic at all - "what is the
tower height?", "hello there", "is 2 + 2 = 4?", "what is 5?" - were
declined by both tiers, every time. A native reader that answered
any of them would have been as wrong as one that failed to add.

Run it::

    python -m ultraquant.experiments.nativecalc_gate
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.reason import calculate

__all__ = ["CalcParityReport", "run_gate", "probe_path", "record_of"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_calc.exe"


def probe_path() -> Path:
    """Where the built reader lives, so tests can skip without it."""
    return _PROBE


@dataclass
class CalcParityReport:
    """Parity, per family, and the first place the tiers parted.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        checked: Questions asked of both tiers.
        mismatches: Records that differed in any field.
        python_only: Inputs Python answered and the native tier did not.
        native_only: Inputs the native tier answered and Python did not.
        families: Family -> how many cases it contributed.
        first_mismatch: The first disagreement, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    checked: int = 0
    mismatches: int = 0
    python_only: int = 0
    native_only: int = 0
    families: dict = field(default_factory=dict)
    first_mismatch: str = ""
    reason: str = ""


def record_of(result) -> str:
    """One MathResult as the flat record both tiers are compared on."""
    if result is None:
        return "none"
    if result.refusal:
        return "refuse|" + result.refusal
    if result.bounds:
        low, high = result.bounds
        return f"bounds|{result.expression}|{low}|{high}"
    places = "-" if result.rounded_to is None else str(result.rounded_to)
    return (f"ok|{result.expression}|{result.shown}|"
            f"{1 if result.fractional else 0}|{places}|"
            f"{1 if result.was_rounded else 0}|{result.exact_shown}")


def _python_record(question: str) -> str:
    """What the tier that DEFINES the semantics says."""
    if question.startswith("list "):
        return record_of(calculate.read_list(question[5:]))
    return record_of(calculate.evaluate(question))


_UNITS = ["meters", "kilometers", "kilograms", "tonnes", "seconds",
          "minutes", "feet", "miles", "grams", "hours"]
_NOT_ARITHMETIC = [
    "what is the tower height?", "what is the tower material?",
    "hello there", "what is 5?", "what is the plus size?",
    "is 2 + 2 = 4?", "is the tower height greater than 200 meters?",
    "what was the tower material?", "why is the tower height 300 meters?",
    "the tower height is 300 meters", "what is the average height?",
    "what is the sum of the tower height and the bridge height?",
]


def _questions(rng: random.Random, cases: int):
    """The corpus, tagged by family so coverage is reportable."""
    out: list[tuple[str, str]] = []
    while len(out) < cases:
        pick = rng.random()
        n = lambda lo=2, hi=99: rng.randint(lo, hi)          # noqa: E731
        if pick < 0.16:
            shape = rng.choice([
                f"what is {n()} + {n()} * {n(2, 12)}?",
                f"what is ({n()} + {n()}) * {n(2, 12)}?",
                f"what is {n()} - {n()}?",
                f"what is {n()} / {n()}?",
                f"what is -{n()} + {n()}?",
                f"what is {n()} - -{n()}?",
                f"what is {n()} * ({n()} - {n()}) / {n()}?",
                f"{n()} * {n()}",
            ])
            out.append((shape, "literal"))
        elif pick < 0.24:
            out.append((rng.choice([
                f"what is {n()} plus {n()}?",
                f"what is {n()} minus {n()}?",
                f"what is {n()} times {n()}?",
                f"what is {n()} divided by {n()}?",
                f"what is {n()} over {n()}?",
                f"what is {n()} multiplied by {n()}?",
            ]), "spoken operator"))
        elif pick < 0.34:
            out.append((rng.choice([
                f"what is {rng.randrange(5, 95, 5)}% of {n(10, 900)}?",
                f"what is {rng.randrange(5, 95, 5)} percent of {n(10, 900)}?",
                f"what is {rng.randrange(5, 95, 5)}% of {n(10, 900)} + {n()}?",
                f"what is {n(10, 900)} + {rng.randrange(5, 95, 5)}%?",
            ]), "percent"))
        elif pick < 0.46:
            out.append((rng.choice([
                f"what is {n(2, 12)} ^ {n(2, 6)}?",
                f"what is {n(2, 12)} ^ -{n(1, 4)}?",
                f"what is -{n(2, 9)} ^ 2?",
                f"what is {n(2, 9)} squared?",
                f"what is {n(2, 9)} cubed?",
                f"what is {n(2, 9)} to the power of {n(2, 5)}?",
                f"what is {n(2, 9)} ^ {n(2, 4)} ^ {n(2, 3)}?",
                "what is 0 ^ 0?",
                f"what is 2 ^ {rng.randint(1001, 4000)}?",
            ]), "power"))
        elif pick < 0.60:
            root = n(2, 40)
            out.append((rng.choice([
                f"what is the square root of {root * root}?",
                f"what is the square root of {n(2, 400)}?",
                f"what is the cube root of {n(2, 200)}?",
                f"what is the cube root of {n(2, 12) ** 3}?",
                f"what is the square root of -{n(2, 50)}?",
                f"what is sqrt({n(2, 200)})?",
            ]), "root"))
        elif pick < 0.74:
            places = rng.randint(1, 5)
            word = ["one", "two", "three", "four", "five"][places - 1]
            out.append((rng.choice([
                f"what is {n(10, 900)} / {rng.choice((3, 6, 7, 9, 11))} "
                f"to {places} decimal places?",
                f"what is {n(10, 900)} / {rng.choice((3, 7, 9))} "
                f"to {word} decimal places?",
                f"what is {n(10, 900)} / {rng.choice((3, 7, 9))} "
                f"to the nearest whole number?",
                f"what is the square root of {n(2, 200)} to {places} "
                f"decimal places?",
                f"what is {n(10, 900)} / {rng.choice((3, 7))} to "
                f"{rng.randint(2, 3)} significant figures?",
            ]), "rounding"))
        elif pick < 0.88:
            count = rng.choice((2, 3, 4))
            op = rng.choice(["sum", "average", "largest", "smallest",
                             "mean", "total", "maximum", "minimum"])
            if rng.random() < 0.5:
                items = [str(n()) for _ in range(count)]
            else:
                unit = rng.choice(_UNITS)
                items = [f"{n()} {unit}" for _ in range(count)]
                if rng.random() < 0.2:
                    items[-1] = f"{n()} {rng.choice(_UNITS)}"
            body = ", ".join(items[:-1]) + " and " + items[-1]
            out.append((f"list the {op} of {body}", "list"))
        else:
            out.append((rng.choice(_NOT_ARITHMETIC), "not arithmetic"))
    return out[:cases]


def run_gate(cases: int = 3000, seed: int = 0) -> CalcParityReport:
    """Ask both tiers the same corpus and compare the records."""
    if not _PROBE.exists():
        return CalcParityReport(
            passes=False,
            reason="FAIL - the native reader is not built; run "
                   "native/uq/build.ps1")
    rng = random.Random(seed)
    plan = _questions(rng, cases)
    families: dict = {}
    for _question, family in plan:
        families[family] = families.get(family, 0) + 1

    got = subprocess.run(
        [str(_PROBE)],
        input="\n".join(question for question, _f in plan) + "\n",
        capture_output=True, text=True, check=True)
    native = got.stdout.splitlines()

    mismatches = 0
    python_only = 0
    native_only = 0
    first = ""
    for index, (question, family) in enumerate(plan):
        want = _python_record(question)
        have = native[index] if index < len(native) else "(no answer)"
        if have == want:
            continue
        mismatches += 1
        if want == "none" and have != "none":
            native_only += 1
        elif have == "none" and want != "none":
            python_only += 1
        if not first:
            first = (f"[{family}] {question!r}\n"
                     f"    python: {want}\n"
                     f"    native: {have}")

    passes = mismatches == 0
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {len(plan)} questions asked "
        f"of both tiers, {mismatches} records differed "
        f"({python_only} answered only by Python, {native_only} only "
        f"by the native tier)"
        + (f"\nfirst mismatch:\n{first}" if first else ""))
    return CalcParityReport(passes=passes, checked=len(plan),
                            mismatches=mismatches,
                            python_only=python_only,
                            native_only=native_only, families=families,
                            first_mismatch=first, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"questions asked of both tiers: {report.checked}")
    print(f"records that differed: {report.mismatches}")
    print(f"  answered only by Python: {report.python_only}")
    print(f"  answered only by the native tier: {report.native_only}")
    print()
    for family, count in sorted(report.families.items()):
        print(f"  {family:<18} {count:5d} cases")
    print()
    print(report.reason)
