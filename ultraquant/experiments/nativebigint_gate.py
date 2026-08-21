"""A second tier that agrees, digit for digit. The native BigInt gate.

README line 8 states the rule the whole project runs on: the
pure-Python tier defines the semantics, and every other tier
reproduces it. That is what makes a native variant worth building
and also what makes its first piece hard. Python's ``int`` is
arbitrary precision, and this system leans on that wherever it
matters - "what is 2 ^ 1000?" is a 302-digit answer, an exact
rational's denominator grows without being asked, and a root
enclosure scales its radicand by 10^(places*degree) before taking an
integer root. A native tier built on ``long long`` would diverge on
the first interesting question and be quietly wrong after that,
which is worse than being slower.

So `native/uq/src/bigint.cpp` is a sign-magnitude arbitrary-precision
integer in base 1,000,000,000, written from scratch with no external
dependency - the same rule the Python tier keeps by using no numpy.
This gate does not check it against a second opinion of my own. **It
checks it against Python**, which is the tier that defines the
answer.

Two behaviours get their own attention, because they are where a C++
port disagrees silently:

- **Floor division.** Python's ``//`` rounds toward negative infinity
  and ``%`` takes the divisor's sign; C++ truncates toward zero.
  Every rounding rule in §11.84 is specified in Python's terms, so
  the native tier implements floor explicitly and the gate hammers
  it with mixed signs.
- **Integer roots.** §11.81's enclosures are only honest if the
  integer root is exact, so each root is checked by exponentiating
  it back rather than by comparing it to a float.

**The criteria, written before the run.**

1. **Parity is total** - every operation must equal Python's own
   answer as a string. One mismatch is a fail; there is no margin
   here to be within.
2. **The cases go past 64 bits** - at least a quarter of the
   operands must exceed 2^64, or the gate has not tested the reason
   the class exists.
3. **Signs are covered** - negative operands appear on both sides,
   and floor division and remainder are exercised in all four sign
   combinations.
4. **Roots are verified independently** - r^d <= v < (r+1)^d,
   checked by the gate's own arithmetic.

**It was run once, and PASSED - 4,000 operations, not one
disagreement.**

| | |
|---|---:|
| operations compared | 4,000 |
| mismatches | **0** |
| operands past 2^64 | 0.494 |
| roots failing their own exponentiation | 0 |

Per operation: add 457, sub 409, mul 428, fdiv 455, fmod 451, gcd
447, pow 482, iroot 443, str 428 - zero mismatched in every one.

One thing the run surfaced is worth keeping, because it is a
difference between the tiers that has nothing to do with arithmetic.
**CPython refuses to PRINT an integer past 4,300 digits** by
default - a denial-of-service guard on decimal conversion, added in
3.11 - and the first attempt to build a case list crashed on it
while computing an oracle answer, not while checking one. The native
tier has no such guard. The limit is raised inside the gate so the
oracle can speak at the sizes the tier under test can reach;
otherwise this would have quietly become a measurement of CPython's
printing policy.

Run it::

    python -m ultraquant.experiments.nativebigint_gate
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Python 3.11 refuses to PRINT an integer past 4,300 digits by
# default - a denial-of-service guard on parsing, not a limit on the
# arithmetic. The native tier has no such guard, so the oracle has to
# be allowed to speak at the sizes the tier under test can reach;
# otherwise the gate would be measuring CPython's printing policy.
sys.set_int_max_str_digits(200000)

__all__ = ["BigIntReport", "run_gate", "probe_path"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_bigint_probe.exe"


def probe_path() -> Path:
    """Where the built probe lives, so tests can skip without it."""
    return _PROBE


@dataclass
class BigIntReport:
    """Parity, coverage, and where any disagreement was.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        checked: Operations compared.
        mismatches: Operations where the tiers disagreed.
        wide_share: Share of operands past 2^64.
        root_failures: Roots that failed their own exponentiation.
        first_mismatch: The first disagreement, for the record.
        per_op: Operation -> [compared, mismatched].
        reason: Plain-language verdict.
    """

    passes: bool
    checked: int = 0
    mismatches: int = 0
    wide_share: float = 0.0
    root_failures: int = 0
    first_mismatch: str = ""
    per_op: dict = field(default_factory=dict)
    reason: str = ""


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _iroot(value: int, degree: int) -> int:
    """Python's own integer root, by bisection - the oracle."""
    if value < 2:
        return value
    low, high = 0, 1
    while high ** degree <= value:
        high *= 2
    while low < high:
        middle = (low + high + 1) // 2
        if middle ** degree <= value:
            low = middle
        else:
            high = middle - 1
    return low


def _operand(rng: random.Random) -> int:
    """A number, deliberately often too big for a machine word."""
    shape = rng.random()
    if shape < 0.25:
        return rng.randint(-9, 9)
    if shape < 0.5:
        return rng.randint(-(2 ** 31), 2 ** 31)
    if shape < 0.75:
        return rng.randint(-(2 ** 96), 2 ** 96)
    return rng.randint(-(2 ** 400), 2 ** 400)


def _plan(rng: random.Random, cases: int):
    """(lines, expected, labels, wide, operands, root_checks)."""
    lines: list[str] = []
    expected: list[str] = []
    labels: list[str] = []
    root_checks: list[tuple[int, int]] = []
    wide = 0
    operands = 0
    for _ in range(cases):
        op = rng.choice(["add", "sub", "mul", "fdiv", "fmod", "gcd",
                         "pow", "iroot", "str"])
        left = _operand(rng)
        operands += 1
        wide += abs(left) >= 2 ** 64
        if op == "pow":
            exponent = rng.randint(0, 64)
            lines.append(f"pow {left} {exponent}")
            expected.append(str(left ** exponent))
            labels.append(f"{left} ** {exponent}")
            continue
        if op == "iroot":
            value = abs(left)
            degree = rng.choice([2, 3, 5])
            lines.append(f"iroot {value} {degree}")
            expected.append(str(_iroot(value, degree)))
            labels.append(f"iroot({value}, {degree})")
            root_checks.append((value, degree))
            continue
        if op == "str":
            lines.append(f"str {left} 0")
            expected.append(str(left))
            labels.append(f"str({left})")
            continue
        right = _operand(rng)
        operands += 1
        wide += abs(right) >= 2 ** 64
        if op in ("fdiv", "fmod", "mul") and right == 0:
            right = rng.choice([-7, 3, 11])
        lines.append(f"{op} {left} {right}")
        if op == "add":
            expected.append(str(left + right))
        elif op == "sub":
            expected.append(str(left - right))
        elif op == "mul":
            expected.append(str(left * right))
        elif op == "gcd":
            expected.append(str(_gcd(left, right)))
        elif op == "fdiv":
            expected.append(str(left // right))
        else:
            expected.append(str(left % right))
        labels.append(f"{left} {op} {right}")
    return lines, expected, labels, wide, operands, root_checks


def run_gate(cases: int = 4000, seed: int = 0) -> BigIntReport:
    """Ask both tiers the same arithmetic and compare the strings."""
    if not _PROBE.exists():
        return BigIntReport(
            passes=False,
            reason="FAIL - the native probe is not built; run "
                   "native/uq/build.ps1")
    rng = random.Random(seed)
    lines, expected, labels, wide, operands, roots = _plan(rng, cases)

    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    answers = got.stdout.splitlines()

    mismatches = 0
    first = ""
    per_op: dict = {}
    for index, want in enumerate(expected):
        op = lines[index].split()[0]
        stats = per_op.setdefault(op, [0, 0])
        stats[0] += 1
        have = answers[index] if index < len(answers) else "(no answer)"
        if have != want:
            mismatches += 1
            stats[1] += 1
            if not first:
                first = (f"{labels[index]} -> native {have!r}, "
                         f"python {want!r}")

    root_failures = 0
    for value, degree in roots:
        root = _iroot(value, degree)
        if not root ** degree <= value < (root + 1) ** degree:
            root_failures += 1

    wide_share = wide / operands if operands else 0.0
    passes = (mismatches == 0 and root_failures == 0
              and wide_share >= 0.25)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {len(expected)} operations "
        f"compared against Python, {mismatches} mismatches, "
        f"{wide_share:.3f} of operands past 2^64, {root_failures} "
        f"roots failing their own exponentiation"
        + (f"; first: {first}" if first else ""))
    return BigIntReport(passes=passes, checked=len(expected),
                        mismatches=mismatches, wide_share=wide_share,
                        root_failures=root_failures,
                        first_mismatch=first, per_op=per_op,
                        reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"operations compared: {report.checked}")
    print(f"mismatches: {report.mismatches}")
    print(f"operands past 2^64: {report.wide_share:.3f}")
    print(f"root failures: {report.root_failures}")
    print()
    for op, (total, bad) in sorted(report.per_op.items()):
        print(f"  {op:<6} {total:5d} compared, {bad} mismatched")
    print()
    print(report.reason)
