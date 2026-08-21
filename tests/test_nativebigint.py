"""The native tier's integers agree with Python's - §11.88, pinned.

The probe is a built executable, so these pins skip when it is
absent. That is deliberate rather than lenient: the Python tier
defines the semantics and answers every question by itself, so a
machine with no compiler must still see a green suite. What must
never happen is a native tier that is present and WRONG, and that is
what the gate is for.
"""

from __future__ import annotations

import subprocess
import unittest

from ultraquant.experiments import nativebigint_gate

_PROBE = nativebigint_gate.probe_path()
_BUILT = _PROBE.exists()


def _ask(lines: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


@unittest.skipUnless(_BUILT, "native probe not built")
class ParityTests(unittest.TestCase):
    """The cases a 64-bit port would get wrong."""

    def test_arithmetic_past_a_machine_word(self) -> None:
        big = 2 ** 300 + 12345
        answers = _ask([f"add {big} {big}", f"mul {big} {big}",
                        f"pow 2 1000"])
        self.assertEqual(answers[0], str(big + big))
        self.assertEqual(answers[1], str(big * big))
        self.assertEqual(answers[2], str(2 ** 1000))

    def test_floor_division_takes_pythons_side(self) -> None:
        """C++ truncates toward zero; every rounding rule in this
        system is written in Python's floor terms."""
        for top, bottom in ((-7, 2), (7, -2), (-7, -2), (7, 2),
                            (-1, 3), (1, -3)):
            answers = _ask([f"fdiv {top} {bottom}",
                            f"fmod {top} {bottom}"])
            self.assertEqual(answers[0], str(top // bottom),
                             f"floor division of {top}/{bottom}")
            self.assertEqual(answers[1], str(top % bottom),
                             f"remainder of {top}/{bottom}")

    def test_integer_roots_are_exact(self) -> None:
        for value, degree in ((2, 2), (999999999999, 2),
                              (10 ** 40 + 7, 2), (27, 3), (10 ** 33, 3)):
            root = int(_ask([f"iroot {value} {degree}"])[0])
            self.assertLessEqual(root ** degree, value)
            self.assertGreater((root + 1) ** degree, value)

    def test_zero_has_one_spelling(self) -> None:
        answers = _ask(["sub 5 5", "mul 0 12345", "str -0"])
        self.assertEqual(answers, ["0", "0", "0"])


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS, and the oracle's own limit."""

    def setUp(self) -> None:
        self.doc = " ".join(nativebigint_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("4,000 operations, not one disagreement",
                      self.doc)
        self.assertIn("| mismatches | **0** |", self.doc)

    def test_the_oracles_own_limit_is_recorded(self) -> None:
        self.assertIn("CPython refuses to PRINT an integer past "
                      "4,300 digits", self.doc)

    def test_parity_is_gated_without_a_margin(self) -> None:
        self.assertIn("One mismatch is a fail", self.doc)


if __name__ == "__main__":
    unittest.main()
