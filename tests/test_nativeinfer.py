"""Two spreads that converge the same way - §11.92, pinned.

The decoys matter more than the chains here: a port that derives
MORE than Python is worse than one that derives less, because every
extra answer is a pun the rule stack exists to refuse.
"""

from __future__ import annotations

import subprocess
import unittest

from ultraquant.experiments import nativeinfer_gate

_PROBE = nativeinfer_gate.probe_path()
_BUILT = _PROBE.exists()


def _native(lines: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return [line for line in got.stdout.splitlines() if line != "ok"]


def _python(statements, questions) -> list[str]:
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.reason.inference import infer, missing_premise

    memory = SystematicMemory()
    for key, value, negated in statements:
        memory.remember_fact(key, value, 0.6, negated)
    out = []
    for question in questions:
        result = infer(question, memory)
        out.append("infer|" + (result.describe() if result else "none"))
        gap = missing_premise(question, memory)
        out.append("gap|" + (f"{gap['premise_key']}|{gap['via_key']}|"
                             f"{gap['via_value']}" if gap else "none"))
    return out


def _both(statements, questions):
    lines = [f"remember 0.6 {1 if neg else 0} {key}|{value}"
             for key, value, neg in statements]
    for question in questions:
        lines.append(f"infer {question}")
        lines.append(f"gap {question}")
    return _native(lines), _python(statements, questions)


@unittest.skipUnless(_BUILT, "native engine not built")
class ParityTests(unittest.TestCase):
    """Chains, decoys, denials and gaps, in both tiers."""

    def test_a_two_fact_chain(self) -> None:
        native, python = _both(
            [("tower material", "iron", False),
             ("iron hardness", "490 units", False)],
            ["what is the tower hardness?"])
        self.assertEqual(native, python)
        self.assertIn("via iron", native[0])

    def test_a_four_fact_chain(self) -> None:
        native, python = _both(
            [("tower architect", "wren", False),
             ("wren hometown", "bristol", False),
             ("bristol region", "west", False),
             ("west climate", "temperate", False)],
            ["what is the tower architect hometown region climate?"])
        self.assertEqual(native, python)

    def test_a_denial_never_bridges(self) -> None:
        native, python = _both(
            [("dome material", "steel", True),
             ("steel hardness", "700 units", False)],
            ["what is the dome hardness?"])
        self.assertEqual(native, python)
        self.assertEqual(native[0], "infer|none")

    def test_an_epithet_does_not_answer_for_the_plain_name(self) -> None:
        native, python = _both(
            [("wren the younger hometown", "norwich", False)],
            ["what is the wren hometown?"])
        self.assertEqual(native, python)

    def test_a_modifier_is_read_away_and_named(self) -> None:
        native, python = _both(
            [("tower material", "iron", False),
             ("iron hardness", "490 units", False)],
            ["what is the weathered tower hardness?"])
        self.assertEqual(native, python)
        self.assertIn("as a modifier", native[0])

    def test_a_curiosity_gap_is_minted_identically(self) -> None:
        native, python = _both(
            [("tower material", "iron", False)],
            ["what is the tower conductivity?"])
        self.assertEqual(native, python)
        self.assertEqual(native[1], "gap|iron conductivity|"
                                    "tower material|iron")

    def test_generated_worlds_agree(self) -> None:
        report = nativeinfer_gate.run_gate(worlds=8, seed=3)
        self.assertEqual(report.mismatches, 0, report.reason)
        self.assertEqual(report.gap_mismatches, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The PASS, and why converging more would be worse."""

    def setUp(self) -> None:
        self.doc = " ".join(nativeinfer_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| derivations that differed | **0** |", self.doc)

    def test_over_derivation_is_named_as_the_worse_failure(self) -> None:
        self.assertIn("A tier that converges MORE is worse than one "
                      "that converges less", self.doc)

    def test_the_first_time_pass_is_attributed(self) -> None:
        """A clean first run is a claim about preparation, not luck."""
        self.assertIn("the rules were not re-derived here, they were "
                      "copied", self.doc)


if __name__ == "__main__":
    unittest.main()
