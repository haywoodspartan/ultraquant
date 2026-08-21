"""Containment corroboration: subsets credit, negations never do.

The bounded exception §11.41 measured into the consensus machinery: a
position and its elaboration are one position; a denial containing the
same words is not; different words are never equated. These tests pin
the mechanism and the verdict that licensed it.
"""

from __future__ import annotations

import unittest

from ultraquant.interpreter.llmls import _containment_core


class ContainmentCoreTests(unittest.TestCase):
    """The core-finding rules, one by one."""

    def test_an_elaboration_backs_its_kernel(self) -> None:
        split = {
            "system communication": ["m1"],
            "system communication using symbols": ["m2"],
            "unrelated notion": ["m3"],
        }
        result = _containment_core(split)
        self.assertIsNotNone(result)
        core, merged = result
        self.assertEqual(core, "system communication")
        self.assertEqual(sorted(merged), ["m1", "m2"])

    def test_the_deepest_shared_core_wins_by_backers(self) -> None:
        """'communication' is inside all three - the live run's shape."""
        split = {
            "communication": ["m1"],
            "system communication": ["m2"],
            "system communication using symbols": ["m3"],
        }
        core, merged = _containment_core(split)
        self.assertEqual(core, "communication")
        self.assertEqual(len(merged), 3)

    def test_a_negated_superset_backs_nothing(self) -> None:
        """'earth round hoax' contains 'earth round' and denies it."""
        split = {
            "earth round": ["m1"],
            "earth round hoax": ["m2"],
        }
        self.assertIsNone(_containment_core(split))

    def test_a_negated_core_is_never_credited(self) -> None:
        split = {
            "not real": ["m1"],
            "not real at all": ["m2"],
        }
        self.assertIsNone(_containment_core(split))

    def test_different_words_never_merge(self) -> None:
        split = {"it fast": ["m1"], "it quick": ["m2"]}
        self.assertIsNone(_containment_core(split))

    def test_a_single_position_does_not_self_credit(self) -> None:
        split = {"earth round": ["m1", "m2"]}
        self.assertIsNone(_containment_core(split))

    def test_overlap_without_nesting_is_the_ceiling(self) -> None:
        """Shared tokens, neither a subset: honestly uncreditable."""
        split = {
            "system communication": ["m1"],
            "system altered idea": ["m2"],
        }
        self.assertIsNone(_containment_core(split))


class GateVerdictTests(unittest.TestCase):
    """The narrow pass and its controls, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import containment_gate

        doc = " ".join(containment_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.16x seed sd", doc)

    def test_both_zero_controls_are_recorded(self) -> None:
        from ultraquant.experiments import containment_gate

        doc = " ".join(containment_gate.__doc__.split())
        self.assertIn("zero negation falls", doc)
        self.assertIn("zero paraphrase falls", doc)

    def test_the_ceiling_is_overlap_without_nesting(self) -> None:
        from ultraquant.experiments import containment_gate

        doc = " ".join(containment_gate.__doc__.split())
        self.assertIn("overlap-without-nesting", doc)


if __name__ == "__main__":
    unittest.main()
