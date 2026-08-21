"""Unit conversion: definitions convert, everything else still refuses.

§11.30's blanket refusal narrowed to exactly the pairs a definition
connects (§11.42): 300 meters + 2 kilometers is 2.3 kilometers, never
302; degrees, florins, and bare numbers refuse as they always did.
"""

from __future__ import annotations

import unittest

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import _convert, infer


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class ConvertTests(unittest.TestCase):
    """The table itself: exact factors, family boundaries."""

    def test_within_family_is_exact(self) -> None:
        self.assertAlmostEqual(_convert(300, "meter", "kilometer"), 0.3)
        self.assertAlmostEqual(_convert(2, "hour", "minute"), 120.0)

    def test_across_families_is_none(self) -> None:
        self.assertIsNone(_convert(300, "meter", "second"))

    def test_unknown_units_are_none(self) -> None:
        self.assertIsNone(_convert(4, "florin", "meter"))
        self.assertIsNone(_convert(4, "meter", "florin"))


class CombineConversionTests(unittest.TestCase):
    """The live combine path with the table in play."""

    def test_the_result_reads_in_the_larger_unit_and_says_so(self) -> None:
        memory = _memory({"tower height": "300 meters",
                          "bridge length": "2 kilometers"})
        result = infer("what is the sum of the tower height and the "
                       "bridge length?", memory)
        self.assertIsNotNone(result)
        self.assertIn("2.3 kilometers", result.answer)
        self.assertIn("(units converted)", result.answer)

    def test_comparison_converts_and_labels(self) -> None:
        """2 kilometers beats 300 meters, shown in the result unit -
        an unlabelled '2' beside a '300' misled before the fix."""
        memory = _memory({"tower height": "300 meters",
                          "bridge length": "2 kilometers"})
        result = infer("which is larger, the tower height or the bridge "
                       "length?", memory)
        self.assertIsNotNone(result)
        self.assertIn("bridge length", result.answer)
        self.assertIn("2 kilometers", result.answer)

    def test_a_bare_number_refuses(self) -> None:
        """Assuming a missing unit would be invention."""
        memory = _memory({"tower height": "300 meters",
                          "bridge length": "2"})
        self.assertIsNone(infer("what is the sum of the tower height and "
                                "the bridge length?", memory))

    def test_same_unit_answers_carry_no_conversion_label(self) -> None:
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters"})
        result = infer("what is the sum of the tower height and the "
                       "bridge height?", memory)
        self.assertIsNotNone(result)
        self.assertNotIn("converted", result.answer)

    def test_same_unit_answers_still_name_the_unit(self) -> None:
        """§11.77. The pin above asserted only what the answer must
        NOT say, so it could not notice that the answer had stopped
        saying what it was an answer IN: "= 420" shipped for two
        premises that both read in meters.
        """
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters"})
        result = infer("what is the sum of the tower height and the "
                       "bridge height?", memory)
        self.assertIn("420 meters", result.answer)

    def test_every_combine_shape_names_the_unit(self) -> None:
        memory = _memory({"tower height": "300 meters",
                          "bridge height": "120 meters"})
        difference = infer("what is the difference between the tower "
                           "height and the bridge height?", memory)
        self.assertIn("180 meters", difference.answer)
        larger = infer("which is larger, the tower height or the "
                       "bridge height?", memory)
        self.assertIn("300 meters", larger.answer)

    def test_unitless_operands_never_gain_a_unit(self) -> None:
        """The opposite failure, and the reason the fix reads the
        operands rather than assuming a default."""
        memory = _memory({"north count": "30", "south count": "20"})
        result = infer("what is the sum of the north count and the "
                       "south count?", memory)
        self.assertIn("= 50", result.answer)
        for unit in ("meter", "kilogram", "second"):
            self.assertNotIn(unit, result.answer)


class GateVerdictTests(unittest.TestCase):
    """The PASS and its boundaries, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import conversion_gate

        doc = " ".join(conversion_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("1.29x seed sd", doc)
        self.assertIn("the refusal narrowed, nothing widened", doc)

    def test_the_zero_falls_are_recorded(self) -> None:
        from ultraquant.experiments import conversion_gate

        doc = " ".join(conversion_gate.__doc__.split())
        self.assertIn("zero refusal falls", doc)

    def test_the_power_correction_is_recorded(self) -> None:
        from ultraquant.experiments import conversion_gate

        doc = " ".join(conversion_gate.__doc__.split())
        self.assertIn("cannot clear one sd whatever the mechanism does",
                      doc)


if __name__ == "__main__":
    unittest.main()
