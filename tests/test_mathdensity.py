"""The arithmetic capstone - §11.85, pinned.

Ten arithmetic forms at density, eight refusal families at zero, and
the price line that located Recall's cost on a question that needs
no recall.
"""

from __future__ import annotations

import unittest


class GateVerdictTests(unittest.TestCase):
    """The recorded floors, controls and price."""

    def setUp(self) -> None:
        from ultraquant.experiments import mathdensity_gate

        self.doc = " ".join(mathdensity_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("Ten forms at 1.000, zero fabrications across "
                      "all eight refusal families, zero false "
                      "bounds", self.doc)

    def test_every_form_is_named_with_its_floor(self) -> None:
        for form in ("literal", "percent", "power", "root exact",
                     "root enclosed", "quantity", "reached",
                     "comparison", "polar arithmetic", "rounding"):
            self.assertIn(f"| {form} | 1.000 |", self.doc,
                          f"{form} lost its recorded floor")

    def test_the_price_line_located_the_cost(self) -> None:
        """A price line that reports a number and stops has done
        half its job."""
        self.assertIn("the Recall stage is ~90% of the price of a "
                      "question it cannot possibly help with",
                      self.doc)
        self.assertIn("0.016 ms", self.doc)

    def test_the_opportunity_is_registered_not_taken(self) -> None:
        self.assertIn("a measured opportunity, not a change",
                      self.doc)


class ShapeTests(unittest.TestCase):
    """The gate's own structure: floors that leave no room."""

    def test_the_floors_are_gated_at_one(self) -> None:
        import inspect

        from ultraquant.experiments import mathdensity_gate

        source = inspect.getsource(mathdensity_gate.run_gate)
        self.assertIn("value >= 1.0 for value in forms.values()",
                      source)

    def test_every_refusal_family_is_carried(self) -> None:
        import inspect

        from ultraquant.experiments import mathdensity_gate

        source = inspect.getsource(mathdensity_gate.run_gate)
        for family in ("unheld operand", "denial operand",
                       "non-numeric operand", "unit times unit",
                       "division by zero", "dangling percent",
                       "unknown relation", "zero to the zero"):
            self.assertIn(family, source)


if __name__ == "__main__":
    unittest.main()
