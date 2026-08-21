"""The question that needed no memory - §11.86, a negative result.

The skip is real, correct, and worthless: 20-60x on expression
questions at perfect parity, and no gain at all for the session,
because the paging it removed was warming the cache for everything
that followed. Pinned so the finding is not rediscovered as an idea.
"""

from __future__ import annotations

import unittest


class MechanismTests(unittest.TestCase):
    """It stays off, and it stays here."""

    def test_the_skip_is_switched_off(self) -> None:
        from ultraquant.interpreter import thoughts

        self.assertFalse(thoughts._ARITHMETIC_SKIPS_RECALL,
                         "a measured-and-rejected mechanism must not "
                         "ship switched on")

    def test_the_reason_is_beside_the_switch(self) -> None:
        """A flag set to False with no explanation is an invitation
        to switch it back on."""
        import inspect

        from ultraquant.interpreter import thoughts

        source = inspect.getsource(thoughts)
        marker = source.split("_ARITHMETIC_SKIPS_RECALL = False")[0]
        tail = marker[-900:]
        self.assertIn("MEASURED AND REJECTED", tail)
        self.assertIn("no faster at all", tail)

    def test_the_predicate_still_exists_for_the_gate(self) -> None:
        from ultraquant.interpreter.thoughts import _is_arithmetic

        self.assertTrue(_is_arithmetic("what is 12 + 3 * 4?"))
        self.assertFalse(_is_arithmetic("what is the tower height?"))


class GateVerdictTests(unittest.TestCase):
    """The recorded FAIL, and the finding it bought."""

    def setUp(self) -> None:
        from ultraquant.experiments import recallskip_gate

        self.doc = " ".join(recallskip_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_failure(self) -> None:
        self.assertIn("FAILED - on criterion 4", self.doc)

    def test_parity_held_even_though_the_gate_failed(self) -> None:
        """The mechanism was correct. It was simply not worth
        anything, which is a different verdict."""
        self.assertIn("Zero response differences and zero "
                      "recalled-fact differences", self.doc)

    def test_the_total_is_recorded(self) -> None:
        self.assertIn("1497.5 ms", self.doc)
        self.assertIn("1530.3 ms", self.doc)
        self.assertIn("The skip saves nothing", self.doc)

    def test_the_real_target_is_named(self) -> None:
        self.assertIn("the price of a question at this density is "
                      "the bucket cache, not the candidate ladder",
                      self.doc)


if __name__ == "__main__":
    unittest.main()
