"""What it costs to change your mind — and the ties that came back.

§11.106 measured §12.3's three open axes. Two were ties. The most
useful part was an artifact the gate caught in its own favour: a
reasoning model with no `reasoning_effort` and an 80-token ceiling
returned EMPTY answers that scored as wrong. These pins hold the
battery's fairness, the fix, and the recorded result.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import correction_gate as gate
from ultraquant.experiments.headtohead_gate import classify


class BatteryTests(unittest.TestCase):
    """The corrections both systems receive."""

    def test_every_correction_actually_changes_the_value(self) -> None:
        for name, old, new in gate.SUBJECTS:
            self.assertNotEqual(old, new,
                                f"{name} is not corrected by {new!r}")

    def test_no_new_value_appears_as_another_subject_s_old_value(self):
        """Otherwise a stale answer could score as a correct one."""
        olds = {old for _n, old, _new in gate.SUBJECTS}
        for name, _old, new in gate.SUBJECTS:
            self.assertNotIn(new, olds,
                             f"{name}'s new value collides with some "
                             "subject's old one")

    def test_the_context_carries_both_the_fact_and_its_correction(self):
        """Criterion 1: the transformer must be able to know."""
        context = gate._llm_context(3)
        for name, old, new in gate.SUBJECTS[:3]:
            self.assertIn(f"the {name} is {old}", context)
            self.assertIn(f"the {name} is now {new}", context)

    def test_the_context_grows_with_the_corrections(self) -> None:
        """The cost being measured, checked rather than assumed."""
        sizes = [len(gate._llm_context(n).encode("utf-8"))
                 for n in (1, 5, 10, 20)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0] * 5)

    def test_the_transformer_is_told_a_correction_replaces(self) -> None:
        """Otherwise the accuracy measures the instruction."""
        self.assertIn("correction replaces", gate._SYSTEM)
        self.assertIn("CURRENT value", gate._SYSTEM)


class ReasoningBudgetTests(unittest.TestCase):
    """Run one's artifact, which pointed the flattering way."""

    def test_the_reasoning_effort_is_passed(self) -> None:
        """llmls.py documented this failure; the gate had ignored it."""
        import inspect

        source = inspect.getsource(gate)
        self.assertIn("reasoning_effort=NO_REASONING", source)
        self.assertNotIn("max_tokens=80", source)

    def test_an_empty_answer_is_not_scored_as_a_wrong_one(self) -> None:
        self.assertEqual(classify(""), "empty")
        self.assertFalse(gate._scored("", ["350 meters"]))

    def test_empties_are_counted_so_the_artifact_cannot_hide(self):
        import inspect

        self.assertIn('out["empty"]', inspect.getsource(gate))


class GateVerdictTests(unittest.TestCase):
    """A pass whose interesting claims were ties."""

    def setUp(self) -> None:
        self.doc = " ".join(gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Identical facts, identical corrections",
                       "Corrections are honoured",
                       "The cost of holding N corrections is reported",
                       "Derived answers name their premises",
                       "The transformer must win or tie somewhere"):
            self.assertIn(phrase, self.doc)

    def test_the_ties_are_recorded_as_ties(self) -> None:
        self.assertIn("Accuracy: a tie", self.doc)
        self.assertIn("Auditability: a tie", self.doc)

    def test_the_flattering_artifact_is_recorded(self) -> None:
        self.assertIn("entirely an artifact of the harness", self.doc)
        self.assertIn("check the instrument before believing the result",
                      self.doc)

    def test_the_cost_claim_is_bounded_honestly(self) -> None:
        """The slope was measured; the wall was not."""
        self.assertIn("the slope, not the wall", self.doc)
        self.assertIn("crossover is somewhere in the thousands", self.doc)


if __name__ == "__main__":
    unittest.main()
