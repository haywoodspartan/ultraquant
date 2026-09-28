"""A why-question takes the stance of the polar question inside it.

11.117: the why handler compared two ways and the polar handler three,
so a denial of one value was answered "It isn't" for another, and a
true claim was denied. These pins hold the shared rule, the arm that
keeps it measurable, and the record of the gate.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T


def _ask(facts: list, question: str, three_way: bool = True) -> str:
    previous = T._WHY_THREE_WAY
    T._WHY_THREE_WAY = three_way
    root = Path(tempfile.mkdtemp(prefix="uq_whystance_t_"))
    try:
        session = T.build_session(root, seed=0)
        for fact in facts:
            T.run_pipeline(fact, session)
        return T.run_pipeline(question, session)[0]
    finally:
        T._WHY_THREE_WAY = previous
        shutil.rmtree(root, ignore_errors=True)


class ClaimStanceTests(unittest.TestCase):
    """The one rule both handlers now use."""

    def test_the_whole_table(self) -> None:
        # (matches, claim_negated, held_negated) -> stance
        table = {
            (True, False, False): True,     # held V, claim V
            (True, True, False): False,     # held V, claim not V
            (False, False, False): False,   # held V, claim W
            (False, True, False): True,     # held V, claim not W
            (True, False, True): False,     # held not V, claim V
            (True, True, True): True,       # held not V, claim not V
            (False, False, True): None,     # held not V, claim W
            (False, True, True): None,      # held not V, claim not W
        }
        for args, stance in table.items():
            with self.subTest(args=args):
                self.assertIs(T._claim_stance(*args), stance)


class WhyStanceTests(unittest.TestCase):
    """The reproductions, the controls, and the arm."""

    def test_a_denial_of_one_value_is_not_a_denial_of_another(self) -> None:
        reply = _ask(["the tower material is not steel"],
                     "why is the tower material iron?")
        self.assertTrue(reply.startswith("I don't know that it is"), reply)
        self.assertIn("I hold only that tower material is not steel", reply)

    def test_a_true_claim_is_explained_not_denied(self) -> None:
        reply = _ask(["the tower material is steel"],
                     "why is the tower material not iron?")
        self.assertTrue(reply.startswith("Because"), reply)

    def test_a_derived_denial_is_not_a_denial_of_another_value(self) -> None:
        reply = _ask(["the tower material is steel",
                      "the steel hardness is not high"],
                     "why is the tower hardness low?")
        self.assertTrue(reply.startswith("I don't know that it is"), reply)

    def test_real_contradictions_are_still_corrected(self) -> None:
        """The anti-rationalisation line, unmoved."""
        self.assertTrue(_ask(["the tower material is steel"],
                             "why is the tower material iron?")
                        .startswith("It isn't"))
        self.assertTrue(_ask(["the tower material is not steel"],
                             "why is the tower material steel?")
                        .startswith("It isn't"))

    def test_the_flag_restores_the_two_way_test(self) -> None:
        reply = _ask(["the tower material is not steel"],
                     "why is the tower material iron?", three_way=False)
        self.assertTrue(reply.startswith("It isn't"), reply)

    def test_polar_still_says_what_it_said(self) -> None:
        """Polar moved onto the shared helper; its words did not move."""
        self.assertTrue(_ask(["the tower material is not steel"],
                             "is the tower material iron?")
                        .startswith("I don't know - I hold only"))
        self.assertTrue(_ask(["the tower material is steel"],
                             "is the tower material not iron?")
                        .startswith("Yes - tower material is steel, not"))

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(T._WHY_THREE_WAY)


class GateVerdictTests(unittest.TestCase):
    """The pass, the prediction it had to meet, and the reference check."""

    def setUp(self) -> None:
        from ultraquant.experiments import whystance_gate
        self.gate = whystance_gate
        self.doc = " ".join(whystance_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Why agrees with polar on every cell",
                       "The old arm disagrees exactly where the two-way "
                       "test is wrong",
                       "Anti-rationalisation holds",
                       "Nothing else moves",
                       "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_prediction_is_six_cells(self) -> None:
        self.assertEqual(len(self.gate.PREDICTED_DEFECT), 6)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("16 of 16", self.doc)
        self.assertIn("exactly the six predicted cells", self.doc)

    def test_the_reference_was_checked_against_the_old_code(self) -> None:
        self.assertIn("0 of 16 differ", self.doc)


if __name__ == "__main__":
    unittest.main()
