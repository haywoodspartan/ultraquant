"""A conjunction of polar questions is answered as polar questions.

11.121: the compound rung rewrote every part of a conjunction as a
"what is" question, so "is A x and B y?" came back "none of it is held"
with junk curiosities - on every polar conjunction tried. These pins
hold the fix, the arm, the gate's record, and the review finding the
gate could not see: an "and" that joins two VALUES is not an "and" that
joins two questions.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T

_FACTS = ["the tower material is steel", "the bridge material is iron",
          "the tower height is 300 meters"]


def _ask(question: str, on: bool = True) -> str:
    previous = T._COMPOUND_POLAR
    T._COMPOUND_POLAR = on
    root = Path(tempfile.mkdtemp(prefix="uq_compoundpolar_t_"))
    try:
        session = T.build_session(root, seed=0)
        for fact in _FACTS:
            T.run_pipeline(fact, session)
        return T.run_pipeline(question, session)[0]
    finally:
        T._COMPOUND_POLAR = previous
        shutil.rmtree(root, ignore_errors=True)


class PolarConjunctionTests(unittest.TestCase):

    def test_each_part_gets_its_polar_answer(self) -> None:
        reply = _ask("is the tower material steel and the bridge material "
                     "steel?")
        self.assertTrue(reply.startswith("Yes - tower material is steel"),
                        reply)
        self.assertIn("; No - bridge material is iron, not steel", reply)

    def test_no_false_claim_that_nothing_is_held(self) -> None:
        reply = _ask("is the tower material steel and the bridge material "
                     "iron?")
        self.assertNotIn("none of it is held", reply)

    def test_arithmetic_conjunctions_are_answered(self) -> None:
        reply = _ask("is 3 + 4 equal to 7 and 2 + 2 equal to 4?")
        self.assertTrue(reply.startswith("Yes"), reply)

    def test_the_flag_restores_the_old_rung(self) -> None:
        reply = _ask("is the tower material steel and the bridge material "
                     "iron?", on=False)
        self.assertIn("none of it is held", reply)

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(T._COMPOUND_POLAR)


class ValueConjunctionTests(unittest.TestCase):
    """Review finding: "and" inside a value is not a conjunction.

    Version 1 routed every non-wh question holding " and " to the polar
    conjunction path, ahead of everything else, and so abstained on
    questions the system had answered correctly: "is the tower material
    steel and iron?" had been "No - tower material is steel, not steel
    and iron". These must keep their pre-fix replies byte for byte.
    """

    def test_a_conjoined_value_keeps_its_answer(self) -> None:
        question = "is the tower material steel and iron?"
        self.assertEqual(_ask(question), _ask(question, on=False))

    def test_and_inside_a_list_or_sum_is_left_alone(self) -> None:
        for question in ("is 7 the sum of 3 and 4?",
                         "is 5 the largest of 3, 5 and 2?"):
            with self.subTest(question=question):
                self.assertEqual(_ask(question), _ask(question, on=False))


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import compoundpolar_gate
        self.gate = compoundpolar_gate
        self.doc = " ".join(compoundpolar_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Polar conjunctions answered per part",
                       "No junk curiosities",
                       "Supported compounds unchanged",
                       "Nothing else moves", "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_battery_can_fail(self) -> None:
        self.assertIn("The OLD arm must fail on every case", self.doc)
        self.assertEqual(len(self.gate.CASES), 8)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("PASSED on all four measured criteria", self.doc)

    def test_the_version_that_was_not_shipped_is_recorded(self) -> None:
        self.assertIn("Version one passed too, and was not shipped", self.doc)


if __name__ == "__main__":
    unittest.main()
