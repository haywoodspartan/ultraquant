"""Aggregates: the family counted, totalled, and averaged - honestly.

§11.57's registered claim: count, total, and average over the same
enumerated family the superlative ranks, scope named, exclusions
named, denials never entering arithmetic. The gate passed at +0.657,
5.05x seed sd with zero drifted numbers; these tests pin the matrix.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class AggregateTests(unittest.TestCase):
    """Count, total, average, and every honesty rule."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import (build_session,
                                                     run_pipeline)

        self.dir = Path(tempfile.mkdtemp(prefix="uq_aggregate_"))
        self.session = build_session(self.dir, seed=0)
        self._run = run_pipeline
        for statement in ("the tower height is 300 meters",
                          "the bridge height is 120 meters",
                          "the spire height is 2 kilometers",
                          "the keep height is not 900 meters",
                          "the mill height is unknown"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, question: str) -> str:
        response, _trace = self._run(question, self.session)
        return response

    def test_the_count_reports_denials_beside_facts(self) -> None:
        """A denial is a belief too - counted as what it is, never as
        what it is not."""
        response = self._ask("how many height facts do you hold?")
        self.assertIn("I hold 4 height fact(s)", response)
        self.assertIn("plus 1 denial(s)", response)

    def test_the_total_converts_and_names_its_scope(self) -> None:
        response = self._ask("what is the total height?")
        self.assertIn("2.42 kilometers", response)
        self.assertIn("of the 3 height facts I hold", response)
        self.assertIn("units converted", response)

    def test_the_denied_value_never_moves_the_number(self) -> None:
        """The keep's negated 900 meters would push the total to 3.32
        kilometers. It must not move it a millimeter."""
        response = self._ask("what is the total height?")
        self.assertIn("2.42", response)
        self.assertNotIn("3.32", response)

    def test_the_average_divides_the_same_family(self) -> None:
        response = self._ask("what is the average height?")
        self.assertIn("average", response)
        self.assertIn("of the 3 height facts I hold", response)

    def test_the_wordy_fact_is_excluded_by_name(self) -> None:
        response = self._ask("what is the total height?")
        self.assertIn("mill height (no number)", response)

    def test_an_empty_family_refuses_both_ways(self) -> None:
        self.assertIn("I hold no depth facts",
                      self._ask("how many depth facts do you hold?"))
        self.assertIn("I hold no numeric depth facts",
                      self._ask("what is the total depth?"))

    def test_the_pairwise_door_is_untouched(self) -> None:
        """"the sum of the tower height and the bridge height" names
        its operands - §11.30's combine, not an aggregate, and the
        aggregate door must not shadow it."""
        response = self._ask("what is the sum of the tower height "
                             "and the bridge height?")
        self.assertIn("420", response)
        self.assertIn("inferred", response)
        self.assertNotIn("facts I hold", response)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the hardening, and the shared rules, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import aggregate_gate

        doc = " ".join(aggregate_gate.__doc__.split())
        self.assertIn("+0.657 at 5.05x seed sd", doc)
        self.assertIn("zero drifted numbers", doc)

    def test_the_float_hardening_is_recorded(self) -> None:
        """A harness hazard closed before it fired - recorded so the
        next gate with an exact-string float check inherits the
        lesson."""
        from ultraquant.experiments import aggregate_gate

        doc = " ".join(aggregate_gate.__doc__.split())
        self.assertIn("a spurious drift waiting on a seed", doc)
        self.assertIn("exact binary eighths", doc)

    def test_the_shared_rules_are_recorded(self) -> None:
        from ultraquant.experiments import aggregate_gate

        doc = " ".join(aggregate_gate.__doc__.split())
        self.assertIn("name the scope, name the exclusions, and never "
                      "let a denial touch a number", doc)


if __name__ == "__main__":
    unittest.main()
