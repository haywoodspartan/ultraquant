"""A semantic reading keeps the denial it read.

11.116: the suggester's reply was built from the stored value alone,
so a held denial reached that way came back as an assertion. These
pins hold the fix, the arm that keeps it measurable, and the record
of how it was found.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.interpreter import thoughts as T
from ultraquant.reason.semantic import SemanticSuggester, Suggestion


class _Constant:
    """Every text embeds to the same vector: every cosine is 1.0."""

    def embed(self, texts, model=None):
        return [[1.0, 0.0] for _ in texts]

    def available(self):
        return True


class PolarityTests(unittest.TestCase):
    """The semantic route against the lexical one, on the same record."""

    def setUp(self) -> None:
        self.roots: list[Path] = []

    def tearDown(self) -> None:
        for root in self.roots:
            shutil.rmtree(root, ignore_errors=True)

    def _reply(self, statement: str, question: str,
               polarity_on: bool = True, semantic: bool = True) -> str:
        previous = T._SEMANTIC_POLARITY
        T._SEMANTIC_POLARITY = polarity_on
        root = Path(tempfile.mkdtemp(prefix="uq_polarity_"))
        self.roots.append(root)
        try:
            kw = ({"semantic": SemanticSuggester(embedder=_Constant())}
                  if semantic else {})
            session = T.build_session(root, seed=0, **kw)
            T.run_pipeline(statement, session)
            return T.run_pipeline(question, session)[0]
        finally:
            T._SEMANTIC_POLARITY = previous

    def test_a_semantic_reading_keeps_the_denial(self) -> None:
        reply = self._reply("the kettle is not hot",
                            "what is the copper kettle?")
        self.assertIn("Reading that as 'kettle'", reply)
        self.assertIn("kettle is not hot", reply)

    def test_multiword_and_numeric_denials_survive(self) -> None:
        for statement, question, denial in (
                ("the granary roof is not slate grey",
                 "what is the east granary roof?", "not slate grey"),
                ("the mill height is not 300 meters",
                 "what is the stone mill height?", "not 300 meters")):
            with self.subTest(statement=statement):
                self.assertIn(denial, self._reply(statement, question))

    def test_the_flag_restores_the_old_render(self) -> None:
        """The arm: off, the defect is back byte for byte."""
        reply = self._reply("the kettle is not hot",
                            "what is the copper kettle?", polarity_on=False)
        self.assertIn("kettle is hot", reply)
        self.assertNotIn("not hot", reply)

    def test_an_affirmed_reading_is_unchanged_by_the_flag(self) -> None:
        on = self._reply("the kettle is hot", "what is the copper kettle?")
        off = self._reply("the kettle is hot", "what is the copper kettle?",
                          polarity_on=False)
        self.assertEqual(on, off)

    def test_the_lexical_route_agrees(self) -> None:
        """The control that was right all along."""
        reply = self._reply("the kettle is not hot", "what is the kettle?",
                            semantic=False)
        self.assertIn("kettle is not hot", reply)

    def test_the_fix_is_the_shipped_path(self) -> None:
        self.assertTrue(T._SEMANTIC_POLARITY)


class SuggestionTests(unittest.TestCase):
    """The carrier that dropped the flag, and what must not break."""

    def test_old_constructors_still_work(self) -> None:
        reading = Suggestion(key="k", value="v", confidence=0.5,
                             similarity=0.9)
        self.assertFalse(reading.negated)

    def test_suggest_carries_the_polarity(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="uq_suggest_"))
        try:
            session = T.build_session(root, seed=0)
            T.run_pipeline("the kettle is not hot", session)
            reading = SemanticSuggester(embedder=_Constant()).suggest(
                "what is the copper kettle?", session.memory)
            self.assertIsNotNone(reading)
            self.assertTrue(reading.negated)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class GateVerdictTests(unittest.TestCase):
    """The pass, what it counted, and how the defect was found."""

    def setUp(self) -> None:
        from ultraquant.experiments import polarity_gate
        self.gate = polarity_gate
        self.doc = " ".join(polarity_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("No inverted assertion",
                       "Positives untouched",
                       "Changes are exactly the negated semantic readings",
                       "Nothing else moves",
                       "The full suite is green"):
            self.assertIn(phrase, self.doc)

    def test_the_battery_can_fail(self) -> None:
        """A battery that never reaches the path passes anything."""
        self.assertIn("The OLD arm must invert all ten", self.doc)
        self.assertGreaterEqual(len(self.gate.BATTERY), 8)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("29 replies changed, and they were exactly the 29",
                      self.doc)
        self.assertIn("0 spurious, 0 missed", self.doc)

    def test_the_exposure_figure_keeps_its_caveat(self) -> None:
        self.assertIn("not a field rate", self.doc)

    def test_the_implementer_did_not_write_its_own_exam(self) -> None:
        self.assertIn("The implementer did not write its own exam",
                      self.doc)


if __name__ == "__main__":
    unittest.main()
