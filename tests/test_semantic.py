"""The embedding suggester: verified readings, and nothing else.

The boundary translator §11.37 measured and §11.39 adopted: cosine
proposes, the lexical core verifies (threshold + transitive positional
rule), the reading is named, and absence costs nothing. Every test that
needs an embedder injects a fake one - the suite must not depend on LM
Studio being up.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.semantic import SemanticSuggester, Suggestion


class _FakeClient:
    """Deterministic embedder: vectors chosen per text by a lookup."""

    def __init__(self, vectors: dict) -> None:
        self.vectors = vectors

    def available(self) -> bool:
        return True

    def embedding_models(self) -> list[str]:
        return ["fake-embed"]

    def embed(self, texts, model=None):
        items = [texts] if isinstance(texts, str) else list(texts)
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in items]


def _memory(facts: dict) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts.items():
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class SuggesterVerificationTests(unittest.TestCase):
    """Cosine proposes; the rules dispose."""

    def test_a_synonym_reading_clears_both_checks(self) -> None:
        memory = _memory({"tower height": "417 meters"})
        client = _FakeClient({
            "how tall is the tower?": [1.0, 0.0, 0.0],
            "tower height": [0.95, 0.05, 0.0],
        })
        suggester = SemanticSuggester(client=client)
        reading = suggester.suggest("how tall is the tower?", memory)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.key, "tower height")
        self.assertGreater(reading.similarity, 0.75)

    def test_below_the_floor_is_refused(self) -> None:
        memory = _memory({"tower height": "417 meters"})
        client = _FakeClient({
            "how tall is the tower?": [1.0, 0.0, 0.0],
            "tower height": [0.5, 0.86, 0.0],
        })
        suggester = SemanticSuggester(client=client)
        self.assertIsNone(suggester.suggest("how tall is the tower?",
                                            memory))

    def test_a_trailing_unknown_subject_is_never_read_away(self) -> None:
        """'the height of the obelisk' at cosine 1.0 must still refuse:
        the transitive positional rule - the first gate run's catch."""
        memory = _memory({"tower height": "417 meters"})
        client = _FakeClient({
            "what is the height of the obelisk?": [1.0, 0.0, 0.0],
            "tower height": [1.0, 0.0, 0.0],
        })
        suggester = SemanticSuggester(client=client)
        self.assertIsNone(suggester.suggest(
            "what is the height of the obelisk?", memory))

    def test_no_anchor_token_means_no_candidates(self) -> None:
        memory = _memory({"tower height": "417 meters"})
        client = _FakeClient({})
        suggester = SemanticSuggester(client=client)
        self.assertIsNone(suggester.suggest("how warm is the citadel?",
                                            memory))

    def test_a_down_server_costs_nothing_and_backs_off(self) -> None:
        class _Down:
            def available(self) -> bool:
                return False

        suggester = SemanticSuggester(client=_Down())
        memory = _memory({"tower height": "417 meters"})
        self.assertIsNone(suggester.suggest("how tall is the tower?",
                                            memory))
        self.assertGreater(suggester._down_until, 0.0)


class WiringTests(unittest.TestCase):
    """Sessions default OFF; enabled sessions read and say so."""

    def test_sessions_default_to_no_suggester(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        root = Path(tempfile.mkdtemp(prefix="uq_semwire_"))
        try:
            session = build_session(root, seed=0)
            self.assertIsNone(session.semantic)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_the_reading_reaches_the_reply_named(self) -> None:
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        root = Path(tempfile.mkdtemp(prefix="uq_semwire_"))
        try:
            session = build_session(root, seed=0)
            run_pipeline("the tower height is 417 meters", session)
            client = _FakeClient({
                "how tall is the tower?": [1.0, 0.0, 0.0],
                "tower height": [0.9, 0.1, 0.0],
            })
            session.semantic = SemanticSuggester(client=client)
            response, _trace = run_pipeline("how tall is the tower?",
                                            session)
            self.assertIn("Reading that as 'tower height'", response)
            self.assertIn("417", response)
            self.assertIn("embedding match", response)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_settings_carry_the_default_on_flag(self) -> None:
        from ultraquant.config import DEFAULTS

        self.assertTrue(DEFAULTS["lmstudio"]["semantic_suggest"])


class GateVerdictTests(unittest.TestCase):
    """The PASS, the new fabrication shape, and the ceiling, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import semantic_gate

        doc = " ".join(semantic_gate.__doc__.split())
        self.assertIn("then PASSED", doc)
        self.assertIn("2.04x seed sd", doc)

    def test_the_attribute_anchor_catch_is_recorded(self) -> None:
        from ultraquant.experiments import semantic_gate

        doc = " ".join(semantic_gate.__doc__.split())
        self.assertIn("wrong-metal fabrication wearing an embedding", doc)
        self.assertIn("subjects are never read away", doc.lower())

    def test_the_chained_ceiling_is_in_the_metric(self) -> None:
        from ultraquant.experiments import semantic_gate

        doc = " ".join(semantic_gate.__doc__.split())
        self.assertIn("the ceiling in the metric instead of outside it",
                      doc)

    def test_a_missing_model_skips_and_never_passes(self) -> None:
        from ultraquant.experiments.semantic_gate import SemanticReport

        report = SemanticReport(skipped=True, reason="test")
        self.assertFalse(report.passes)
        self.assertIn("SKIPPED", report.as_text())


if __name__ == "__main__":
    unittest.main()
