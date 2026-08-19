"""The language wall, pinned: what holds, what hedges, what needs help.

Reorder and verbose forms hold at 1.000 with zero fabrications; the
synonym wall stands at 0.000 by the assertion standard while still
delivering the right number hedged; and the embedding comparison
quantifies the hybrid tradeoff the architecture's honest assessment
rests on.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path


class FormFamilyTests(unittest.TestCase):
    """Live behavior per surface form."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        self.dir = Path(tempfile.mkdtemp(prefix="uq_para_"))
        self.session = build_session(self.dir, seed=0)
        for statement in ("the tower height is 417 meters",
                          "the tower material is steel",
                          "the steel hardness is 7712 units"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_reorder_is_free(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline("what is the height of the tower?",
                                        self.session)
        self.assertIn("417", response)

    def test_politeness_filler_is_free(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline(
            "could you please tell me the tower height?", self.session)
        self.assertIn("417", response)

    def test_a_reordered_chain_still_infers(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline(
            "what is the hardness of the tower?", self.session)
        self.assertIn("7712", response)
        self.assertIn("inferred", response)

    def test_the_synonym_hedges_but_delivers(self) -> None:
        """'how tall' does not assert 'height' - but the demoted reply
        carries the right number, which is the wall's practical shape."""
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline("how tall is the tower?",
                                        self.session)
        self.assertTrue(response.startswith("I don't hold that exactly"))
        self.assertIn("417", response)

    def test_a_paraphrased_ghost_still_refuses(self) -> None:
        from ultraquant.interpreter.thoughts import run_pipeline

        response, _trace = run_pipeline(
            "could you please tell me the obelisk height?", self.session)
        self.assertTrue(response.startswith("I don't hold"))


class GateVerdictTests(unittest.TestCase):
    """The PASS, the wall's height and shape, and the tradeoff, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import paraphrase_gate

        doc = " ".join(paraphrase_gate.__doc__.split())
        self.assertIn("PASSED", doc)
        self.assertIn("synonym | **0.000**", doc.replace(" | ", " | "))

    def test_the_hedged_delivery_nuance_is_recorded(self) -> None:
        from ultraquant.experiments import paraphrase_gate

        doc = " ".join(paraphrase_gate.__doc__.split())
        self.assertIn("hedged rather than asserted", doc)

    def test_the_threshold_safety_finding_is_recorded(self) -> None:
        """0.833 fabrication at cosine 0.65 vs zero at 0.75: the threshold
        IS the safety argument, and losing that sentence loses the reason
        embeddings stay at the boundary."""
        from ultraquant.experiments import paraphrase_gate

        doc = " ".join(paraphrase_gate.__doc__.split())
        self.assertIn("0.833", doc)
        self.assertIn("not a tuning detail", doc)

    def test_the_division_of_labor_is_recorded(self) -> None:
        from ultraquant.experiments import paraphrase_gate

        doc = " ".join(paraphrase_gate.__doc__.split())
        self.assertIn("neither can do the other's job", doc)


if __name__ == "__main__":
    unittest.main()
