"""Command-R's next token, computed by our own engine.

11.126: the whole Command-R 08-2024 forward pass in UltraQuant. These
pins hold the scoring the exam used, llama.cpp's own answer to one
prompt wherever the model is on disk, and - on request, because it
takes minutes and two llama.cpp servers - the whole gate.

Set ULTRAQUANT_SLOW_GATES=1 to run the full two-oracle gate.
"""

from __future__ import annotations

import math
import os
import unittest
import urllib.request

from ultraquant.experiments import forward_gate as G


class ScoringTests(unittest.TestCase):
    """The exam's arithmetic, on data it can be checked against."""

    def test_log_softmax_normalises(self) -> None:
        lp = G.log_softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(math.fsum(math.exp(v) for v in lp), 1.0,
                               places=12)

    def test_a_near_tie_under_the_floor_is_excluded(self) -> None:
        ours = [-1.0, -1.001, -9.0, -9.5, -9.9]
        cpu = {0: -1.0, 1: -1.004, 2: -9.0, 3: -9.5, 4: -9.9}
        gpu = {0: -1.02, 1: -1.0, 2: -9.0, 3: -9.5, 4: -9.9}
        row = G.score(ours, cpu, gpu)
        self.assertTrue(row["near_tie"])
        self.assertTrue(row["top1_ok"])

    def test_the_bound_is_twice_the_spread_or_a_tenth(self) -> None:
        cpu = {0: -0.5, 1: -2.0, 2: -3.0, 3: -4.0, 4: -5.0}
        gpu = {0: -0.8, 1: -2.0, 2: -3.0, 3: -4.0, 4: -5.0}
        row = G.score([-0.5, -2.0, -3.0, -4.0, -5.0], cpu, gpu)
        self.assertAlmostEqual(row["bound"], 0.6)
        self.assertEqual(G.score([-0.5, -2.0, -3.0, -4.0, -5.0], cpu,
                                 cpu)["bound"], 0.1)

    def test_the_planted_rope_really_is_a_different_rotation(self) -> None:
        """The defect the gate plants must change the computation."""
        from ultraquant.infer import command_r as C

        rotations = [(math.cos(0.3 * (i + 1)), math.sin(0.3 * (i + 1)))
                     for i in range(2)]

        class Probe:
            """An engine that rotates a vector with the module's _rope."""

            def logits(self, ids):
                vector = [1.0, 2.0, 3.0, 4.0]
                C._rope(vector, 4, rotations)
                return vector

        original = C._rope
        real = Probe().logits(None)
        planted = G.planted_rope(Probe()).logits(None)
        self.assertNotEqual(real, planted)
        self.assertIs(C._rope, original)     # swapped back after the call


class GoldenFranceTests(unittest.TestCase):
    """llama.cpp CPU's top five for one prompt, captured from the oracle."""

    IDS = [5, 2162, 7784, 1719, 5334, 1801]     # "The capital of France is"
    GOLDEN = {5641: -0.803532, 1671: -2.191792, 1690: -2.589105,
              2371: -2.789494, 6090: -3.332356}  # " Paris", " a", " the"...

    def test_our_engine_lands_within_the_floor(self) -> None:
        if not G.COMMAND_R.exists():
            self.skipTest("Command-R is not on this machine")
        from ultraquant.infer.command_r import CommandR
        from ultraquant.native import accel
        if accel.load_cpu() is None:
            self.skipTest("the native CPU tier is not built here")
        lp = G.log_softmax(CommandR.from_gguf(G.COMMAND_R).logits(self.IDS))
        self.assertEqual(max(range(len(lp)), key=lp.__getitem__), 5641)
        for token, want in self.GOLDEN.items():
            with self.subTest(token=token):
                # the gate measured 0.035 here; 0.1 is its floor minimum
                self.assertLess(abs(lp[token] - want), 0.1)


class GateVerdictTests(unittest.TestCase):

    def setUp(self) -> None:
        self.doc = " ".join(G.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Top-1 agreement", "Within the noise floor",
                       "The exam can fail", "Deterministic"):
            self.assertIn(phrase, self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("Command-R runs in UltraQuant", self.doc)
        self.assertIn("failed all 12 prompts", self.doc)

    def test_the_whole_gate(self) -> None:
        if os.environ.get("ULTRAQUANT_SLOW_GATES") != "1":
            self.skipTest("set ULTRAQUANT_SLOW_GATES=1 for the two-oracle gate")
        for base in (G.CPU, G.GPU):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
            except OSError:
                self.skipTest(f"the llama.cpp oracle at {base} is not running")
        self.assertTrue(G.run_gate().passes)


if __name__ == "__main__":
    unittest.main()
