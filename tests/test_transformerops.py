"""The operations between the matmuls, pinned against their definitions.

§11.100 added what an MLP never needed: normalisation, attention,
and the activation functions. These pins hold the two bugs run one
found - a fully masked row coming out NaN, and a naive form the
harness never provoked - plus the values a wrong implementation
still looks right without.
"""

from __future__ import annotations

import math
import unittest
from decimal import Decimal

from ultraquant.experiments import transformerops_gate as gate
from ultraquant.infer import ops


class DefinitionTests(unittest.TestCase):
    """Values a wrong implementation still passes a property test on."""

    def test_softmax_is_the_distribution_not_merely_a_distribution(self):
        got = ops.softmax([1.0, 2.0, 3.0])
        want = [float(v) for v in gate.ref_softmax(
            [Decimal(1), Decimal(2), Decimal(3)])]
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=15)
        self.assertAlmostEqual(sum(got), 1.0, places=15)

    def test_layer_norm_uses_the_biased_variance(self) -> None:
        """Dividing by n-1 still centres and is silently wrong."""
        xs = [1.0, 2.0, 3.0, 4.0]
        got = ops.layer_norm(xs, None, None, eps=0.0)
        mean = 2.5
        biased = sum((x - mean) ** 2 for x in xs) / 4
        want = [(x - mean) / math.sqrt(biased) for x in xs]
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=12)
        unbiased = sum((x - mean) ** 2 for x in xs) / 3
        self.assertNotAlmostEqual(got[0], (xs[0] - mean)
                                  / math.sqrt(unbiased), places=6)

    def test_rms_norm_does_not_centre(self) -> None:
        """Not a LayerNorm with the mean left out by accident."""
        xs = [1.0, 2.0, 3.0, 4.0]
        got = ops.rms_norm(xs, None, eps=0.0)
        self.assertGreater(min(got), 0.0,
                           "RMSNorm must not shift the sign of a positive "
                           "vector; centring would")

    def test_the_two_gelus_are_not_the_same_function(self) -> None:
        """A model trained with one and run with the other is wrong.

        Measured rather than guessed: they diverge most at
        x = -2.698, by 0.000473. The first version of this pin
        asserted a bound of 1e-4 at four hand-picked points and
        failed, because 9.8e-5 is what the gap happens to be there -
        an assertion about a number nobody had looked up.
        """
        xs = [i / 500 - 8 for i in range(8001)]
        gaps = [abs(a - b) for a, b in zip(ops.gelu(xs), ops.gelu_tanh(xs))]
        worst = max(gaps)
        self.assertAlmostEqual(worst, 0.000473, places=6)
        self.assertAlmostEqual(xs[gaps.index(worst)], -2.698, places=3)


class MaskTests(unittest.TestCase):
    """Run one's NaN bug, and the mask behaviour around it."""

    def test_a_fully_masked_row_is_zero_not_nan(self) -> None:
        got = ops.softmax([-math.inf] * 4)
        self.assertEqual(got, [0.0, 0.0, 0.0, 0.0])

    def test_a_fully_masked_row_is_zero_not_uniform(self) -> None:
        """Uniform would invent attention where the mask said none."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        blind = [[False, False], [False, False]]
        for row in ops.attention_weights(q, q, mask=blind):
            self.assertEqual(row, [0.0, 0.0])

    def test_one_live_position_takes_all_the_weight(self) -> None:
        got = ops.softmax([-math.inf, 0.0, -math.inf])
        self.assertEqual(got, [0.0, 1.0, 0.0])

    def test_the_causal_mask_leaks_nothing(self) -> None:
        q = [[float(i), 1.0] for i in range(6)]
        k = [list(row) for row in q]
        v = [[float(i) * 2, 1.0] for i in range(6)]
        before = ops.attention(q, k, v, causal=True)
        k[-1] = [99.0, -99.0]
        v[-1] = [1234.0, -1234.0]
        after = ops.attention(q, k, v, causal=True)
        for i in range(5):
            self.assertEqual(before[i], after[i],
                             f"position {i} moved when the future did")

    def test_masked_rows_still_sum_to_one(self) -> None:
        """Zeroing after the softmax would scale attention down."""
        q = [[float(i), 1.0] for i in range(5)]
        for row in ops.attention_weights(q, q, causal=True):
            self.assertAlmostEqual(sum(row), 1.0, places=15)


class HeadTests(unittest.TestCase):
    """The split every shape check passes and every subspace fails."""

    def test_the_split_is_contiguous(self) -> None:
        rows = [[float(i) for i in range(8)]]
        heads = ops.split_heads(rows, 4)
        self.assertEqual(heads[0][0], [0.0, 1.0])
        self.assertEqual(heads[3][0], [6.0, 7.0])

    def test_merging_a_split_returns_the_original(self) -> None:
        rows = [[float(i * 7 % 13) for i in range(12)] for _ in range(3)]
        for count in (1, 2, 3, 4, 6, 12):
            self.assertEqual(ops.merge_heads(ops.split_heads(rows, count)),
                             rows)

    def test_a_width_that_does_not_divide_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            ops.split_heads([[1.0, 2.0, 3.0]], 2)

    def test_multi_head_scales_per_head_not_per_model(self) -> None:
        rows = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        one = ops.attention(rows, rows, rows)
        many = ops.multi_head_attention(rows, rows, rows, heads=2)
        self.assertNotEqual(one, many,
                            "if these matched the head scale would be "
                            "wrong in one of them")


class GateVerdictTests(unittest.TestCase):
    """The PASS, and the four failures run one produced."""

    def setUp(self) -> None:
        self.doc = " ".join(gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Every operation matches its definition",
                       "The stabilisation is demonstrated, not asserted",
                       "The causal mask leaks nothing",
                       "Attention rows are distributions",
                       "The head split is contiguous and reversible",
                       "The cost is named"):
            self.assertIn(phrase, self.doc)

    def test_both_code_bugs_are_kept(self) -> None:
        self.assertIn("came out NaN", self.doc)
        self.assertIn("SiLU's naive form was never provoked", self.doc)

    def test_both_criterion_corrections_are_kept(self) -> None:
        """A gate that quietly rewrites its own bar is not a gate."""
        self.assertIn("per-element relative error measures float64", self.doc)
        self.assertIn("unsatisfiable by construction", self.doc)
        self.assertIn("one regime was gated that no format can pass",
                      self.doc)

    def test_the_layer_norm_finding_is_recorded(self) -> None:
        self.assertIn("9.81e-11", self.doc)
        self.assertIn("the only one that can care", self.doc)

    def test_the_gate_passes(self) -> None:
        report = gate.run_gate(cases=8, seed=2)
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(report.leaks, 0, report.reason)
        self.assertEqual(report.masked_rows_nonzero, 0, report.reason)
        self.assertEqual(report.split_errors, 0, report.reason)


if __name__ == "__main__":
    unittest.main()
