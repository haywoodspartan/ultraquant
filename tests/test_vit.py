"""The 27-block encoder: its shape, its refusals, and its wiring.

§11.101 assembled the real encoder from `infer/ops`. These pins hold
the decisions that would otherwise be silent: every hyperparameter
read from the file rather than defaulted, the 4-D patch kernel read
whole rather than in the 256-value fragment `TensorInfo.rows`
implies, and the block wired pre-norm with both residuals bypassing
their norm.
"""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.infer import ops, vit

_METADATA = {
    "clip.vision.block_count": 27,
    "clip.vision.embedding_length": 1152,
    "clip.vision.attention.head_count": 16,
    "clip.vision.feed_forward_length": 4304,
    "clip.vision.attention.layer_norm_epsilon": 1e-6,
    "clip.vision.patch_size": 16,
    "clip.vision.image_size": 768,
    "clip.vision.spatial_merge_size": 2,
    "clip.use_gelu": True,
    "clip.projector_type": "qwen3vl_merger",
}


class ConfigTests(unittest.TestCase):
    """Nothing is defaulted, because the default would be wrong."""

    def test_the_shape_is_read_from_the_file(self) -> None:
        config = vit.load_config(_METADATA)
        self.assertEqual(config.blocks, 27)
        self.assertEqual(config.width, 1152)
        self.assertEqual(config.heads, 16)
        self.assertEqual(config.head_dim, 72)
        self.assertEqual(config.grid, 48)
        self.assertEqual(config.grid ** 2, 2304)

    def test_a_missing_key_refuses_rather_than_defaulting(self) -> None:
        for key in _METADATA:
            partial = {k: v for k, v in _METADATA.items() if k != key}
            with self.assertRaises(KeyError, msg=f"{key} was defaulted"):
                vit.load_config(partial)

    def test_the_epsilon_is_the_checkpoint_s_not_the_function_s(self):
        """1e-6 in the file; ops.layer_norm defaults to 1e-5."""
        self.assertAlmostEqual(vit.load_config(_METADATA).eps, 1e-6)
        self.assertNotAlmostEqual(vit.load_config(_METADATA).eps, 1e-5)


def _block(width: int, ffn: int, rng: random.Random) -> vit.BlockWeights:
    def matrix(rows: int, cols: int) -> list:
        return [[rng.gauss(0, 0.05) for _ in range(cols)]
                for _ in range(rows)]
    return vit.BlockWeights(
        ln1_w=[1.0] * width, ln1_b=[0.0] * width,
        qkv_w=matrix(3 * width, width), qkv_b=[0.0] * (3 * width),
        out_w=matrix(width, width), out_b=[0.0] * width,
        ln2_w=[1.0] * width, ln2_b=[0.0] * width,
        up_w=matrix(ffn, width), up_b=[0.0] * ffn,
        down_w=matrix(width, ffn), down_b=[0.0] * width)


class BlockTests(unittest.TestCase):
    """Wiring properties, on weights small enough to test."""

    def setUp(self) -> None:
        self.config = vit.VisionConfig(
            blocks=2, width=16, heads=4, ffn=32, eps=1e-6, patch=2,
            image_size=8, merge=2, use_gelu=True, projector="test")
        self.rng = random.Random(4)
        self.weights = _block(16, 32, self.rng)
        self.stream = [[self.rng.gauss(0, 1) for _ in range(16)]
                       for _ in range(4)]

    def test_a_block_preserves_the_stream_shape(self) -> None:
        out = vit.apply_block(self.stream, self.weights, self.config)
        self.assertEqual(len(out), len(self.stream))
        self.assertEqual(len(out[0]), self.config.width)
        self.assertTrue(all(math.isfinite(v) for t in out for v in t))

    def test_the_residual_carries_the_input_through(self) -> None:
        """With zeroed output projections a block must be identity."""
        weights = self.weights
        weights.out_w = [[0.0] * 16 for _ in range(16)]
        weights.down_w = [[0.0] * 32 for _ in range(16)]
        out = vit.apply_block(self.stream, weights, self.config)
        for got, want in zip(out, self.stream):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=12)

    def test_the_norm_is_bypassed_by_the_residual(self) -> None:
        """Post-norm would rescale the stream; pre-norm does not."""
        big = [[v * 1000.0 for v in token] for token in self.stream]
        weights = self.weights
        weights.out_w = [[0.0] * 16 for _ in range(16)]
        weights.down_w = [[0.0] * 32 for _ in range(16)]
        out = vit.apply_block(big, weights, self.config)
        self.assertGreater(max(abs(v) for t in out for v in t), 100.0,
                           "the residual was normalised; this is post-norm")

    def test_the_tail_merges_four_patches_into_one(self) -> None:
        width = self.config.width
        span = self.config.merge ** 2
        tail = {"post_ln_w": [1.0] * width, "post_ln_b": [0.0] * width,
                "mm0_w": [[0.0] * (span * width) for _ in range(8)],
                "mm0_b": [0.0] * 8,
                "mm2_w": [[0.0] * 8 for _ in range(5)],
                "mm2_b": [0.0] * 5}
        out = vit.apply_tail(self.stream, tail, self.config)
        self.assertEqual(len(out), len(self.stream) // span)
        self.assertEqual(len(out[0]), 5)


class NormaliseTests(unittest.TestCase):
    """The step whose omission nothing downstream would notice."""

    def test_pixels_land_in_the_range_the_model_trained_on(self) -> None:
        planes = [[[[0.0, 127.5, 255.0]]]]
        got = vit.normalise(planes)[0][0][0]
        self.assertAlmostEqual(got[0], -1.0)
        self.assertAlmostEqual(got[1], 0.0)
        self.assertAlmostEqual(got[2], 1.0)


class GateVerdictTests(unittest.TestCase):
    """A failure recorded as a failure, and the follow-up that placed it."""

    def setUp(self) -> None:
        from ultraquant.experiments import encoder_gate
        self.doc = " ".join(encoder_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The tower completes on real weights",
                       "End-to-end fidelity decides usability",
                       "The measurement can tell a bug from quantisation",
                       "The compounding is reported as a curve"):
            self.assertIn(phrase, self.doc)

    def test_the_bar_was_registered_before_the_run(self) -> None:
        self.assertIn("At or above 0.90, converted weights are usable",
                      self.doc)
        self.assertIn("the honest expectation is that it FAILS", self.doc)

    def test_the_headline_failure_is_recorded(self) -> None:
        self.assertIn("FAILED criteria 2 and 3", self.doc)
        self.assertIn("0.0185 against a bar of", self.doc)

    def test_the_methodological_trap_is_recorded(self) -> None:
        """0.87 before the final norm, 0.0185 after it."""
        self.assertIn("would have read 0.87 and called it a success",
                      self.doc)

    def test_criterion_three_was_localised_not_excused(self) -> None:
        """The failure was placed, not explained away."""
        self.assertIn("discriminates at every depth up to 8", self.doc)
        self.assertIn("a fact about quantisation, not about the method",
                      self.doc)

    def test_the_unverifiable_part_is_admitted(self) -> None:
        self.assertIn("does NOT verify that the encoder reproduces "
                      "llama.cpp", self.doc)
        self.assertIn("they are assumptions", self.doc)


if __name__ == "__main__":
    unittest.main()
