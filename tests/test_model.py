"""Tests for ultraquant.model.quantize and ultraquant.model.network."""

from __future__ import annotations

import json
import random
import unittest

from ultraquant.model.network import UltraQuantNet
from ultraquant.model.quantize import (
    TernaryLinear,
    dequantize,
    fake_quantize_activation,
    quantize_ternary,
)

XOR_XS: list[list[float]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_YS: list[int] = [0, 1, 1, 0]


class TestQuantizeTernary(unittest.TestCase):
    def test_hand_built_matrix(self) -> None:
        # mean(|w|) = (0.5 + 0.4 + 0.1 + 0.0) / 4 = 0.25 -> delta = 0.1875
        w = [[0.5, -0.4], [0.1, 0.0]]
        q, alpha = quantize_ternary(w)
        self.assertEqual(q, [[1, -1], [0, 0]])
        # alpha = mean(|w|) over surviving entries = (0.5 + 0.4) / 2 = 0.45
        self.assertAlmostEqual(alpha, 0.45, places=12)

    def test_all_zero_matrix_edge_case(self) -> None:
        q, alpha = quantize_ternary([[0.0, 0.0], [0.0, 0.0]])
        self.assertEqual(q, [[0, 0], [0, 0]])
        self.assertEqual(alpha, 1.0)

    def test_dequantize(self) -> None:
        q = [[1, -1], [0, 1]]
        self.assertEqual(dequantize(q, 0.5), [[0.5, -0.5], [0.0, 0.5]])

    def test_quantize_dequantize_shapes(self) -> None:
        rng = random.Random(7)
        w = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(3)]
        q, alpha = quantize_ternary(w)
        self.assertEqual(len(q), 3)
        self.assertTrue(all(len(row) == 5 for row in q))
        self.assertTrue(all(v in (-1, 0, 1) for row in q for v in row))
        self.assertGreater(alpha, 0.0)
        deq = dequantize(q, alpha)
        self.assertEqual(len(deq), 3)
        self.assertTrue(all(len(row) == 5 for row in deq))


class TestFakeQuantizeActivation(unittest.TestCase):
    def test_clamps_to_range(self) -> None:
        out = fake_quantize_activation([-100.0, 100.0], bits=8, clip=4.0)
        self.assertAlmostEqual(out[0], -4.0, places=12)
        self.assertAlmostEqual(out[1], 4.0, places=12)

    def test_quantization_error_bounded(self) -> None:
        bits, clip = 8, 4.0
        step = 2.0 * clip / ((2 ** bits - 1) - 1)
        rng = random.Random(3)
        xs = [rng.uniform(-clip, clip) for _ in range(200)]
        out = fake_quantize_activation(xs, bits=bits, clip=clip)
        for original, quantized in zip(xs, out):
            self.assertLessEqual(abs(original - quantized), step / 2 + 1e-12)

    def test_zero_is_representable_neighborhood(self) -> None:
        out = fake_quantize_activation([0.0], bits=8, clip=4.0)
        self.assertEqual(len(out), 1)
        self.assertLessEqual(abs(out[0]), 2.0 * 4.0 / (2 ** 8 - 2))


class TestTernaryLinear(unittest.TestCase):
    def test_forward_uses_quantized_weights(self) -> None:
        layer = TernaryLinear(2, 2, random.Random(0), quantized=True)
        layer.w = [[0.5, -0.4], [0.1, 0.0]]
        layer.b = [0.0, 1.0]
        out = layer.forward([1.0, 1.0])
        # Quantized weights are alpha * [[1, -1], [0, 0]] with alpha = 0.45.
        self.assertAlmostEqual(out[0], 0.45 - 0.45, places=12)
        self.assertAlmostEqual(out[1], 1.0, places=12)

    def test_backward_straight_through_gradients(self) -> None:
        layer = TernaryLinear(2, 1, random.Random(0), quantized=True)
        layer.w = [[0.5, -0.4]]
        x = [2.0, 3.0]
        layer.forward(x)
        grad_in = layer.backward([1.0])
        # Input grad uses the quantized weights (alpha = 0.45).
        self.assertAlmostEqual(grad_in[0], 0.45, places=12)
        self.assertAlmostEqual(grad_in[1], -0.45, places=12)
        # STE: master-weight grads are grad_out * x, untouched by quantization.
        self.assertAlmostEqual(layer.gw[0][0], 2.0, places=12)
        self.assertAlmostEqual(layer.gw[0][1], 3.0, places=12)
        self.assertAlmostEqual(layer.gb[0], 1.0, places=12)

    def test_step_updates_clamps_and_zeroes_grads(self) -> None:
        layer = TernaryLinear(2, 1, random.Random(0), quantized=False)
        layer.w = [[0.95, -0.95]]
        layer.gw = [[-1.0, 1.0]]
        layer.gb = [0.5]
        layer.step(0.2)
        self.assertEqual(layer.w[0], [1.0, -1.0])  # clamped to [-1, 1]
        self.assertAlmostEqual(layer.b[0], -0.1, places=12)
        self.assertEqual(layer.gw, [[0.0, 0.0]])
        self.assertEqual(layer.gb, [0.0])

    def test_state_dict_round_trip(self) -> None:
        layer = TernaryLinear(3, 2, random.Random(11), quantized=True)
        payload = json.loads(json.dumps(layer.state_dict()))
        other = TernaryLinear(3, 2, random.Random(99), quantized=False)
        other.load_state_dict(payload)
        x = [0.3, -0.7, 1.2]
        self.assertEqual(layer.forward(x), other.forward(x))


class TestUltraQuantNetXor(unittest.TestCase):
    def test_xor_trains_to_perfect_accuracy(self) -> None:
        # Seed 3 converges within tens of epochs at every tested learning
        # rate; some seeds hit ternary-quantization dead spots at width 8.
        net = UltraQuantNet(
            input_dim=2, hidden_dims=[8], num_classes=2, seed=3,
            quantize_head=False,
        )
        solved = False
        for _ in range(500):
            net.train_batch(XOR_XS, XOR_YS, lr=0.5)
            if all(net.predict(x)[0] == y for x, y in zip(XOR_XS, XOR_YS)):
                solved = True
                break
        self.assertTrue(solved, "XOR did not reach 4/4 accuracy in 500 epochs")

    def test_train_batch_returns_finite_mean_loss(self) -> None:
        net = UltraQuantNet(2, [8], 2, seed=1)
        loss = net.train_batch(XOR_XS, XOR_YS, lr=0.1)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 100.0)

    def test_predict_returns_label_and_confidence(self) -> None:
        net = UltraQuantNet(2, [8], 2, seed=2)
        label, confidence = net.predict([1.0, 0.0])
        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestUltraQuantNetStateDict(unittest.TestCase):
    def test_round_trip_identical_logits(self) -> None:
        net = UltraQuantNet(4, [6, 5], 3, seed=13)
        rng = random.Random(21)
        xs = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(8)]
        ys = [rng.randrange(3) for _ in range(8)]
        for _ in range(5):
            net.train_batch(xs, ys, lr=0.05)

        payload = json.loads(json.dumps(net.state_dict()))  # JSON-safe check
        restored = UltraQuantNet(4, [6, 5], 3, seed=999)
        restored.load_state_dict(payload)

        for x in xs:
            self.assertEqual(net.forward(x), restored.forward(x))
            self.assertEqual(net.predict(x), restored.predict(x))

    def test_architecture_mismatch_raises(self) -> None:
        net = UltraQuantNet(4, [6], 3, seed=0)
        other = UltraQuantNet(4, [7], 3, seed=0)
        with self.assertRaises(ValueError):
            other.load_state_dict(net.state_dict())


if __name__ == "__main__":
    unittest.main()
