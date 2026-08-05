"""Tests for the quantum feature map layer."""

from __future__ import annotations

import unittest

from ultraquant.quantum.backend import SimulatorBackend
from ultraquant.quantum.qlayer import QuantumFeatureMap


class TestQuantumFeatureMap(unittest.TestCase):
    def test_extract_length_and_range(self) -> None:
        qmap = QuantumFeatureMap(4, SimulatorBackend(seed=0))
        out = qmap.extract([0.3, -1.2, 0.5, 2.0])
        self.assertEqual(len(out), 4)
        for v in out:
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_exact_mode_determinism(self) -> None:
        feats = [0.1, -0.4, 0.9]
        qmap = QuantumFeatureMap(3, SimulatorBackend(seed=3))
        first = qmap.extract(feats)
        second = qmap.extract(feats)
        self.assertEqual(first, second)
        # Exact mode is also independent of the backend seed entirely.
        other = QuantumFeatureMap(3, SimulatorBackend(seed=99)).extract(feats)
        for a, b in zip(first, other):
            self.assertAlmostEqual(a, b, places=12)

    def test_data_reuploading_seven_features_three_qubits(self) -> None:
        qmap = QuantumFeatureMap(3, SimulatorBackend(seed=1))
        feats = [0.5, -0.5, 1.0, 2.0, -2.0, 0.25, 0.75]
        circ = qmap.encode(feats)
        ry_ops = [op for op in circ.ops if op.name == "ry"]
        cnot_ops = [op for op in circ.ops if op.name == "cnot"]
        # ceil(7/3) = 3 upload blocks: 3 RY layers of 3 (last chunk padded
        # with angle 0) and 3 entangler rings of 3 CNOTs.
        self.assertEqual(len(ry_ops), 9)
        self.assertEqual(len(cnot_ops), 9)
        out = qmap.extract(feats)
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertGreaterEqual(v, -1.0 - 1e-9)
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_missing_features_pad_with_zero_angle(self) -> None:
        qmap = QuantumFeatureMap(4, SimulatorBackend(seed=0))
        circ = qmap.encode([0.5])
        ry_ops = [op for op in circ.ops if op.name == "ry"]
        self.assertEqual(len(ry_ops), 4)
        self.assertEqual(ry_ops[1].params, (0.0,))
        self.assertEqual(ry_ops[2].params, (0.0,))
        self.assertEqual(ry_ops[3].params, (0.0,))
        out = qmap.extract([0.5])
        self.assertEqual(len(out), 4)

    def test_single_qubit_skips_entangler_ring(self) -> None:
        qmap = QuantumFeatureMap(1, SimulatorBackend(seed=2))
        circ = qmap.encode([0.7, -0.2])
        self.assertFalse(any(op.name == "cnot" for op in circ.ops))
        self.assertEqual(len([op for op in circ.ops if op.name == "ry"]), 2)
        out = qmap.extract([0.7, -0.2])
        self.assertEqual(len(out), 1)

    def test_shots_mode_estimates_near_exact(self) -> None:
        feats = [0.4, -0.9]
        sampled = QuantumFeatureMap(2, SimulatorBackend(seed=11), shots=4096).extract(
            feats
        )
        exact = QuantumFeatureMap(2, SimulatorBackend(seed=11)).extract(feats)
        self.assertEqual(len(sampled), 2)
        for s, e in zip(sampled, exact):
            self.assertLess(abs(s - e), 0.1)
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
