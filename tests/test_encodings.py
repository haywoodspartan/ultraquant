"""Tests for the alternative data encodings and their scaling."""

from __future__ import annotations

import math
import unittest

from ultraquant.quantum.analysis import entanglement_profile
from ultraquant.quantum.ansatz import AmplitudeEncoder, circuit_stats
from ultraquant.quantum.encodings import (
    AngleEncoder,
    SparseEncoder,
    ZZFeatureMap,
    compare_encodings,
)
from ultraquant.quantum.kernel import inversion_test
from ultraquant.quantum.state import QuantumState


class AngleEncodingTests(unittest.TestCase):
    """One qubit per feature, and nothing exponential."""

    def test_gate_count_is_linear(self) -> None:
        for width in (4, 16, 64, 256):
            with self.subTest(width=width):
                cost = AngleEncoder(width).cost(width)
                self.assertLessEqual(cost["gates"], 3 * width)

    def test_produces_a_valid_state(self) -> None:
        circuit = AngleEncoder(5).encode([0.3, -0.7, 1.2, 0.0, 2.5])
        state = QuantumState(5)
        circuit.apply_to(state)
        self.assertAlmostEqual(state.norm(), 1.0, places=12)

    def test_distinct_inputs_give_distinct_states(self) -> None:
        encoder = AngleEncoder(4)
        a = encoder.encode([1.0, 0.0, 0.0, 0.0])
        b = encoder.encode([0.0, 1.0, 0.0, 0.0])
        self.assertLess(inversion_test(a, b), 0.99)

    def test_identical_inputs_give_identical_states(self) -> None:
        encoder = AngleEncoder(4)
        values = [0.4, 0.9, 0.1, 0.6]
        self.assertAlmostEqual(
            inversion_test(encoder.encode(values), encoder.encode(values)), 1.0, places=10
        )

    def test_reuploading_consumes_more_features(self) -> None:
        single = AngleEncoder(3, reuploads=1)
        double = AngleEncoder(3, reuploads=2)
        values = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
        self.assertLess(
            inversion_test(single.encode(values), double.encode(values)), 0.999,
            "a second upload round must change the state",
        )

    def test_entangles_when_asked(self) -> None:
        entangled = entanglement_profile(AngleEncoder(4, entangle=True).encode([0.5] * 4))
        plain = entanglement_profile(AngleEncoder(4, entangle=False).encode([0.5] * 4))
        self.assertTrue(plain["separable"], "without an entangler it must be a product state")
        self.assertFalse(entangled["separable"])

    def test_scales_where_amplitude_encoding_cannot(self) -> None:
        """The point of this encoding: 50 qubits is routine, not impossible."""
        cost = AngleEncoder(50).cost(50)
        self.assertLess(cost["two_qubit_gates"], 100)
        self.assertGreater((1 << 50) - 2, 10 ** 15)


class ZZFeatureMapTests(unittest.TestCase):
    """The entangling map: polynomial cost, genuinely entangling."""

    def test_cost_is_polynomial(self) -> None:
        linear = ZZFeatureMap(50, repetitions=2, entanglement="linear").cost()
        full = ZZFeatureMap(50, repetitions=2, entanglement="full").cost()
        self.assertLess(linear["two_qubit_gates"], 250)
        self.assertLess(full["two_qubit_gates"], 6000)
        self.assertGreater(full["two_qubit_gates"], linear["two_qubit_gates"])

    def test_entanglement_variants(self) -> None:
        for mode, expected_pairs in (("linear", 4), ("circular", 5), ("full", 10)):
            with self.subTest(mode=mode):
                zz = ZZFeatureMap(5, repetitions=1, entanglement=mode)
                self.assertEqual(len(zz._pairs()), expected_pairs)

    def test_produces_entanglement(self) -> None:
        profile = entanglement_profile(
            ZZFeatureMap(4, repetitions=2, entanglement="full").encode([0.4, 0.8, 0.2, 0.6])
        )
        self.assertFalse(profile["separable"])

    def test_state_stays_normalised(self) -> None:
        state = QuantumState(5)
        ZZFeatureMap(5, repetitions=2).encode([0.1, 0.5, 0.9, 0.3, 0.7]).apply_to(state)
        self.assertAlmostEqual(state.norm(), 1.0, places=12)

    def test_kernel_is_symmetric_and_self_similar(self) -> None:
        zz = ZZFeatureMap(4, repetitions=2)
        a = zz.encode([0.2, 0.4, 0.6, 0.8])
        b = zz.encode([0.9, 0.1, 0.5, 0.3])
        self.assertAlmostEqual(inversion_test(a, a), 1.0, places=10)
        self.assertAlmostEqual(inversion_test(a, b), inversion_test(b, a), places=10)

    def test_reaches_widths_amplitude_encoding_cannot(self) -> None:
        cost = ZZFeatureMap(50, repetitions=2, entanglement="linear").cost()
        # Feasible gate count at a width beyond exact classical simulation.
        self.assertLess(cost["two_qubit_gates"], 250)
        self.assertGreater(cost["qubits"], 40)


class SparseEncodingTests(unittest.TestCase):
    """Sparsity is worth exploiting, and the fallback stays exact."""

    def test_reports_when_sparsity_pays(self) -> None:
        encoder = SparseEncoder(8)
        sparse = [0.0] * 256
        for index in (3, 17, 200):
            sparse[index] = 1.0
        self.assertTrue(encoder.worth_it(sparse))
        self.assertFalse(encoder.worth_it([1.0] * 256))

    def test_single_basis_state_is_exact_and_cheap(self) -> None:
        encoder = SparseEncoder(4)
        values = [0.0] * 16
        values[11] = 1.0
        circuit = encoder.encode(values)
        state = QuantumState(4)
        circuit.apply_to(state)
        self.assertAlmostEqual(state.probabilities()[11], 1.0, places=12)
        self.assertEqual(circuit_stats(circuit)["two_qubit_gates"], 0)

    def test_dense_fallback_is_still_exact(self) -> None:
        encoder = SparseEncoder(3)
        values = [0.3, 0.5, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6]
        state = QuantumState(3)
        encoder.encode(values).apply_to(state)
        target = AmplitudeEncoder(3).normalise(values)
        overlap = sum(
            complex(a).conjugate() * t for a, t in zip(state.amplitudes, target)
        )
        self.assertAlmostEqual(abs(overlap) ** 2, 1.0, places=10)

    def test_zero_vector_is_handled(self) -> None:
        circuit = SparseEncoder(3).encode([0.0] * 8)
        self.assertEqual(len(circuit.ops), 0)

    def test_cost_report_shape(self) -> None:
        values = [0.0] * 32
        values[5] = 1.0
        cost = SparseEncoder(5).cost(values)
        self.assertEqual(cost["nonzero"], 1)
        self.assertTrue(cost["worth_it"])
        self.assertLess(cost["sparse_two_qubit_gates"], cost["dense_two_qubit_gates"])


class ScalingComparisonTests(unittest.TestCase):
    """The headline claim: amplitude encoding is the exponential one, alone."""

    def test_amplitude_is_exponential_and_the_others_are_not(self) -> None:
        rows = {r["qubits"]: r for r in compare_encodings((4, 8, 12))}
        # Amplitude doubles with each qubit; angle grows by one.
        self.assertAlmostEqual(rows[8]["amplitude_2q"] / rows[4]["amplitude_2q"], 18.1, delta=2)
        self.assertEqual(rows[8]["angle_2q"] - rows[4]["angle_2q"], 4)
        self.assertLess(rows[12]["zz_2q"], rows[12]["amplitude_2q"])

    def test_measured_where_feasible(self) -> None:
        rows = compare_encodings((4, 16))
        self.assertTrue(rows[0]["amplitude_measured"])
        self.assertFalse(rows[1]["amplitude_measured"],
                         "a 16-qubit amplitude circuit must be projected, not built")

    def test_amplitude_cost_matches_the_closed_form(self) -> None:
        for row in compare_encodings((4, 8, 12)):
            with self.subTest(qubits=row["qubits"]):
                self.assertEqual(row["amplitude_2q"], (1 << row["qubits"]) - 2)


if __name__ == "__main__":
    unittest.main()
