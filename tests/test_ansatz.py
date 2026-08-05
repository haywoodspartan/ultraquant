"""Tests for the trainable quantum circuit: encoding, ansatz, gradients, training."""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.quantum.ansatz import (
    AmplitudeEncoder,
    QuantumCircuitModel,
    VariationalAnsatz,
    circuit_stats,
    draw,
    parameter_shift_gradient,
)
from ultraquant.quantum.state import QuantumState
from ultraquant.quantum.vqc import VariationalClassifier


def _prepare(encoder: AmplitudeEncoder, values: list[float]) -> QuantumState:
    """Run the encoder's circuit and return the resulting state."""
    state = QuantumState(encoder.num_qubits)
    encoder.encode(values).apply_to(state)
    return state


def _fidelity(state: QuantumState, target: list[float]) -> float:
    """|<state|target>|**2."""
    overlap = sum(complex(a).conjugate() * t for a, t in zip(state.amplitudes, target))
    return abs(overlap) ** 2


class AmplitudeEncoderTests(unittest.TestCase):
    """State preparation must produce exactly the state it claims."""

    def test_non_negative_vectors_are_exact(self) -> None:
        rng = random.Random(0)
        for num_qubits in range(1, 7):
            with self.subTest(qubits=num_qubits):
                encoder = AmplitudeEncoder(num_qubits)
                values = [rng.uniform(0, 1) for _ in range(1 << num_qubits)]
                state = _prepare(encoder, values)
                self.assertAlmostEqual(
                    _fidelity(state, encoder.normalise(values)), 1.0, places=10
                )
                self.assertAlmostEqual(state.norm(), 1.0, places=12)

    def test_signed_vectors_are_exact(self) -> None:
        """Negative amplitudes need the phase operator, not just the rotations."""
        rng = random.Random(7)
        for num_qubits in range(1, 6):
            with self.subTest(qubits=num_qubits):
                encoder = AmplitudeEncoder(num_qubits)
                values = [rng.uniform(-1, 1) for _ in range(1 << num_qubits)]
                state = _prepare(encoder, values)
                self.assertAlmostEqual(
                    _fidelity(state, encoder.normalise(values)), 1.0, places=10
                )

    def test_all_positive_input_adds_no_phase_gates(self) -> None:
        encoder = AmplitudeEncoder(4)
        positive = encoder.encode([1.0] * 16)
        self.assertNotIn("rz", circuit_stats(positive)["by_gate"])

    def test_basis_state_is_prepared_exactly(self) -> None:
        """A one-hot vector must land entirely on one basis index."""
        encoder = AmplitudeEncoder(3)
        for index in range(8):
            with self.subTest(index=index):
                values = [0.0] * 8
                values[index] = 1.0
                state = _prepare(encoder, values)
                probabilities = state.probabilities()
                self.assertAlmostEqual(probabilities[index], 1.0, places=10)

    def test_uniform_superposition(self) -> None:
        encoder = AmplitudeEncoder(3)
        state = _prepare(encoder, [1.0] * 8)
        for probability in state.probabilities():
            self.assertAlmostEqual(probability, 1 / 8, places=10)

    def test_padding_and_truncation(self) -> None:
        encoder = AmplitudeEncoder(5)
        # 25 pixels into 32 amplitudes: the rest must be zero, not garbage.
        values = [1.0] * 25
        state = _prepare(encoder, values)
        for index in range(25, 32):
            self.assertAlmostEqual(state.probabilities()[index], 0.0, places=12)
        self.assertEqual(len(encoder.normalise([1.0] * 100)), 32)

    def test_zero_vector_becomes_uniform(self) -> None:
        """A state must be normalised; all-zero has no normalisation."""
        encoder = AmplitudeEncoder(2)
        normalised = encoder.normalise([0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(v * v for v in normalised), 1.0, places=12)

    def test_rejects_zero_qubits(self) -> None:
        with self.assertRaises(ValueError):
            AmplitudeEncoder(0)

    def test_glyph_encodes_all_25_pixels(self) -> None:
        """The point of amplitude encoding: no row-mean summary in between."""
        from ultraquant.pattern.recognition import PATTERNS, render

        encoder = AmplitudeEncoder(5)
        plus = render(PATTERNS["plus"])
        square = render(PATTERNS["square"])
        overlap = sum(
            a * b for a, b in zip(encoder.normalise(plus), encoder.normalise(square))
        )
        self.assertLess(abs(overlap), 0.99, "distinct glyphs must give distinct states")


class VariationalAnsatzTests(unittest.TestCase):
    """The trainable part is shaped as declared."""

    def test_parameter_count(self) -> None:
        self.assertEqual(VariationalAnsatz(5, layers=2).num_parameters, 20)
        self.assertEqual(VariationalAnsatz(3, layers=4).num_parameters, 24)

    def test_wrong_parameter_count_rejected(self) -> None:
        ansatz = VariationalAnsatz(3, layers=1)
        with self.assertRaises(ValueError):
            ansatz.build([0.0])

    def test_entangler_variants_build(self) -> None:
        for entangler in ("ring", "linear", "all-to-all"):
            with self.subTest(entangler=entangler):
                ansatz = VariationalAnsatz(4, layers=1, entangler=entangler)
                circuit = ansatz.build([0.1] * ansatz.num_parameters)
                self.assertGreater(circuit_stats(circuit)["two_qubit_gates"], 0)

    def test_single_qubit_ansatz_has_no_entangler(self) -> None:
        ansatz = VariationalAnsatz(1, layers=2)
        circuit = ansatz.build([0.1] * ansatz.num_parameters)
        self.assertEqual(circuit_stats(circuit)["two_qubit_gates"], 0)


class ParameterShiftTests(unittest.TestCase):
    """The gradient rule is an identity, so it must match finite differences."""

    def test_matches_finite_differences(self) -> None:
        from ultraquant.pattern.recognition import PATTERNS, render

        model = QuantumCircuitModel(num_qubits=4, layers=2, seed=1)
        pixels = render(PATTERNS["plus"])[:16]
        weights = [1.0, -0.5, 0.25, 0.75]

        def evaluate(params):
            return sum(w * z for w, z in zip(weights, model.forward(pixels, params)))

        exact = parameter_shift_gradient(evaluate, model.parameters)

        epsilon = 1e-6
        working = list(model.parameters)
        approximate = []
        for index in range(len(working)):
            original = working[index]
            working[index] = original + epsilon
            plus = evaluate(working)
            working[index] = original - epsilon
            minus = evaluate(working)
            working[index] = original
            approximate.append((plus - minus) / (2 * epsilon))

        for got, want in zip(exact, approximate):
            self.assertAlmostEqual(got, want, places=6)

    def test_gradient_is_nonzero(self) -> None:
        """A vanishing gradient everywhere would mean nothing can be learned."""
        model = QuantumCircuitModel(num_qubits=3, layers=2, seed=2)
        values = [0.3, 0.1, 0.9, 0.2, 0.5, 0.4, 0.8, 0.6]
        gradient = model.gradient(values, [1.0, 0.0, -1.0])
        self.assertGreater(sum(g * g for g in gradient) ** 0.5, 1e-6)

    def test_constant_function_has_zero_gradient(self) -> None:
        gradient = parameter_shift_gradient(lambda _p: 3.0, [0.1, 0.2, 0.3])
        for value in gradient:
            self.assertAlmostEqual(value, 0.0, places=12)

    def test_costs_two_evaluations_per_parameter(self) -> None:
        calls = []

        def evaluate(params):
            calls.append(tuple(params))
            return sum(params)

        parameter_shift_gradient(evaluate, [0.0] * 5)
        self.assertEqual(len(calls), 10)


class CircuitModelTests(unittest.TestCase):
    """The end-to-end circuit behaves and reports itself honestly."""

    def test_forward_is_deterministic_and_bounded(self) -> None:
        model = QuantumCircuitModel(num_qubits=4, layers=2, seed=3)
        values = [0.1 * i for i in range(16)]
        first = model.forward(values)
        self.assertEqual(first, model.forward(values))
        self.assertEqual(len(first), 4)
        for value in first:
            self.assertGreaterEqual(value, -1.0000001)
            self.assertLessEqual(value, 1.0000001)

    def test_circuit_stats_counts_depth_and_two_qubit_gates(self) -> None:
        model = QuantumCircuitModel(num_qubits=5, layers=2, seed=0)
        stats = circuit_stats(model.circuit([1.0] * 25))
        self.assertEqual(stats["num_qubits"], 5)
        self.assertGreater(stats["depth"], 0)
        self.assertGreater(stats["two_qubit_gates"], 0)
        self.assertEqual(stats["gates"], sum(stats["by_gate"].values()))

    def test_draw_renders_one_line_per_qubit(self) -> None:
        model = QuantumCircuitModel(num_qubits=3, layers=1, seed=0)
        diagram = draw(model.ansatz.build(model.parameters))
        lines = diagram.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("q0:"))


class VariationalClassifierTests(unittest.TestCase):
    """The circuit can actually be trained on the glyph task."""

    def _dataset(self, classes=("plus", "square"), n_per_class=4, flips=1, seed=0):
        from ultraquant.pattern.recognition import PATTERNS, render

        rng = random.Random(seed)
        xs, ys = [], []
        for label, name in enumerate(classes):
            base = render(PATTERNS[name])
            for _ in range(n_per_class):
                variant = list(base)
                for pixel in rng.sample(range(len(variant)), flips):
                    variant[pixel] = 1.0 - variant[pixel]
                xs.append(variant)
                ys.append(label)
        return xs, ys

    def test_learns_a_two_class_task(self) -> None:
        xs, ys = self._dataset()
        model = VariationalClassifier(num_classes=2, seed=0)
        before = model.score(xs, ys)
        report = model.fit(xs, ys, epochs=6, seed=0)
        self.assertGreaterEqual(report["accuracy"], 0.75)
        self.assertGreaterEqual(report["accuracy"], before)

    def test_training_moves_the_quantum_parameters(self) -> None:
        """The circuit must be doing some of the learning, not just the head."""
        xs, ys = self._dataset()
        model = VariationalClassifier(num_classes=2, seed=1)
        before = list(model.quantum.parameters)
        model.fit(xs, ys, epochs=2, seed=1)
        moved = max(abs(a - b) for a, b in zip(before, model.quantum.parameters))
        self.assertGreater(moved, 1e-6)

    def test_predict_shape(self) -> None:
        xs, _ys = self._dataset()
        model = VariationalClassifier(num_classes=2, seed=0)
        label, confidence = model.predict(xs[0])
        self.assertIn(label, (0, 1))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_describe_reports_the_real_cost(self) -> None:
        model = VariationalClassifier(num_classes=4, seed=0)
        info = model.describe()
        self.assertEqual(info["quantum_parameters"], 20)
        self.assertEqual(info["evaluations_per_training_step"], 41)
        self.assertIn("depth", info["circuit"])

    def test_softmax_is_stable_on_large_logits(self) -> None:
        probabilities = VariationalClassifier.softmax([1000.0, 1001.0])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertTrue(all(math.isfinite(p) for p in probabilities))


if __name__ == "__main__":
    unittest.main()
