"""Tests for the advanced quantum functions: kernels, noise, search, diagnostics."""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.quantum.analysis import (
    barren_plateau_scan,
    eigenvalues_hermitian,
    entanglement_entropy,
    entanglement_profile,
    expressibility,
    reduced_density_matrix,
)
from ultraquant.quantum.ansatz import AmplitudeEncoder, QuantumCircuitModel
from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.kernel import (
    KernelClassifier,
    QuantumKernel,
    fidelity,
    inversion_test,
    swap_test,
)
from ultraquant.quantum.noise import (
    NoiseModel,
    NoisyBackend,
    fold_circuit,
    mitigation_report,
    zero_noise_extrapolate,
)
from ultraquant.quantum.search import (
    diffusion,
    indices_oracle,
    optimal_iterations,
    pattern_oracle,
    phase_operator,
    search,
)
from ultraquant.quantum.state import QuantumState


class CircuitInverseTests(unittest.TestCase):
    """Every gate must have a correct adjoint."""

    def test_random_circuit_undoes_itself(self) -> None:
        rng = random.Random(0)
        for trial in range(5):
            with self.subTest(trial=trial):
                circuit = Circuit(4)
                for _gate in range(30):
                    choice = rng.randrange(7)
                    q = rng.randrange(4)
                    if choice == 0:
                        circuit.h(q)
                    elif choice == 1:
                        circuit.ry(q, rng.uniform(0, 6))
                    elif choice == 2:
                        circuit.rz(q, rng.uniform(0, 6))
                    elif choice == 3:
                        circuit.rx(q, rng.uniform(0, 6))
                    elif choice == 4:
                        circuit.s(q)
                    elif choice == 5:
                        circuit.t(q)
                    else:
                        other = rng.randrange(4)
                        if other != q:
                            circuit.cnot(q, other)
                state = QuantumState(4)
                circuit.compose(circuit.inverse()).apply_to(state)
                self.assertAlmostEqual(state.probabilities()[0], 1.0, places=10)

    def test_compose_rejects_mismatched_width(self) -> None:
        with self.assertRaises(ValueError):
            Circuit(2).compose(Circuit(3))


class KernelTests(unittest.TestCase):
    """Overlap measured three ways must agree."""

    def test_inversion_and_swap_agree_with_direct_fidelity(self) -> None:
        encoder = AmplitudeEncoder(3)
        rng = random.Random(0)
        for trial in range(4):
            with self.subTest(trial=trial):
                a = [rng.uniform(0, 1) for _ in range(8)]
                b = [rng.uniform(0, 1) for _ in range(8)]
                ca, cb = encoder.encode(a), encoder.encode(b)
                sa, sb = QuantumState(3), QuantumState(3)
                ca.apply_to(sa)
                cb.apply_to(sb)
                direct = fidelity(sa.amplitudes, sb.amplitudes)
                self.assertAlmostEqual(inversion_test(ca, cb), direct, places=10)
                self.assertAlmostEqual(swap_test(ca, cb), direct, places=10)

    def test_identical_states_have_unit_overlap(self) -> None:
        encoder = AmplitudeEncoder(3)
        circuit = encoder.encode([0.2, 0.5, 0.1, 0.9, 0.4, 0.3, 0.7, 0.6])
        self.assertAlmostEqual(inversion_test(circuit, circuit), 1.0, places=10)

    def test_orthogonal_states_have_zero_overlap(self) -> None:
        encoder = AmplitudeEncoder(2)
        a = encoder.encode([1.0, 0.0, 0.0, 0.0])
        b = encoder.encode([0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(inversion_test(a, b), 0.0, places=10)

    def test_mismatched_widths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inversion_test(Circuit(2), Circuit(3))
        with self.assertRaises(ValueError):
            swap_test(Circuit(2), Circuit(3))

    def test_sampling_approaches_the_exact_value(self) -> None:
        encoder = AmplitudeEncoder(3)
        rng = random.Random(1)
        a = encoder.encode([rng.uniform(0, 1) for _ in range(8)])
        b = encoder.encode([rng.uniform(0, 1) for _ in range(8)])
        exact = inversion_test(a, b)
        sampled = inversion_test(a, b, shots=8000, seed=3)
        self.assertAlmostEqual(sampled, exact, delta=0.05)

    def test_kernel_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        from ultraquant.pattern.recognition import PATTERNS, render

        kernel = QuantumKernel(num_qubits=5)
        glyphs = [render(PATTERNS[n]) for n in ("plus", "square", "diamond")]
        matrix = kernel.matrix(glyphs)
        for i in range(3):
            self.assertAlmostEqual(matrix[i][i], 1.0, places=10)
            for j in range(3):
                self.assertAlmostEqual(matrix[i][j], matrix[j][i], places=12)

    def test_kernel_classifier_separates_glyphs(self) -> None:
        from ultraquant.pattern.recognition import PATTERNS, render

        rng = random.Random(0)
        xs, ys = [], []
        for label, name in enumerate(("plus", "square")):
            base = render(PATTERNS[name])
            for _ in range(4):
                variant = list(base)
                variant[rng.randrange(25)] = 1.0 - variant[rng.randrange(25)]
                xs.append(variant)
                ys.append(label)
        model = KernelClassifier(QuantumKernel(num_qubits=5)).fit(xs, ys)
        self.assertGreaterEqual(model.score(xs, ys), 0.75)


class NoiseTests(unittest.TestCase):
    """Noise degrades results, folding preserves the unitary, ZNE recovers."""

    def _circuit(self) -> Circuit:
        model = QuantumCircuitModel(num_qubits=3, layers=2, seed=1)
        return model.circuit([0.2, 0.7, 0.1, 0.4, 0.9, 0.3, 0.5, 0.6])

    def test_zero_noise_matches_the_exact_simulator(self) -> None:
        circuit = self._circuit()
        exact = QuantumState(3)
        circuit.apply_to(exact)
        backend = NoisyBackend(NoiseModel(0.0, 0.0, 0.0))
        got = backend.expectations(circuit)
        for value, reference in zip(got, [exact.expectation_z(q) for q in range(3)]):
            self.assertAlmostEqual(value, reference, places=12)

    def test_noise_moves_the_answer(self) -> None:
        circuit = self._circuit()
        exact = NoisyBackend(NoiseModel(0.0, 0.0)).expectations(circuit)
        noisy = NoisyBackend(NoiseModel(0.01, 0.05), trajectories=200, seed=0)
        got = noisy.expectations(circuit)
        self.assertGreater(max(abs(a - b) for a, b in zip(got, exact)), 1e-3)

    def test_folding_preserves_the_unitary(self) -> None:
        circuit = self._circuit()
        clean = NoisyBackend(NoiseModel(0.0, 0.0))
        base = clean.expectations(circuit)
        for scale in (3, 5):
            with self.subTest(scale=scale):
                folded = clean.expectations(fold_circuit(circuit, scale))
                for a, b in zip(folded, base):
                    self.assertAlmostEqual(a, b, places=10)

    def test_folding_multiplies_the_gate_count(self) -> None:
        circuit = self._circuit()
        self.assertEqual(len(fold_circuit(circuit, 1).ops), len(circuit.ops))
        self.assertEqual(len(fold_circuit(circuit, 3).ops), 3 * len(circuit.ops))
        self.assertEqual(len(fold_circuit(circuit, 5).ops), 5 * len(circuit.ops))

    def test_even_fold_scale_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fold_circuit(self._circuit(), 2)

    def test_mitigation_reduces_error_at_modest_noise(self) -> None:
        """The claim that mitigation helps, measured rather than asserted."""
        report = mitigation_report(
            self._circuit(), NoiseModel(single_qubit=0.001, two_qubit=0.01),
            trajectories=400, seed=0,
        )
        self.assertLess(report["mitigated_error"], report["unmitigated_error"])
        self.assertGreater(report["error_removed"], 0.2)

    def test_extrapolation_needs_two_scales(self) -> None:
        backend = NoisyBackend(NoiseModel(0.001, 0.01), trajectories=10)
        with self.assertRaises(ValueError):
            zero_noise_extrapolate(self._circuit(), backend, scales=(1,))
        with self.assertRaises(ValueError):
            zero_noise_extrapolate(self._circuit(), backend, method="magic")

    def test_invalid_error_rate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NoiseModel(single_qubit=1.5)


class SearchTests(unittest.TestCase):
    """Grover finds the marked states, and the round count matters."""

    def test_finds_a_single_marked_index(self) -> None:
        for num_qubits in (3, 4, 5):
            with self.subTest(qubits=num_qubits):
                target = (1 << num_qubits) - 3
                result = search(num_qubits, indices_oracle(num_qubits, [target]), 1)
                self.assertEqual(result.best, target)
                self.assertGreater(result.probability, 0.9)

    def test_speedup_grows_with_the_search_space(self) -> None:
        small = search(3, indices_oracle(3, [5]), 1)
        large = search(6, indices_oracle(6, [61]), 1)
        self.assertGreater(large.speedup, small.speedup)

    def test_structured_oracle_amplifies_every_match(self) -> None:
        """A catalog query marks a whole field value, not one index."""
        from ultraquant.quantum.search import grover_circuit

        mask, pattern = 0b000111, 0b000110
        result = search(6, pattern_oracle(6, pattern, mask), num_marked=8)
        state = QuantumState(6)
        grover_circuit(pattern_oracle(6, pattern, mask), result.iterations).apply_to(state)
        matched = sum(
            p for i, p in enumerate(state.probabilities()) if (i & mask) == pattern
        )
        self.assertGreater(matched, 0.8, "matching shards must dominate the outcome")

    def test_pattern_oracle_is_cheap(self) -> None:
        """The structured oracle must not scale with the number of matches."""
        wide = pattern_oracle(8, 0b0110, mask=0b1111)
        narrow = pattern_oracle(8, 0b0110, mask=0b111111)
        self.assertLess(len(wide.ops), 64)
        self.assertLess(len(narrow.ops), 400)

    def test_optimal_iterations(self) -> None:
        self.assertEqual(optimal_iterations(4, 16), 0)
        self.assertGreaterEqual(optimal_iterations(6, 1), 5)
        self.assertEqual(optimal_iterations(4, 0), 0)

    def test_success_probability_oscillates_with_rounds(self) -> None:
        """Grover rotates the state; it does not converge.

        Success probability is ``sin^2((2k+1) arcsin(sqrt(M/N)))``, so running
        past the optimum sweeps the amplitude away from the target and, later,
        back towards it. More rounds is therefore not "more accurate" — which is
        why :func:`optimal_iterations` exists rather than a convergence loop.
        """
        oracle = indices_oracle(4, [9])
        probabilities = [
            search(4, oracle, 1, iterations=k).probability for k in range(13)
        ]
        best_round = max(range(13), key=lambda k: probabilities[k])
        worst_after = min(probabilities[best_round + 1:])
        self.assertGreater(probabilities[best_round], 0.9)
        self.assertLess(worst_after, 0.2, "the amplitude must swing away again")
        self.assertAlmostEqual(
            probabilities[optimal_iterations(4, 1)], probabilities[best_round],
            delta=0.05, msg="the computed round count should be near the peak",
        )

    def test_phase_operator_length_checked(self) -> None:
        with self.assertRaises(ValueError):
            phase_operator(2, [0.0, 0.0])

    def test_diffusion_is_unitary(self) -> None:
        state = QuantumState(3)
        diffusion(3).apply_to(state)
        self.assertAlmostEqual(state.norm(), 1.0, places=12)


class AnalysisTests(unittest.TestCase):
    """Entanglement and trainability diagnostics report the truth."""

    def test_product_state_has_zero_entropy(self) -> None:
        state = QuantumState(2)
        Circuit(2).h(0).apply_to(state)
        self.assertAlmostEqual(entanglement_entropy(state.amplitudes, 2, [0]), 0.0, places=10)

    def test_bell_state_is_maximally_entangled(self) -> None:
        state = QuantumState(2)
        Circuit(2).h(0).cnot(0, 1).apply_to(state)
        self.assertAlmostEqual(entanglement_entropy(state.amplitudes, 2, [0]), 1.0, places=10)

    def test_ghz_state(self) -> None:
        state = QuantumState(3)
        Circuit(3).h(0).cnot(0, 1).cnot(1, 2).apply_to(state)
        self.assertAlmostEqual(entanglement_entropy(state.amplitudes, 3, [0]), 1.0, places=10)

    def test_reduced_density_matrix_has_unit_trace(self) -> None:
        state = QuantumState(3)
        Circuit(3).h(0).cnot(0, 1).ry(2, 0.7).apply_to(state)
        rho = reduced_density_matrix(state.amplitudes, 3, [0, 1])
        trace = sum(rho[i][i] for i in range(len(rho)))
        self.assertAlmostEqual(complex(trace).real, 1.0, places=10)

    def test_eigenvalues_of_a_known_matrix(self) -> None:
        values = eigenvalues_hermitian([[2 + 0j, 0j], [0j, 5 + 0j]])
        self.assertAlmostEqual(max(values), 5.0, places=9)
        self.assertAlmostEqual(min(values), 2.0, places=9)

    def test_our_model_circuit_actually_entangles(self) -> None:
        """A separable circuit would mean the quantum framing buys nothing."""
        from ultraquant.pattern.recognition import PATTERNS, render

        model = QuantumCircuitModel(num_qubits=4, layers=2, seed=1)
        profile = entanglement_profile(model.circuit(render(PATTERNS["plus"])[:16]))
        self.assertFalse(profile["separable"])
        self.assertGreater(profile["mean"], 0.1)

    def test_barren_plateau_variance_falls_with_width(self) -> None:
        points = barren_plateau_scan(qubit_range=(2, 5), layers=3, samples=16)
        self.assertEqual(len(points), 2)
        self.assertLess(points[-1].variance, points[0].variance,
                        "gradient variance must shrink as the register widens")

    def test_expressibility_reports_a_divergence(self) -> None:
        report = expressibility(num_qubits=3, layers=2, samples=30)
        self.assertGreaterEqual(report["kl_divergence"], 0.0)
        self.assertAlmostEqual(report["haar_mean_fidelity"], 1 / 8, places=12)


if __name__ == "__main__":
    unittest.main()
