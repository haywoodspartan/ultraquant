"""Tests for the quantum engine: gates, statevector, circuit, backends."""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.quantum import gates
from ultraquant.quantum.backend import (
    Backend,
    BackendResult,
    QPUBackend,
    SimulatorBackend,
    auto_backend,
)
from ultraquant.quantum.circuit import Circuit, Op
from ultraquant.quantum.state import QuantumState


class TestQuantumState(unittest.TestCase):
    def test_h_on_zero_gives_half_half(self) -> None:
        state = QuantumState(1)
        state.apply_gate(gates.H, 0)
        probs = state.probabilities()
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], 0.5, places=9)
        self.assertAlmostEqual(probs[1], 0.5, places=9)

    def test_bell_state_counts_and_expectations(self) -> None:
        state = QuantumState(2, seed=42)
        state.apply_gate(gates.H, 0)
        state.apply_controlled(gates.X, 0, 1)  # CNOT
        counts = state.measure_all(2000)
        self.assertTrue(set(counts) <= {"00", "11"}, f"unexpected keys: {counts}")
        self.assertEqual(sum(counts.values()), 2000)
        self.assertAlmostEqual(state.expectation_z(0), 0.0, delta=1e-9)
        self.assertAlmostEqual(state.expectation_z(1), 0.0, delta=1e-9)

    def test_bell_via_circuit(self) -> None:
        state = QuantumState(2, seed=7)
        Circuit(2).h(0).cnot(0, 1).apply_to(state)
        probs = state.probabilities()
        self.assertAlmostEqual(probs[0], 0.5, places=9)
        self.assertAlmostEqual(probs[3], 0.5, places=9)
        self.assertAlmostEqual(probs[1], 0.0, places=9)
        self.assertAlmostEqual(probs[2], 0.0, places=9)

    def test_rotations_preserve_norm(self) -> None:
        rng = random.Random(123)
        state = QuantumState(3)
        for _ in range(25):
            theta = rng.uniform(-2.0 * math.pi, 2.0 * math.pi)
            q = rng.randrange(3)
            state.apply_gate(gates.rx(theta), q)
            state.apply_gate(gates.ry(theta), q)
            state.apply_gate(gates.rz(theta), q)
            state.apply_gate(gates.H, q)
            self.assertAlmostEqual(state.norm(), 1.0, places=9)

    def test_little_endian_convention(self) -> None:
        # X on qubit 0 of a 2-qubit register -> basis index 1 (LSB set),
        # bitstring "01" (qubit 0 is the rightmost character).
        state = QuantumState(2, seed=0)
        state.apply_gate(gates.X, 0)
        self.assertAlmostEqual(abs(state.amplitudes[1] - 1.0), 0.0, places=9)
        for i in (0, 2, 3):
            self.assertAlmostEqual(abs(state.amplitudes[i]), 0.0, places=9)
        counts = state.measure_all(10)
        self.assertEqual(counts, {"01": 10})
        self.assertAlmostEqual(state.expectation_z(0), -1.0, places=9)
        self.assertAlmostEqual(state.expectation_z(1), 1.0, places=9)

    def test_seeded_measure_all_is_deterministic(self) -> None:
        def build_and_measure() -> dict[str, int]:
            state = QuantumState(3, seed=7)
            Circuit(3).h(0).ry(1, 0.9).cnot(0, 2).rx(2, -0.4).apply_to(state)
            return state.measure_all(500)

        self.assertEqual(build_and_measure(), build_and_measure())

    def test_swap_op(self) -> None:
        state = QuantumState(2, seed=1)
        Circuit(2).x(0).swap(0, 1).apply_to(state)
        # |01> (index 1) swapped to |10> (index 2); counts key "10".
        self.assertAlmostEqual(abs(state.amplitudes[2] - 1.0), 0.0, places=9)
        self.assertEqual(state.measure_all(5), {"10": 5})


class TestCircuit(unittest.TestCase):
    def test_builders_chain_and_record_ops(self) -> None:
        circ = Circuit(2)
        returned = circ.h(0).x(1).ry(0, 0.5).cnot(0, 1).cz(1, 0)
        self.assertIs(returned, circ)
        self.assertEqual(
            [op.name for op in circ.ops], ["h", "x", "ry", "cnot", "cz"]
        )
        self.assertEqual(circ.ops[2], Op("ry", (0,), (0.5,)))

    def test_out_of_range_qubit_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Circuit(2).h(2)


class TestSimulatorBackend(unittest.TestCase):
    def test_exact_run(self) -> None:
        circ = Circuit(2).h(0).cnot(0, 1)
        result = SimulatorBackend(seed=0).run(circ)
        self.assertIsNone(result.counts)
        self.assertEqual(len(result.expectations_z), 2)
        self.assertAlmostEqual(result.expectations_z[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(result.expectations_z[1], 0.0, delta=1e-9)

    def test_shots_estimate_close_to_exact(self) -> None:
        circ = Circuit(2).ry(0, 0.7).ry(1, -1.3).cnot(0, 1)
        exact = SimulatorBackend(seed=5).run(circ).expectations_z
        sampled = SimulatorBackend(seed=5).run(circ, shots=4096)
        self.assertIsNotNone(sampled.counts)
        assert sampled.counts is not None
        self.assertEqual(sum(sampled.counts.values()), 4096)
        for e, s in zip(exact, sampled.expectations_z):
            self.assertLess(abs(e - s), 0.1)
        for v in sampled.expectations_z:
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_seeded_backend_run_sequence_deterministic(self) -> None:
        circ = Circuit(2).h(0).cnot(0, 1)

        def run_twice() -> list[dict[str, int] | None]:
            backend = SimulatorBackend(seed=9)
            return [backend.run(circ, shots=200).counts for _ in range(2)]

        first = run_twice()
        second = run_twice()
        self.assertEqual(first, second)

    def test_backend_name(self) -> None:
        self.assertEqual(SimulatorBackend().name, "ultraquant-statevector")
        self.assertIsInstance(SimulatorBackend(), Backend)


class TestBackendDispatch(unittest.TestCase):
    def test_qpu_available_returns_bool_and_never_raises(self) -> None:
        self.assertIsInstance(QPUBackend.available(), bool)

    def test_auto_backend_without_qpu_preference(self) -> None:
        backend = auto_backend(prefer_qpu=False, seed=3)
        self.assertIsInstance(backend, SimulatorBackend)
        result = backend.run(Circuit(1).h(0))
        self.assertIsInstance(result, BackendResult)
        self.assertAlmostEqual(result.expectations_z[0], 0.0, delta=1e-9)

    def test_auto_backend_never_raises(self) -> None:
        backend = auto_backend(prefer_qpu=True, seed=1)
        self.assertIsInstance(backend, Backend)


if __name__ == "__main__":
    unittest.main()
