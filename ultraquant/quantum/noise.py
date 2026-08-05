"""Noise models and error mitigation.

Section 7 of ARCHITECTURE.md lists error mitigation among the things that would
be needed for real quantum hardware to be worth using, and records that none was
implemented. This module implements it, and — just as importantly — implements
the noise you would be mitigating, so the claim can be tested rather than
asserted.

**Noise.** Real gates are imperfect. :class:`NoiseModel` applies the standard
Pauli-twirled approximation: after each gate, with some probability, a random
Pauli error is inserted — heavier for two-qubit gates, which is where most of the
error budget goes on real devices. Running many such trajectories and averaging
approximates the noisy expectation value without needing a density matrix, so it
composes with the existing statevector simulator instead of replacing it.

**Mitigation.** :func:`zero_noise_extrapolate` implements zero-noise
extrapolation: run the circuit at several *deliberately amplified* noise levels,
then extrapolate back to the zero-noise limit. Noise is amplified by unitary
folding — replacing ``G`` with ``G G† G``, which is mathematically the identity
substitution but takes three times as long to execute and so accumulates three
times the error. It is the workhorse mitigation technique precisely because it
needs no knowledge of the noise model and no extra qubits.

What this does not do is make noisy hardware exact. Extrapolation reduces bias at
the cost of variance; the honest measure is whether it lands closer to the true
value than the unmitigated run, which :func:`mitigation_report` measures directly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ultraquant.quantum.circuit import Circuit, Op
from ultraquant.quantum.state import QuantumState

__all__ = [
    "NoiseModel",
    "NoisyBackend",
    "fold_circuit",
    "zero_noise_extrapolate",
    "mitigation_report",
]

#: Single-qubit Pauli errors an insertion picks from.
_PAULIS = ("x", "y", "z")


@dataclass
class NoiseModel:
    """Stochastic Pauli noise, applied after each gate.

    Defaults are loosely representative of current superconducting hardware:
    two-qubit gates are roughly an order of magnitude noisier than single-qubit
    ones, which is why circuit *depth in two-qubit gates* is the figure that
    decides whether an algorithm survives.
    """

    single_qubit: float = 0.001
    two_qubit: float = 0.01
    readout: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the probabilities."""
        for name, value in (
            ("single_qubit", self.single_qubit),
            ("two_qubit", self.two_qubit),
            ("readout", self.readout),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} error must be in [0, 1], got {value}")

    @property
    def noiseless(self) -> bool:
        """Whether this model would leave a circuit untouched."""
        return self.single_qubit == 0.0 and self.two_qubit == 0.0 and self.readout == 0.0

    def apply(self, circuit: Circuit, rng: random.Random) -> Circuit:
        """Return a copy of ``circuit`` with random Pauli errors inserted."""
        noisy = Circuit(circuit.num_qubits)
        for op in circuit.ops:
            noisy.ops.append(op)
            probability = self.two_qubit if len(op.qubits) > 1 else self.single_qubit
            for qubit in op.qubits:
                if probability and rng.random() < probability:
                    noisy.ops.append(Op(rng.choice(_PAULIS), (qubit,), ()))
        return noisy


class NoisyBackend:
    """Estimate expectation values under a noise model, by trajectories.

    Each trajectory is one sampled error pattern run on the exact simulator;
    averaging over many of them approximates the noisy expectation. More
    trajectories mean less sampling error, at linear cost.
    """

    name = "noisy-trajectory"

    def __init__(self, noise: NoiseModel, trajectories: int = 200,
                 seed: int | None = 0) -> None:
        """Configure the estimator.

        Args:
            noise: The noise model to apply.
            trajectories: How many sampled error patterns to average.
            seed: Base seed for reproducibility.
        """
        self.noise = noise
        self.trajectories = max(1, int(trajectories))
        self.seed = seed
        self.circuits_run = 0

    def expectations(self, circuit: Circuit) -> list[float]:
        """Per-qubit ⟨Z⟩ under the noise model."""
        if self.noise.noiseless:
            state = QuantumState(circuit.num_qubits)
            circuit.apply_to(state)
            self.circuits_run += 1
            return [state.expectation_z(q) for q in range(circuit.num_qubits)]

        rng = random.Random(self.seed)
        totals = [0.0] * circuit.num_qubits
        for _trajectory in range(self.trajectories):
            state = QuantumState(circuit.num_qubits)
            self.noise.apply(circuit, rng).apply_to(state)
            self.circuits_run += 1
            for q in range(circuit.num_qubits):
                value = state.expectation_z(q)
                if self.noise.readout and rng.random() < self.noise.readout:
                    value = -value        # a flipped readout reports the opposite
                totals[q] += value
        return [total / self.trajectories for total in totals]


def fold_circuit(circuit: Circuit, scale: int) -> Circuit:
    """Amplify noise by unitary folding: ``G -> G (G† G)^k``.

    Mathematically this changes nothing — ``G† G`` is the identity — but it makes
    the hardware do three, five, seven times the work, and so accumulate that
    multiple of the error. That is what lets noise be *dialled* without knowing
    anything about its source.

    Args:
        circuit: The circuit to fold.
        scale: Odd multiplier (1, 3, 5, …). 1 returns the circuit unchanged.

    Returns:
        The folded circuit.

    Raises:
        ValueError: If ``scale`` is not a positive odd integer.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"fold scale must be a positive odd integer, got {scale}")
    if scale == 1:
        return circuit

    folded = Circuit(circuit.num_qubits)
    folded.ops.extend(circuit.ops)
    inverse = circuit.inverse()
    for _repeat in range((scale - 1) // 2):
        folded.ops.extend(inverse.ops)
        folded.ops.extend(circuit.ops)
    return folded


def zero_noise_extrapolate(
    circuit: Circuit,
    backend: NoisyBackend,
    scales: tuple[int, ...] = (1, 3, 5),
    method: str = "linear",
) -> list[float]:
    """Estimate the noise-free expectation by extrapolating from noisier runs.

    Args:
        circuit: The circuit to evaluate.
        backend: A noisy estimator.
        scales: Odd folding factors to sample. More points fit a better curve
            and cost proportionally more.
        method: ``"linear"`` (least squares in the noise scale) or
            ``"richardson"`` (exact polynomial through every point — lower bias,
            but it amplifies sampling noise, so it wants many trajectories).

    Returns:
        One extrapolated ⟨Z⟩ per qubit.

    Raises:
        ValueError: If fewer than two scales are given, or the method is unknown.
    """
    if len(scales) < 2:
        raise ValueError("zero-noise extrapolation needs at least two scales")
    if method not in ("linear", "richardson"):
        raise ValueError(f"unknown extrapolation method {method!r}")

    measured = [backend.expectations(fold_circuit(circuit, s)) for s in scales]
    width = circuit.num_qubits
    out = []
    for q in range(width):
        ys = [row[q] for row in measured]
        out.append(
            _linear_intercept(scales, ys) if method == "linear"
            else _richardson(scales, ys)
        )
    return out


def _linear_intercept(xs: tuple[int, ...], ys: list[float]) -> float:
    """Least-squares intercept at x = 0."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return mean_y - slope * mean_x


def _richardson(xs: tuple[int, ...], ys: list[float]) -> float:
    """Lagrange interpolation evaluated at x = 0."""
    total = 0.0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        term = yi
        for j, xj in enumerate(xs):
            if i != j:
                term *= (0.0 - xj) / (xi - xj)
        total += term
    return total


def mitigation_report(
    circuit: Circuit,
    noise: NoiseModel,
    trajectories: int = 400,
    scales: tuple[int, ...] = (1, 3, 5),
    seed: int = 0,
) -> dict:
    """Measure whether mitigation actually helped, against the exact answer.

    Returns:
        Exact, unmitigated and mitigated expectations plus the mean absolute
        error of each, and how much of the error the mitigation removed.
    """
    exact_state = QuantumState(circuit.num_qubits)
    circuit.apply_to(exact_state)
    exact = [exact_state.expectation_z(q) for q in range(circuit.num_qubits)]

    backend = NoisyBackend(noise, trajectories=trajectories, seed=seed)
    unmitigated = backend.expectations(circuit)
    mitigated = zero_noise_extrapolate(circuit, backend, scales=scales)

    def error(values: list[float]) -> float:
        return sum(abs(a - b) for a, b in zip(values, exact)) / len(exact)

    raw_error = error(unmitigated)
    fixed_error = error(mitigated)
    return {
        "exact": exact,
        "unmitigated": unmitigated,
        "mitigated": mitigated,
        "unmitigated_error": raw_error,
        "mitigated_error": fixed_error,
        "error_removed": (
            (raw_error - fixed_error) / raw_error if raw_error > 1e-12 else 0.0
        ),
        "circuits_run": backend.circuits_run,
        "two_qubit_gates": sum(1 for op in circuit.ops if len(op.qubits) > 1),
    }
