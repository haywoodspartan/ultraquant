"""Diagnostics: is this circuit doing anything quantum, and can it be trained?

Two failure modes make a variational circuit useless while it still runs happily
and returns numbers. Both are measurable, and neither is visible from accuracy
alone.

**No entanglement.** A circuit whose qubits never become correlated is a product
state, and a product state is simulable classically in linear time — whatever it
computes, a laptop can do without the quantum framing.
:func:`entanglement_entropy` measures the von Neumann entropy of a qubit's
reduced state: 0 means separable, 1 means maximally entangled with the rest.

**Barren plateaus.** As a random ansatz gets deeper or wider, the variance of its
gradients falls off exponentially. The circuit is then untrainable for reasons
that have nothing to do with the data: every gradient is indistinguishable from
zero, and no learning rate rescues it. This is the central practical obstacle in
variational quantum algorithms, and :func:`barren_plateau_scan` measures it
directly on the ansatz being used rather than trusting a rule of thumb.

Also here: :func:`expressibility`, which asks whether an ansatz can reach a good
spread of states at all, since a circuit that only ever produces nearly the same
state cannot fit anything.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass

from ultraquant.quantum.ansatz import VariationalAnsatz
from ultraquant.quantum.state import QuantumState

__all__ = [
    "reduced_density_matrix",
    "eigenvalues_hermitian",
    "entanglement_entropy",
    "entanglement_profile",
    "barren_plateau_scan",
    "expressibility",
]


def reduced_density_matrix(
    amplitudes: list[complex], num_qubits: int, keep: list[int]
) -> list[list[complex]]:
    """Trace out every qubit except ``keep``.

    Args:
        amplitudes: The full state vector.
        num_qubits: Register width.
        keep: Qubits to retain, in the order they should index the result.

    Returns:
        A ``2**len(keep)`` square Hermitian matrix.
    """
    keep = list(keep)
    kept = len(keep)
    rest = [q for q in range(num_qubits) if q not in keep]
    size = 1 << kept
    rho = [[0j] * size for _ in range(size)]

    for index, amplitude in enumerate(amplitudes):
        if amplitude == 0:
            continue
        row = sum(((index >> q) & 1) << p for p, q in enumerate(keep))
        environment = sum(((index >> q) & 1) << p for p, q in enumerate(rest))
        for other, other_amplitude in enumerate(amplitudes):
            if other_amplitude == 0:
                continue
            other_env = sum(((other >> q) & 1) << p for p, q in enumerate(rest))
            if other_env != environment:
                continue
            column = sum(((other >> q) & 1) << p for p, q in enumerate(keep))
            rho[row][column] += amplitude * complex(other_amplitude).conjugate()
    return rho


def eigenvalues_hermitian(matrix: list[list[complex]]) -> list[float]:
    """Eigenvalues of a Hermitian matrix, by Jacobi rotation.

    A complex Hermitian ``H`` has the same spectrum as the real symmetric block
    matrix ``[[Re H, -Im H], [Im H, Re H]]``, with every eigenvalue doubled —
    which lets one small real Jacobi solver serve without a linear-algebra
    dependency.
    """
    size = len(matrix)
    real = [[complex(matrix[i][j]).real for j in range(size)] for i in range(size)]
    imag = [[complex(matrix[i][j]).imag for j in range(size)] for i in range(size)]
    n = 2 * size
    a = [[0.0] * n for _ in range(n)]
    for i in range(size):
        for j in range(size):
            a[i][j] = real[i][j]
            a[i][j + size] = -imag[i][j]
            a[i + size][j] = imag[i][j]
            a[i + size][j + size] = real[i][j]

    for _sweep in range(100):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < 1e-12:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk

    doubled = sorted((a[i][i] for i in range(n)), reverse=True)
    # Each true eigenvalue appears twice in the embedding.
    return [doubled[i] for i in range(0, n, 2)]


def entanglement_entropy(
    amplitudes: list[complex], num_qubits: int, keep: list[int], base: float = 2.0
) -> float:
    """Von Neumann entropy of a subsystem, in bits by default.

    Zero means the subsystem is in a pure state — unentangled with the rest, so
    the circuit has produced nothing a classical product state could not. One bit
    per qubit is the maximum.
    """
    rho = reduced_density_matrix(amplitudes, num_qubits, keep)
    entropy = 0.0
    for value in eigenvalues_hermitian(rho):
        if value > 1e-12:
            entropy -= value * math.log(value, base)
    return max(0.0, entropy)


def entanglement_profile(circuit, num_qubits: int | None = None) -> dict:
    """Entropy of each qubit against the rest of the register.

    Returns:
        ``{"per_qubit": [...], "mean": float, "max": float, "separable": bool}``.
    """
    width = num_qubits if num_qubits is not None else circuit.num_qubits
    state = QuantumState(width)
    circuit.apply_to(state)
    per_qubit = [
        entanglement_entropy(state.amplitudes, width, [q]) for q in range(width)
    ]
    return {
        "per_qubit": per_qubit,
        "mean": sum(per_qubit) / max(len(per_qubit), 1),
        "max": max(per_qubit) if per_qubit else 0.0,
        "separable": all(v < 1e-6 for v in per_qubit),
    }


@dataclass
class PlateauPoint:
    """Gradient statistics at one circuit size."""

    num_qubits: int
    layers: int
    variance: float
    mean_abs: float
    samples: int


def barren_plateau_scan(
    qubit_range: tuple[int, ...] = (2, 3, 4, 5),
    layers: int = 3,
    samples: int = 24,
    seed: int = 0,
) -> list[PlateauPoint]:
    """Measure how gradient variance falls as the register widens.

    For each width, random parameter vectors are drawn and the gradient of
    ``<Z_0>`` with respect to the first parameter is computed exactly by the
    parameter-shift rule. An exponential decay in the variance is the barren
    plateau: the circuit is untrainable at that size regardless of the data or
    the optimiser.

    Returns:
        One :class:`PlateauPoint` per width.
    """
    from ultraquant.quantum.ansatz import SHIFT

    out: list[PlateauPoint] = []
    for num_qubits in qubit_range:
        ansatz = VariationalAnsatz(num_qubits, layers=layers)
        rng = random.Random(seed + num_qubits)
        gradients = []
        for _sample in range(samples):
            parameters = [
                rng.uniform(0, 2 * math.pi) for _ in range(ansatz.num_parameters)
            ]

            def measure(params: list[float]) -> float:
                state = QuantumState(num_qubits)
                for q in range(num_qubits):
                    state.apply_gate(
                        [[1 / math.sqrt(2), 1 / math.sqrt(2)],
                         [1 / math.sqrt(2), -1 / math.sqrt(2)]], q
                    )
                ansatz.build(params).apply_to(state)
                return state.expectation_z(0)

            shifted_up = list(parameters)
            shifted_up[0] += SHIFT
            shifted_down = list(parameters)
            shifted_down[0] -= SHIFT
            gradients.append((measure(shifted_up) - measure(shifted_down)) / 2.0)

        mean = sum(gradients) / len(gradients)
        variance = sum((g - mean) ** 2 for g in gradients) / len(gradients)
        out.append(PlateauPoint(
            num_qubits=num_qubits,
            layers=layers,
            variance=variance,
            mean_abs=sum(abs(g) for g in gradients) / len(gradients),
            samples=samples,
        ))
    return out


def expressibility(
    num_qubits: int = 4, layers: int = 2, samples: int = 60, bins: int = 20, seed: int = 0
) -> dict:
    """How widely an ansatz spreads over state space, against the Haar ideal.

    Random parameter pairs are drawn, their state fidelity measured, and the
    resulting distribution compared with the Haar-random one,
    ``P(F) = (N - 1)(1 - F)^(N - 2)``. A low divergence means the ansatz can
    reach most of the space; a high one means it keeps returning nearly the same
    state and cannot fit much, however many parameters it has.

    Returns:
        ``{"kl_divergence", "mean_fidelity", "haar_mean_fidelity", "samples"}``.
    """
    from ultraquant.quantum.kernel import fidelity

    ansatz = VariationalAnsatz(num_qubits, layers=layers)
    rng = random.Random(seed)
    dimension = 1 << num_qubits
    fidelities = []

    for _sample in range(samples):
        states = []
        for _side in range(2):
            parameters = [
                rng.uniform(0, 2 * math.pi) for _ in range(ansatz.num_parameters)
            ]
            state = QuantumState(num_qubits)
            for q in range(num_qubits):
                state.apply_gate(
                    [[1 / math.sqrt(2), 1 / math.sqrt(2)],
                     [1 / math.sqrt(2), -1 / math.sqrt(2)]], q
                )
            ansatz.build(parameters).apply_to(state)
            states.append(state.amplitudes)
        fidelities.append(fidelity(states[0], states[1]))

    observed = [0.0] * bins
    for value in fidelities:
        index = min(bins - 1, int(value * bins))
        observed[index] += 1.0 / len(fidelities)

    divergence = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        # Integral of the Haar fidelity density over the bin.
        haar = (1 - low) ** (dimension - 1) - (1 - high) ** (dimension - 1)
        if observed[index] > 0 and haar > 0:
            divergence += observed[index] * math.log(observed[index] / haar)

    return {
        "kl_divergence": divergence,
        "mean_fidelity": sum(fidelities) / len(fidelities),
        "haar_mean_fidelity": 1.0 / dimension,
        "samples": samples,
    }
