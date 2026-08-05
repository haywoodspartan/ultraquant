"""Quantum kernels: similarity measured as the overlap of two encoded states.

A kernel method never needs the feature vectors themselves, only the inner
products between them. That is a natural fit for a quantum computer, because a
state living in a 2**n-dimensional Hilbert space has an inner product that is
cheap to *measure* and, for a suitable encoding, expensive to compute classically.
This is the one corner of quantum machine learning with a clean theoretical
story: the kernel is the quantum part, and everything after it is ordinary
classical learning.

Two ways to get the overlap, both implemented:

* **Inversion test** — prepare ``|φ(x)⟩``, apply the *inverse* of the circuit
  that prepares ``|φ(y)⟩``, and measure the probability of returning to
  ``|0…0⟩``. That probability is exactly ``|⟨φ(y)|φ(x)⟩|²``. It needs no extra
  qubits, which is why it is what hardware actually uses.
* **Swap test** — an ancilla, two Hadamards and a controlled swap. It needs
  ``2n + 1`` qubits instead of ``n`` but leaves both registers intact, and it is
  the textbook construction, so it is here to check the inversion test against.

Both are exact on the simulator and become sampling problems on hardware, with
the usual ``1/sqrt(shots)`` error.

Honest note on advantage: with the amplitude encoding used here the kernel is a
plain squared dot product, which a classical computer computes instantly. A
kernel that is genuinely hard to compute classically needs a feature map with
deep entanglement and no efficient classical description. The machinery is
general — the encoding is what would have to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ultraquant.quantum.ansatz import AmplitudeEncoder
from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.state import QuantumState

__all__ = [
    "fidelity",
    "inversion_test",
    "swap_test",
    "QuantumKernel",
    "KernelClassifier",
]


def fidelity(state_a: Sequence[complex], state_b: Sequence[complex]) -> float:
    """``|<a|b>|**2`` for two state vectors of equal length."""
    overlap = sum(complex(a).conjugate() * complex(b) for a, b in zip(state_a, state_b))
    return abs(overlap) ** 2


def inversion_test(prepare_x: Circuit, prepare_y: Circuit, shots: int | None = None,
                   seed: int | None = None) -> float:
    """Overlap of two prepared states, by running one preparation backwards.

    Args:
        prepare_x: Circuit preparing ``|φ(x)⟩`` from ``|0…0⟩``.
        prepare_y: Circuit preparing ``|φ(y)⟩`` from ``|0…0⟩``.
        shots: Sample this many measurements instead of computing exactly.
        seed: Seed for sampling.

    Returns:
        ``|<φ(y)|φ(x)>|**2``.

    Raises:
        ValueError: If the two circuits have different widths.
    """
    if prepare_x.num_qubits != prepare_y.num_qubits:
        raise ValueError("both preparations must act on the same number of qubits")

    state = QuantumState(prepare_x.num_qubits, seed=seed)
    prepare_x.compose(prepare_y.inverse()).apply_to(state)

    if shots is None:
        return state.probabilities()[0]
    counts = state.measure_all(shots)
    zeros = "0" * prepare_x.num_qubits
    return counts.get(zeros, 0) / shots


def swap_test(prepare_x: Circuit, prepare_y: Circuit, shots: int | None = None,
              seed: int | None = None) -> float:
    """Overlap via the textbook swap test, using an ancilla.

    Costs ``2n + 1`` qubits against the inversion test's ``n``, and is kept as an
    independent check on that cheaper construction rather than as the default.

    Returns:
        ``|<φ(y)|φ(x)>|**2``, recovered from the ancilla statistics.

    Raises:
        ValueError: If the two circuits have different widths.
    """
    if prepare_x.num_qubits != prepare_y.num_qubits:
        raise ValueError("both preparations must act on the same number of qubits")

    width = prepare_x.num_qubits
    total = 2 * width + 1
    ancilla = 2 * width           # the ancilla is the most significant qubit
    circuit = Circuit(total)

    for op in prepare_x.ops:
        circuit.ops.append(type(op)(op.name, op.qubits, op.params))
    for op in prepare_y.ops:
        shifted = tuple(q + width for q in op.qubits)
        circuit.ops.append(type(op)(op.name, shifted, op.params))

    circuit.h(ancilla)
    for q in range(width):
        _controlled_swap(circuit, ancilla, q, q + width)
    circuit.h(ancilla)

    state = QuantumState(total, seed=seed)
    circuit.apply_to(state)

    if shots is None:
        probability_zero = sum(
            p for index, p in enumerate(state.probabilities())
            if not (index >> ancilla) & 1
        )
    else:
        counts = state.measure_all(shots)
        hits = sum(
            count for key, count in counts.items()
            if key[-1 - ancilla] == "0"
        )
        probability_zero = hits / shots

    # P(ancilla = 0) = (1 + |<x|y>|**2) / 2
    return max(0.0, min(1.0, 2.0 * probability_zero - 1.0))


def _controlled_swap(circuit: Circuit, control: int, a: int, b: int) -> None:
    """Fredkin gate, built from CNOTs and a Toffoli decomposition."""
    circuit.cnot(b, a)
    _toffoli(circuit, control, a, b)
    circuit.cnot(b, a)


def _toffoli(circuit: Circuit, c1: int, c2: int, target: int) -> None:
    """Standard six-CNOT Toffoli decomposition with T gates."""
    circuit.h(target)
    circuit.cnot(c2, target)
    _tdg(circuit, target)
    circuit.cnot(c1, target)
    circuit.t(target)
    circuit.cnot(c2, target)
    _tdg(circuit, target)
    circuit.cnot(c1, target)
    circuit.t(c2)
    circuit.t(target)
    circuit.h(target)
    circuit.cnot(c1, c2)
    circuit.t(c1)
    _tdg(circuit, c2)
    circuit.cnot(c1, c2)


def _tdg(circuit: Circuit, qubit: int) -> None:
    """T dagger, as Z · S · T (since T**8 = I)."""
    circuit.z(qubit)
    circuit.s(qubit)
    circuit.t(qubit)


@dataclass
class QuantumKernel:
    """Kernel matrix from a state-preparation feature map."""

    num_qubits: int = 5
    encoder: AmplitudeEncoder = field(init=False)
    shots: int | None = None
    seed: int | None = None
    #: Optional extra circuit applied after encoding, to deepen the feature map.
    entangling_layers: int = 0

    def __post_init__(self) -> None:
        """Build the encoder."""
        self.encoder = AmplitudeEncoder(self.num_qubits)
        self.evaluations = 0

    def feature_circuit(self, values: list[float]) -> Circuit:
        """Circuit preparing the feature state for one sample."""
        circuit = self.encoder.encode(values)
        # Extra entanglement makes the kernel less classically reproducible;
        # with none, this reduces to a squared dot product.
        for _layer in range(self.entangling_layers):
            for q in range(self.num_qubits):
                circuit.cz(q, (q + 1) % self.num_qubits)
            for q in range(self.num_qubits):
                circuit.ry(q, math.pi / 4)
        return circuit

    def evaluate(self, x: list[float], y: list[float]) -> float:
        """Kernel entry ``k(x, y)`` — the overlap of the two feature states."""
        self.evaluations += 1
        return inversion_test(
            self.feature_circuit(x), self.feature_circuit(y),
            shots=self.shots, seed=self.seed,
        )

    def matrix(self, xs: Sequence[list[float]],
               ys: Sequence[list[float]] | None = None) -> list[list[float]]:
        """Full kernel matrix.

        When ``ys`` is omitted the matrix is symmetric with a unit diagonal, so
        only the upper triangle is computed — half the circuit runs.
        """
        if ys is None:
            size = len(xs)
            out = [[0.0] * size for _ in range(size)]
            for i in range(size):
                out[i][i] = 1.0
                for j in range(i + 1, size):
                    value = self.evaluate(xs[i], xs[j])
                    out[i][j] = value
                    out[j][i] = value
            return out
        return [[self.evaluate(x, y) for y in ys] for x in xs]


@dataclass
class KernelClassifier:
    """Kernel nearest-centroid classification over quantum feature states.

    Deliberately the simplest kernel method that works: a test point is assigned
    to whichever class its mean kernel similarity is highest for. It needs no
    training beyond storing the support set, which keeps the quantum part —
    the kernel — the only interesting thing in the pipeline.
    """

    kernel: QuantumKernel
    xs: list[list[float]] = field(default_factory=list)
    ys: list[int] = field(default_factory=list)

    def fit(self, xs: Sequence[list[float]], ys: Sequence[int]) -> "KernelClassifier":
        """Store the support set."""
        self.xs = [list(x) for x in xs]
        self.ys = list(ys)
        return self

    def scores(self, x: list[float]) -> dict[int, float]:
        """Mean kernel similarity to each class."""
        totals: dict[int, list[float]] = {}
        for support, label in zip(self.xs, self.ys):
            totals.setdefault(label, []).append(self.kernel.evaluate(x, support))
        return {label: sum(v) / len(v) for label, v in totals.items()}

    def predict(self, x: list[float]) -> tuple[int, float]:
        """Predicted class and its mean similarity."""
        scores = self.scores(x)
        best = max(scores, key=lambda k: scores[k])
        return best, scores[best]

    def score(self, xs: Sequence[list[float]], ys: Sequence[int]) -> float:
        """Accuracy over a dataset."""
        correct = sum(1 for x, y in zip(xs, ys) if self.predict(x)[0] == y)
        return correct / max(len(list(xs)), 1)
