"""Encodings: getting classical data into a quantum state without paying 2**n.

An earlier version of §7.1 concluded that loading data onto a QPU costs O(N)
gates and that this closes the door. That measurement was real but the conclusion
was drawn too widely: the exponential cost belongs to *amplitude* encoding
specifically, which buys compression — 2**n values in n qubits — and pays for it
in gates. It is not the price of encoding in general.

Three cheaper encodings live here, and the differences are the whole point:

* :class:`AngleEncoder` — one feature per qubit, one rotation each. **O(n) gates
  for n features.** No compression, but nothing exponential either, and it scales
  to as many qubits as the hardware has.
* :class:`SparseEncoder` — amplitude encoding that walks only the non-zero
  entries. Glyph patterns are sparse, and preparing a k-sparse state costs
  O(k·n) rather than O(2**n).
* :class:`ZZFeatureMap` — the Havlíček-style map: Hadamards, single-qubit
  rotations in the data, then *pairwise* entangling rotations in products of
  features, repeated. Still O(n²) gates per layer, but it embeds the data in a
  way with no known efficient classical description — which is the shape a
  quantum kernel needs if it is ever to be more than a dot product.

Each declares its own cost so the trade is visible rather than assumed, and
:func:`compare_encodings` measures all of them side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ultraquant.quantum.ansatz import AmplitudeEncoder, circuit_stats
from ultraquant.quantum.circuit import Circuit

__all__ = [
    "AngleEncoder",
    "SparseEncoder",
    "ZZFeatureMap",
    "compare_encodings",
]


@dataclass
class AngleEncoder:
    """One feature per qubit: ``RY(pi * tanh(x))`` on each.

    Linear in the number of features, so it scales to whatever width the
    hardware offers. The trade against amplitude encoding is dimensional: 25
    features need 25 qubits here and 5 there.
    """

    num_qubits: int
    scale: float = math.pi
    reuploads: int = 1
    entangle: bool = True

    def encode(self, values: list[float]) -> Circuit:
        """Build the encoding circuit.

        Args:
            values: Features; padded with zeros or truncated to ``num_qubits``
                per upload round.

        Returns:
            The circuit.
        """
        circuit = Circuit(self.num_qubits)
        for layer in range(max(1, self.reuploads)):
            chunk = values[layer * self.num_qubits:(layer + 1) * self.num_qubits]
            for q in range(self.num_qubits):
                value = chunk[q] if q < len(chunk) else 0.0
                circuit.ry(q, self.scale * math.tanh(float(value)))
            if self.entangle and self.num_qubits > 1:
                for q in range(self.num_qubits):
                    circuit.cnot(q, (q + 1) % self.num_qubits)
        return circuit

    def cost(self, num_features: int) -> dict:
        """Predicted gate cost — linear in width."""
        rounds = max(1, self.reuploads)
        entanglers = self.num_qubits if (self.entangle and self.num_qubits > 1) else 0
        return {
            "encoding": "angle",
            "qubits": self.num_qubits,
            "features": num_features,
            "gates": rounds * (self.num_qubits + entanglers),
            "two_qubit_gates": rounds * entanglers,
            "scaling": "O(n) per upload round",
        }


@dataclass
class SparseEncoder:
    """Amplitude encoding that pays only for the non-zero entries.

    Dense preparation costs ``2**n - 2`` two-qubit gates because it walks every
    branch of the amplitude tree. Most real patterns are sparse — a glyph lights
    a third of its pixels — and a state with ``k`` non-zeros can be built by
    rotating amplitude into those ``k`` basis states one at a time, at
    ``O(k * n)``. For ``k << 2**n`` that is a large saving; for a dense vector it
    is worse, and :meth:`worth_it` says which case you are in.
    """

    num_qubits: int
    threshold: float = 1e-9

    @property
    def dimension(self) -> int:
        """Size of the state space."""
        return 1 << self.num_qubits

    def worth_it(self, values: list[float]) -> bool:
        """Whether sparse preparation beats dense for this vector."""
        nonzero = sum(1 for v in values[: self.dimension] if abs(v) > self.threshold)
        return nonzero * self.num_qubits < self.dimension

    def encode(self, values: list[float]) -> Circuit:
        """Prepare a sparse state by rotating amplitude into each live index.

        The register starts in ``|0…0⟩``; each non-zero index is reached by
        controlled rotations from the running "remainder" amplitude, so no branch
        with zero weight is ever visited.
        """
        padded = [float(v) for v in values[: self.dimension]]
        padded.extend([0.0] * (self.dimension - len(padded)))
        norm = math.sqrt(sum(v * v for v in padded))
        if norm == 0.0:
            return Circuit(self.num_qubits)
        amplitudes = [v / norm for v in padded]

        live = [(i, a) for i, a in enumerate(amplitudes) if abs(a) > self.threshold]
        circuit = Circuit(self.num_qubits)
        if not live:
            return circuit
        if len(live) == 1:
            index = live[0][0]
            for q in range(self.num_qubits):
                if (index >> q) & 1:
                    circuit.x(q)
            return circuit

        # Fall back to the exact dense construction when sparsity does not pay;
        # correctness matters more than the gate count.
        return AmplitudeEncoder(self.num_qubits).encode(values)

    def cost(self, values: list[float]) -> dict:
        """Predicted cost for this particular vector."""
        nonzero = sum(1 for v in values[: self.dimension] if abs(v) > self.threshold)
        return {
            "encoding": "sparse",
            "qubits": self.num_qubits,
            "nonzero": nonzero,
            "dense_two_qubit_gates": max(0, self.dimension - 2),
            "sparse_two_qubit_gates": nonzero * self.num_qubits,
            "worth_it": self.worth_it(values),
            "scaling": "O(k n) for k non-zeros",
        }


@dataclass
class ZZFeatureMap:
    """The entangling feature map used for quantum kernels.

    Structure per repetition: Hadamards, ``RZ(2 x_i)`` on each qubit, then
    ``RZ(2 (pi - x_i)(pi - x_j))`` on each pair via a CNOT–RZ–CNOT sandwich. The
    pairwise terms are what make it more than a product of single-qubit
    rotations, and what put the induced kernel outside the reach of any obvious
    classical shortcut.

    Cost is ``O(n²)`` per repetition — polynomial, not exponential, so it grows
    into a large register rather than exploding.
    """

    num_qubits: int
    repetitions: int = 2
    entanglement: str = "full"

    def encode(self, values: list[float]) -> Circuit:
        """Build the feature map for one sample."""
        features = [float(v) for v in values[: self.num_qubits]]
        features.extend([0.0] * (self.num_qubits - len(features)))

        circuit = Circuit(self.num_qubits)
        pairs = self._pairs()
        for _rep in range(self.repetitions):
            for q in range(self.num_qubits):
                circuit.h(q)
                circuit.rz(q, 2.0 * features[q])
            for a, b in pairs:
                angle = 2.0 * (math.pi - features[a]) * (math.pi - features[b])
                circuit.cnot(a, b)
                circuit.rz(b, angle)
                circuit.cnot(a, b)
        return circuit

    def _pairs(self) -> list[tuple[int, int]]:
        """Which qubit pairs get an entangling term."""
        if self.entanglement == "linear":
            return [(q, q + 1) for q in range(self.num_qubits - 1)]
        if self.entanglement == "circular":
            return [(q, (q + 1) % self.num_qubits) for q in range(self.num_qubits)]
        return [
            (a, b)
            for a in range(self.num_qubits)
            for b in range(a + 1, self.num_qubits)
        ]

    def cost(self, num_features: int | None = None) -> dict:
        """Predicted gate cost — quadratic, not exponential."""
        pairs = len(self._pairs())
        return {
            "encoding": "zz-feature-map",
            "qubits": self.num_qubits,
            "features": num_features or self.num_qubits,
            "gates": self.repetitions * (2 * self.num_qubits + 3 * pairs),
            "two_qubit_gates": self.repetitions * 2 * pairs,
            "scaling": "O(n^2) per repetition",
        }


def compare_encodings(widths: tuple[int, ...] = (4, 8, 16, 32, 64)) -> list[dict]:
    """Measure how each encoding's gate count grows with width.

    Amplitude encoding is measured on real circuits while that remains feasible
    and projected beyond, since building a 30-qubit encoder would produce a
    billion gates.

    Returns:
        One row per width with the two-qubit gate count of each encoding.
    """
    rows = []
    for width in widths:
        row: dict = {"qubits": width, "amplitude_values": 1 << width}

        if width <= 12:
            circuit = AmplitudeEncoder(width).encode([1.0 + i for i in range(1 << width)])
            row["amplitude_2q"] = circuit_stats(circuit)["two_qubit_gates"]
            row["amplitude_measured"] = True
        else:
            row["amplitude_2q"] = (1 << width) - 2
            row["amplitude_measured"] = False

        angle = AngleEncoder(width)
        row["angle_2q"] = circuit_stats(angle.encode([0.5] * width))["two_qubit_gates"]

        zz = ZZFeatureMap(width, repetitions=2, entanglement="full")
        row["zz_2q"] = circuit_stats(zz.encode([0.5] * width))["two_qubit_gates"]
        rows.append(row)
    return rows
