"""Dense statevector simulator for a small register of qubits.

Bit convention is **little-endian**: qubit ``q`` is bit ``q`` of the basis-state
index, so qubit 0 is the least significant bit. Bitstring keys returned by
:meth:`QuantumState.measure_all` put qubit 0 as the *rightmost* character,
matching ordinary binary notation of the index.
"""

from __future__ import annotations

import math
import random

from ultraquant.quantum.gates import Matrix


class QuantumState:
    """Dense complex statevector over ``2**num_qubits`` amplitudes.

    The state is initialised to ``|0...0>``. All sampling randomness flows
    through an internal ``random.Random(seed)`` instance so results are
    deterministic when seeded.
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Create the ``|0...0>`` state on ``num_qubits`` qubits.

        Args:
            num_qubits: Number of qubits in the register (must be >= 1).
            seed: Optional seed for the internal RNG used by ``measure_all``.
        """
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits: int = num_qubits
        self.amplitudes: list[complex] = [0j] * (2**num_qubits)
        self.amplitudes[0] = 1 + 0j
        self._rng: random.Random = random.Random(seed)

    def apply_gate(self, matrix: Matrix, qubit: int) -> None:
        """Apply a 2x2 gate ``matrix`` to ``qubit`` in place.

        Iterates over index pairs differing only in bit ``qubit``.
        """
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"qubit {qubit} out of range for {self.num_qubits} qubits")
        m00, m01 = matrix[0][0], matrix[0][1]
        m10, m11 = matrix[1][0], matrix[1][1]
        bit = 1 << qubit
        amps = self.amplitudes
        for i in range(len(amps)):
            if i & bit:
                continue
            j = i | bit
            a0, a1 = amps[i], amps[j]
            amps[i] = m00 * a0 + m01 * a1
            amps[j] = m10 * a0 + m11 * a1

    def apply_controlled(self, matrix: Matrix, control: int, target: int) -> None:
        """Apply ``matrix`` to ``target`` only where the ``control`` bit is 1.

        CNOT is ``apply_controlled(X, control, target)``; CZ likewise with Z.
        """
        if control == target:
            raise ValueError("control and target must differ")
        for q in (control, target):
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} out of range for {self.num_qubits} qubits")
        m00, m01 = matrix[0][0], matrix[0][1]
        m10, m11 = matrix[1][0], matrix[1][1]
        cbit = 1 << control
        tbit = 1 << target
        amps = self.amplitudes
        for i in range(len(amps)):
            if not i & cbit:
                continue
            if i & tbit:
                continue
            j = i | tbit
            a0, a1 = amps[i], amps[j]
            amps[i] = m00 * a0 + m01 * a1
            amps[j] = m10 * a0 + m11 * a1

    def probabilities(self) -> list[float]:
        """Return ``|amplitude|**2`` for every basis-state index."""
        return [abs(a) ** 2 for a in self.amplitudes]

    def expectation_z(self, qubit: int) -> float:
        """Exact ``<Z_qubit>``: sum of ``p(i) * (+1 if bit q of i == 0 else -1)``."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"qubit {qubit} out of range for {self.num_qubits} qubits")
        bit = 1 << qubit
        total = 0.0
        for i, a in enumerate(self.amplitudes):
            p = abs(a) ** 2
            total += -p if i & bit else p
        return total

    def measure_all(self, shots: int) -> dict[str, int]:
        """Sample ``shots`` basis states from ``probabilities()`` with the internal RNG.

        Keys are bitstrings with qubit 0 as the rightmost character (i.e. the
        plain binary rendering of the basis-state index, zero-padded to
        ``num_qubits`` characters).
        """
        if shots < 1:
            raise ValueError("shots must be >= 1")
        probs = self.probabilities()
        indices = self._rng.choices(range(len(probs)), weights=probs, k=shots)
        counts: dict[str, int] = {}
        width = self.num_qubits
        for i in indices:
            key = format(i, f"0{width}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def norm(self) -> float:
        """L2 norm of the amplitude vector (should stay 1.0 under unitaries)."""
        return math.sqrt(sum(abs(a) ** 2 for a in self.amplitudes))
