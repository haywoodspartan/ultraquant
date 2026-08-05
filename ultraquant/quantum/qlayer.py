"""Quantum feature map: the quantum layer of the UltraQuant model.

Classical features are encoded as RY rotation angles (squashed through tanh
for numerical safety), entangled with a ring of CNOTs, and re-uploaded in
chunks when there are more features than qubits. The extracted feature vector
is the per-qubit Z expectation, each value in [-1, 1].
"""

from __future__ import annotations

import math

from ultraquant.quantum.backend import Backend
from ultraquant.quantum.circuit import Circuit


class QuantumFeatureMap:
    """Encode classical features into a circuit and extract Z expectations."""

    def __init__(
        self, num_qubits: int, backend: Backend, shots: int | None = None
    ) -> None:
        """Create a feature map.

        Args:
            num_qubits: Width of the quantum register (and of the output vector).
            backend: Execution backend used by :meth:`extract`.
            shots: Passed through to ``backend.run``; ``None`` means exact
                expectations where the backend supports them.
        """
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self.backend = backend
        self.shots = shots

    def encode(self, features: list[float]) -> Circuit:
        """Build the encoding circuit for ``features``.

        Angles are ``theta_i = pi * tanh(f_i)``. The first ``num_qubits``
        features fill an RY layer (missing features contribute angle 0),
        followed by a ring of CNOTs ``cnot(q, (q+1) % num_qubits)`` (skipped
        for a single qubit). Remaining features are data re-uploaded: the
        (RY layer, entangler ring) block repeats with the next chunk until all
        features are consumed.
        """
        n = self.num_qubits
        circuit = Circuit(n)
        feats = [float(f) for f in features]
        if not feats:
            feats = [0.0] * n
        for start in range(0, len(feats), n):
            chunk = feats[start : start + n]
            for q in range(n):
                value = chunk[q] if q < len(chunk) else 0.0
                circuit.ry(q, math.pi * math.tanh(value))
            if n > 1:
                for q in range(n):
                    circuit.cnot(q, (q + 1) % n)
        return circuit

    def extract(self, features: list[float]) -> list[float]:
        """Run the encoding circuit and return per-qubit ``<Z>`` values.

        The result has length ``num_qubits`` with every entry in [-1, 1].
        """
        result = self.backend.run(self.encode(features), self.shots)
        return list(result.expectations_z)
