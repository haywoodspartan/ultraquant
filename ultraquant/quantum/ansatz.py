"""A trainable quantum circuit for this system.

The feature map in :mod:`ultraquant.quantum.qlayer` is a fixed encoding: it takes
five row means, rotates them onto five qubits, entangles once, and reads ⟨Z⟩ back
out. It has no parameters, so it cannot learn anything, and it throws away
twenty of a glyph's twenty-five pixels before the circuit even starts. That is
why it buys nothing — and saying so has been the honest position all along.

This module builds the circuit that answers the obvious question: what would a
quantum layer look like if it were actually doing the work?

**Amplitude encoding.** Five qubits span 32 basis states, which is enough to hold
all 25 pixels of a glyph *at once*, one per amplitude. :class:`AmplitudeEncoder`
synthesises the state-preparation circuit for an arbitrary real vector using
Möttönen's scheme — a binary tree of uniformly-controlled RY rotations, decomposed
into plain RY and CNOT gates through a Gray-code walk. The whole pattern lives in
the state, not a five-number summary of it.

**A variational ansatz.** :class:`VariationalAnsatz` is a hardware-efficient
circuit: per layer, an RY and RZ on every qubit followed by a ring of CNOTs. Its
angles are free parameters, so the circuit is trained rather than fixed.

**Parameter-shift gradients.** The gradient of an expectation value with respect
to a rotation angle is exactly ``[f(θ + π/2) − f(θ − π/2)] / 2`` — no finite
differences, no backpropagation through the state. This matters because you
cannot backpropagate through real hardware: parameter-shift is how a variational
circuit is trained on a QPU, and using it here means the same training code runs
on the simulator and on a device.

Everything is exact on the simulator and reduces to sampling on hardware, exactly
like the rest of the quantum layer. Pure Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.state import QuantumState

__all__ = [
    "AmplitudeEncoder",
    "VariationalAnsatz",
    "QuantumCircuitModel",
    "parameter_shift_gradient",
    "circuit_stats",
    "draw",
]

#: The parameter-shift rule's displacement. Exact for gates of the form
#: exp(-i θ P / 2) with P a Pauli, which covers RX, RY and RZ.
SHIFT = math.pi / 2


# ---------------------------------------------------------------------------
# State preparation
# ---------------------------------------------------------------------------

def _gray(index: int) -> int:
    """Gray code of ``index``."""
    return index ^ (index >> 1)


def _multiplexor_angles(thetas: list[float]) -> list[float]:
    """Transform target angles into uniformly-controlled rotation angles.

    A uniformly-controlled RY applies a different angle for each control basis
    state. Implementing it directly would need one multi-controlled gate per
    state; instead the angles are transformed by the matrix
    ``M_ij = (-1)^{b(i)·g(j)} / 2^k`` and emitted as plain rotations separated by
    CNOTs, which is what makes the decomposition cheap.
    """
    count = len(thetas)
    bits = max(1, count.bit_length() - 1)
    out = []
    for j in range(count):
        total = 0.0
        for i in range(count):
            parity = bin(i & _gray(j)).count("1") & 1
            total += thetas[i] * (-1 if parity else 1)
        out.append(total / count)
    return out


class AmplitudeEncoder:
    """Prepare an arbitrary real vector as the amplitudes of a qubit register."""

    def __init__(self, num_qubits: int) -> None:
        """Configure an encoder for ``num_qubits`` qubits.

        Args:
            num_qubits: Register width; encodes up to ``2**num_qubits`` values.

        Raises:
            ValueError: If ``num_qubits`` is not positive.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self.num_qubits = int(num_qubits)
        self.dimension = 1 << self.num_qubits

    def normalise(self, values: list[float]) -> list[float]:
        """Pad and L2-normalise ``values`` to a legal state vector.

        A quantum state must have unit norm, so the data is normalised rather
        than clipped; an all-zero input becomes the uniform state, which is the
        only sensible choice that stays normalised.
        """
        padded = [float(v) for v in values[: self.dimension]]
        padded.extend([0.0] * (self.dimension - len(padded)))
        norm = math.sqrt(sum(v * v for v in padded))
        if norm == 0.0:
            uniform = 1.0 / math.sqrt(self.dimension)
            return [uniform] * self.dimension
        return [v / norm for v in padded]

    def encode(self, values: list[float]) -> Circuit:
        """Build a circuit preparing ``values`` as amplitudes from |0…0>.

        Args:
            values: Real values; padded or truncated to ``2**num_qubits``.

        Returns:
            A circuit whose final state has the (normalised) values as its
            amplitudes, in the project's little-endian index convention.
        """
        amplitudes = self.normalise(values)
        circuit = Circuit(self.num_qubits)

        # Walk the binary tree of subtree norms from the most significant qubit
        # down. At each level, one rotation per control pattern turns the parent
        # amplitude into the correct split between its two children.
        for level in range(self.num_qubits):
            # Chunk m of this level is selected by the qubits above the target,
            # with bit p of m being qubit (n - level + p). The control list must
            # follow that order: reversing it silently mis-assigns every angle,
            # which shows up as a fidelity just short of 1 rather than an error.
            control_qubits = [self.num_qubits - level + p for p in range(level)]
            target = self.num_qubits - 1 - level
            block = 1 << (self.num_qubits - level)
            thetas: list[float] = []
            for start in range(0, self.dimension, block):
                chunk = amplitudes[start:start + block]
                half = block >> 1
                left = math.sqrt(sum(v * v for v in chunk[:half]))
                right = math.sqrt(sum(v * v for v in chunk[half:]))
                if left == 0.0 and right == 0.0:
                    thetas.append(0.0)
                else:
                    thetas.append(2.0 * math.atan2(right, left))
            self._uniformly_controlled_ry(circuit, thetas, control_qubits, target)

        # Möttönen's construction assumes non-negative amplitudes; a sign flip is
        # a phase, applied afterwards where the data is actually negative.
        self._apply_signs(circuit, amplitudes)
        return circuit

    @staticmethod
    def _uniformly_controlled_ry(
        circuit: Circuit, thetas: list[float], controls: list[int], target: int
    ) -> None:
        """Emit a uniformly-controlled RY as RY/CNOT pairs in Gray-code order."""
        if not controls:
            if thetas and thetas[0] != 0.0:
                circuit.ry(target, thetas[0])
            return
        angles = _multiplexor_angles(thetas)
        count = len(angles)
        for j, angle in enumerate(angles):
            circuit.ry(target, angle)
            # The control that flips between this Gray code and the next is the
            # position of the changed bit; the last one wraps to the top control.
            if j == count - 1:
                control_index = len(controls) - 1
            else:
                control_index = (_gray(j) ^ _gray(j + 1)).bit_length() - 1
            circuit.cnot(controls[control_index], target)

    def _apply_signs(self, circuit: Circuit, amplitudes: list[float]) -> None:
        """Give negative amplitudes their sign, as a diagonal phase operator.

        The rotation tree above is built from subtree *norms*, so it can only
        produce non-negative amplitudes. The signs are therefore a separate
        diagonal operator ``diag((-1)^b_x)``, and any such operator expands over
        the Walsh–Hadamard basis: each non-empty qubit subset contributes a term
        ``exp(i α_S (-1)^{parity(S·x)})``, realised by a CNOT ladder that
        computes the parity of ``S``, an RZ, and the ladder undone.

        Terms with zero coefficient are skipped, so an all-positive vector — the
        usual case here, since pixels and row means are non-negative — adds no
        gates at all.
        """
        signs = [1.0 if v >= 0 else -1.0 for v in amplitudes]
        if all(s > 0 for s in signs):
            return

        n = self.num_qubits
        dimension = self.dimension
        phases = [0.0 if s > 0 else math.pi for s in signs]

        for subset in range(1, dimension):
            coefficient = 0.0
            for x in range(dimension):
                parity = bin(subset & x).count("1") & 1
                coefficient += phases[x] * (-1 if parity else 1)
            coefficient /= dimension
            if abs(coefficient) < 1e-12:
                continue

            members = [q for q in range(n) if (subset >> q) & 1]
            target = members[-1]
            for q in members[:-1]:
                circuit.cnot(q, target)
            # RZ(θ) contributes exp(-i θ (-1)^parity / 2), so θ = -2 α_S.
            circuit.rz(target, -2.0 * coefficient)
            for q in reversed(members[:-1]):
                circuit.cnot(q, target)


# ---------------------------------------------------------------------------
# Variational ansatz
# ---------------------------------------------------------------------------

@dataclass
class VariationalAnsatz:
    """A hardware-efficient trainable circuit.

    Per layer: RY and RZ on every qubit, then a ring of CNOTs. The rotations are
    the free parameters; the entangler is what makes the layer more than a
    product of single-qubit operations.
    """

    num_qubits: int
    layers: int = 2
    entangler: str = "ring"

    @property
    def num_parameters(self) -> int:
        """How many free angles this ansatz has."""
        return self.layers * self.num_qubits * 2

    def build(self, parameters: list[float]) -> Circuit:
        """Build the circuit for a parameter vector.

        Args:
            parameters: ``num_parameters`` angles.

        Returns:
            The parameterised circuit.

        Raises:
            ValueError: If the wrong number of parameters is supplied.
        """
        if len(parameters) != self.num_parameters:
            raise ValueError(
                f"ansatz needs {self.num_parameters} parameters, got {len(parameters)}"
            )
        circuit = Circuit(self.num_qubits)
        index = 0
        for _layer in range(self.layers):
            for q in range(self.num_qubits):
                circuit.ry(q, parameters[index])
                index += 1
                circuit.rz(q, parameters[index])
                index += 1
            if self.num_qubits > 1:
                if self.entangler == "ring":
                    for q in range(self.num_qubits):
                        circuit.cnot(q, (q + 1) % self.num_qubits)
                elif self.entangler == "linear":
                    for q in range(self.num_qubits - 1):
                        circuit.cnot(q, q + 1)
                elif self.entangler == "all-to-all":
                    for a in range(self.num_qubits):
                        for b in range(a + 1, self.num_qubits):
                            circuit.cz(a, b)
        return circuit


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class QuantumCircuitModel:
    """Encode data, apply a trainable ansatz, measure observables.

    The output is one ⟨Z⟩ per qubit — a real vector that downstream classical
    layers consume exactly like any other feature vector, except that these
    features have *trainable* quantum parameters behind them.
    """

    num_qubits: int = 5
    layers: int = 2
    entangler: str = "ring"
    parameters: list[float] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        """Prepare the encoder, ansatz and initial angles."""
        self.encoder = AmplitudeEncoder(self.num_qubits)
        self.ansatz = VariationalAnsatz(self.num_qubits, self.layers, self.entangler)
        if not self.parameters:
            import random

            rng = random.Random(self.seed)
            # Small initial angles keep the first forward pass near |0...0>,
            # which avoids the flat regions a random deep circuit starts in.
            self.parameters = [
                rng.uniform(-0.1, 0.1) for _ in range(self.ansatz.num_parameters)
            ]

    def circuit(self, values: list[float], parameters: list[float] | None = None) -> Circuit:
        """The full circuit: state preparation followed by the ansatz."""
        params = self.parameters if parameters is None else parameters
        full = self.encoder.encode(values)
        for op in self.ansatz.build(params).ops:
            full.ops.append(op)
        return full

    def forward(self, values: list[float], parameters: list[float] | None = None) -> list[float]:
        """Run the circuit and return one ⟨Z⟩ per qubit."""
        state = QuantumState(self.num_qubits)
        self.circuit(values, parameters).apply_to(state)
        return [state.expectation_z(q) for q in range(self.num_qubits)]

    def gradient(self, values: list[float], weights: list[float]) -> list[float]:
        """Gradient of ``sum(weights * <Z>)`` with respect to every parameter.

        Uses the parameter-shift rule, so the same call works against a
        simulator or real hardware.

        Args:
            values: The input to encode.
            weights: One weight per qubit, defining the scalar being
                differentiated.

        Returns:
            One gradient per ansatz parameter.
        """
        return parameter_shift_gradient(
            lambda params: sum(
                w * z for w, z in zip(weights, self.forward(values, params))
            ),
            self.parameters,
        )


def parameter_shift_gradient(evaluate, parameters: list[float]) -> list[float]:
    """Exact gradient of ``evaluate`` by the parameter-shift rule.

    For a gate ``exp(-i θ P / 2)`` the derivative of any expectation value is
    ``[f(θ + π/2) − f(θ − π/2)] / 2`` — an identity, not an approximation, so
    unlike a finite difference it has no step-size error and is well conditioned.

    Args:
        evaluate: Maps a parameter vector to a scalar expectation.
        parameters: The point to differentiate at.

    Returns:
        The gradient, one entry per parameter. Costs ``2 * len(parameters)``
        circuit evaluations, which is the honest price of hardware-compatible
        gradients.
    """
    gradient = []
    working = list(parameters)
    for index in range(len(working)):
        original = working[index]
        working[index] = original + SHIFT
        plus = evaluate(working)
        working[index] = original - SHIFT
        minus = evaluate(working)
        working[index] = original
        gradient.append((plus - minus) / 2.0)
    return gradient


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def circuit_stats(circuit: Circuit) -> dict:
    """Gate counts, two-qubit count and depth.

    Depth is what decides whether a circuit is runnable on real hardware, since
    coherence time is finite; two-qubit gate count is the usual proxy for error
    rate, being far noisier than single-qubit gates.
    """
    counts: dict[str, int] = {}
    two_qubit = 0
    frontier = [0] * circuit.num_qubits
    for op in circuit.ops:
        counts[op.name] = counts.get(op.name, 0) + 1
        touched = list(op.qubits)
        if len(touched) > 1:
            two_qubit += 1
        level = max(frontier[q] for q in touched) + 1
        for q in touched:
            frontier[q] = level
    return {
        "num_qubits": circuit.num_qubits,
        "gates": len(circuit.ops),
        "two_qubit_gates": two_qubit,
        "depth": max(frontier) if frontier else 0,
        "by_gate": dict(sorted(counts.items())),
    }


def draw(circuit: Circuit, max_columns: int = 60) -> str:
    """Render a circuit as a text diagram.

    Args:
        circuit: The circuit to draw.
        max_columns: Truncate after this many time steps.

    Returns:
        One line per qubit, most significant last.
    """
    lanes = [[] for _ in range(circuit.num_qubits)]
    frontier = [0] * circuit.num_qubits

    for op in circuit.ops:
        touched = list(op.qubits)
        level = max(frontier[q] for q in touched)
        for lane in lanes:
            while len(lane) < level:
                lane.append("---")
        if len(touched) == 1:
            lanes[touched[0]].append(f"-{op.name.upper()[:1]}-")
        else:
            control, target = touched[0], touched[1]
            lanes[control].append("-*-")
            lanes[target].append("-+-" if op.name == "cnot" else "-@-")
            low, high = min(control, target), max(control, target)
            for q in range(low + 1, high):
                while len(lanes[q]) <= level:
                    lanes[q].append("-|-")
        for q in touched:
            frontier[q] = level + 1
        span = max(len(lane) for lane in lanes)
        for lane in lanes:
            while len(lane) < span:
                lane.append("---")

    lines = []
    for q, lane in enumerate(lanes):
        body = "".join(lane[:max_columns])
        suffix = " ..." if len(lane) > max_columns else ""
        lines.append(f"q{q}: {body}{suffix}")
    return "\n".join(lines)
