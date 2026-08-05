"""Amplitude amplification: Grover search over the shard catalog.

The library is an unstructured collection of shards, and "which shard matches
this pattern?" is unstructured search — the one problem where Grover's algorithm
gives a proven speedup: ``O(sqrt(N))`` oracle queries against a classical scan's
``O(N)``. With a million shards that is a thousand queries instead of a million.

Two kinds of oracle are provided, and the difference between them is the whole
honesty of the subject:

* :func:`pattern_oracle` marks every index matching a bit pattern under a mask —
  "category field equals 6", say. It is built from one multi-controlled phase
  flip, costs a handful of gates, and is the case that matters for a catalog,
  because catalog keys *have* structure.
* :func:`indices_oracle` marks an arbitrary set of indices, synthesised through
  the same Walsh–Hadamard phase decomposition the amplitude encoder uses. It is
  fully general and costs up to ``2**n`` terms — which means that if you have to
  enumerate the marked set to build the oracle, you have already done the linear
  scan Grover was supposed to save you. It is here for testing and to make that
  trap explicit.

Grover speeds up querying an oracle you already possess. It does not conjure one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.state import QuantumState

__all__ = [
    "phase_operator",
    "indices_oracle",
    "pattern_oracle",
    "diffusion",
    "grover_circuit",
    "optimal_iterations",
    "search",
    "SearchResult",
]


def phase_operator(num_qubits: int, phases: Sequence[float]) -> Circuit:
    """Synthesise ``diag(exp(i * phases[x]))`` as a circuit.

    Any diagonal unitary expands over the Walsh–Hadamard basis: each non-empty
    qubit subset contributes a term realised by a CNOT ladder computing that
    subset's parity, an RZ, and the ladder undone. Zero coefficients are skipped,
    so a structured phase pattern stays cheap while an arbitrary one does not.

    Args:
        num_qubits: Register width.
        phases: ``2**num_qubits`` phase angles.

    Returns:
        The circuit implementing the diagonal operator.

    Raises:
        ValueError: If ``phases`` has the wrong length.
    """
    dimension = 1 << num_qubits
    if len(phases) != dimension:
        raise ValueError(f"expected {dimension} phases, got {len(phases)}")

    circuit = Circuit(num_qubits)
    for subset in range(1, dimension):
        coefficient = 0.0
        for x in range(dimension):
            parity = bin(subset & x).count("1") & 1
            coefficient += phases[x] * (-1 if parity else 1)
        coefficient /= dimension
        if abs(coefficient) < 1e-12:
            continue
        members = [q for q in range(num_qubits) if (subset >> q) & 1]
        target = members[-1]
        for q in members[:-1]:
            circuit.cnot(q, target)
        circuit.rz(target, -2.0 * coefficient)
        for q in reversed(members[:-1]):
            circuit.cnot(q, target)
    return circuit


def indices_oracle(num_qubits: int, marked: Sequence[int]) -> Circuit:
    """Phase-flip an arbitrary set of indices.

    General but expensive: synthesising it requires knowing the marked set, which
    is the very thing a search is for. Use :func:`pattern_oracle` for anything
    with structure.
    """
    dimension = 1 << num_qubits
    marked_set = set(marked)
    phases = [math.pi if x in marked_set else 0.0 for x in range(dimension)]
    return phase_operator(num_qubits, phases)


def pattern_oracle(num_qubits: int, pattern: int, mask: int | None = None) -> Circuit:
    """Phase-flip every index whose masked bits equal ``pattern``.

    This is the oracle a catalog actually needs — "the category field is 6" — and
    it costs one multi-controlled phase flip regardless of how many indices match.

    Args:
        num_qubits: Register width.
        pattern: The value the masked bits must take.
        mask: Which bits participate; defaults to all of them.

    Returns:
        The oracle circuit.
    """
    if mask is None:
        mask = (1 << num_qubits) - 1
    members = [q for q in range(num_qubits) if (mask >> q) & 1]
    circuit = Circuit(num_qubits)
    if not members:
        return circuit

    # Conjugate by X so the target pattern becomes all-ones, then apply a
    # multi-controlled Z, then undo.
    flips = [q for q in members if not (pattern >> q) & 1]
    for q in flips:
        circuit.x(q)
    _multi_controlled_z(circuit, members)
    for q in flips:
        circuit.x(q)
    return circuit


def _multi_controlled_z(circuit: Circuit, qubits: list[int]) -> None:
    """Phase-flip the all-ones state of ``qubits``."""
    if len(qubits) == 1:
        circuit.z(qubits[0])
        return
    if len(qubits) == 2:
        circuit.cz(qubits[0], qubits[1])
        return
    # A parity-ladder phase: flip when every listed qubit is 1. Built from the
    # Walsh-Hadamard identity for the projector onto |1...1>.
    n = len(qubits)
    phases = []
    total = 1 << n
    for x in range(total):
        phases.append(math.pi if x == total - 1 else 0.0)
    sub = phase_operator(n, phases)
    for op in sub.ops:
        circuit.ops.append(type(op)(op.name, tuple(qubits[q] for q in op.qubits), op.params))


def diffusion(num_qubits: int) -> Circuit:
    """The Grover diffusion operator: reflection about the uniform state."""
    circuit = Circuit(num_qubits)
    for q in range(num_qubits):
        circuit.h(q)
    # Phase-flip |0...0>, which is pattern_oracle with pattern 0.
    for op in pattern_oracle(num_qubits, 0).ops:
        circuit.ops.append(op)
    for q in range(num_qubits):
        circuit.h(q)
    return circuit


def optimal_iterations(num_qubits: int, num_marked: int) -> int:
    """How many Grover rounds maximise the success probability.

    ``floor((pi/4) * sqrt(N/M))`` — and note that *more* rounds than this makes
    things worse, because the amplitude rotates past the target and back.
    """
    if num_marked <= 0:
        return 0
    total = 1 << num_qubits
    if num_marked >= total:
        return 0
    return max(1, int(math.floor((math.pi / 4.0) * math.sqrt(total / num_marked))))


def grover_circuit(oracle: Circuit, iterations: int) -> Circuit:
    """Build the full Grover circuit: superposition, then oracle+diffusion rounds."""
    num_qubits = oracle.num_qubits
    circuit = Circuit(num_qubits)
    for q in range(num_qubits):
        circuit.h(q)
    spread = diffusion(num_qubits)
    for _round in range(iterations):
        circuit.ops.extend(oracle.ops)
        circuit.ops.extend(spread.ops)
    return circuit


@dataclass
class SearchResult:
    """Outcome of an amplitude-amplification search."""

    best: int
    probability: float
    iterations: int
    oracle_queries: int
    classical_queries: int
    distribution: dict[int, float]

    @property
    def speedup(self) -> float:
        """Classical queries per quantum query."""
        return self.classical_queries / max(self.oracle_queries, 1)


def search(
    num_qubits: int,
    oracle: Circuit,
    num_marked: int,
    iterations: int | None = None,
    top: int = 4,
) -> SearchResult:
    """Run Grover search and report where the amplitude ended up.

    Args:
        num_qubits: Register width; the search space is ``2**num_qubits``.
        oracle: A phase oracle marking the wanted indices.
        num_marked: How many indices the oracle marks — needed to pick the
            iteration count, since overshooting rotates the amplitude away again.
        iterations: Override the computed round count.
        top: How many of the most likely outcomes to report.

    Returns:
        A :class:`SearchResult`.
    """
    rounds = optimal_iterations(num_qubits, num_marked) if iterations is None else iterations
    state = QuantumState(num_qubits)
    grover_circuit(oracle, rounds).apply_to(state)

    probabilities = state.probabilities()
    ranked = sorted(range(len(probabilities)), key=lambda i: -probabilities[i])[:top]
    best = ranked[0] if ranked else 0
    total = 1 << num_qubits
    return SearchResult(
        best=best,
        probability=probabilities[best],
        iterations=rounds,
        oracle_queries=rounds,
        # A classical scan needs N/2 probes on average to find one of M targets.
        classical_queries=max(1, total // (2 * max(num_marked, 1))),
        distribution={i: probabilities[i] for i in ranked},
    )
