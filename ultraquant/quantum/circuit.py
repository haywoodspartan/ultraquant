"""Gate-list circuit representation and replay onto a :class:`QuantumState`.

A :class:`Circuit` is an ordered list of :class:`Op` records built with
chainable helper methods, decoupled from any particular execution backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from ultraquant.quantum import gates
from ultraquant.quantum.gates import Matrix
from ultraquant.quantum.state import QuantumState


@dataclass(frozen=True)
class Op:
    """One circuit operation: gate ``name``, ``qubits`` acted on, float ``params``."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


_FIXED_GATES: dict[str, Matrix] = {
    "h": gates.H,
    "x": gates.X,
    "y": gates.Y,
    "z": gates.Z,
    "s": gates.S,
    "t": gates.T,
}

_PARAM_GATES = {
    "rx": gates.rx,
    "ry": gates.ry,
    "rz": gates.rz,
}


class Circuit:
    """An ordered sequence of gate operations on ``num_qubits`` qubits."""

    def __init__(self, num_qubits: int) -> None:
        """Create an empty circuit on ``num_qubits`` qubits."""
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits: int = num_qubits
        self.ops: list[Op] = []

    # -- single-qubit fixed gates -------------------------------------------

    def h(self, q: int) -> "Circuit":
        """Append a Hadamard on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("h", (q,))

    def x(self, q: int) -> "Circuit":
        """Append a Pauli-X on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("x", (q,))

    def y(self, q: int) -> "Circuit":
        """Append a Pauli-Y on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("y", (q,))

    def z(self, q: int) -> "Circuit":
        """Append a Pauli-Z on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("z", (q,))

    def s(self, q: int) -> "Circuit":
        """Append an S (phase) gate on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("s", (q,))

    def t(self, q: int) -> "Circuit":
        """Append a T gate on qubit ``q``. Returns ``self`` for chaining."""
        return self._append("t", (q,))

    # -- single-qubit rotations ---------------------------------------------

    def rx(self, q: int, theta: float) -> "Circuit":
        """Append an X rotation by ``theta`` on qubit ``q``. Returns ``self``."""
        return self._append("rx", (q,), (float(theta),))

    def ry(self, q: int, theta: float) -> "Circuit":
        """Append a Y rotation by ``theta`` on qubit ``q``. Returns ``self``."""
        return self._append("ry", (q,), (float(theta),))

    def rz(self, q: int, theta: float) -> "Circuit":
        """Append a Z rotation by ``theta`` on qubit ``q``. Returns ``self``."""
        return self._append("rz", (q,), (float(theta),))

    # -- two-qubit gates ----------------------------------------------------

    def cnot(self, control: int, target: int) -> "Circuit":
        """Append a CNOT with the given ``control`` and ``target``. Returns ``self``."""
        return self._append("cnot", (control, target))

    def cz(self, control: int, target: int) -> "Circuit":
        """Append a CZ with the given ``control`` and ``target``. Returns ``self``."""
        return self._append("cz", (control, target))

    def swap(self, a: int, b: int) -> "Circuit":
        """Append a SWAP of qubits ``a`` and ``b`` (dedicated op). Returns ``self``."""
        return self._append("swap", (a, b))

    # -- transformations ----------------------------------------------------

    def inverse(self) -> "Circuit":
        """Return the adjoint circuit, undoing this one.

        Every gate here is unitary, so the inverse is the reversed sequence of
        per-gate adjoints: rotations negate their angle, S and T become their
        daggers, and the self-inverse gates are unchanged. Running a circuit
        followed by its inverse must return the register to where it started —
        which is exactly what the fidelity kernel in
        :mod:`ultraquant.quantum.kernel` measures.

        Raises:
            ValueError: On an op with no known adjoint.
        """
        out = Circuit(self.num_qubits)
        for op in reversed(self.ops):
            name, qubits, params = op.name, op.qubits, op.params
            if name in ("h", "x", "y", "z", "cnot", "cz", "swap"):
                out._append(name, qubits, params)
            elif name in ("rx", "ry", "rz"):
                out._append(name, qubits, (-params[0],))
            elif name == "s":
                # S dagger = Z · S, since S**2 = Z and S**4 = I.
                out._append("z", qubits)
                out._append("s", qubits)
            elif name == "t":
                # T dagger = S dagger · T, as T**2 = S and T**8 = I.
                out._append("z", qubits)
                out._append("s", qubits)
                out._append("t", qubits)
            else:  # pragma: no cover - defensive
                raise ValueError(f"no adjoint defined for op {name!r}")
        return out

    def compose(self, other: "Circuit") -> "Circuit":
        """Return this circuit followed by ``other``.

        Raises:
            ValueError: If the registers differ in width.
        """
        if other.num_qubits != self.num_qubits:
            raise ValueError(
                f"cannot compose a {other.num_qubits}-qubit circuit onto "
                f"{self.num_qubits} qubits"
            )
        out = Circuit(self.num_qubits)
        out.ops.extend(self.ops)
        out.ops.extend(other.ops)
        return out

    # -- execution ----------------------------------------------------------

    def apply_to(self, state: QuantumState) -> None:
        """Replay every op of this circuit onto ``state`` in order."""
        for op in self.ops:
            if op.name in _FIXED_GATES:
                state.apply_gate(_FIXED_GATES[op.name], op.qubits[0])
            elif op.name in _PARAM_GATES:
                state.apply_gate(_PARAM_GATES[op.name](op.params[0]), op.qubits[0])
            elif op.name == "cnot":
                state.apply_controlled(gates.X, op.qubits[0], op.qubits[1])
            elif op.name == "cz":
                state.apply_controlled(gates.Z, op.qubits[0], op.qubits[1])
            elif op.name == "swap":
                a, b = op.qubits
                state.apply_controlled(gates.X, a, b)
                state.apply_controlled(gates.X, b, a)
                state.apply_controlled(gates.X, a, b)
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown op: {op.name!r}")

    # -- internals ----------------------------------------------------------

    def _append(
        self, name: str, qubits: tuple[int, ...], params: tuple[float, ...] = ()
    ) -> "Circuit":
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(
                    f"qubit {q} out of range for {self.num_qubits}-qubit circuit"
                )
        self.ops.append(Op(name, qubits, params))
        return self
