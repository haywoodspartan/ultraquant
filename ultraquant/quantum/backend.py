"""Execution backends: the built-in statevector simulator and an IBM QPU bridge.

``SimulatorBackend`` is pure stdlib and always available. ``QPUBackend`` talks
to IBM Quantum through qiskit / qiskit-ibm-runtime, but every qiskit import is
guarded inside methods so the package works perfectly when qiskit is absent.
``auto_backend`` performs the "real QPU if reachable, simulator otherwise"
dispatch and never raises.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.state import QuantumState


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot execute (e.g. QPU unreachable or qiskit missing)."""


@dataclass
class BackendResult:
    """Result of one backend run.

    Attributes:
        expectations_z: ``<Z_q>`` per qubit (exact or shot-estimated), in circuit
            qubit order.
        counts: Sampled measurement counts (bitstring -> occurrences, qubit 0
            rightmost) when shots were taken, else ``None``.
    """

    expectations_z: list[float]
    counts: dict[str, int] | None


class Backend(abc.ABC):
    """Abstract execution backend for :class:`Circuit` objects."""

    name: str = "abstract"

    @abc.abstractmethod
    def run(self, circuit: Circuit, shots: int | None = None) -> BackendResult:
        """Execute ``circuit`` and return Z expectations (and counts if sampled).

        Args:
            circuit: The circuit to execute.
            shots: ``None`` for exact expectations (where supported), otherwise
                the number of measurement samples.
        """
        raise NotImplementedError


class SimulatorBackend(Backend):
    """Local dense-statevector backend (pure Python, always available)."""

    name: str = "ultraquant-statevector"

    def __init__(self, seed: int | None = None) -> None:
        """Create the simulator backend.

        Args:
            seed: Base seed. Each ``run`` seeds its state from this base plus an
                internal per-run counter, so repeated runs are deterministic as a
                sequence but do not reuse identical random draws.
        """
        self._base_seed = seed
        self._run_counter = 0

    def run(self, circuit: Circuit, shots: int | None = None) -> BackendResult:
        """Simulate ``circuit`` exactly, sampling measurements only if ``shots``.

        With ``shots=None`` the result carries the exact ``<Z_q>`` for every
        qubit and ``counts=None``. With ``shots`` given, ``measure_all(shots)``
        is sampled and the Z expectations are *estimated* from those counts.
        """
        self._run_counter += 1
        if self._base_seed is None:
            state_seed: int | None = None
        else:
            state_seed = self._base_seed + self._run_counter
        state = QuantumState(circuit.num_qubits, seed=state_seed)
        circuit.apply_to(state)

        if shots is None:
            exact = [state.expectation_z(q) for q in range(circuit.num_qubits)]
            return BackendResult(expectations_z=exact, counts=None)

        counts = state.measure_all(shots)
        total = sum(counts.values())
        estimates: list[float] = []
        for q in range(circuit.num_qubits):
            acc = 0
            for key, c in counts.items():
                acc += c if key[-1 - q] == "0" else -c
            estimates.append(acc / total)
        return BackendResult(expectations_z=estimates, counts=counts)


class QPUBackend(Backend):
    """IBM Quantum hardware backend via the Runtime Estimator primitive.

    All qiskit imports live inside methods and are guarded; construction is
    always safe even without qiskit installed.
    """

    name: str = "ibm-qpu"

    def __init__(self, backend_name: str | None = None) -> None:
        """Create the QPU backend.

        Args:
            backend_name: Specific IBM backend to target, or ``None`` to pick
                the least busy operational real device at run time.
        """
        self.backend_name = backend_name

    @classmethod
    def available(cls) -> bool:
        """True only if qiskit + qiskit-ibm-runtime import cleanly *and* a saved
        IBM account exists. Never raises: every failure path returns False.
        """
        try:
            import qiskit  # noqa: F401
            from qiskit_ibm_runtime import QiskitRuntimeService

            return bool(QiskitRuntimeService.saved_accounts())
        except Exception:
            return False

    def run(self, circuit: Circuit, shots: int | None = None) -> BackendResult:
        """Estimate ``<Z_q>`` for every qubit on IBM hardware.

        Translates the circuit ops to a Qiskit circuit and evaluates one
        single-qubit Z observable per qubit with the Runtime Estimator
        (default 1024 shots when ``shots`` is ``None``). Raises
        :class:`BackendUnavailable` if anything at all fails.
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

            service = QiskitRuntimeService()
            if self.backend_name:
                device = service.backend(self.backend_name)
            else:
                device = service.least_busy(operational=True, simulator=False)

            qc = QuantumCircuit(circuit.num_qubits)
            for op in circuit.ops:
                self._translate_op(qc, op)

            n = circuit.num_qubits
            # Qiskit Pauli strings are little-endian with qubit 0 rightmost,
            # matching our convention: qubit q gets Z at string position n-1-q.
            observables = [
                SparsePauliOp("I" * (n - 1 - q) + "Z" + "I" * q) for q in range(n)
            ]

            pass_manager = generate_preset_pass_manager(
                backend=device, optimization_level=1
            )
            isa_circuit = pass_manager.run(qc)
            isa_observables = [
                obs.apply_layout(isa_circuit.layout) for obs in observables
            ]

            estimator = EstimatorV2(mode=device)
            estimator.options.default_shots = 1024 if shots is None else int(shots)
            job = estimator.run([(isa_circuit, isa_observables)])
            evs = job.result()[0].data.evs
            return BackendResult(
                expectations_z=[float(v) for v in evs], counts=None
            )
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"QPU run failed: {exc}") from exc

    @staticmethod
    def _translate_op(qc: object, op: object) -> None:
        """Append one UltraQuant :class:`~ultraquant.quantum.circuit.Op` to a
        Qiskit ``QuantumCircuit`` (typed loosely to keep qiskit out of the
        module namespace)."""
        name = op.name  # type: ignore[attr-defined]
        qubits = op.qubits  # type: ignore[attr-defined]
        params = op.params  # type: ignore[attr-defined]
        if name in ("h", "x", "y", "z", "s", "t"):
            getattr(qc, name)(qubits[0])
        elif name in ("rx", "ry", "rz"):
            getattr(qc, name)(params[0], qubits[0])
        elif name == "cnot":
            qc.cx(qubits[0], qubits[1])  # type: ignore[attr-defined]
        elif name == "cz":
            qc.cz(qubits[0], qubits[1])  # type: ignore[attr-defined]
        elif name == "swap":
            qc.swap(qubits[0], qubits[1])  # type: ignore[attr-defined]
        else:
            raise ValueError(f"unknown op: {name!r}")


def auto_backend(prefer_qpu: bool = True, seed: int | None = None) -> Backend:
    """Return a ready backend: the QPU if preferred and available, else the simulator.

    This is the "uses qubits on a QPU when reachable, simulator otherwise"
    dispatch point. It must never raise.
    """
    if prefer_qpu:
        try:
            if QPUBackend.available():
                return QPUBackend()
        except Exception:
            pass
    return SimulatorBackend(seed)
