"""Native execution backends: C++ (CPU) and CUDA (GPU) statevector simulators.

Both classes implement the exact :class:`~ultraquant.quantum.backend.Backend`
contract of the pure-Python :class:`~ultraquant.quantum.backend.SimulatorBackend`
(SPEC-NATIVE.md section 4):

- ``shots is None`` -> exact per-qubit ``<Z>`` expectations, ``counts=None``.
- ``shots`` given -> basis states are sampled in Python with a seeded
  ``random.Random`` from the native probabilities, expectations are *estimated*
  from those counts, and the counts are returned (bitstring keys, qubit 0
  rightmost -- identical to ``QuantumState.measure_all``).

Statevector evolution itself runs inside the DLLs via the typed wrappers in
:mod:`ultraquant.native.accel`; every value agrees with the pure-Python tier
to 1e-9. Constructors raise :class:`BackendUnavailable` when their tier's DLL
cannot be loaded, so callers (see :mod:`ultraquant.native.dispatch`) can fall
through the tier order without special-casing.
"""

from __future__ import annotations

import random
from array import array

from ultraquant.native import accel
from ultraquant.quantum.backend import Backend, BackendResult, BackendUnavailable
from ultraquant.quantum.circuit import Circuit


class _AcceleratedSimulatorBackend(Backend):
    """Shared run/sampling plumbing for the two native statevector backends.

    Subclasses provide the tier-specific ``_evolve`` / ``_expectations`` /
    ``_probabilities`` hooks; this base class reproduces the seeding and
    shot-estimation semantics of the pure-Python ``SimulatorBackend`` (a base
    seed plus an internal per-run counter, so repeated runs are deterministic
    as a sequence without reusing identical random draws).
    """

    def __init__(self, seed: int | None = None) -> None:
        """Store the base seed and reset the per-run counter.

        Args:
            seed: Base seed for measurement sampling. Each ``run`` derives its
                own RNG from this base plus an internal counter; ``None`` means
                nondeterministic sampling.
        """
        self._base_seed = seed
        self._run_counter = 0

    # -- tier hooks ----------------------------------------------------------

    def _evolve(self, circuit: Circuit) -> array:
        """Run ``circuit`` from |0...0>; return the final interleaved re/im state."""
        raise NotImplementedError

    def _expectations(self, amp: array, num_qubits: int) -> list[float]:
        """Exact per-qubit ``<Z>`` from a final statevector."""
        raise NotImplementedError

    def _probabilities(self, amp: array, num_qubits: int) -> list[float]:
        """Basis-state probabilities from a final statevector."""
        raise NotImplementedError

    # -- Backend contract ----------------------------------------------------

    def run(self, circuit: Circuit, shots: int | None = None) -> BackendResult:
        """Execute ``circuit`` natively; exact expectations or shot estimates.

        Args:
            circuit: The circuit to execute (encoded via the shared wire format).
            shots: ``None`` for exact ``<Z>`` per qubit (``counts=None``);
                otherwise the number of measurement samples to draw.

        Returns:
            A :class:`BackendResult` mirroring ``SimulatorBackend.run``.
        """
        self._run_counter += 1
        amp = self._evolve(circuit)
        num_qubits = circuit.num_qubits

        if shots is None:
            exact = self._expectations(amp, num_qubits)
            return BackendResult(expectations_z=exact, counts=None)

        if shots < 1:
            raise ValueError("shots must be >= 1")
        probs = self._probabilities(amp, num_qubits)
        if self._base_seed is None:
            rng_seed: int | None = None
        else:
            rng_seed = self._base_seed + self._run_counter
        rng = random.Random(rng_seed)
        indices = rng.choices(range(len(probs)), weights=probs, k=shots)
        counts: dict[str, int] = {}
        for i in indices:
            key = format(i, f"0{num_qubits}b")
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        estimates: list[float] = []
        for q in range(num_qubits):
            acc = 0
            for key, c in counts.items():
                acc += c if key[-1 - q] == "0" else -c
            estimates.append(acc / total)
        return BackendResult(expectations_z=estimates, counts=counts)


class NativeSimulatorBackend(_AcceleratedSimulatorBackend):
    """Dense-statevector backend running in the C++ CPU DLL (ultraquant_native.dll)."""

    name: str = "ultraquant-native-cpu"

    def __init__(self, seed: int | None = None) -> None:
        """Create the backend; requires the native CPU DLL.

        Args:
            seed: Base seed for measurement sampling (see base class).

        Raises:
            BackendUnavailable: If ``ultraquant_native.dll`` cannot be loaded.
        """
        if accel.load_cpu() is None:
            raise BackendUnavailable(
                "native CPU tier unavailable: cannot load "
                f"{accel.CPU_DLL_PATH!r} (build with native\\build.ps1 -Target cpu)"
            )
        super().__init__(seed)

    def _evolve(self, circuit: Circuit) -> array:
        return accel.run_circuit_cpu(circuit)

    def _expectations(self, amp: array, num_qubits: int) -> list[float]:
        return accel.expectations_z_cpu(amp, num_qubits)

    def _probabilities(self, amp: array, num_qubits: int) -> list[float]:
        return accel.probabilities_cpu(amp, num_qubits)


class CudaSimulatorBackend(_AcceleratedSimulatorBackend):
    """Dense-statevector backend running on the GPU (ultraquant_cuda.dll).

    Evolution uses ``uqg_run_circuit`` (single large statevector in global
    memory, one kernel launch per gate); expectations use
    ``uqg_expectations_z_from``; probabilities are computed host-side from the
    copied-back amplitudes (the GPU tier has no probabilities export).
    """

    name: str = "ultraquant-cuda"

    def __init__(self, seed: int | None = None) -> None:
        """Create the backend; requires the CUDA DLL and a usable device.

        Args:
            seed: Base seed for measurement sampling (see base class).

        Raises:
            BackendUnavailable: If ``ultraquant_cuda.dll`` cannot be loaded or
                reports no CUDA device.
        """
        if accel.load_gpu() is None:
            raise BackendUnavailable(
                "CUDA tier unavailable: cannot load "
                f"{accel.GPU_DLL_PATH!r} or no CUDA device "
                "(build with native\\build.ps1 -Target cuda)"
            )
        super().__init__(seed)

    def _evolve(self, circuit: Circuit) -> array:
        return accel.run_circuit_gpu(circuit)

    def _expectations(self, amp: array, num_qubits: int) -> list[float]:
        return accel.expectations_z_gpu(amp, num_qubits)

    def _probabilities(self, amp: array, num_qubits: int) -> list[float]:
        return accel.probabilities_from_amp(amp)
