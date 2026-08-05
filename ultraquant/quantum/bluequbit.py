"""BlueQubit cloud backend: managed CPU/GPU simulators and QPU access.

BlueQubit (https://www.bluequbit.io) runs quantum circuits on hosted hardware —
large GPU-cluster statevector and MPS simulators, and real quantum devices —
which is exactly the tier this project cannot provide locally. Our own CUDA
simulator tops out at whatever fits in one 24 GB card; theirs does not.

This is an *optional* tier, in the same sense as :class:`QPUBackend`: it needs
the ``bluequbit`` SDK and an API token, and its absence is normal. Every other
tier remains available and computes the same values, so nothing here is load
bearing — it is a way to reach circuits too wide for this machine.

Devices, per the SDK:

======================  ==================================================
``cpu``                 hosted CPU statevector simulator
``gpu``                 GPU-cluster statevector simulator
``quantum``             real quantum hardware
``mps.cpu`` ``mps.gpu`` matrix-product-state simulators for wide, low-entanglement
                        circuits, where a full statevector would not fit at all
======================  ==================================================

Bit ordering is assumed, not documented: the platform consumes Qiskit circuits,
so its counts keys are taken to follow Qiskit's little-endian convention (qubit 0
rightmost), the same one :class:`QPUBackend` relies on. Because a mirrored
ordering would corrupt every expectation value while still looking plausible,
:meth:`BlueQubitBackend.verify_bit_order` settles the question against the live
service — it is opt-in, since it costs a billable job.

**Not verified against the live service.** The request shapes here follow the
published SDK API, and the tests drive them through a stub SDK. Running against
the real platform needs an account, a token, and — for the GPU and QPU devices —
credit. Treat this as written-to-documentation until you have run it yourself.

The API token is read from the ``BLUEQUBIT_API_TOKEN`` environment variable by
default, is never logged, and never appears in :meth:`describe`.
"""

from __future__ import annotations

import os
from typing import Any

from ultraquant.quantum.backend import Backend, BackendResult, BackendUnavailable
from ultraquant.quantum.circuit import Circuit

__all__ = ["BlueQubitBackend", "BLUEQUBIT_DEVICES", "MAX_STATEVECTOR_QUBITS"]

#: The platform returns at most this many outcomes from ``get_counts()``.
_COUNTS_LIMIT = 131_072      # 2**17, per the SDK documentation

#: Default ceiling for asking the service for a full statevector. 2**22
#: complex128 amplitudes is 64 MB to download; the simulators go to 40 qubits,
#: where the same request would be 17 TB.
MAX_STATEVECTOR_QUBITS = 22

#: Devices the platform exposes, with a one-line description.
BLUEQUBIT_DEVICES: dict[str, str] = {
    "cpu": "hosted CPU statevector simulator",
    "gpu": "GPU-cluster statevector simulator",
    "quantum": "real quantum hardware",
    "mps.cpu": "matrix-product-state simulator (CPU)",
    "mps.gpu": "matrix-product-state simulator (GPU)",
}


class BlueQubitBackend(Backend):
    """Run circuits on BlueQubit's hosted simulators or quantum hardware."""

    name = "bluequbit"

    def __init__(
        self,
        device: str = "cpu",
        api_token: str | None = None,
        job_name: str = "ultraquant",
        timeout: float | None = None,
        options: dict | None = None,
        bit_order: str = "auto",
        max_statevector_qubits: int = MAX_STATEVECTOR_QUBITS,
    ) -> None:
        """Connect to BlueQubit.

        Args:
            device: One of :data:`BLUEQUBIT_DEVICES`.
            api_token: Token; defaults to ``$BLUEQUBIT_API_TOKEN``.
            job_name: Label attached to submitted jobs.
            timeout: Per-job timeout in seconds.
            options: Device options passed through, e.g.
                ``{"mps_bond_dimension": 2}``.
            bit_order: ``"little"`` (qubit 0 rightmost) or ``"big"``. Defaults
                to little-endian, which is Qiskit's convention and therefore the
                likely one for a platform that consumes Qiskit circuits — but the
                platform does not document it, so :meth:`verify_bit_order` exists
                to settle it against the real service. That check is *not* run
                automatically: it costs a billable job.
            max_statevector_qubits: Above this width, never ask for a
                statevector. The simulators reach 40 qubits, where a returned
                statevector would be 17 TB; the default keeps any download to
                tens of megabytes and samples beyond it.

        Raises:
            BackendUnavailable: If the SDK or a token is missing.
            ValueError: If ``device`` or ``bit_order`` is not recognised.
        """
        if device not in BLUEQUBIT_DEVICES:
            raise ValueError(
                f"unknown BlueQubit device {device!r}; "
                f"expected one of {sorted(BLUEQUBIT_DEVICES)}"
            )
        if bit_order not in ("auto", "little", "big"):
            raise ValueError(
                f"bit_order must be 'auto', 'little' or 'big', got {bit_order!r}"
            )
        self.device = device
        self.job_name = job_name
        self.timeout = timeout
        self.options = dict(options or {})
        self.bit_order = bit_order
        self.max_statevector_qubits = int(max_statevector_qubits)
        self.name = f"bluequbit-{device}"
        #: Set when a reply looked truncated or renormalised; see run().
        self.last_warnings: list[str] = []

        token = api_token or os.environ.get("BLUEQUBIT_API_TOKEN")
        if not token:
            raise BackendUnavailable(
                "BlueQubit needs an API token: set BLUEQUBIT_API_TOKEN, or pass "
                "api_token. Get one by registering at https://app.bluequbit.io"
            )
        self._token = token

        try:
            import bluequbit  # noqa: PLC0415 - optional dependency, probed on purpose
        except ImportError as exc:
            raise BackendUnavailable(
                "the 'bluequbit' package is not installed (pip install bluequbit)"
            ) from exc

        # The Quick Start uses init(); the API reference documents BQClient.
        # Accept whichever this SDK version actually exposes.
        try:
            if hasattr(bluequbit, "init"):
                self._client = bluequbit.init(token)
            else:
                self._client = bluequbit.BQClient(api_token=token)
        except Exception as exc:  # noqa: BLE001 - SDK raises its own hierarchy
            raise BackendUnavailable(f"BlueQubit client init failed: {exc}") from exc

    # -- availability ------------------------------------------------------

    @classmethod
    def available(cls, api_token: str | None = None) -> bool:
        """Whether the SDK is installed and a token is present. Never raises."""
        if not (api_token or os.environ.get("BLUEQUBIT_API_TOKEN")):
            return False
        try:
            import bluequbit  # noqa: F401, PLC0415
        except Exception:  # noqa: BLE001
            return False
        return True

    # -- translation -------------------------------------------------------

    @staticmethod
    def to_qiskit(circuit: Circuit):
        """Translate an UltraQuant circuit into a Qiskit circuit.

        Raises:
            BackendUnavailable: If Qiskit is not installed.
            ValueError: On an op this translation does not cover.
        """
        try:
            from qiskit import QuantumCircuit  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                "BlueQubit submits Qiskit circuits; install qiskit"
            ) from exc

        qc = QuantumCircuit(circuit.num_qubits)
        for op in circuit.ops:
            name, qubits, params = op.name, op.qubits, op.params
            if name == "h":
                qc.h(qubits[0])
            elif name == "x":
                qc.x(qubits[0])
            elif name == "y":
                qc.y(qubits[0])
            elif name == "z":
                qc.z(qubits[0])
            elif name == "s":
                qc.s(qubits[0])
            elif name == "t":
                qc.t(qubits[0])
            elif name == "rx":
                qc.rx(params[0], qubits[0])
            elif name == "ry":
                qc.ry(params[0], qubits[0])
            elif name == "rz":
                qc.rz(params[0], qubits[0])
            elif name == "cnot":
                qc.cx(qubits[0], qubits[1])
            elif name == "cz":
                qc.cz(qubits[0], qubits[1])
            elif name == "swap":
                qc.swap(qubits[0], qubits[1])
            else:
                raise ValueError(f"cannot translate op {name!r} for BlueQubit")
        return qc

    # -- expectation values ------------------------------------------------

    @staticmethod
    def _expectations_from_counts(
        counts: dict, num_qubits: int, bit_order: str = "little"
    ) -> list[float]:
        """Estimate per-qubit <Z> from measurement counts or probabilities.

        ``get_counts()`` returns integer counts when shots were requested and
        probabilities otherwise, so the total is normalised rather than assumed
        to be the shot count. Both cases work out the same way.

        Args:
            counts: Bitstring -> count or probability.
            num_qubits: Width of the circuit.
            bit_order: ``"little"`` puts qubit 0 at the rightmost character.
        """
        totals = [0.0] * num_qubits
        total_weight = float(sum(counts.values())) or 1.0
        for bitstring, weight in counts.items():
            clean = bitstring.replace(" ", "")
            for q in range(num_qubits):
                if bit_order == "little":
                    bit = clean[-1 - q] if q < len(clean) else "0"
                else:
                    bit = clean[q] if q < len(clean) else "0"
                totals[q] += float(weight) * (1.0 if bit == "0" else -1.0)
        return [value / total_weight for value in totals]

    def verify_bit_order(self) -> str:
        """Settle empirically which end of a counts key is qubit 0.

        Getting this backwards would silently mirror every expectation value —
        an error that yields plausible numbers rather than an exception, so it is
        worth one deliberate check against the real service. A circuit with an
        asymmetric outcome is run (X on qubit 0 of a two-qubit register, which
        must read ``01`` little-endian and ``10`` big-endian) and the result is
        remembered on this instance.

        **This submits a billable job.** It is never called automatically; run it
        once when you first point at the service, then pass the answer as
        ``bit_order`` thereafter.

        Returns:
            ``"little"`` or ``"big"``.

        Raises:
            BackendUnavailable: If the probe cannot be run or is ambiguous.
        """
        result = self._submit(Circuit(2).x(0), shots=64, allow_statevector=False)
        counts = result.get("counts") or {}
        winner = max(counts, key=lambda k: counts[k], default="")
        clean = winner.replace(" ", "")
        if clean == "01":
            self.bit_order = "little"
        elif clean == "10":
            self.bit_order = "big"
        else:
            raise BackendUnavailable(
                f"could not determine BlueQubit bit ordering from probe counts "
                f"{counts!r}; pass bit_order='little' or 'big' explicitly"
            )
        return self.bit_order

    @staticmethod
    def _expectations_from_statevector(amplitudes, num_qubits: int) -> list[float]:
        """Exact per-qubit <Z> from a statevector.

        Iterates the amplitudes without importing numpy: the SDK returns an
        ndarray, but indexing and ``abs`` are enough, and this package does not
        take a numpy dependency of its own.
        """
        totals = [0.0] * num_qubits
        for index in range(len(amplitudes)):
            probability = abs(complex(amplitudes[index])) ** 2
            if not probability:
                continue
            for q in range(num_qubits):
                totals[q] += probability * (1.0 if not (index >> q) & 1 else -1.0)
        return totals

    # -- execution ---------------------------------------------------------

    def _submit(
        self, circuit: Circuit, shots: int | None, allow_statevector: bool
    ) -> dict:
        """Submit one job and return ``{"counts": ..., "statevector": ...}``.

        Raises:
            BackendUnavailable: If the job fails or returns nothing usable.
        """
        qc = self.to_qiskit(circuit)
        kwargs: dict[str, Any] = {"device": self.device, "job_name": self.job_name}
        if shots is not None:
            kwargs["shots"] = int(shots)
        if self.options:
            kwargs["options"] = dict(self.options)
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        try:
            result = self._client.run(qc, **kwargs)
        except Exception as exc:  # noqa: BLE001 - SDK raises its own hierarchy
            raise BackendUnavailable(f"BlueQubit job failed: {exc}") from exc

        error = getattr(result, "error_message", None)
        if error:
            raise BackendUnavailable(f"BlueQubit job failed: {error}")
        status = getattr(result, "run_status", None)
        if status is not None and str(status).upper() not in ("COMPLETED", "", "NONE"):
            raise BackendUnavailable(f"BlueQubit job did not complete: {status}")

        payload: dict = {"counts": None, "statevector": None}
        if allow_statevector:
            try:
                amplitudes = result.get_statevector()
            except Exception:  # noqa: BLE001 - not every device returns one
                amplitudes = None
            if amplitudes is not None and len(amplitudes):
                payload["statevector"] = amplitudes
                return payload

        try:
            counts = result.get_counts()
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"BlueQubit returned no counts: {exc}") from exc
        if not counts:
            raise BackendUnavailable("BlueQubit returned empty counts")
        payload["counts"] = counts
        return payload

    def run(self, circuit: Circuit, shots: int | None = None) -> BackendResult:
        """Execute ``circuit`` on the configured BlueQubit device.

        Args:
            circuit: The circuit to run.
            shots: Measurement count. ``None`` asks for exact expectations from
                the statevector — but only while the circuit is narrow enough
                for that to be sane, and never on real hardware, which cannot
                produce one. Wider circuits fall back to sampling.

        Returns:
            Per-qubit ``<Z>`` and, when sampled, the counts.

        Raises:
            BackendUnavailable: If the job fails or returns nothing usable.
        """
        self.last_warnings = []

        # A 40-qubit statevector is 17 TB. Asking for one because the caller
        # said "exact" would be a denial of service against our own process,
        # so exactness is bounded by width and degrades to sampling.
        want_statevector = (
            shots is None
            and self.device != "quantum"
            and circuit.num_qubits <= self.max_statevector_qubits
        )
        if shots is None and not want_statevector:
            shots = 1024
            if self.device != "quantum":
                self.last_warnings.append(
                    f"{circuit.num_qubits} qubits exceeds the "
                    f"{self.max_statevector_qubits}-qubit statevector limit; "
                    f"sampled {shots} shots instead of computing exactly"
                )

        payload = self._submit(circuit, shots, allow_statevector=want_statevector)

        if payload["statevector"] is not None:
            return BackendResult(
                expectations_z=self._expectations_from_statevector(
                    payload["statevector"], circuit.num_qubits
                ),
                counts=None,
            )

        counts = payload["counts"]
        order = "little" if self.bit_order == "auto" else self.bit_order

        # get_counts() returns only the top 2**17 outcomes. For a wide circuit
        # the rest of the distribution is simply absent, so the expectation is
        # computed over the mass that came back — say so rather than present it
        # as exact.
        total = float(sum(counts.values()))
        if len(counts) >= _COUNTS_LIMIT:
            self.last_warnings.append(
                f"BlueQubit returns at most {_COUNTS_LIMIT:,} outcomes; the "
                f"distribution was truncated and <Z> is estimated over the "
                f"{total:.4g} of probability mass returned"
            )

        integral = all(float(v).is_integer() for v in counts.values())
        return BackendResult(
            expectations_z=self._expectations_from_counts(
                counts, circuit.num_qubits, order
            ),
            counts={str(k): int(v) for k, v in counts.items()} if integral else None,
        )

    # -- introspection -----------------------------------------------------

    def estimate(self, circuit: Circuit) -> dict:
        """Ask the platform what a circuit would cost and how long it would take.

        Returns:
            ``{"estimated_runtime", "estimated_cost"}``, or ``{}`` if the SDK
            does not support estimation.
        """
        try:
            result = self._client.estimate(self.to_qiskit(circuit), device=self.device)
        except Exception:  # noqa: BLE001 - estimation is a convenience
            return {}
        return {
            "estimated_runtime": getattr(result, "estimated_runtime", None),
            "estimated_cost": getattr(result, "estimated_cost", None),
        }

    def describe(self) -> dict:
        """A JSON-safe summary. Never includes the API token."""
        return {
            "backend": self.name,
            "device": self.device,
            "description": BLUEQUBIT_DEVICES[self.device],
            "job_name": self.job_name,
            "options": dict(self.options),
            "bit_order": self.bit_order,
            "max_statevector_qubits": self.max_statevector_qubits,
            "authenticated": bool(self._token),
            "warnings": list(self.last_warnings),
        }
