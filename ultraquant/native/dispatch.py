"""Backend tier selection: pick the fastest available execution backend.

SPEC-NATIVE.md section 6. :func:`best_backend` walks the tier order

    QPU (only when ``prefer_qpu`` and reachable) -> CUDA -> native CPU ->
    pure-Python ``SimulatorBackend``

and returns the first backend that constructs; the pure-Python simulator is
always available, so the function never raises (and never logs).
:func:`tier_report` summarizes what each tier looks like from this machine.
"""

from __future__ import annotations

from ultraquant.native import accel
from ultraquant.native.backends import CudaSimulatorBackend, NativeSimulatorBackend
from ultraquant.quantum.backend import Backend, QPUBackend, SimulatorBackend


#: Selectable quantum execution tiers, in the order they are preferred.
#: Every one of these computes the same expectation values; they differ in speed,
#: and the QPU differs additionally in being noisy (see ARCHITECTURE.md section 7).
QUANTUM_TIERS: list[tuple[str, str]] = [
    ("auto", "Auto (fastest available)"),
    ("qpu", "QPU - real quantum hardware (IBM)"),
    ("bluequbit.gpu", "BlueQubit - GPU cluster simulator (cloud)"),
    ("bluequbit.cpu", "BlueQubit - CPU simulator (cloud)"),
    ("bluequbit.quantum", "BlueQubit - quantum hardware (cloud)"),
    ("bluequbit.mps.gpu", "BlueQubit - MPS simulator, wide circuits (cloud)"),
    ("cuda", "CUDA statevector simulator"),
    ("cpp", "C++ statevector simulator"),
    ("python", "Pure-Python simulator (reference)"),
]


def backend_by_name(name: str, seed: int | None = None) -> Backend:
    """Return a specific quantum tier by name, falling back when it is absent.

    Args:
        name: One of the keys in :data:`QUANTUM_TIERS`.
        seed: Seed for whichever simulator ends up being used.

    Returns:
        The requested backend, or the best available one if the request cannot
        be honoured. Never raises: an unreachable QPU or a missing DLL degrades
        rather than failing, because every tier computes the same values.

    Raises:
        ValueError: If ``name`` is not a known tier.
    """
    known = {key for key, _label in QUANTUM_TIERS}
    if name not in known:
        raise ValueError(f"unknown quantum tier {name!r}; expected one of {sorted(known)}")

    if name.startswith("bluequbit."):
        from ultraquant.quantum.bluequbit import BlueQubitBackend

        try:
            return BlueQubitBackend(device=name.split(".", 1)[1])
        except Exception:  # noqa: BLE001 - no token or no SDK is normal
            return best_backend(prefer_qpu=False, seed=seed)

    if name == "auto":
        return best_backend(prefer_qpu=True, seed=seed)
    if name == "qpu":
        try:
            if QPUBackend.available():
                return QPUBackend()
        except Exception:  # noqa: BLE001 - unreachable hardware is normal
            pass
        return best_backend(prefer_qpu=False, seed=seed)
    if name == "cuda":
        try:
            return CudaSimulatorBackend(seed=seed)
        except Exception:  # noqa: BLE001 - tier absent
            return best_backend(prefer_qpu=False, seed=seed)
    if name == "cpp":
        try:
            return NativeSimulatorBackend(seed=seed)
        except Exception:  # noqa: BLE001 - tier absent
            return SimulatorBackend(seed=seed)
    return SimulatorBackend(seed=seed)


def best_backend(prefer_qpu: bool = False, seed: int | None = None) -> Backend:
    """Return the best available backend for this machine.

    Tier order: QPU (only if ``prefer_qpu`` and ``QPUBackend.available()``),
    then CUDA, then native CPU, then the pure-Python simulator. Never raises;
    logs nothing; returns the backend instance.

    Args:
        prefer_qpu: Consider real IBM hardware first when it is reachable.
        seed: Seed passed to whichever simulator backend is chosen (base seed
            for measurement sampling; ignored by the QPU tier).

    Returns:
        A ready-to-use :class:`Backend` instance.
    """
    if prefer_qpu:
        try:
            if QPUBackend.available():
                return QPUBackend()
        except Exception:
            pass
    try:
        return CudaSimulatorBackend(seed=seed)
    except Exception:
        pass
    try:
        return NativeSimulatorBackend(seed=seed)
    except Exception:
        pass
    return SimulatorBackend(seed)


def cloud_report() -> dict:
    """Whether the BlueQubit cloud tier is reachable. Never raises."""
    try:
        from ultraquant.quantum.bluequbit import BlueQubitBackend

        return {"bluequbit": BlueQubitBackend.available()}
    except Exception:  # noqa: BLE001
        return {"bluequbit": False}


def tier_report() -> dict:
    """Report the availability of every execution tier.

    Returns:
        ``{"qpu": bool, "cuda": bool | str, "native_cpu": bool, "python": True}``
        where ``"cuda"`` is the CUDA device 0 name when the GPU tier is up
        (``True`` if the tier works but the name lookup fails, ``False`` when
        the tier is absent). Never raises.
    """
    try:
        qpu = bool(QPUBackend.available())
    except Exception:
        qpu = False

    cuda: bool | str = False
    if accel.gpu_available():
        name = accel.gpu_device_name()
        cuda = name if name else True

    return {
        "qpu": qpu,
        "cuda": cuda,
        "native_cpu": accel.load_cpu() is not None,
        "python": True,
    }
