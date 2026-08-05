"""ctypes loaders and typed wrappers for the UltraQuant native DLLs.

This module is the FFI boundary of the native acceleration tier
(SPEC-NATIVE.md section 3). It exposes:

- ``load_cpu()`` / ``load_gpu()`` -- cached ``ctypes.CDLL`` handles for
  ``_bin/ultraquant_native.dll`` and ``_bin/ultraquant_cuda.dll`` with full
  ``argtypes``/``restype`` declarations on every export. Both return ``None``
  on any failure (missing file, load error, no CUDA device) and never raise.
- ``encode_circuit(circuit)`` -- lower a ``Circuit`` to the shared wire
  format: three parallel arrays ``int32 opcodes[N]``, ``int32 qubits[2N]``
  (unused slot = -1) and ``double params[N]`` (unused slot = 0.0).
- Thin typed wrappers over every DLL export, using ``array.array`` buffers
  (no numpy). Statevectors cross the FFI as interleaved re/im doubles
  (``array('d')`` of length ``2 * 2**num_qubits``), little-endian qubit
  indexing exactly as SPEC.md section 2 (qubit q is bit q of the basis index).

Importing this module never imports the rest of the ultraquant package and
never raises because a DLL is missing; the wrappers raise ``RuntimeError``
only when called while their tier is unavailable.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from array import array
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - annotation-only; never at runtime
    from ultraquant.quantum.circuit import Circuit

__all__ = [
    "OPCODES",
    "CPU_DLL_PATH",
    "GPU_DLL_PATH",
    "load_cpu",
    "load_gpu",
    "encode_circuit",
    "version_cpu",
    "init_state_cpu",
    "run_ops_cpu",
    "run_circuit_cpu",
    "probabilities_cpu",
    "expectations_z_cpu",
    "feature_map_cpu",
    "feature_map_batch_cpu",
    "ternary_forward_cpu",
    "ternary_forward_batch_cpu",
    "gpu_available",
    "gpu_device_name",
    "run_ops_gpu",
    "run_circuit_gpu",
    "expectations_z_gpu",
    "feature_map_batch_gpu",
    "ternary_forward_batch_gpu",
    "probabilities_from_amp",
]

#: Wire-format opcodes shared by both DLLs (SPEC-NATIVE.md).
OPCODES: dict[str, int] = {
    "h": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "s": 4,
    "t": 5,
    "rx": 6,
    "ry": 7,
    "rz": 8,
    "cnot": 9,
    "cz": 10,
    "swap": 11,
}

_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bin")


def _library_name(stem: str) -> str:
    """Platform-correct shared-library filename for ``stem``.

    The accelerators were built on Windows and the paths were hardcoded to
    ``.dll``, which meant the native tiers could never load anywhere else — the
    pure-Python fallback would silently take over on Linux and the only symptom
    would be that everything is slower. Naming is resolved per platform instead.
    """
    if os.name == "nt":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"




def _build_hint(target: str) -> str:
    """How to build a missing accelerator on this platform."""
    if os.name == "nt":
        return f"build with native\\build.ps1 -Target {target}"
    return f"build with native/build.sh --target {target}"


CPU_DLL_PATH: str = os.path.join(_BIN_DIR, _library_name("ultraquant_native"))
GPU_DLL_PATH: str = os.path.join(_BIN_DIR, _library_name("ultraquant_cuda"))

# GPU feature-map kernel keeps the per-sample statevector in shared memory.
_GPU_FEATURE_MAP_MAX_QUBITS = 10
# Both run_circuit implementations guard num_qubits at 30.
_MAX_QUBITS = 30

_UNSET = object()  # sentinel: "load not attempted yet" (None = attempted, failed)
_cpu_dll: object = _UNSET
_gpu_dll: object = _UNSET
_load_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _declare_cpu(dll: ctypes.CDLL) -> None:
    """Set argtypes/restype on every ultraquant_native.dll export."""
    c_int = ctypes.c_int
    c_dbl = ctypes.c_double
    p_dbl = ctypes.POINTER(ctypes.c_double)
    p_int = ctypes.POINTER(ctypes.c_int)
    p_i8 = ctypes.POINTER(ctypes.c_int8)

    dll.uq_version.argtypes = []
    dll.uq_version.restype = c_int
    dll.uq_init_state.argtypes = [p_dbl, c_int]
    dll.uq_init_state.restype = None
    dll.uq_run_circuit.argtypes = [p_dbl, c_int, p_int, p_int, p_dbl, c_int]
    dll.uq_run_circuit.restype = None
    dll.uq_probabilities.argtypes = [p_dbl, c_int, p_dbl]
    dll.uq_probabilities.restype = None
    dll.uq_expectations_z.argtypes = [p_dbl, c_int, p_dbl]
    dll.uq_expectations_z.restype = None
    dll.uq_feature_map.argtypes = [p_dbl, c_int, c_int, p_dbl]
    dll.uq_feature_map.restype = None
    dll.uq_feature_map_batch.argtypes = [p_dbl, c_int, c_int, c_int, p_dbl, c_int]
    dll.uq_feature_map_batch.restype = None
    dll.uq_ternary_forward.argtypes = [p_i8, c_dbl, p_dbl, p_dbl, c_int, c_int, p_dbl]
    dll.uq_ternary_forward.restype = None
    dll.uq_ternary_forward_batch.argtypes = [
        p_i8, c_dbl, p_dbl, p_dbl, c_int, c_int, c_int, p_dbl, c_int,
    ]
    dll.uq_ternary_forward_batch.restype = None


def _declare_gpu(dll: ctypes.CDLL) -> None:
    """Set argtypes/restype on every ultraquant_cuda.dll export."""
    c_int = ctypes.c_int
    c_dbl = ctypes.c_double
    p_dbl = ctypes.POINTER(ctypes.c_double)
    p_int = ctypes.POINTER(ctypes.c_int)
    p_i8 = ctypes.POINTER(ctypes.c_int8)

    dll.uqg_available.argtypes = []
    dll.uqg_available.restype = c_int
    dll.uqg_device_name.argtypes = [ctypes.c_char_p, c_int]
    dll.uqg_device_name.restype = c_int
    dll.uqg_run_circuit.argtypes = [p_dbl, c_int, p_int, p_int, p_dbl, c_int]
    dll.uqg_run_circuit.restype = None
    dll.uqg_expectations_z_from.argtypes = [p_dbl, c_int, p_dbl]
    dll.uqg_expectations_z_from.restype = None
    dll.uqg_feature_map_batch.argtypes = [p_dbl, c_int, c_int, c_int, p_dbl]
    dll.uqg_feature_map_batch.restype = None
    dll.uqg_ternary_forward_batch.argtypes = [
        p_i8, c_dbl, p_dbl, p_dbl, c_int, c_int, c_int, p_dbl,
    ]
    dll.uqg_ternary_forward_batch.restype = None


def _try_load_cpu() -> ctypes.CDLL | None:
    try:
        if not os.path.isfile(CPU_DLL_PATH):
            return None
        dll = ctypes.CDLL(CPU_DLL_PATH)
        _declare_cpu(dll)
        if dll.uq_version() < 1:
            return None
        return dll
    except Exception:
        return None


def _try_load_gpu() -> ctypes.CDLL | None:
    try:
        if not os.path.isfile(GPU_DLL_PATH):
            return None
        dll = ctypes.CDLL(GPU_DLL_PATH)
        _declare_gpu(dll)
        if dll.uqg_available() < 1:
            return None
        return dll
    except Exception:
        return None


def load_cpu() -> ctypes.CDLL | None:
    """Load and cache the CPU DLL (ultraquant_native.dll).

    Returns the module-level cached ``ctypes.CDLL`` handle with all argtypes
    declared, or ``None`` if the DLL is missing or fails to load. The result
    (including failure) is cached; this function never raises.
    """
    global _cpu_dll
    if _cpu_dll is _UNSET:
        with _load_lock:
            if _cpu_dll is _UNSET:
                _cpu_dll = _try_load_cpu()
    return _cpu_dll  # type: ignore[return-value]


def load_gpu() -> ctypes.CDLL | None:
    """Load and cache the GPU DLL (ultraquant_cuda.dll).

    Returns ``None`` if the DLL is missing, fails to load, or reports no CUDA
    device (``uqg_available() == 0``). Cached module-level; never raises.
    """
    global _gpu_dll
    if _gpu_dll is _UNSET:
        with _load_lock:
            if _gpu_dll is _UNSET:
                _gpu_dll = _try_load_gpu()
    return _gpu_dll  # type: ignore[return-value]


def _require_cpu() -> ctypes.CDLL:
    dll = load_cpu()
    if dll is None:
        raise RuntimeError(
            "UltraQuant native CPU tier unavailable "
            f"(cannot load {CPU_DLL_PATH!r}; build with native\\build.ps1 -Target cpu)"
        )
    return dll


def _require_gpu() -> ctypes.CDLL:
    dll = load_gpu()
    if dll is None:
        raise RuntimeError(
            "UltraQuant CUDA tier unavailable "
            f"(cannot load {GPU_DLL_PATH!r} or no CUDA device; "
            "build with native\\build.ps1 -Target cuda)"
        )
    return dll


# ---------------------------------------------------------------------------
# Buffer helpers (array.array <-> ctypes, zero-copy)
# ---------------------------------------------------------------------------

def _darr(values: Sequence[float]) -> array:
    if isinstance(values, array) and values.typecode == "d":
        return values
    return array("d", values)


def _iarr(values: Sequence[int]) -> array:
    if isinstance(values, array) and values.typecode == "i":
        return values
    return array("i", values)


def _c_dbl(arr: array):
    """View an array('d') as a ctypes double array (shared memory)."""
    return (ctypes.c_double * len(arr)).from_buffer(arr)


def _c_int(arr: array):
    return (ctypes.c_int * len(arr)).from_buffer(arr)


def _c_i8(arr: array):
    return (ctypes.c_int8 * len(arr)).from_buffer(arr)


def _flatten2d(rows: Sequence[Sequence], typecode: str, what: str) -> tuple[array, int, int]:
    """Flatten a rectangular list-of-rows to a flat array. -> (flat, n_rows, width)."""
    n_rows = len(rows)
    width = len(rows[0]) if n_rows else 0
    flat = array(typecode)
    for r in rows:
        if len(r) != width:
            raise ValueError(f"{what}: rows must all have the same length "
                             f"(expected {width}, got {len(r)})")
        flat.extend(r)
    return flat, n_rows, width


def _check_num_qubits(num_qubits: int, upper: int = _MAX_QUBITS) -> None:
    if not 1 <= num_qubits <= upper:
        raise ValueError(f"num_qubits must be in 1..{upper}, got {num_qubits}")


def _check_amp(amp: array, num_qubits: int) -> None:
    want = 2 * (1 << num_qubits)
    if not (isinstance(amp, array) and amp.typecode == "d"):
        raise TypeError("amp must be an array('d') of interleaved re/im doubles")
    if len(amp) != want:
        raise ValueError(f"amp length {len(amp)} != 2 * 2**{num_qubits} = {want}")


def _check_ops(opcodes: array, qubits: array, params: array) -> None:
    if len(qubits) != 2 * len(opcodes) or len(params) != len(opcodes):
        raise ValueError(
            "wire arrays inconsistent: need len(qubits) == 2*len(opcodes) and "
            f"len(params) == len(opcodes); got {len(opcodes)}/{len(qubits)}/{len(params)}"
        )


def _fresh_state(num_qubits: int) -> array:
    """array('d') of length 2*2**n holding |0...0> (built host-side)."""
    amp = array("d", bytes(16 * (1 << num_qubits)))  # zeros
    amp[0] = 1.0
    return amp


def _chunk(flat: array, n_rows: int, width: int) -> list[list[float]]:
    return [list(flat[s * width:(s + 1) * width]) for s in range(n_rows)]


# ---------------------------------------------------------------------------
# Circuit encoding
# ---------------------------------------------------------------------------

def encode_circuit(circuit: "Circuit") -> tuple[array, array, array]:
    """Lower a Circuit to the native wire format.

    Returns ``(opcodes, qubits, params)`` as
    ``(array('i')[N], array('i')[2N], array('d')[N])`` with unused qubit
    slots as ``-1`` and unused params as ``0.0``, per SPEC-NATIVE.md.

    Raises ``ValueError`` on op names outside the shared opcode table and
    ``TypeError`` if ``circuit`` is not Circuit-like (no ``.ops``).
    """
    # Lazy on purpose: the base package is built concurrently and must not be
    # imported at module load. Fall back to duck-typing if it is absent.
    try:
        from ultraquant.quantum.circuit import Circuit as _Circuit
    except Exception:
        _Circuit = None
    if not (_Circuit is not None and isinstance(circuit, _Circuit)):
        if not hasattr(circuit, "ops"):
            raise TypeError(
                f"encode_circuit expects a Circuit, got {type(circuit).__name__}"
            )

    opcodes = array("i")
    qubits = array("i")
    params = array("d")
    for op in circuit.ops:
        name = str(op.name)
        try:
            code = OPCODES[name]
        except KeyError:
            raise ValueError(
                f"unknown op name for native tier: {name!r} "
                f"(known: {sorted(OPCODES)})"
            ) from None
        qs = tuple(op.qubits)
        ps = tuple(getattr(op, "params", ()) or ())
        opcodes.append(code)
        qubits.append(int(qs[0]) if len(qs) > 0 else -1)
        qubits.append(int(qs[1]) if len(qs) > 1 else -1)
        params.append(float(ps[0]) if ps else 0.0)
    return opcodes, qubits, params


def _circuit_num_qubits(circuit: "Circuit", qubits: array) -> int:
    n = getattr(circuit, "num_qubits", None)
    if n is not None:
        return int(n)
    # Fallback: derive from the highest qubit index actually used.
    return max((q for q in qubits if q >= 0), default=0) + 1


# ---------------------------------------------------------------------------
# CPU wrappers (ultraquant_native.dll)
# ---------------------------------------------------------------------------

def version_cpu() -> int:
    """``uq_version()`` from the CPU DLL. RuntimeError if the tier is absent."""
    return int(_require_cpu().uq_version())


def init_state_cpu(num_qubits: int) -> array:
    """Allocate a statevector and set it to |0...0> via ``uq_init_state``.

    Returns an ``array('d')`` of length ``2 * 2**num_qubits`` (interleaved
    re/im, little-endian qubit indexing).
    """
    _check_num_qubits(num_qubits)
    dll = _require_cpu()
    amp = array("d", bytes(16 * (1 << num_qubits)))
    dll.uq_init_state(_c_dbl(amp), num_qubits)
    return amp


def run_ops_cpu(
    num_qubits: int,
    opcodes: Sequence[int],
    qubits: Sequence[int],
    params: Sequence[float],
    amp: array | None = None,
) -> array:
    """Replay pre-encoded wire arrays through ``uq_run_circuit``.

    ``amp`` (optional) is an existing statevector to continue from, mutated
    in place; when omitted a fresh |0...0> state is created. Returns the
    statevector (``array('d')``, interleaved re/im).
    """
    _check_num_qubits(num_qubits)
    opc, qub, par = _iarr(opcodes), _iarr(qubits), _darr(params)
    _check_ops(opc, qub, par)
    dll = _require_cpu()
    if amp is None:
        amp = _fresh_state(num_qubits)
    else:
        _check_amp(amp, num_qubits)
    if len(opc):
        dll.uq_run_circuit(
            _c_dbl(amp), num_qubits, _c_int(opc), _c_int(qub), _c_dbl(par), len(opc)
        )
    return amp


def run_circuit_cpu(circuit: "Circuit") -> array:
    """Encode + run a Circuit on the CPU DLL from |0...0>.

    Returns the final statevector as ``array('d')`` (interleaved re/im);
    pair with ``probabilities_cpu`` / ``expectations_z_cpu``.
    """
    opc, qub, par = encode_circuit(circuit)
    n = _circuit_num_qubits(circuit, qub)
    return run_ops_cpu(n, opc, qub, par)


def probabilities_cpu(amp: array, num_qubits: int) -> list[float]:
    """``uq_probabilities``: basis-state probabilities (length ``2**n``)."""
    _check_num_qubits(num_qubits)
    _check_amp(amp, num_qubits)
    dll = _require_cpu()
    out = array("d", bytes(8 * (1 << num_qubits)))
    dll.uq_probabilities(_c_dbl(amp), num_qubits, _c_dbl(out))
    return list(out)


def expectations_z_cpu(amp: array, num_qubits: int) -> list[float]:
    """``uq_expectations_z``: exact per-qubit <Z> (length ``num_qubits``)."""
    _check_num_qubits(num_qubits)
    _check_amp(amp, num_qubits)
    dll = _require_cpu()
    out = array("d", bytes(8 * num_qubits))
    dll.uq_expectations_z(_c_dbl(amp), num_qubits, _c_dbl(out))
    return list(out)


def feature_map_cpu(features: Sequence[float], num_qubits: int) -> list[float]:
    """``uq_feature_map``: single-sample SPEC.md section 5 feature map.

    Returns per-qubit <Z> (length ``num_qubits``), matching
    ``QuantumFeatureMap.extract`` to 1e-9.
    """
    _check_num_qubits(num_qubits)
    dll = _require_cpu()
    feats = _darr(features)
    fbuf = feats if len(feats) else array("d", [0.0])  # non-null, never read
    out = array("d", bytes(8 * num_qubits))
    dll.uq_feature_map(_c_dbl(fbuf), len(feats), num_qubits, _c_dbl(out))
    return list(out)


def feature_map_batch_cpu(
    features_batch: Sequence[Sequence[float]],
    num_qubits: int,
    n_threads: int = 0,
) -> list[list[float]]:
    """``uq_feature_map_batch``: feature map over a batch of samples.

    ``features_batch`` is a rectangular list of per-sample feature rows.
    ``n_threads <= 0`` means hardware concurrency (resolved inside the DLL).
    Returns one ``num_qubits``-long <Z> row per sample, in input order.
    """
    _check_num_qubits(num_qubits)
    flat, n_samples, n_features = _flatten2d(features_batch, "d", "features_batch")
    if n_samples == 0:
        return []
    dll = _require_cpu()
    fbuf = flat if len(flat) else array("d", [0.0])
    out = array("d", bytes(8 * n_samples * num_qubits))
    dll.uq_feature_map_batch(
        _c_dbl(fbuf), n_samples, n_features, num_qubits, _c_dbl(out), int(n_threads)
    )
    return _chunk(out, n_samples, num_qubits)


def flatten_features(
    features_batch: Sequence[Sequence[float]],
) -> tuple[array, int, int]:
    """Marshal a rectangular batch once into a flat ``array('d')``.

    Callers that dispatch one batch across several tiers concurrently should
    marshal here (single-threaded, once) and then hand each tier a slice via
    :func:`feature_map_slice_cpu` / :func:`feature_map_slice_gpu`.  Converting
    inside the workers instead would serialise them on the GIL, which is the
    dominant cost at large batch sizes.

    Returns:
        ``(flat, n_samples, n_features)``.
    """
    return _flatten2d(features_batch, "d", "features_batch")


def _offset_dbl(buf, sample_offset: int, width: int):
    """Pointer to sample ``sample_offset`` inside a flat double buffer."""
    p_dbl = ctypes.POINTER(ctypes.c_double)
    return ctypes.cast(
        ctypes.byref(buf, sample_offset * width * ctypes.sizeof(ctypes.c_double)),
        p_dbl,
    )


def feature_map_slice_cpu(
    flat: array,
    n_features: int,
    num_qubits: int,
    start: int,
    count: int,
    out: array,
    n_threads: int = 0,
) -> None:
    """Run the CPU feature map over ``count`` samples starting at ``start``.

    Reads directly from ``flat`` and writes directly into ``out`` — no Python
    level conversion happens, so this releases the GIL for its whole duration.
    """
    if count <= 0:
        return
    _check_num_qubits(num_qubits)
    dll = _require_cpu()
    fbuf = (ctypes.c_double * len(flat)).from_buffer(flat)
    obuf = (ctypes.c_double * len(out)).from_buffer(out)
    dll.uq_feature_map_batch(
        _offset_dbl(fbuf, start, n_features),
        int(count),
        int(n_features),
        int(num_qubits),
        _offset_dbl(obuf, start, num_qubits),
        int(n_threads),
    )


def feature_map_slice_gpu(
    flat: array,
    n_features: int,
    num_qubits: int,
    start: int,
    count: int,
    out: array,
) -> None:
    """Run the CUDA feature map over ``count`` samples starting at ``start``.

    Same zero-copy contract as :func:`feature_map_slice_cpu`.
    """
    if count <= 0:
        return
    _check_num_qubits(num_qubits, _GPU_FEATURE_MAP_MAX_QUBITS)
    dll = _require_gpu()
    fbuf = (ctypes.c_double * len(flat)).from_buffer(flat)
    obuf = (ctypes.c_double * len(out)).from_buffer(out)
    dll.uqg_feature_map_batch(
        _offset_dbl(fbuf, start, n_features),
        int(count),
        int(n_features),
        int(num_qubits),
        _offset_dbl(obuf, start, num_qubits),
    )


def _prep_ternary(
    qw: Sequence[Sequence[int]],
    bias: Sequence[float] | None,
) -> tuple[array, array | None, int, int]:
    qflat, out_dim, in_dim = _flatten2d(qw, "b", "quantized weights")
    if out_dim < 1 or in_dim < 1:
        raise ValueError(f"quantized weights must be non-empty, got {out_dim}x{in_dim}")
    barr: array | None = None
    if bias is not None:
        barr = _darr(bias)
        if len(barr) != out_dim:
            raise ValueError(f"bias length {len(barr)} != out_dim {out_dim}")
    return qflat, barr, out_dim, in_dim


def ternary_forward_cpu(
    qw: Sequence[Sequence[int]],
    alpha: float,
    bias: Sequence[float] | None,
    x: Sequence[float],
) -> list[float]:
    """``uq_ternary_forward``: ``out = alpha * (qw @ x) + bias``.

    ``qw`` is the ternary matrix as rows ``[out_dim][in_dim]`` with entries
    in {-1, 0, +1}; ``bias`` may be ``None`` (treated as zeros).
    """
    qflat, barr, out_dim, in_dim = _prep_ternary(qw, bias)
    xarr = _darr(x)
    if len(xarr) != in_dim:
        raise ValueError(f"x length {len(xarr)} != in_dim {in_dim}")
    dll = _require_cpu()
    out = array("d", bytes(8 * out_dim))
    dll.uq_ternary_forward(
        _c_i8(qflat), float(alpha), _c_dbl(barr) if barr is not None else None,
        _c_dbl(xarr), in_dim, out_dim, _c_dbl(out),
    )
    return list(out)


def ternary_forward_batch_cpu(
    qw: Sequence[Sequence[int]],
    alpha: float,
    bias: Sequence[float] | None,
    xs: Sequence[Sequence[float]],
    n_threads: int = 0,
) -> list[list[float]]:
    """``uq_ternary_forward_batch``: ternary forward over a batch of inputs.

    ``xs`` is a rectangular batch ``[n_samples][in_dim]``; ``n_threads <= 0``
    means hardware concurrency. Returns ``[n_samples][out_dim]`` in order.
    """
    qflat, barr, out_dim, in_dim = _prep_ternary(qw, bias)
    xflat, n_samples, x_width = _flatten2d(xs, "d", "xs")
    if n_samples == 0:
        return []
    if x_width != in_dim:
        raise ValueError(f"xs row length {x_width} != in_dim {in_dim}")
    dll = _require_cpu()
    out = array("d", bytes(8 * n_samples * out_dim))
    dll.uq_ternary_forward_batch(
        _c_i8(qflat), float(alpha), _c_dbl(barr) if barr is not None else None,
        _c_dbl(xflat), n_samples, in_dim, out_dim, _c_dbl(out), int(n_threads),
    )
    return _chunk(out, n_samples, out_dim)


# ---------------------------------------------------------------------------
# GPU wrappers (ultraquant_cuda.dll)
# ---------------------------------------------------------------------------

def gpu_available() -> bool:
    """True iff the CUDA DLL loads and reports at least one device. Never raises."""
    return load_gpu() is not None


def gpu_device_name() -> str | None:
    """Name of CUDA device 0 (``uqg_device_name``), or ``None``. Never raises."""
    dll = load_gpu()
    if dll is None:
        return None
    try:
        buf = ctypes.create_string_buffer(256)
        if dll.uqg_device_name(buf, 256) != 0:
            return None
        return buf.value.decode("utf-8", errors="replace")
    except Exception:
        return None


def run_ops_gpu(
    num_qubits: int,
    opcodes: Sequence[int],
    qubits: Sequence[int],
    params: Sequence[float],
    amp: array | None = None,
) -> array:
    """Replay pre-encoded wire arrays through ``uqg_run_circuit``.

    The host statevector is copied to the device, evolved one kernel launch
    per gate, and copied back; ``amp`` is mutated in place when given
    (fresh |0...0> otherwise). Returns the statevector.
    """
    _check_num_qubits(num_qubits)
    opc, qub, par = _iarr(opcodes), _iarr(qubits), _darr(params)
    _check_ops(opc, qub, par)
    dll = _require_gpu()
    if amp is None:
        amp = _fresh_state(num_qubits)
    else:
        _check_amp(amp, num_qubits)
    if len(opc):
        dll.uqg_run_circuit(
            _c_dbl(amp), num_qubits, _c_int(opc), _c_int(qub), _c_dbl(par), len(opc)
        )
    return amp


def run_circuit_gpu(circuit: "Circuit") -> array:
    """Encode + run a Circuit on the GPU DLL from |0...0>.

    Returns the final statevector as ``array('d')`` (interleaved re/im);
    pair with ``expectations_z_gpu`` / ``probabilities_from_amp``.
    """
    opc, qub, par = encode_circuit(circuit)
    n = _circuit_num_qubits(circuit, qub)
    return run_ops_gpu(n, opc, qub, par)


def expectations_z_gpu(amp: array, num_qubits: int) -> list[float]:
    """``uqg_expectations_z_from``: exact per-qubit <Z> from a host statevector."""
    _check_num_qubits(num_qubits)
    _check_amp(amp, num_qubits)
    dll = _require_gpu()
    out = array("d", bytes(8 * num_qubits))
    dll.uqg_expectations_z_from(_c_dbl(amp), num_qubits, _c_dbl(out))
    return list(out)


def feature_map_batch_gpu(
    features_batch: Sequence[Sequence[float]],
    num_qubits: int,
) -> list[list[float]]:
    """``uqg_feature_map_batch``: SPEC.md section 5 feature map on the GPU.

    One thread block per sample; requires ``1 <= num_qubits <= 10`` (the
    per-sample statevector lives in shared memory). Returns one
    ``num_qubits``-long <Z> row per sample, in input order.
    """
    _check_num_qubits(num_qubits, upper=_GPU_FEATURE_MAP_MAX_QUBITS)
    flat, n_samples, n_features = _flatten2d(features_batch, "d", "features_batch")
    if n_samples == 0:
        return []
    dll = _require_gpu()
    fbuf = flat if len(flat) else array("d", [0.0])
    out = array("d", bytes(8 * n_samples * num_qubits))
    dll.uqg_feature_map_batch(
        _c_dbl(fbuf), n_samples, n_features, num_qubits, _c_dbl(out)
    )
    return _chunk(out, n_samples, num_qubits)


def ternary_forward_batch_gpu(
    qw: Sequence[Sequence[int]],
    alpha: float,
    bias: Sequence[float] | None,
    xs: Sequence[Sequence[float]],
) -> list[list[float]]:
    """``uqg_ternary_forward_batch``: batched ternary forward on the GPU.

    Same contract as ``ternary_forward_batch_cpu`` (bias ``None`` = zeros);
    returns ``[n_samples][out_dim]`` in input order.
    """
    qflat, barr, out_dim, in_dim = _prep_ternary(qw, bias)
    xflat, n_samples, x_width = _flatten2d(xs, "d", "xs")
    if n_samples == 0:
        return []
    if x_width != in_dim:
        raise ValueError(f"xs row length {x_width} != in_dim {in_dim}")
    dll = _require_gpu()
    out = array("d", bytes(8 * n_samples * out_dim))
    dll.uqg_ternary_forward_batch(
        _c_i8(qflat), float(alpha), _c_dbl(barr) if barr is not None else None,
        _c_dbl(xflat), n_samples, in_dim, out_dim, _c_dbl(out),
    )
    return _chunk(out, n_samples, out_dim)


# ---------------------------------------------------------------------------
# Pure-Python helper (no DLL): useful for backends built on the GPU tier,
# which has no probabilities export.
# ---------------------------------------------------------------------------

def probabilities_from_amp(amp: Sequence[float]) -> list[float]:
    """Basis-state probabilities |amp[i]|**2 from an interleaved re/im vector."""
    if len(amp) % 2:
        raise ValueError("amp must have even length (interleaved re/im)")
    return [amp[2 * i] ** 2 + amp[2 * i + 1] ** 2 for i in range(len(amp) // 2)]
