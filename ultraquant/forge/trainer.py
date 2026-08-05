"""Multi-expert trainer with pure-Python, C++ and CUDA tiers.

A model in UltraQuant is a *library of pattern experts*, so training one is
really training many small independent networks.  That is the workload shape a
GPU is built for — one block per expert, hundreds resident at once — and it is
why the GPU wins here even though it loses on a single narrow feature map.

All three tiers implement the same algorithm (single hidden layer, ternary
weights, straight-through estimator, softmax cross-entropy, mean-gradient SGD)
and consume the same precomputed shuffle order, so the minibatches are identical
across tiers.  The CPU tiers additionally match the pure-Python reference's
summation order, so they agree to floating-point noise; the GPU reduces in
parallel, so it agrees statistically (same accuracy) rather than bitwise.
"""

from __future__ import annotations

import ctypes
import random
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "ExpertSpec",
    "TrainedExpert",
    "TrainReport",
    "forge_tier_report",
    "train_experts",
]

from ultraquant.native.accel import _library_name

_BIN = Path(__file__).resolve().parent.parent / "native" / "_bin"
# Same platform resolution as the quantum tier: hardcoding ".dll" made the
# forge accelerators unloadable anywhere but Windows, and the only symptom
# was silently falling back to pure Python.
_CPU_DLL = _BIN / _library_name("ultraquant_forge")
_GPU_DLL = _BIN / _library_name("ultraquant_forge_cuda")

_ACT_BITS = 8
_ACT_CLIP = 4.0

_UNSET = object()
#: _UNSET = not tried yet; None = tried and unavailable; else the loaded handle.
#: Measured GPU share per (network shape), so the probe runs once per shape.
_SPLIT_CACHE: dict[tuple, float] = {}

_cpu_handle: ctypes.CDLL | None | object = _UNSET
_gpu_handle: ctypes.CDLL | None | object = _UNSET


@dataclass
class ExpertSpec:
    """One expert's training problem."""

    category: str
    labels: list[str]
    xs: list[list[float]]
    ys: list[int]


@dataclass
class TrainedExpert:
    """A trained expert's parameters and training outcome."""

    category: str
    labels: list[str]
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    loss: float
    accuracy: float
    samples: int


@dataclass
class TrainReport:
    """Outcome of one multi-expert training run."""

    tier: str
    seconds: float
    experts: list[TrainedExpert] = field(default_factory=list)
    #: For the "both" tier: how many experts each device trained.
    split: dict = field(default_factory=dict)

    @property
    def mean_accuracy(self) -> float:
        """Mean training accuracy across experts."""
        return sum(e.accuracy for e in self.experts) / max(len(self.experts), 1)


# ---------------------------------------------------------------------------
# DLL loading
# ---------------------------------------------------------------------------

def _load_cpu() -> ctypes.CDLL | None:
    """Load the forge CPU DLL, or None if unavailable. Never raises."""
    global _cpu_handle
    if _cpu_handle is not _UNSET and _cpu_handle is not None:
        return _cpu_handle  # type: ignore[return-value]
    if _cpu_handle is None:
        return None
    try:
        dll = ctypes.CDLL(str(_CPU_DLL))
        c_i, c_d, p_d, p_i = ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32)
        dll.uqf_version.argtypes = []
        dll.uqf_version.restype = c_i
        dll.uqf_train_experts.argtypes = [
            c_i, c_i, c_i, c_i, c_i,          # n_experts, in, hidden, classes, samples
            p_d, p_i,                          # xs, ys
            p_d, p_d, p_d, p_d,                # w1, b1, w2, b2
            c_i, c_d, c_i, c_i, c_i,           # epochs, lr, batch, act_bits, quantize_head
            c_d, p_i, p_d, p_d, c_i,           # clip, order, loss, acc, threads
        ]
        dll.uqf_train_experts.restype = None
        _cpu_handle = dll
        return dll
    except Exception:  # noqa: BLE001 - absence is normal
        _cpu_handle = None
        return None


def _load_gpu() -> ctypes.CDLL | None:
    """Load the forge CUDA DLL, or None if unavailable. Never raises."""
    global _gpu_handle
    if _gpu_handle is not _UNSET and _gpu_handle is not None:
        return _gpu_handle  # type: ignore[return-value]
    if _gpu_handle is None:
        return None
    try:
        dll = ctypes.CDLL(str(_GPU_DLL))
        c_i, c_d, p_d, p_i = ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32)
        dll.uqfg_available.argtypes = []
        dll.uqfg_available.restype = c_i
        dll.uqfg_shared_doubles.argtypes = [c_i, c_i, c_i, c_i]
        dll.uqfg_shared_doubles.restype = c_i
        dll.uqfg_train_experts.argtypes = [
            c_i, c_i, c_i, c_i, c_i,
            p_d, p_i,
            p_d, p_d, p_d, p_d,
            c_i, c_d, c_i, c_i, c_i,
            c_d, p_i, p_d, p_d,
        ]
        dll.uqfg_train_experts.restype = c_i
        if dll.uqfg_available() < 1:
            _gpu_handle = None
            return None
        _gpu_handle = dll
        return dll
    except Exception:  # noqa: BLE001 - absence is normal
        _gpu_handle = None
        return None


def forge_tier_report() -> dict:
    """Which forge tiers are usable on this machine."""
    return {
        "cuda": _load_gpu() is not None,
        "cpu": _load_cpu() is not None,
        "python": True,
    }


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _init_weights(in_dim: int, hidden: int, n_classes: int, seed: int):
    """Seeded initial weights, identical to UltraQuantNet's construction order."""
    rng = random.Random(seed)
    r1 = (6.0 / (in_dim + hidden)) ** 0.5
    w1 = [[rng.uniform(-r1, r1) for _ in range(in_dim)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    r2 = (6.0 / (hidden + n_classes)) ** 0.5
    w2 = [[rng.uniform(-r2, r2) for _ in range(hidden)] for _ in range(n_classes)]
    b2 = [0.0] * n_classes
    return w1, b1, w2, b2


def _shuffle_order(n_samples: int, epochs: int, seed: int) -> list[int]:
    """Flat [epochs][n_samples] visit order, shared by every tier."""
    rng = random.Random(seed)
    flat: list[int] = []
    for _ in range(epochs):
        idx = list(range(n_samples))
        rng.shuffle(idx)
        flat.extend(idx)
    return flat


def _validate(specs: Sequence[ExpertSpec]) -> tuple[int, int, int]:
    """Check the batch is uniform enough for the native path.

    Returns:
        ``(in_dim, n_classes, n_samples)``.

    Raises:
        ValueError: If specs are empty or ragged.
    """
    if not specs:
        raise ValueError("no experts to train")
    in_dim = len(specs[0].xs[0])
    n_samples = len(specs[0].xs)
    n_classes = len(specs[0].labels)
    for spec in specs:
        if len(spec.xs) != n_samples:
            raise ValueError("every expert must have the same sample count")
        if len(spec.labels) != n_classes:
            raise ValueError("every expert must have the same class count")
        if len(spec.ys) != len(spec.xs):
            raise ValueError(f"{spec.category}: xs/ys length mismatch")
        for row in spec.xs:
            if len(row) != in_dim:
                raise ValueError("every sample must have the same feature count")
    return in_dim, n_classes, n_samples


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

def _train_python(
    specs: Sequence[ExpertSpec], hidden: int, epochs: int, lr: float,
    batch_size: int, seed: int, quantize_head: bool,
) -> list[TrainedExpert]:
    """Reference tier: the pure-Python network, one expert at a time."""
    from ultraquant.model.network import UltraQuantNet

    in_dim, n_classes, n_samples = _validate(specs)
    out: list[TrainedExpert] = []
    for index, spec in enumerate(specs):
        net = UltraQuantNet(in_dim, [hidden], n_classes, seed=seed + index,
                            quantize_head=quantize_head, act_bits=_ACT_BITS)
        w1, b1, w2, b2 = _init_weights(in_dim, hidden, n_classes, seed + index)
        net.hidden[0].w, net.hidden[0].b = w1, b1
        net.head.w, net.head.b = w2, b2

        order = _shuffle_order(n_samples, epochs, seed + index)
        loss = 0.0
        for epoch in range(epochs):
            ep = order[epoch * n_samples:(epoch + 1) * n_samples]
            losses = []
            for start in range(0, n_samples, batch_size):
                idx = ep[start:start + batch_size]
                losses.append(net.train_batch([spec.xs[i] for i in idx],
                                              [spec.ys[i] for i in idx], lr))
            loss = sum(losses) / max(len(losses), 1)

        correct = sum(1 for x, y in zip(spec.xs, spec.ys) if net.predict(x)[0] == y)
        out.append(TrainedExpert(
            category=spec.category, labels=list(spec.labels),
            w1=[list(r) for r in net.hidden[0].w], b1=list(net.hidden[0].b),
            w2=[list(r) for r in net.head.w], b2=list(net.head.b),
            loss=loss, accuracy=correct / n_samples, samples=n_samples,
        ))
    return out


def _marshal(specs: Sequence[ExpertSpec], hidden: int, epochs: int, seed: int) -> dict:
    """Flatten every expert's data and initial weights into shared buffers.

    Done once, on the calling thread. Splitting work across devices afterwards
    is then pure pointer arithmetic — no Python runs inside either worker, which
    is what lets two devices actually overlap instead of taking turns on the GIL
    while each builds its own arrays.
    """
    in_dim, n_classes, n_samples = _validate(specs)
    buf: dict = {
        "xs": array("d"), "ys": array("i"), "w1": array("d"), "b1": array("d"),
        "w2": array("d"), "b2": array("d"), "order": array("i"),
        "in_dim": in_dim, "n_classes": n_classes, "n_samples": n_samples,
        "n_experts": len(specs),
    }
    for index, spec in enumerate(specs):
        for row in spec.xs:
            buf["xs"].extend(row)
        buf["ys"].extend(spec.ys)
        iw1, ib1, iw2, ib2 = _init_weights(in_dim, hidden, n_classes, seed + index)
        for row in iw1:
            buf["w1"].extend(row)
        buf["b1"].extend(ib1)
        for row in iw2:
            buf["w2"].extend(row)
        buf["b2"].extend(ib2)
        buf["order"].extend(_shuffle_order(n_samples, epochs, seed + index))
    buf["loss"] = array("d", bytes(8 * len(specs)))
    buf["acc"] = array("d", bytes(8 * len(specs)))
    return buf


def _offset_ptr(buf: array, element_offset: int, ctype):
    """Pointer into ``buf`` at ``element_offset`` elements from the start."""
    view = (ctype * len(buf)).from_buffer(buf)
    return ctypes.cast(
        ctypes.byref(view, element_offset * ctypes.sizeof(ctype)),
        ctypes.POINTER(ctype),
    )


def _call_native(
    dll, gpu: bool, buf: dict, start: int, count: int, hidden: int, epochs: int,
    lr: float, batch_size: int, quantize_head: bool, n_threads: int,
) -> None:
    """Train experts ``[start, start + count)`` straight out of shared buffers."""
    if count <= 0:
        return
    in_dim, n_classes, n_samples = buf["in_dim"], buf["n_classes"], buf["n_samples"]
    c_d, c_i32 = ctypes.c_double, ctypes.c_int32
    args = (
        count, in_dim, hidden, n_classes, n_samples,
        _offset_ptr(buf["xs"], start * n_samples * in_dim, c_d),
        _offset_ptr(buf["ys"], start * n_samples, c_i32),
        _offset_ptr(buf["w1"], start * hidden * in_dim, c_d),
        _offset_ptr(buf["b1"], start * hidden, c_d),
        _offset_ptr(buf["w2"], start * n_classes * hidden, c_d),
        _offset_ptr(buf["b2"], start * n_classes, c_d),
        epochs, float(lr), batch_size, _ACT_BITS, int(quantize_head),
        float(_ACT_CLIP),
        _offset_ptr(buf["order"], start * epochs * n_samples, c_i32),
        _offset_ptr(buf["loss"], start, c_d),
        _offset_ptr(buf["acc"], start, c_d),
    )
    if gpu:
        rc = dll.uqfg_train_experts(*args)
        if rc != 0:
            raise RuntimeError(f"CUDA forge training failed (code {rc})")
    else:
        dll.uqf_train_experts(*args, int(n_threads))


def _unpack(specs: Sequence[ExpertSpec], buf: dict, hidden: int) -> list[TrainedExpert]:
    """Read trained parameters back out of the shared buffers."""
    in_dim, n_classes = buf["in_dim"], buf["n_classes"]
    s1, s2 = hidden * in_dim, n_classes * hidden
    w1, b1, w2, b2 = buf["w1"], buf["b1"], buf["w2"], buf["b2"]
    out: list[TrainedExpert] = []
    for index, spec in enumerate(specs):
        o1, o2 = index * s1, index * s2
        out.append(TrainedExpert(
            category=spec.category,
            labels=list(spec.labels),
            w1=[list(w1[o1 + h * in_dim:o1 + (h + 1) * in_dim]) for h in range(hidden)],
            b1=list(b1[index * hidden:(index + 1) * hidden]),
            w2=[list(w2[o2 + o * hidden:o2 + (o + 1) * hidden]) for o in range(n_classes)],
            b2=list(b2[index * n_classes:(index + 1) * n_classes]),
            loss=buf["loss"][index], accuracy=buf["acc"][index],
            samples=buf["n_samples"],
        ))
    return out


def _train_native(
    dll: ctypes.CDLL, gpu: bool, specs: Sequence[ExpertSpec], hidden: int,
    epochs: int, lr: float, batch_size: int, seed: int, quantize_head: bool,
    n_threads: int,
) -> list[TrainedExpert]:
    """Native tier: all experts trained concurrently in one call."""
    buf = _marshal(specs, hidden, epochs, seed)
    _call_native(dll, gpu, buf, 0, len(specs), hidden, epochs, lr, batch_size,
                 quantize_head, n_threads)
    return _unpack(specs, buf, hidden)


def _train_both(
    specs: Sequence[ExpertSpec], hidden: int, epochs: int, lr: float,
    batch_size: int, seed: int, quantize_head: bool, n_threads: int,
) -> tuple[list[TrainedExpert], dict]:
    """Train on the GPU and the CPU at the same time, splitting the experts.

    Experts are independent, so a heterogeneous machine can work both processors
    at once rather than picking one. The split is measured, not assumed: a probe
    times each device on a representative slice and the rest is divided in
    proportion. Per-expert seeding means which device trains which expert makes
    no difference to the result.

    Returns:
        ``(experts, split)`` where ``split`` records how many went where.

    Raises:
        RuntimeError: If either native tier is missing.
    """
    gpu, cpu = _load_gpu(), _load_cpu()
    if gpu is None or cpu is None:
        raise RuntimeError("both-device training needs the CUDA and C++ forge tiers")

    n = len(specs)
    if n < 2:
        return (_train_native(gpu, True, specs, hidden, epochs, lr, batch_size,
                              seed, quantize_head, n_threads),
                {"cuda": n, "cpu": 0})

    # The split ratio is a property of this machine and this network shape, not
    # of the particular data, so it is measured once and reused. Probing costs
    # real work — including marshalling the probe slice — and paying it on every
    # forge was enough to cancel out the gain from splitting.
    in_dim, n_classes, n_samples = _validate(specs)
    shape = (hidden, in_dim, n_classes, n_samples, epochs, batch_size)
    gpu_share = _SPLIT_CACHE.get(shape)

    if gpu_share is None:
        probe_n = max(2, min(n, n // 16))
        probe_epochs = max(2, min(epochs, epochs // 4))
        probe = specs[:probe_n]

        started = time.perf_counter()
        _train_native(gpu, True, probe, hidden, probe_epochs, lr, batch_size,
                      seed, quantize_head, n_threads)
        gpu_dt = max(time.perf_counter() - started, 1e-9)
        started = time.perf_counter()
        _train_native(cpu, False, probe, hidden, probe_epochs, lr, batch_size,
                      seed, quantize_head, n_threads)
        cpu_dt = max(time.perf_counter() - started, 1e-9)

        gpu_share = (1.0 / gpu_dt) / ((1.0 / gpu_dt) + (1.0 / cpu_dt))
        _SPLIT_CACHE[shape] = gpu_share
    n_gpu = min(max(int(round(gpu_share * n)), 1), n - 1)

    # Marshal once; each device then works a disjoint slice of the same buffers.
    buf = _marshal(specs, hidden, epochs, seed)
    error: dict[str, BaseException] = {}

    def gpu_worker() -> None:
        try:
            _call_native(gpu, True, buf, 0, n_gpu, hidden, epochs, lr,
                         batch_size, quantize_head, n_threads)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            error["gpu"] = exc

    thread = threading.Thread(target=gpu_worker, name="uq-forge-gpu", daemon=True)
    thread.start()
    try:
        _call_native(cpu, False, buf, n_gpu, n - n_gpu, hidden, epochs, lr,
                     batch_size, quantize_head, n_threads)
    finally:
        thread.join()
    if "gpu" in error:
        raise error["gpu"]

    return _unpack(specs, buf, hidden), {"cuda": n_gpu, "cpu": n - n_gpu}


def train_experts(
    specs: Sequence[ExpertSpec],
    hidden: int = 32,
    epochs: int = 40,
    lr: float = 0.05,
    batch_size: int = 16,
    seed: int = 0,
    quantize_head: bool = False,
    tier: str = "auto",
    n_threads: int = 0,
) -> TrainReport:
    """Train every expert, choosing the fastest available tier.

    Args:
        specs: One training problem per expert.
        hidden: Hidden-layer width (the native tiers support exactly one
            hidden layer; deeper nets fall back to the Python tier).
        epochs: Passes over each expert's data.
        lr: Learning rate.
        batch_size: Minibatch size.
        seed: Base seed; expert *i* uses ``seed + i`` for both initialization
            and shuffling, identically on every tier.
        quantize_head: Ternarize the output layer too.
        tier: ``"auto"``, ``"cuda"``, ``"cpu"`` or ``"python"``.
        n_threads: CPU worker threads (``<= 0`` = hardware concurrency).

    Returns:
        A :class:`TrainReport` naming the tier actually used.

    Raises:
        ValueError: If ``tier`` is unknown or the specs are ragged.
    """
    if tier not in ("auto", "both", "cuda", "cpu", "python"):
        raise ValueError(f"unknown tier {tier!r}")

    start = time.perf_counter()
    if tier == "both":
        experts, split = _train_both(specs, hidden, epochs, lr, batch_size, seed,
                                     quantize_head, n_threads)
        report = TrainReport("both", time.perf_counter() - start, experts)
        report.split = split
        return report
    order: tuple[str, ...]
    if tier == "auto":
        order = ("cuda", "cpu", "python")
    else:
        order = (tier,)

    for candidate in order:
        if candidate == "cuda":
            dll = _load_gpu()
            if dll is None:
                continue
            try:
                experts = _train_native(dll, True, specs, hidden, epochs, lr,
                                        batch_size, seed, quantize_head, n_threads)
                return TrainReport("cuda", time.perf_counter() - start, experts)
            except Exception:  # noqa: BLE001 - fall through to the next tier
                if tier == "cuda":
                    raise
                continue
        if candidate == "cpu":
            dll = _load_cpu()
            if dll is None:
                continue
            try:
                experts = _train_native(dll, False, specs, hidden, epochs, lr,
                                        batch_size, seed, quantize_head, n_threads)
                return TrainReport("cpu", time.perf_counter() - start, experts)
            except Exception:  # noqa: BLE001 - fall through to the next tier
                if tier == "cpu":
                    raise
                continue
        experts = _train_python(specs, hidden, epochs, lr, batch_size, seed, quantize_head)
        return TrainReport("python", time.perf_counter() - start, experts)

    experts = _train_python(specs, hidden, epochs, lr, batch_size, seed, quantize_head)
    return TrainReport("python", time.perf_counter() - start, experts)
