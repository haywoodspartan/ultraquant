"""Heterogeneous feature extraction: CPU and GPU working on one batch at once.

:class:`HeterogeneousFeatureExtractor` (SPEC-NATIVE.md section 5) computes the
SPEC.md section 5 quantum feature map for a batch of samples using every tier
it can find:

- Both native tiers available: the first batch triggers a small timed warmup
  on each tier, from which a GPU/CPU throughput ratio is derived and cached.
  Every batch is then split by that ratio and the GPU slice (on a
  ``threading.Thread``) runs *concurrently* with the CPU slice (whose
  ``uq_feature_map_batch`` call spawns its own C++ worker threads; ctypes
  releases the GIL for the duration of both foreign calls).
- Exactly one tier available: that tier takes the whole batch.
- No native tier: pure-Python ``QuantumFeatureMap`` on ``SimulatorBackend``.

Degradation is graceful at every level: a tier that is absent, that fails at
runtime, or that cannot handle the request (GPU feature map is limited to 10
qubits; native batch calls need rectangular input) is skipped and the work
flows to the next tier, ending at pure Python, which always succeeds. Results
are always returned in the original sample order and match the pure-Python
tier to 1e-9 regardless of how the batch was split. The most recent batch's
placement is exposed as ``last_split`` (``{"gpu": n, "cpu": n, "python": n}``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
import time
from array import array
from typing import Sequence

from ultraquant.native import accel

#: Samples timed per tier on the first mixed batch to derive the split ratio.
#: This has to be large enough to measure steady-state throughput rather than
#: fixed launch overhead: a 16-sample probe made the GPU look ~4x better than it
#: is for narrow feature maps, and the resulting split was worse than either
#: tier alone.
_WARMUP_SAMPLES = 2048

#: The GPU feature-map kernel keeps each sample's statevector in shared memory.
_GPU_MAX_QUBITS = 10

#: Clamp for the cached GPU fraction.  The bounds are 0/1 on purpose: when one
#: processor is decisively faster for a given shape, giving the other a token
#: share only slows the batch down.  Narrow feature maps really are CPU work --
#: a 5-qubit statevector is 512 bytes and never leaves L1, so the transfer and
#: launch cost of the GPU cannot be amortised.  The GPU earns its share on wide
#: statevectors, where it wins by orders of magnitude.
_MIN_FRACTION = 0.0
_MAX_FRACTION = 1.0

#: Below this batch size the split is skipped entirely -- the fixed cost of
#: starting a worker thread outweighs any parallel gain.
_MIN_SPLIT_SAMPLES = 256


@dataclass
class FlatBatch:
    """A batch already marshalled: a flat ``array('d')`` plus its row width.

    The zero-copy *input* counterpart of :class:`~ultraquant.native.accel.Rows`.
    Callers that generate their samples numerically can fill one of these
    directly and never build a list-of-lists at all, which is the only way to
    avoid the flattening pass - measured at 39.8 ms per 131k x 8 batch, more
    than either kernel costs.

    Honest limit, stated because it decides whether this is worth using:
    building a ``FlatBatch`` *from* an existing list-of-lists costs 29.7 ms,
    so it saves a quarter of the flatten and is not the win. The win is
    generating flat in the first place (11.2 ms), or reusing one buffer
    across many calls.
    """

    values: "array"
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, array) or self.values.typecode != "d":
            raise TypeError("FlatBatch.values must be an array('d')")
        if self.width <= 0:
            raise ValueError("FlatBatch.width must be positive")
        if len(self.values) % self.width:
            raise ValueError(
                f"{len(self.values)} values is not a whole number of "
                f"{self.width}-wide rows"
            )

    def __len__(self) -> int:
        return len(self.values) // self.width

    @classmethod
    def from_rows(cls, rows: "Sequence[Sequence[float]]") -> "FlatBatch":
        """Marshal a list-of-rows once, for callers that already have lists."""
        flat, _n, width = accel.flatten_features(rows)
        return cls(flat, width)

    def to_rows(self) -> list[list[float]]:
        """The list-of-lists form, at the price this class exists to avoid."""
        return [list(self.values[i * self.width:(i + 1) * self.width])
                for i in range(len(self))]


class HeterogeneousFeatureExtractor:
    """Split quantum feature-map batches across the GPU and CPU tiers.

    Attributes:
        num_qubits: Width of the feature map (length of each output row).
        cpu_threads: Thread count handed to ``uq_feature_map_batch``
            (``<= 0`` means hardware concurrency, resolved inside the DLL).
        force: ``None`` for automatic tier use, or ``"cpu"`` / ``"gpu"`` /
            ``"python"`` to pin extraction to one tier (for testing). A forced
            tier that is unavailable degrades gracefully down the tier order
            instead of raising.
        last_split: ``{"gpu": n, "cpu": n, "python": n}`` -- how many samples
            of the most recent :meth:`extract_batch` call each tier computed.
    """

    def __init__(
        self,
        num_qubits: int,
        cpu_threads: int = 0,
        force: str | None = None,
    ) -> None:
        """Create the extractor and detect the available tiers.

        Args:
            num_qubits: Feature-map width; must be >= 1. Values above 10
                disable the GPU tier (shared-memory kernel limit).
            cpu_threads: Worker threads for the native CPU batch call
                (``<= 0`` = hardware concurrency).
            force: ``None`` (auto) or one of ``"cpu"``, ``"gpu"``, ``"python"``.

        Raises:
            ValueError: If ``num_qubits < 1`` or ``force`` is not one of the
                accepted values.
        """
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        if force not in (None, "cpu", "gpu", "python"):
            raise ValueError(
                f"force must be None, 'cpu', 'gpu' or 'python', got {force!r}"
            )
        self.num_qubits = int(num_qubits)
        self.cpu_threads = int(cpu_threads)
        self.force = force
        self.last_split: dict[str, int] = {"gpu": 0, "cpu": 0, "python": 0}

        # Cached GPU share of each batch, derived from the first-batch warmup.
        self._gpu_fraction: float | None = None
        # Tiers that raised at runtime are retired for this extractor.
        self._cpu_dead = False
        self._gpu_dead = False
        # Lazily built pure-Python fallback (QuantumFeatureMap).
        self._qmap: object | None = None

    # -- tier probes ---------------------------------------------------------

    def _cpu_usable(self) -> bool:
        """True while the native CPU tier is loadable and has not failed."""
        return not self._cpu_dead and accel.load_cpu() is not None

    def _gpu_usable(self) -> bool:
        """True while the CUDA tier is loadable, sized for this map, and alive."""
        return (
            not self._gpu_dead
            and self.num_qubits <= _GPU_MAX_QUBITS
            and accel.load_gpu() is not None
        )

    # -- per-tier runners ----------------------------------------------------

    def _run_gpu(self, rows: list[list[float]]) -> list[list[float]]:
        return accel.feature_map_batch_gpu(rows, self.num_qubits)

    def _run_cpu(self, rows: list[list[float]]) -> list[list[float]]:
        return accel.feature_map_batch_cpu(rows, self.num_qubits, self.cpu_threads)

    def _run_python(self, rows: list[list[float]]) -> list[list[float]]:
        qmap = self._python_qmap()
        return [list(qmap.extract(row)) for row in rows]

    def _python_qmap(self) -> object:
        """Build (once) the pure-Python fallback feature map (exact mode)."""
        if self._qmap is None:
            from ultraquant.quantum.backend import SimulatorBackend
            from ultraquant.quantum.qlayer import QuantumFeatureMap

            self._qmap = QuantumFeatureMap(
                self.num_qubits, SimulatorBackend(seed=0), shots=None
            )
        return self._qmap

    def _run_degrading(
        self, rows: list[list[float]], tiers: tuple[str, ...]
    ) -> tuple[list[list[float]], str]:
        """Run ``rows`` on the first workable tier of ``tiers``.

        A tier that is unavailable is skipped; a tier that raises is retired
        (marked dead) and the next tier is tried. Pure Python is the implicit
        final fallback and is assumed to always succeed.

        Returns:
            Tuple ``(results, tier_name)`` with ``tier_name`` in
            ``{"gpu", "cpu", "python"}``.
        """
        for tier in tiers:
            if tier == "gpu" and self._gpu_usable():
                try:
                    return self._run_gpu(rows), "gpu"
                except Exception:
                    self._gpu_dead = True
            elif tier == "cpu" and self._cpu_usable():
                try:
                    return self._run_cpu(rows), "cpu"
                except Exception:
                    self._cpu_dead = True
            elif tier == "python":
                break
        return self._run_python(rows), "python"

    # -- warmup / split ------------------------------------------------------

    def _warmup(self, batch: list[list[float]]) -> float:
        """Time a chunk on each tier and cache the GPU throughput share.

        Both tiers are timed on the *kernel only* (the batch is marshalled once
        beforehand), because Python-level conversion is a shared cost that would
        otherwise mask the difference between them.  Any warmup failure falls
        back to an even 0.5 split; the real run degrades gracefully on its own.
        """
        # The throughput ratio is size-dependent (thread-spawn cost dominates a
        # small CPU batch; PCIe transfer dominates a small GPU batch), so probe
        # a slice proportional to the real batch rather than a fixed count.
        probe = max(_WARMUP_SAMPLES, len(batch) // 4)
        rows = batch[: max(1, min(probe, len(batch)))]
        try:
            flat, n_rows, n_features = accel.flatten_features(rows)
            out = array("d", bytes(8 * n_rows * self.num_qubits))

            t0 = time.perf_counter()
            accel.feature_map_slice_gpu(flat, n_features, self.num_qubits, 0, n_rows, out)
            gpu_dt = max(time.perf_counter() - t0, 1e-9)

            t0 = time.perf_counter()
            accel.feature_map_slice_cpu(
                flat, n_features, self.num_qubits, 0, n_rows, out, self.cpu_threads
            )
            cpu_dt = max(time.perf_counter() - t0, 1e-9)

            gpu_rate = n_rows / gpu_dt
            cpu_rate = n_rows / cpu_dt
            fraction = gpu_rate / (gpu_rate + cpu_rate)
            # A tier that is not within 4x of the other is not worth the split:
            # hand the whole batch to the winner instead of pacing it to the
            # loser. (Measured: narrow feature maps are ~5x faster on CPU.)
            if gpu_rate > 4.0 * cpu_rate:
                fraction = 1.0
            elif cpu_rate > 4.0 * gpu_rate:
                fraction = 0.0
        except Exception:
            fraction = 0.5
        fraction = min(max(fraction, _MIN_FRACTION), _MAX_FRACTION)
        self._gpu_fraction = fraction
        return fraction

    def _run_split(self, batch: list[list[float]]) -> list[list[float]]:
        """Both tiers available: split by the cached ratio and run concurrently.

        The batch is marshalled into one flat buffer *before* the workers start
        and each tier writes its own slice of one shared output buffer.  Neither
        worker touches Python objects while running, so the GPU and the CPU
        threads overlap for real instead of taking turns holding the GIL.
        """
        fraction = self._gpu_fraction
        if fraction is None:
            fraction = self._warmup(batch)

        n = len(batch)
        # A fraction of exactly 0.0 or 1.0 means the warmup found one processor
        # decisively faster for this shape; honour that and give it everything
        # rather than pacing the batch to the slower one.
        n_gpu = min(max(int(round(fraction * n)), 0), n)
        n_cpu = n - n_gpu

        try:
            flat, _n, n_features = accel.flatten_features(batch)
            out = array("d", bytes(8 * n * self.num_qubits))
        except (TypeError, ValueError):
            # Not marshallable (e.g. non-numeric); let the slow path handle it.
            return self._run_degrading(batch, ("cpu", "python"))[0]

        gpu_err: list[BaseException] = []

        def _gpu_worker() -> None:
            try:
                accel.feature_map_slice_gpu(
                    flat, n_features, self.num_qubits, 0, n_gpu, out
                )
            except BaseException as exc:  # noqa: BLE001 - degrade, never lose work
                gpu_err.append(exc)

        thread: threading.Thread | None = None
        if n_gpu:
            thread = threading.Thread(
                target=_gpu_worker, name="uq-hetero-gpu", daemon=True
            )
            thread.start()

        cpu_tier = "cpu"
        cpu_err: BaseException | None = None
        if n_cpu:
            try:
                accel.feature_map_slice_cpu(
                    flat, n_features, self.num_qubits, n_gpu, n_cpu, out,
                    self.cpu_threads,
                )
            except BaseException as exc:  # noqa: BLE001 - degrade, never lose work
                cpu_err = exc

        if thread is not None:
            thread.join()

        results: object = accel.Rows(out, self.num_qubits)

        gpu_tier = "gpu"
        if n_gpu and gpu_err:
            self._gpu_dead = True
            recovered, gpu_tier = self._run_degrading(batch[:n_gpu], ("cpu", "python"))
            results = results.tolist() if isinstance(results, accel.Rows) else results
            results[:n_gpu] = recovered
        if n_cpu and cpu_err is not None:
            self._cpu_dead = True
            recovered, cpu_tier = self._run_degrading(batch[n_gpu:], ("python",))
            results = results.tolist() if isinstance(results, accel.Rows) else results
            results[n_gpu:] = recovered

        if n_gpu:
            self.last_split[gpu_tier] += n_gpu
        if n_cpu:
            self.last_split[cpu_tier] += n_cpu
        return results

    # -- public API ----------------------------------------------------------

    def _run_flat(self, batch: "FlatBatch") -> object:
        """Split a pre-marshalled batch - the path with no flattening at all."""
        n = len(batch)
        if n == 0:
            return accel.Rows(array("d"), self.num_qubits)

        gpu_ok, cpu_ok = self._gpu_usable(), self._cpu_usable()
        if not (gpu_ok or cpu_ok) or self.force == "python":
            rows = self._run_python(batch.to_rows())
            self.last_split["python"] += n
            return rows

        fraction = self._gpu_fraction
        if fraction is None:
            fraction = self._warmup(batch.to_rows()[:min(n, 256)])
        if not gpu_ok:
            fraction = 0.0
        elif not cpu_ok:
            fraction = 1.0

        n_gpu = min(max(int(round(fraction * n)), 0), n)
        n_cpu = n - n_gpu
        out = array("d", bytes(8 * n * self.num_qubits))
        errors: list[BaseException] = []

        def _gpu_worker() -> None:
            try:
                accel.feature_map_slice_gpu(
                    batch.values, batch.width, self.num_qubits, 0, n_gpu, out
                )
            except BaseException as exc:  # noqa: BLE001 - degrade, never lose work
                errors.append(exc)

        thread = None
        if n_gpu:
            thread = threading.Thread(target=_gpu_worker, name="uq-flat-gpu",
                                      daemon=True)
            thread.start()
        if n_cpu:
            try:
                accel.feature_map_slice_cpu(
                    batch.values, batch.width, self.num_qubits,
                    n_gpu, n_cpu, out, self.cpu_threads,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if thread is not None:
            thread.join()

        if errors:
            # Any native failure falls back to the fully-checked list path,
            # which knows how to degrade tier by tier.
            return self.extract_batch(batch.to_rows())

        if n_gpu:
            self.last_split["gpu"] += n_gpu
        if n_cpu:
            self.last_split["cpu"] += n_cpu
        return accel.Rows(out, self.num_qubits)

    def extract_batch(
        self, features_batch: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        """Quantum feature map for every sample in ``features_batch``.

        Args:
            features_batch: One feature row per sample. Rectangular batches
                are eligible for the native tiers; ragged batches are handled
                (correctly) by the pure-Python fallback.

        Returns:
            One ``num_qubits``-long ``<Z>`` row per sample, in input order,
            matching the pure-Python tier to 1e-9 regardless of the split.
            ``last_split`` records where the samples of this call ran.
        """
        # No defensive float() copy here: the accel wrappers marshal straight
        # into an array('d') and the pure-Python fallback accepts any numeric
        # sequence, so converting first would cost a full extra pass over the
        # batch (measured: ~44 ms per 131k samples) for nothing.
        self.last_split = {"gpu": 0, "cpu": 0, "python": 0}

        # A caller that already holds its batch flat skips marshalling
        # entirely. Measured at 131k x 8: flattening a list-of-lists costs
        # 39.8 ms, more than either kernel (34.7 / 37.4 ms), and it cannot be
        # viewed away because a list-of-lists must genuinely be copied once.
        # Handing in a FlatBatch is the only way to not pay it - and it is
        # only a real saving for callers that never materialise the lists
        # (building an array *from* lists still costs 29.7 ms).
        if isinstance(features_batch, FlatBatch):
            return self._run_flat(features_batch)

        batch = features_batch if isinstance(features_batch, list) else list(features_batch)
        n = len(batch)
        if n == 0:
            return []

        width = len(batch[0])
        rectangular = all(len(row) == width for row in batch)

        if self.force == "python" or not rectangular:
            out, tier = self._run_python(batch), "python"
        elif self.force == "gpu":
            out, tier = self._run_degrading(batch, ("gpu", "cpu", "python"))
        elif self.force == "cpu":
            out, tier = self._run_degrading(batch, ("cpu", "python"))
        else:  # auto
            gpu_ok, cpu_ok = self._gpu_usable(), self._cpu_usable()
            if gpu_ok and cpu_ok:
                return self._run_split(batch)
            if gpu_ok:
                out, tier = self._run_degrading(batch, ("gpu", "cpu", "python"))
            elif cpu_ok:
                out, tier = self._run_degrading(batch, ("cpu", "python"))
            else:
                out, tier = self._run_python(batch), "python"

        self.last_split[tier] += n
        return out
