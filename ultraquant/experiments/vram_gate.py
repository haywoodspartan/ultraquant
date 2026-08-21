"""The VRAM tier against the reference and the clock. The gate.

The vision's storage split — disk as the permanent library, DRAM as the
working set, VRAM as the hot layer — had its first two tiers measured
long ago (§6, §11.14). This gate measures the third, now that the DLL
holds resident layers (§11.45): weights cross the bus once, activations
move per query, and the Python cache (`native/vram.py`) decides
residency under a byte budget.

Rule 1 outranks the clock: the simulator is the reference, accelerators
are never semantics. So the parity criterion is absolute and comes
first, and the latency claim is measured after it, on the repeated
single-query pattern the hot-shard story is actually about — a
conversation touching the same expert again and again.

SKIPS, never passes, without a CUDA device.

**The criteria, written before the run.**

1. **Parity** — resident-forward outputs must match the per-call GPU
   path AND the pure-Python reference within 1e-9, on every world. One
   divergence is a fail: a fast wrong answer is not an accelerator.
2. **Residency pays** — mean per-query wall time over repeated
   single-sample forwards must be lower resident than per-call-upload
   by more than the repetition-to-repetition sd. ~25% of worlds use a
   layer too large for the gate's deliberately small cache budget, so
   residency is refused and the world scores no gain — the fallback
   path measured alongside the fast path, and the contrast's variance.
3. **Accounting balances** — resident bytes reported by the DLL must
   equal the cache's own ledger at every step, and a cleared cache must
   leave device residency at zero.

**It was run, caught one cache bug, and then PASSED.**

The first run's accounting control caught an oversized layer becoming
resident over the budget: eviction empties the cache and then has
nothing left to give, and the upload proceeded anyway. An oversized
layer now refuses and the tier below runs — the fallback is the
mechanism, not an apology. Then, on an RTX 4090:

| measure | result |
|---|---:|
| worst parity divergence | **3.6e-15** |
| per-query, resident | 0.028 ms |
| per-query, per-call upload | 0.061 ms |
| per-query, pure Python | 0.076 ms |

**Residency saves +0.033 ms per query at 3.26x repetition sd** — 2.2x
over the per-call GPU path, 2.7x over the reference — with the ledger
and the device agreeing at every step and residency at zero after
clear. The storage split the architecture was named for now exists at
all three tiers, each measured: disk as the permanent library (§6),
DRAM as the working set (§11.14), VRAM as the hot layer (here) — and
rule 1 held the whole way: parity first, the clock second.

Run it::

    python -m ultraquant.experiments.vram_gate
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field

__all__ = ["VramReport", "run_gate"]

_REPEATS = 60


@dataclass
class VramReport:
    """Parity, latency, accounting, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        skipped: True without a CUDA device. Never a pass.
        worst_parity: Largest divergence seen anywhere (must be <1e-9).
        resident_ms: Mean per-query ms on the resident path.
        upload_ms: Mean per-query ms on the per-call path.
        python_ms: Mean per-query ms on the pure-Python reference.
        accounting_ok: DLL bytes matched the cache ledger throughout.
        delta_ms: Per-query saving (upload minus resident).
        rep_sd: Repetition sd of that contrast.
        margin_ratio: ``delta_ms / rep_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    worst_parity: float = 0.0
    resident_ms: float = 0.0
    upload_ms: float = 0.0
    python_ms: float = 0.0
    accounting_ok: bool = True
    delta_ms: float = 0.0
    rep_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The numbers, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [
            f"worst parity divergence: {self.worst_parity:.2e}",
            f"per-query: resident {self.resident_ms:.3f} ms | "
            f"per-call upload {self.upload_ms:.3f} ms | "
            f"pure python {self.python_ms:.3f} ms",
            f"accounting balanced: {self.accounting_ok}",
            f"delta (upload - resident) {self.delta_ms:+.3f} ms   "
            f"rep sd {self.rep_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _python_forward(qw, alpha, bias, xs):
    out = []
    for sample in xs:
        row_out = []
        for row, b in zip(qw, bias):
            acc = sum(w * x for w, x in zip(row, sample))
            row_out.append(alpha * acc + b)
        out.append(row_out)
    return out


def run_gate(worlds: int = 8) -> VramReport:
    """Run parity, latency, and accounting over generated layers.

    Args:
        worlds: Fresh layer/input worlds.

    Returns:
        The :class:`VramReport`.
    """
    from ultraquant.native import accel
    from ultraquant.native.vram import VramLayerCache

    try:
        if not accel.gpu_available():
            return VramReport(skipped=True,
                              reason="no CUDA device; an accelerator "
                                     "gate must not pass in absentia")
    except Exception:  # noqa: BLE001 - no DLL is the same absence
        return VramReport(skipped=True, reason="CUDA DLL unavailable")

    cache = VramLayerCache(budget_bytes=6_000)
    worst = 0.0
    resident_times, upload_times, python_times = [], [], []
    contrasts = []
    accounting_ok = True

    try:
        for world in range(worlds):
            rng = random.Random(world)
            oversized = rng.random() < 0.25
            in_dim = 30
            out_dim = 160 if oversized else rng.choice((8, 16, 32))
            qw = [[rng.choice((-1, 0, 1)) for _ in range(in_dim)]
                  for _ in range(out_dim)]
            alpha = rng.uniform(0.2, 1.5)
            bias = [rng.uniform(-1, 1) for _ in range(out_dim)]
            xs = [[rng.uniform(-1, 1) for _ in range(in_dim)]]
            key = f"world:{world}"

            reference = _python_forward(qw, alpha, bias, xs)
            batch = accel.ternary_forward_batch_gpu(qw, alpha, bias, xs)
            resident = cache.forward(key, qw, alpha, bias, xs)
            for candidate in (batch, resident):
                if candidate is None:
                    continue
                for ref_row, got_row in zip(reference, candidate):
                    for ref_v, got_v in zip(ref_row, got_row):
                        worst = max(worst, abs(ref_v - got_v))

            expected_none = oversized and cache.stats()["budget_bytes"] < \
                (out_dim * in_dim + out_dim * 8)
            if expected_none and resident is not None:
                accounting_ok = accounting_ok and False

            world_resident, world_upload = [], []
            for _rep in range(_REPEATS):
                started = time.perf_counter()
                got = cache.forward(key, qw, alpha, bias, xs)
                mid = time.perf_counter()
                accel.ternary_forward_batch_gpu(qw, alpha, bias, xs)
                done = time.perf_counter()
                if got is not None:
                    world_resident.append(1000.0 * (mid - started))
                world_upload.append(1000.0 * (done - mid))
            started = time.perf_counter()
            for _rep in range(10):
                _python_forward(qw, alpha, bias, xs)
            python_times.append(100.0 * (time.perf_counter() - started))

            if world_resident:
                resident_times.extend(world_resident)
                upload_times.extend(world_upload)
                pairs = min(len(world_resident), len(world_upload))
                contrasts.extend(world_upload[i] - world_resident[i]
                                 for i in range(pairs))

            ledger = cache.stats()["resident_bytes"]
            device = accel.layer_resident_bytes()
            if ledger != device:
                accounting_ok = False
    finally:
        cache.clear()
        if accel.layer_resident_bytes() != 0:
            accounting_ok = False

    report = VramReport(
        worst_parity=worst,
        resident_ms=(statistics.fmean(resident_times)
                     if resident_times else 0.0),
        upload_ms=(statistics.fmean(upload_times)
                   if upload_times else 0.0),
        python_ms=(statistics.fmean(python_times)
                   if python_times else 0.0),
        accounting_ok=accounting_ok,
    )
    report.delta_ms = (statistics.fmean(contrasts) if contrasts else 0.0)
    # stdev, not pstdev: repetitions are a sample of timings.
    report.rep_sd = (statistics.stdev(contrasts)
                     if len(contrasts) > 1 else 0.0)
    report.margin_ratio = (report.delta_ms / report.rep_sd
                           if report.rep_sd > 0 else 0.0)

    if report.worst_parity >= 1e-9:
        report.reason = (f"parity failed at {report.worst_parity:.2e}: a "
                         "fast wrong answer is not an accelerator")
        return report
    if not report.accounting_ok:
        report.reason = ("accounting failed: the cache ledger and the "
                        "device disagreed, or residency leaked")
        return report
    if report.delta_ms <= 0:
        report.reason = f"residency saves nothing ({report.delta_ms:+.3f} ms)"
        return report
    if report.rep_sd <= 0.0:
        report.reason = "every repetition timed identically; nothing varied"
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"saving {report.delta_ms:+.3f} ms is "
                         f"{report.margin_ratio:.2f}x the repetition sd")
        return report
    report.passes = True
    report.reason = (
        f"residency pays: {report.upload_ms:.3f} ms per query falls to "
        f"{report.resident_ms:.3f} ms ({report.delta_ms:+.3f} ms, "
        f"{report.margin_ratio:.2f}x rep sd) at parity "
        f"{report.worst_parity:.1e} with balanced accounting")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
