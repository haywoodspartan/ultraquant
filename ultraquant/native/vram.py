"""The VRAM tier: hot expert layers resident on the device.

The vision's remaining half, finally mechanical: the library lives on
disk, the working set lives in RAM (§11.14's context window, the shard
cache), and the HOT layers — the experts a conversation keeps touching —
live in VRAM, weights crossing the bus once and only activations moving
per query. `uqg_layer_upload/forward/free` (§11.45's DLL extension) hold
a fixed slot table on the device; this cache decides what deserves a
slot, byte-budgeted and LRU like every other tier in the system.

Fallback is structural: no GPU, no slots left, upload refused — every
path returns None and the caller runs the tier below, bit-identical
(measured: resident, per-call-GPU, and pure Python agree to 0.0e+00).
The simulator stays the reference; VRAM is an accelerant, never
semantics — rule 1, unchanged since the first line of the book.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

__all__ = ["VramLayerCache"]


class VramLayerCache:
    """LRU cache of ternary layers resident in device memory.

    Args:
        budget_bytes: Ceiling on resident weight bytes (ternary weights
            count one byte each, biases eight).
        accel: The accel module, injectable for tests.
    """

    def __init__(self, budget_bytes: int = 8 * 1024 * 1024,
                 accel: Any = None) -> None:
        if accel is None:
            from ultraquant.native import accel as accel_module

            accel = accel_module
        self.accel = accel
        self.budget_bytes = int(budget_bytes)
        #: key -> (slot, nbytes, in_dim, out_dim)
        self._slots: OrderedDict[str, tuple[int, int, int, int]] = \
            OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _layer_bytes(in_dim: int, out_dim: int, has_bias: bool) -> int:
        return out_dim * in_dim + (out_dim * 8 if has_bias else 0)

    def _evict_until(self, needed: int) -> None:
        used = sum(nbytes for _s, nbytes, _i, _o in self._slots.values())
        while self._slots and used + needed > self.budget_bytes:
            _key, (slot, nbytes, _i, _o) = self._slots.popitem(last=False)
            self.accel.layer_free(slot)
            used -= nbytes
            self.evictions += 1

    # ------------------------------------------------------------ the work

    def forward(self, key: str, qw, alpha: float, bias, xs) -> list | None:
        """Run ``xs`` through the layer, resident if it can be.

        Args:
            key: Stable identity for the layer (shard id + layer index).
            qw: Ternary weight rows.
            alpha: Dequantisation scale.
            bias: Bias vector or None.
            xs: Input batch (list of feature lists).

        Returns:
            The output batch, or None when no GPU path is available -
            the caller falls back to the tier below.
        """
        try:
            if not self.accel.gpu_available():
                return None
        except Exception:  # noqa: BLE001 - no GPU is a normal state
            return None
        in_dim = len(qw[0]) if qw else 0
        out_dim = len(qw)
        if not in_dim or not out_dim:
            return None

        entry = self._slots.get(key)
        if entry is not None:
            self._slots.move_to_end(key)
            self.hits += 1
            slot, _nbytes, in_dim, out_dim = entry
        else:
            self.misses += 1
            nbytes = self._layer_bytes(in_dim, out_dim, bias is not None)
            if nbytes > self.budget_bytes:
                # A layer larger than the whole budget never becomes
                # resident: eviction empties the cache and then has
                # nothing left to give, and the first gate run caught
                # exactly this shape uploading anyway. Refuse; the
                # caller's fallback runs the tier below.
                return None
            self._evict_until(nbytes)
            slot = self.accel.layer_upload(qw, alpha, bias, in_dim,
                                           out_dim)
            if slot < 0:
                return None
            self._slots[key] = (slot, nbytes, in_dim, out_dim)
        return self.accel.layer_forward(slot, xs, in_dim, out_dim)

    def forward_lazy(self, key: str, provider, xs) -> list | None:
        """Like :meth:`forward`, materialising weights only on a miss.

        A frozen expert requantises its float masters on every Python
        forward; on a VRAM hit none of that work is needed - the device
        already holds (q, alpha, bias). ``provider`` is a zero-argument
        callable returning ``(qw, alpha, bias)``, called only when the
        layer is not resident.
        """
        try:
            if not self.accel.gpu_available():
                return None
        except Exception:  # noqa: BLE001 - no GPU is a normal state
            return None
        entry = self._slots.get(key)
        if entry is not None:
            self._slots.move_to_end(key)
            self.hits += 1
            slot, _nbytes, in_dim, out_dim = entry
            return self.accel.layer_forward(slot, xs, in_dim, out_dim)
        qw, alpha, bias = provider()
        return self.forward(key, qw, alpha, bias, xs)

    def drop(self, key: str) -> None:
        """Release one layer, if resident."""
        entry = self._slots.pop(key, None)
        if entry is not None:
            self.accel.layer_free(entry[0])

    def drop_prefix(self, prefix: str) -> None:
        """Release every layer whose key starts with ``prefix``.

        The invalidation retraining requires: a device copy is valid
        only while the expert is frozen, and anything that rewrites an
        expert's weights must take its resident layers down with it.
        """
        for key in [k for k in self._slots if k.startswith(prefix)]:
            self.drop(key)

    def clear(self) -> None:
        """Release everything."""
        for slot, _n, _i, _o in self._slots.values():
            self.accel.layer_free(slot)
        self._slots.clear()

    def __del__(self) -> None:
        # The slot table is a process-wide resource (64 slots in the
        # DLL); a cache dying with its session must hand them back, or
        # the table fills with orphans and every later session falls
        # back forever. The full suite caught exactly that: hundreds of
        # sessions, then slot -1 for whoever asked next.
        try:
            self.clear()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass

    def stats(self) -> dict:
        """Residency accounting, for ':resident' and the gate."""
        return {
            "resident": list(self._slots),
            "resident_bytes": sum(n for _s, n, _i, _o
                                  in self._slots.values()),
            "budget_bytes": self.budget_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
