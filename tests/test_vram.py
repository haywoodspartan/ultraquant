"""The VRAM tier: residency logic without a GPU, parity with one.

Cache behavior runs under a fake accel so the suite needs no device;
the hardware truths (bit parity, balanced accounting) run only when a
CUDA device exists and skip cleanly otherwise.
"""

from __future__ import annotations

import unittest

from ultraquant.native.vram import VramLayerCache


class _FakeAccel:
    """Slot bookkeeping without a device."""

    def __init__(self, slots: int = 4) -> None:
        self.slots = slots
        self.used: dict[int, int] = {}
        self.uploads = 0
        self.forwards = 0

    def gpu_available(self) -> bool:
        return True

    def layer_upload(self, qw, alpha, bias, in_dim, out_dim) -> int:
        for slot in range(self.slots):
            if slot not in self.used:
                self.used[slot] = out_dim * in_dim
                self.uploads += 1
                return slot
        return -1

    def layer_forward(self, slot, xs, in_dim, out_dim):
        self.forwards += 1
        return [[0.0] * out_dim for _ in xs]

    def layer_free(self, slot) -> None:
        self.used.pop(slot, None)

    def layer_resident_bytes(self) -> int:
        return sum(self.used.values())


def _layer(out_dim: int = 4, in_dim: int = 6):
    qw = [[1] * in_dim for _ in range(out_dim)]
    return qw, 1.0, [0.0] * out_dim, [[0.5] * in_dim]


class CacheLogicTests(unittest.TestCase):
    """LRU, budget, refusal, fallback - no device required."""

    def test_a_hit_reuses_the_slot(self) -> None:
        accel = _FakeAccel()
        cache = VramLayerCache(budget_bytes=1000, accel=accel)
        qw, alpha, bias, xs = _layer()
        cache.forward("a", qw, alpha, bias, xs)
        cache.forward("a", qw, alpha, bias, xs)
        self.assertEqual(accel.uploads, 1)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_the_budget_evicts_least_recently_used(self) -> None:
        accel = _FakeAccel()
        qw, alpha, bias, xs = _layer(out_dim=4, in_dim=6)  # 56 bytes
        cache = VramLayerCache(budget_bytes=120, accel=accel)
        cache.forward("a", qw, alpha, bias, xs)
        cache.forward("b", qw, alpha, bias, xs)
        cache.forward("c", qw, alpha, bias, xs)
        stats = cache.stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertNotIn("a", stats["resident"])
        self.assertIn("c", stats["resident"])

    def test_an_oversized_layer_refuses(self) -> None:
        """The first gate run caught this uploading anyway: eviction
        empties the cache and then has nothing left to give."""
        accel = _FakeAccel()
        cache = VramLayerCache(budget_bytes=50, accel=accel)
        qw, alpha, bias, xs = _layer(out_dim=8, in_dim=30)
        self.assertIsNone(cache.forward("big", qw, alpha, bias, xs))
        self.assertEqual(accel.uploads, 0)

    def test_no_gpu_means_none_and_no_crash(self) -> None:
        class _Dead:
            def gpu_available(self) -> bool:
                return False

        cache = VramLayerCache(accel=_Dead())
        qw, alpha, bias, xs = _layer()
        self.assertIsNone(cache.forward("a", qw, alpha, bias, xs))

    def test_clear_frees_everything(self) -> None:
        accel = _FakeAccel()
        cache = VramLayerCache(budget_bytes=1000, accel=accel)
        qw, alpha, bias, xs = _layer()
        cache.forward("a", qw, alpha, bias, xs)
        cache.forward("b", qw, alpha, bias, xs)
        cache.clear()
        self.assertEqual(accel.layer_resident_bytes(), 0)
        self.assertEqual(cache.stats()["resident"], [])


def _gpu_ready() -> bool:
    try:
        from ultraquant.native import accel

        return bool(accel.gpu_available())
    except Exception:  # noqa: BLE001 - no DLL, no device
        return False


@unittest.skipUnless(_gpu_ready(), "no CUDA device")
class HardwareParityTests(unittest.TestCase):
    """Rule 1 on the real device: the simulator is the reference."""

    def test_resident_forward_matches_the_reference(self) -> None:
        import random

        from ultraquant.native import accel

        rng = random.Random(7)
        qw = [[rng.choice((-1, 0, 1)) for _ in range(30)]
              for _ in range(16)]
        bias = [rng.uniform(-1, 1) for _ in range(16)]
        xs = [[rng.uniform(-1, 1) for _ in range(30)] for _ in range(3)]
        slot = accel.layer_upload(qw, 0.9, bias, 30, 16)
        self.assertGreaterEqual(slot, 0)
        try:
            got = accel.layer_forward(slot, xs, 30, 16)
            for sample, got_row in zip(xs, got):
                for row, b, value in zip(qw, bias, got_row):
                    ref = 0.9 * sum(w * x for w, x in zip(row, sample)) + b
                    self.assertLess(abs(ref - value), 1e-9)
        finally:
            accel.layer_free(slot)

    def test_free_returns_the_bytes(self) -> None:
        from ultraquant.native import accel

        before = accel.layer_resident_bytes()
        slot = accel.layer_upload([[1, 0, -1]], 1.0, [0.0], 3, 1)
        self.assertGreaterEqual(slot, 0)
        self.assertGreater(accel.layer_resident_bytes(), before)
        accel.layer_free(slot)
        self.assertEqual(accel.layer_resident_bytes(), before)


class GateVerdictTests(unittest.TestCase):
    """The PASS, the caught bug, and the three-tier story, pinned."""

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        from ultraquant.experiments import vram_gate

        doc = " ".join(vram_gate.__doc__.split())
        self.assertIn("then PASSED", doc)
        self.assertIn("3.26x repetition sd", doc)

    def test_the_oversized_bug_is_recorded(self) -> None:
        from ultraquant.experiments import vram_gate

        doc = " ".join(vram_gate.__doc__.split())
        self.assertIn("the fallback is the mechanism, not an apology", doc)

    def test_the_three_tier_story_is_recorded(self) -> None:
        from ultraquant.experiments import vram_gate

        doc = " ".join(vram_gate.__doc__.split())
        self.assertIn("all three tiers", doc)
        self.assertIn("parity first, the clock second", doc)


if __name__ == "__main__":
    unittest.main()
