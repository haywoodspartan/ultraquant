"""The black-box entropy assessor: judge a source by its output alone.

Built after the chaos-injection experiment was reverted for a wrong
measurement. The framing is the point: an assessor that knows nothing of a
source's internals cannot be talked into trusting one, and this one would have
caught the all-zero jitter trap on its first sample.

The design claim is that each test catches a failure the others miss, and
these tests *demonstrate* it rather than asserting it — the table each case
below encodes was measured:

    source            min-entropy  serial  compression   caught by
    counter                  8.00    0.18         0.60   serial only
    slow ramp                8.00    0.01         0.81   serial only
    whitened zeros           4.42    4.68         0.20   compression only
    biased 90% zero          0.15    1.11         1.46   min-entropy only
    os.urandom               7.09    7.88         8.00   passes all

The whitened-zeros row is the one that matters: a hashed constant passes both
the histogram and the correlation test at ~4.5 bits/byte. Only compression
sees through it. That is the exact laundering that made the reverted work look
sound.
"""

from __future__ import annotations

import hashlib
import os
import random
import unittest

from ultraquant.quantum.entropy_blackbox import assess, assess_bytes

_N = 4096


def _whitened_constant(n: int = _N) -> bytes:
    """A constant laundered through BLAKE2 - the trap, reproduced."""
    return (hashlib.blake2b(bytes(64), digest_size=64).digest() * (n // 64 + 1))[:n]


class RealEntropyTests(unittest.TestCase):
    """A genuine source must be credited."""

    def test_os_urandom_is_trusted(self) -> None:
        verdict = assess_bytes(os.urandom(_N))
        self.assertGreater(verdict.credited_bits_per_byte, 6.5)
        self.assertTrue(verdict.trustworthy)

    def test_a_strong_source_passes_every_test(self) -> None:
        tests = assess_bytes(os.urandom(_N)).tests
        self.assertGreater(tests["min_entropy_per_byte"], 6.0)
        self.assertGreater(tests["serial_bits_per_byte"], 6.0)
        self.assertGreater(tests["compression_bits_per_byte"], 7.0)


class EachTestEarnsItsPlaceTests(unittest.TestCase):
    """Every test catches a failure that at least one other test misses."""

    def test_the_collapse_guard_kills_a_constant(self) -> None:
        """The all-zero jitter trap's natural death, in one line."""
        verdict = assess_bytes(bytes(_N))
        self.assertEqual(verdict.credited_bits_per_byte, 0.0)
        self.assertFalse(verdict.trustworthy)
        self.assertIn("collapsed", verdict.reason)

    def test_serial_correlation_catches_a_perfect_histogram(self) -> None:
        """A counter has *ideal* min-entropy (8.00) and no randomness at all.

        A histogram-only assessor would credit it the full 8 bits/byte.
        """
        counter = bytes(i % 256 for i in range(_N))
        verdict = assess_bytes(counter)
        self.assertGreater(verdict.tests["min_entropy_per_byte"], 7.9,
                           "the counter should look perfect to a histogram")
        self.assertLess(verdict.tests["serial_bits_per_byte"], 1.0)
        self.assertLess(verdict.credited_bits_per_byte, 1.0)

    def test_serial_correlation_catches_a_slow_sensor(self) -> None:
        ramp = bytes((i // 16) % 256 for i in range(_N))
        verdict = assess_bytes(ramp)
        self.assertGreater(verdict.tests["min_entropy_per_byte"], 7.9)
        self.assertLess(verdict.credited_bits_per_byte, 0.5)

    def test_compression_catches_the_laundered_constant(self) -> None:
        """The trap that fooled the reverted work, and the only test that sees it.

        Hashing a constant produces output that passes the histogram test
        (4.42 bits/byte) and the correlation test (4.68) - both plausible.
        Only compressibility reveals that the whole stream is one repeated
        digest.
        """
        verdict = assess_bytes(_whitened_constant())
        self.assertGreater(verdict.tests["min_entropy_per_byte"], 3.0,
                           "laundering should fool the histogram - that is why "
                           "the histogram cannot be trusted alone")
        self.assertGreater(verdict.tests["serial_bits_per_byte"], 3.0,
                           "and it should fool the correlation test too")
        self.assertLess(verdict.tests["compression_bits_per_byte"], 1.0,
                        "compression is the test that must catch it")
        self.assertLess(verdict.credited_bits_per_byte, 1.0)
        self.assertFalse(verdict.trustworthy)

    def test_min_entropy_catches_a_biased_source(self) -> None:
        """A 90%-zero stream is well-spread and uncorrelated, but guessable."""
        rng = random.Random(0)
        biased = bytes(0 if i % 10 else rng.randrange(1, 256) for i in range(_N))
        verdict = assess_bytes(biased)
        self.assertLess(verdict.tests["min_entropy_per_byte"], 1.0)
        self.assertGreater(verdict.tests["compression_bits_per_byte"], 1.0,
                           "compression is more generous here than min-entropy")
        self.assertLess(verdict.credited_bits_per_byte, 1.0)

    def test_credit_is_the_weakest_measured_property(self) -> None:
        """A source is only as trusted as its worst test - never its best."""
        for data in (bytes(i % 256 for i in range(_N)), _whitened_constant()):
            with self.subTest(data=data[:8]):
                verdict = assess_bytes(data)
                floor = min(verdict.tests["min_entropy_per_byte"],
                            verdict.tests["serial_bits_per_byte"],
                            verdict.tests["compression_bits_per_byte"])
                self.assertAlmostEqual(verdict.credited_bits_per_byte, floor,
                                       places=9)


class MonobitIsNeverCreditTests(unittest.TestCase):
    """The lesson from the revert, encoded as a test."""

    def test_monobit_is_reported_but_never_credited(self) -> None:
        """A laundered constant has textbook bit balance and zero entropy.

        Crediting the test that whitening passes is exactly how the reverted
        work convinced itself; monobit is therefore informational only.
        """
        verdict = assess_bytes(_whitened_constant())
        balance = verdict.tests["monobit_ones_fraction"]
        self.assertGreater(balance, 0.45)
        self.assertLess(balance, 0.55)
        self.assertLess(verdict.credited_bits_per_byte, 1.0,
                        "perfect bit balance must not buy any credit")


class BlackBoxInterfaceTests(unittest.TestCase):
    """Sources are callables; nothing about their internals is consulted."""

    def test_a_source_is_judged_only_by_what_it_emits(self) -> None:
        verdict = assess(lambda n: os.urandom(n), sample_bytes=2048)
        self.assertTrue(verdict.trustworthy)

    def test_cross_draw_repetition_is_caught(self) -> None:
        """A source fine within a call but identical across calls is not random."""
        fixed = os.urandom(512)
        verdict = assess(lambda n: fixed[:n], sample_bytes=2048, draws=4)
        self.assertEqual(verdict.credited_bits_per_byte, 0.0)
        self.assertIn("identical output", verdict.reason)

    def test_a_short_sample_is_survivable(self) -> None:
        verdict = assess(lambda n: os.urandom(n), sample_bytes=32)
        self.assertGreaterEqual(verdict.credited_bits_per_byte, 0.0)

    def test_no_bytes_is_a_verdict_not_a_crash(self) -> None:
        verdict = assess_bytes(b"")
        self.assertEqual(verdict.credited_bits_per_byte, 0.0)
        self.assertFalse(verdict.trustworthy)

    def test_the_verdict_is_json_safe(self) -> None:
        import json

        payload = assess_bytes(os.urandom(1024)).as_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)


class RealHardwareTests(unittest.TestCase):
    """The machine's own timing wobble, judged honestly."""

    def test_timing_jitter_is_measured_not_assumed(self) -> None:
        """Measured ~1.2 bits/byte here: real, weak, and correctly refused as
        a standalone source. The assessor's job is to say so."""
        import time

        def jitter(n: int) -> bytes:
            deltas = []
            for _ in range(n):
                start = time.perf_counter_ns()
                churn: dict[str, list[int]] = {}
                for i in range(60):
                    churn[str(i)] = [i] * 7
                deltas.append(time.perf_counter_ns() - start)
            floor = min(deltas)
            return bytes(((d - floor) // 100) & 0xFF for d in deltas)

        verdict = assess(jitter, sample_bytes=2048, draws=2)
        self.assertLess(verdict.credited_bits_per_byte, 6.0,
                        "timing jitter must not be credited as a strong source")
        self.assertFalse(verdict.trustworthy)


if __name__ == "__main__":
    unittest.main()
