"""A black-box entropy assessor: judge any source by its output alone.

The chaos-injection experiment was reverted because its measurement was wrong.
What survives, and is worth building properly, is the *measurement* — and the
black-box framing is exactly the right one for it. An assessor that knows
nothing of a source's internals, only the bytes it emits, cannot be fooled by
a plausible story about where the randomness supposedly comes from. It would
have caught the all-zero jitter trap on the first sample, automatically,
before any of that shipped.

The interface is deliberately opaque: hand :func:`assess` a callable that
returns bytes, and get back a verdict — a credited **bits of entropy per byte**
(conservative: the minimum across several independent tests, because an
adversarial or broken source only has to pass the ones it happens to pass) and
a plain-language reason. No configuration describes the source; the tests are
universal.

The tests, each catching a failure the others miss:

* **collapse** — distinct byte values seen. The jitter trap emitted one value;
  min-entropy over the observed histogram is 0 and the verdict is 0, whatever
  else looks fine. This is the test the reverted work needed and lacked.
* **min-entropy** — ``-log2(max p)`` over the byte histogram: the entropy an
  adversary who guesses the single likeliest byte cannot beat. The
  conservative floor, and the headline number.
* **serial correlation** — successive bytes predicting each other. A counter
  or a slow-moving sensor passes a histogram test and fails this one hard.
* **compression** — zlib ratio. A source with hidden structure a histogram
  cannot see (a repeating block, a low-order pattern) compresses, and the
  credited entropy is capped at the compressed density.
* **monobit** — the bit-level balance a whitener fakes perfectly; kept only as
  a *sanity* signal, never as a credit, precisely because it is the one a
  laundered constant sails through. (The lesson from the revert, encoded: a
  test that whitening passes cannot be trusted to credit entropy.)

The credited figure is the **minimum** of the byte-level tests, so a source is
only as trusted as its weakest measured property. A source claiming to be
random is guilty until its output proves otherwise — the sibling of
found-is-not-believed, applied to entropy.

Pure Python standard library.
"""

from __future__ import annotations

import collections
import math
import zlib
from dataclasses import dataclass, field
from typing import Callable

from ultraquant.quantum.symbols import min_entropy, shannon

__all__ = ["EntropyVerdict", "assess", "assess_bytes"]


@dataclass
class EntropyVerdict:
    """What the black box concluded about a source.

    Attributes:
        credited_bits_per_byte: The conservative entropy estimate — the minimum
            across the byte-level tests. 0.0 means "do not trust this as
            random".
        trustworthy: Whether the credit clears a usable floor (>= 4 bits/byte,
            half of ideal). A source below this may still be *mixed* with a
            strong one, but must never be relied on alone.
        reason: Plain-language account of what limited the credit.
        tests: Every test's raw score, for inspection.
        n_bytes: How many bytes were assessed.
    """

    credited_bits_per_byte: float
    trustworthy: bool
    reason: str
    tests: dict = field(default_factory=dict)
    n_bytes: int = 0

    def as_dict(self) -> dict:
        """JSON-safe summary."""
        return {
            "credited_bits_per_byte": round(self.credited_bits_per_byte, 4),
            "trustworthy": self.trustworthy,
            "reason": self.reason,
            "tests": {k: round(v, 4) for k, v in self.tests.items()},
            "n_bytes": self.n_bytes,
        }


#: A source is usable alone only above this credited entropy (half of ideal 8).
_TRUST_FLOOR = 4.0


def _serial_bits_per_byte(data: bytes) -> float:
    """Entropy discounted by how well each byte predicts the next.

    Lag-1 correlation of the byte stream; a source whose bytes march (a
    counter, a warming sensor) scores high correlation and is discounted
    toward zero. Independent bytes score ~0 correlation and keep full credit.
    """
    if len(data) < 3:
        return 0.0
    values = list(data)
    mean = sum(values) / len(values)
    num = sum((values[i] - mean) * (values[i + 1] - mean)
              for i in range(len(values) - 1))
    den = sum((v - mean) ** 2 for v in values)
    if den == 0.0:
        return 0.0  # a constant: no variance, no entropy
    correlation = abs(num / den)
    # Full histogram entropy scaled by (1 - |r|): perfectly correlated -> 0.
    histogram = collections.Counter(data)
    return shannon(list(histogram.values())) * (1.0 - min(1.0, correlation))


def _compression_bits_per_byte(data: bytes) -> float:
    """Bits per byte the best cheap compressor cannot remove.

    A histogram is blind to sequential structure — ``abababab`` is uniform by
    byte frequency yet trivially compressible. zlib is a fast, universal
    catcher of that structure, and the compressed size in bits per input byte
    caps the credit at what genuinely could not be predicted away.
    """
    if not data:
        return 0.0
    compressed = zlib.compress(bytes(data), 9)
    # zlib has ~a few bytes of unavoidable header; discount it for small inputs.
    payload = max(0, len(compressed) - 8)
    return min(8.0, 8.0 * payload / len(data))


def _monobit_balance(data: bytes) -> float:
    """Fraction of set bits — a sanity signal, never a credit.

    Kept because a wildly unbalanced stream is worth *reporting*, but excluded
    from the credited minimum on purpose: this is precisely the test a whitener
    passes while laundering a constant, so trusting it would reintroduce the
    exact hole the reverted work fell into.
    """
    ones = sum(bin(byte).count("1") for byte in data)
    return ones / (len(data) * 8) if data else 0.0


def assess_bytes(data: bytes) -> EntropyVerdict:
    """Judge a fixed sample of bytes.

    Returns:
        The :class:`EntropyVerdict`. Credit is the minimum of the byte-level
        tests (collapse, min-entropy, serial, compression); monobit is
        reported but never credited.
    """
    if not data:
        return EntropyVerdict(0.0, False, "no bytes to assess", {}, 0)

    histogram = collections.Counter(data)
    distinct = len(histogram)
    min_ent = min_entropy(list(histogram.values()))
    serial = _serial_bits_per_byte(data)
    compression = _compression_bits_per_byte(data)
    monobit = _monobit_balance(data)

    tests = {
        "distinct_values": float(distinct),
        "min_entropy_per_byte": min_ent,
        "serial_bits_per_byte": serial,
        "compression_bits_per_byte": compression,
        "monobit_ones_fraction": monobit,
    }

    # The collapse guard is categorical: one or two distinct values is not a
    # weak source, it is not a source. This is the all-zero jitter trap's
    # natural death.
    if distinct <= 2:
        return EntropyVerdict(
            0.0, False,
            f"collapsed to {distinct} distinct byte value(s) - not random",
            tests, len(data),
        )

    credited = min(min_ent, serial, compression)
    floors = {
        "min-entropy (histogram is peaked)": min_ent,
        "serial correlation (bytes predict each other)": serial,
        "compressibility (hidden sequential structure)": compression,
    }
    limiter = min(floors, key=floors.get)
    reason = (f"credited {credited:.2f} bits/byte; limited by {limiter}"
              if credited < 7.5 else
              f"credited {credited:.2f} bits/byte; strong on every test")
    return EntropyVerdict(credited, credited >= _TRUST_FLOOR, reason,
                          tests, len(data))


def assess(source: Callable[[int], bytes], sample_bytes: int = 4096,
           draws: int = 1) -> EntropyVerdict:
    """Judge a source callable by drawing from it — the black box.

    Args:
        source: ``f(n) -> bytes`` returning at least ``n`` bytes. Its internals
            are irrelevant and never inspected; only its output is judged.
        sample_bytes: Total bytes to assess.
        draws: How many separate draws to concatenate. More draws catch a
            source that is fine within a call but repeats *across* calls — the
            cross-call collapse a single big draw would miss.

    Returns:
        The verdict over the concatenated sample.
    """
    per_draw = max(1, sample_bytes // max(1, draws))
    chunks = []
    for _ in range(draws):
        chunk = bytes(source(per_draw))[:per_draw]
        chunks.append(chunk)
    data = b"".join(chunks)

    verdict = assess_bytes(data)
    # A source that is internally fine but returns the *same* bytes every call
    # is caught here: identical draws compress to nearly nothing.
    if draws > 1 and len(set(chunks)) == 1 and len(chunks[0]) > 0:
        return EntropyVerdict(
            0.0, False,
            f"identical output across {draws} draws - deterministic, not random",
            verdict.tests, len(data),
        )
    return verdict
