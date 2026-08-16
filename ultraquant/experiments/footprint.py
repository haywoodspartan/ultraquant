"""What ternary quantization actually saves on disk here. Measured, not asserted.

Written because a summary of this codebase claimed ternary weights are "64x
smaller than fp32", and a number like that drives real hardware decisions. It is
wrong twice over, and the second error is the interesting one.

**Ternary cannot reach 64x against fp32.** Three states need 2 bits packed
(1.58 information-theoretically), so the ceiling is 32/2 = **16x**. There is no
arrangement of {-1, 0, +1} that gets to 64x.

**And this codebase does not pack bits at all.** Shards are stored as JSON inside
zlib-compressed payloads (`shards/vault.py`), because the container has to stay
inspectable and portable. So the theoretical ceiling is not what a user gets
either. What they get is measured here.

The honest headline is **~11x smaller than packed fp32** — good, real, and
roughly a sixth of what the 64x claim promised. Sizing a machine on 64x would
under-provision disk by 6x.

Two further things worth knowing before planning around any of it:

* The saving is *not* mostly the ternary alphabet. About 46% of weights fall
  below the ``0.75 * mean|w|`` threshold and become exact zeros, and zlib
  compresses long zero runs almost for free. Sparsity is doing much of the work,
  so the ratio moves with the weight distribution rather than being a constant.
* Comparing against fp32 *stored the same way* gives a much flattering ~27x, and
  that comparison should be treated as an artifact: JSON is a terrible container
  for floats. Packed fp32 is the fair baseline and the one reported first.

Run it::

    python -m ultraquant.experiments.footprint
"""

from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass

from ultraquant.model.quantize import quantize_ternary

__all__ = ["FootprintReport", "measure"]


@dataclass
class FootprintReport:
    """Sizes for one weight matrix, in bytes unless named otherwise.

    Attributes:
        weights: How many weights were measured.
        nonzero: How many survived the ternarization threshold.
        fp32_packed: The fair baseline - 4 bytes per weight, no container.
        fp32_stored: fp32 through this project's json+zlib path, for contrast.
        ternary_packed: The theoretical 2-bits-per-weight ceiling.
        ternary_stored: What a shard actually costs on disk.
        ratio_vs_packed_fp32: The honest headline.
        ratio_vs_stored_fp32: Flattering; an artifact of JSON float formatting.
    """

    weights: int
    nonzero: int
    fp32_packed: int
    fp32_stored: int
    ternary_packed: int
    ternary_stored: int

    @property
    def ratio_vs_packed_fp32(self) -> float:
        """The number to plan disk on."""
        return self.fp32_packed / max(1, self.ternary_stored)

    @property
    def ratio_vs_stored_fp32(self) -> float:
        """Reported for contrast, and not the number to quote."""
        return self.fp32_stored / max(1, self.ternary_stored)

    @property
    def sparsity(self) -> float:
        """Fraction of weights the threshold zeroed."""
        return 1.0 - (self.nonzero / max(1, self.weights))

    def as_text(self) -> str:
        """A printable report, with the misleading comparison labelled."""
        return "\n".join([
            f"{self.weights:,} weights, {self.sparsity:.0%} zeroed by the "
            f"0.75*mean|w| threshold",
            f"  fp32, packed (fair baseline) {self.fp32_packed:>9,} B   1.00x",
            f"  ternary, theoretical 2-bit   {self.ternary_packed:>9,} B  "
            f"{self.fp32_packed / max(1, self.ternary_packed):>5.1f}x  (ceiling; "
            "not what this stores)",
            f"  ternary, as actually stored  {self.ternary_stored:>9,} B  "
            f"{self.ratio_vs_packed_fp32:>5.1f}x  <- the honest headline",
            f"  fp32, as stored (json+zlib)  {self.fp32_stored:>9,} B  "
            f"{self.ratio_vs_stored_fp32:>5.1f}x vs ternary - FLATTERING, JSON "
            "is a poor float container",
        ])


def _stored_bytes(payload) -> int:
    """Size through this project's actual shard path: JSON, then zlib."""
    return len(zlib.compress(json.dumps(payload).encode("utf-8"), 9))


def measure(rows: int = 64, cols: int = 64, seed: int = 0) -> FootprintReport:
    """Measure one Gaussian weight matrix end to end.

    Args:
        rows: Output dimension.
        cols: Input dimension.
        seed: For reproducibility; the ratio moves with the weight
            distribution, so this is not a constant of the design.

    Returns:
        The :class:`FootprintReport`.
    """
    rng = random.Random(seed)
    weights = [[rng.gauss(0.0, 0.5) for _ in range(cols)] for _ in range(rows)]
    count = rows * cols
    quantized, _alpha = quantize_ternary(weights)
    nonzero = sum(1 for row in quantized for value in row if value)
    return FootprintReport(
        weights=count,
        nonzero=nonzero,
        fp32_packed=count * 4,
        fp32_stored=_stored_bytes(weights),
        ternary_packed=(count * 2 + 7) // 8,
        ternary_stored=_stored_bytes(quantized),
    )


if __name__ == "__main__":                        # pragma: no cover
    print(measure().as_text())
