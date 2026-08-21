"""A glyph for every imported tensor.

A packed tensor arrives in the library addressed three ways: its
shard id names it exactly, its category names its layer, and its
name tokens name its kind. All three are things somebody already
knew how to type. None of them says anything about what is actually
IN the tensor.

This adds a fourth address that does: a 5x5 glyph rendered from the
weights themselves. The system already has a recognizer for exactly
that shape - eight named classes, a trained classifier, and a
memory that logs what it saw - so a tensor that carries a glyph can
be named by the machinery that was built for glyphs, and that name
becomes another way into the catalog.

**What a cell measures, and why that one.** Each cell is the density
of surviving trits in its block. For a ternary tensor that is the
structural quantity: the threshold already discarded everything
small, so where the survivors CLUSTER is where the tensor keeps its
weight. A cell is set when its block is denser than the tensor's own
mean, which removes overall sparsity from the picture - otherwise a
uniformly sparse tensor renders blank and a dense one renders solid,
and both would carry exactly nothing.

A tensor with no structure at all - every block equally dense -
renders blank, and that is the honest glyph for it rather than a
tie-break dressed up as a pattern.

**What this is NOT for.** Measured on 2-D tensors against the
library's own 64-bit sketch on identical input, the glyph reaches
0.19 of the one-sd separation its gate asked for and the sketch
reaches -0.04: **neither identifies a tensor's kind.** That is not a
glyph problem but a summary problem - block density over a ternary
matrix does not tell an attention projection from a feed-forward
one. So nothing routes by glyph distance, and nothing should route
by sketch distance over these summaries either. The glyph is kept
for the job it does do: being rendered once, looked at, named by the
recognizer, and retrieved by that name. A signature you can read and
a signature you can screen with are different jobs, and this is
only the first.

**Deterministic by construction.** Integer counts, a mean, and a
strict comparison; no sampling, no seed, no floating-point
accumulation order that could differ between machines. The same
tensor renders the same glyph anywhere, which is the property a
stored signature has to have.
"""

from __future__ import annotations

__all__ = ["GLYPH_SIDE", "densities_of", "glyph_of", "glyph_bits",
           "hamming", "as_flat"]

#: The recognizer's grid. Not a parameter: PATTERNS is 5x5, and a
#: signature the recognizer cannot read is not a glyph.
GLYPH_SIDE = 5


def _blocks(count: int) -> list[tuple[int, int]]:
    """Split ``count`` positions into GLYPH_SIDE contiguous ranges.

    Integer arithmetic so the split is identical everywhere. Short
    axes give empty blocks rather than an error - a 3-wide tensor has
    two columns of the glyph with nothing in them, which is true.
    """
    edges = [(count * i) // GLYPH_SIDE for i in range(GLYPH_SIDE + 1)]
    return [(edges[i], edges[i + 1]) for i in range(GLYPH_SIDE)]


def densities_of(quantised: list[list[int]]) -> list[list[float]]:
    """The 5x5 grid of surviving-trit densities, before thresholding.

    Split out so the same summary can be spelled two ways and the
    spellings compared fairly. The glyph thresholds it into 25 bits;
    the library's existing sketch projects it into 64. Handing the
    two different inputs would make whichever one got more data look
    better for a reason that has nothing to do with the spelling.
    """
    if not quantised or not quantised[0]:
        return [[0.0] * GLYPH_SIDE for _ in range(GLYPH_SIDE)]
    row_blocks = _blocks(len(quantised))
    col_blocks = _blocks(len(quantised[0]))
    density: list[list[float]] = []
    for top, bottom in row_blocks:
        line: list[float] = []
        for left, right in col_blocks:
            cells = (bottom - top) * (right - left)
            if cells <= 0:
                line.append(0.0)
                continue
            alive = 0
            for row in quantised[top:bottom]:
                for value in row[left:right]:
                    if value:
                        alive += 1
            line.append(alive / cells)
        density.append(line)
    return density


def glyph_of(quantised: list[list[int]]) -> list[str]:
    """A 5x5 glyph rendered from a ternary matrix.

    Args:
        quantised: Ternary rows, entries in {-1, 0, +1}.

    Returns:
        Five strings of five characters, ``#`` and ``.``, in the
        form :mod:`ultraquant.pattern.recognition` reads.
    """
    if not quantised or not quantised[0]:
        return ["." * GLYPH_SIDE] * GLYPH_SIDE
    density = densities_of(quantised)
    flat = [value for line in density for value in line]
    mean = sum(flat) / len(flat) if flat else 0.0
    return ["".join("#" if value > mean else "." for value in line)
            for line in density]


def glyph_bits(rows: list[str]) -> int:
    """A glyph as 25 bits, for cheap distance work."""
    bits = 0
    for index, char in enumerate(char for row in rows for char in row):
        if char == "#":
            bits |= 1 << index
    return bits


def hamming(left: int, right: int) -> int:
    """Differing cells between two glyphs."""
    return (left ^ right).bit_count()


def as_flat(rows: list[str]) -> list[float]:
    """The glyph as the 25 floats the recognizer takes."""
    return [1.0 if char == "#" else 0.0 for row in rows for char in row]
