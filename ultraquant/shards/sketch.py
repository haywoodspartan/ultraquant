"""Sign-random-projection sketches: cheap cosine screening for pattern routing.

Content-based routing compares an input pattern against every category's stored
prototypes ([§4.1.1]). That is exact and it is linear: measured at 20,000
categories with three prototypes each, one route cost **128 ms** and touched
1.5 million multiply-accumulates in pure Python, against **7.6 us** for the
keyword path which has an inverted index behind it. Restoring an O(library) scan
to the one place it had been eliminated is not acceptable at the 1.2-trillion
parameter design point.

The fix is the standard one for cosine similarity. Project each vector onto a
fixed set of random hyperplanes and keep only the *signs* as a bit string. For
two vectors separated by angle ``theta``, a random hyperplane separates them with
probability ``theta / pi``, so

    P[bit agrees] = 1 - theta / pi = 1 - arccos(cos_sim) / pi

and the fraction of agreeing bits estimates the angle directly. Comparing two
64-bit sketches is one XOR and one ``bit_count`` — a single machine instruction
each on CPython's fast path — instead of 25 float multiplies.

This is a **screen, not an answer**. Sketches rank candidates approximately; the
survivors are then scored exactly, so the similarity a caller sees is always the
true cosine. The only error the screen can make is dropping a genuine match
before it is scored, which is why the candidate pool is generous and why recall
is measured rather than assumed (see ``tests/test_sketch.py``).

Pure standard library: the hyperplanes come from a seeded ``random.Random``, so
sketches are byte-identical across runs and machines and can be persisted.
"""

from __future__ import annotations

import random

__all__ = ["SKETCH_BITS", "hyperplanes", "sketch", "hamming", "estimate_cosine"]

#: Bits per sketch. 64 keeps a sketch in one machine word.
SKETCH_BITS = 64

#: Fixed seed for the projections. Changing it invalidates every stored sketch,
#: so it is a constant rather than a parameter.
_SEED = 0x5157_4C42

#: dim -> list of ``SKETCH_BITS`` unit-ish random vectors.
_PLANES: dict[int, list[list[float]]] = {}


def hyperplanes(dim: int, bits: int = SKETCH_BITS) -> list[list[float]]:
    """Return the fixed random hyperplanes for ``dim``-dimensional vectors.

    Generated once per dimension and cached. Deterministic: the same dimension
    always yields the same planes, on any machine, so a sketch computed today
    still screens correctly against one computed last month.

    Args:
        dim: Vector dimension.
        bits: How many hyperplanes (one bit each).

    Returns:
        ``bits`` vectors of length ``dim``.
    """
    cached = _PLANES.get(dim)
    if cached is not None and len(cached) >= bits:
        return cached
    rng = random.Random(_SEED ^ (dim * 0x9E3779B1))
    planes = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(bits)]
    _PLANES[dim] = planes
    return planes


def sketch(vector: list[float], bits: int = SKETCH_BITS) -> int:
    """Sketch ``vector`` into a ``bits``-wide integer of projection signs.

    Args:
        vector: The vector to sketch.
        bits: Sketch width.

    Returns:
        An integer whose bit *i* is 1 when the vector falls on the positive side
        of hyperplane *i*. ``0`` for an empty vector.
    """
    if not vector:
        return 0
    planes = hyperplanes(len(vector), bits)
    out = 0
    for index in range(bits):
        plane = planes[index]
        total = 0.0
        for position, value in enumerate(vector):
            total += value * plane[position]
        if total > 0.0:
            out |= 1 << index
    return out


def hamming(left: int, right: int) -> int:
    """Number of differing bits between two sketches."""
    return (left ^ right).bit_count()


def estimate_cosine(distance: int, bits: int = SKETCH_BITS) -> float:
    """Estimate cosine similarity from a sketch Hamming distance.

    Inverts ``P[bit differs] = arccos(sim) / pi``. Only accurate to about
    ``1 / sqrt(bits)``, which is why it is used for ranking and never reported
    as the similarity itself.

    Args:
        distance: Hamming distance between two sketches.
        bits: Sketch width.

    Returns:
        The estimated cosine similarity in ``[-1, 1]``.
    """
    import math

    return math.cos(math.pi * distance / max(1, bits))
