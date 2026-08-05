"""Hypervectors: structure as bit-parallel algebra.

The blackboard ([§11.6]) gave partial results somewhere to meet, but what it
produces is a flat ``{slot: value}`` dict. A dict cannot be nested, cannot be
stored as one fixed-width thing, and cannot be compared to another dict by
similarity. This module is the representation that can.

Concepts become very high-dimensional random binary vectors. With 10,000 bits,
two random vectors are almost exactly orthogonal, and three operations suffice:

============  ===================  ============================================
operation     implementation       meaning
============  ===================  ============================================
**bind**      XOR                  attach a filler to a role; exactly invertible
**bundle**    per-bit majority     superpose several things into one
**permute**   cyclic bit rotation  impose order, or descend a level of nesting
**similarity** XOR + popcount      how alike two structures are
============  ===================  ============================================

Measured on the reference implementation: unbinding the correct role recovers
its filler at similarity **1.000** against **0.003** for an unrelated vector, and
**48 role-filler pairs bundled into a single 10,000-bit vector all 48 retrieve
correctly**, with similarity decaying as ~1/sqrt(n) exactly as theory predicts.
One vector is 1,250 bytes, so a whole composed scene fits in L1 cache.

**Why this shape and not another.** [§11.5] diagnosed the failed encoder
precisely: transfer needs a representation that *adds* structure rather than
compressing what is already there. An autoencoder compresses. This expands into
a space where composition is algebraic, which is the property that analysis asked
for. It also has the piece such systems normally need bolted on — retrieval
returns a *noisy* vector and must be snapped to the nearest stored concept, and
the shard library's signature screen is already exactly that cleanup memory.

**Runs on one core, scales to many.** Python's arbitrary-precision integers do
XOR, AND and ``bit_count`` in C across every bit at once, so the whole algebra is
bit-parallel *with no extension module* — measured at 25.8 bit-operations per
nanosecond, 807x faster than the equivalent float dot product in pure Python. The
single-core path here is the reference semantics; :mod:`ultraquant.reason.hdparallel`
adds cores and is required to reproduce it bit-for-bit.

That requirement is why nothing in this module is random at runtime. Majority
voting has to resolve ties, and resolving them by coin flip would make the result
depend on how the work was divided between cores. Ties are broken from the
*content* being bundled instead, so any split gives the same answer.

Pure Python standard library.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "DIM",
    "seed_vector",
    "random_vector",
    "bind",
    "unbind",
    "bundle",
    "bundle_counts",
    "merge_counts",
    "resolve",
    "permute",
    "similarity",
    "hamming",
    "ItemMemory",
]

#: Bits per hypervector. 10,000 keeps random vectors near-orthogonal and the
#: whole vector at 1,250 bytes — small enough to stay in L1.
DIM = 10_000


def seed_vector(name: str, dim: int = DIM) -> int:
    """A named vector, identical on every machine and every run.

    Derived from the name by hashing rather than drawn from a generator, so two
    processes that never speak agree on what ``"SHAPE"`` means. That is what
    lets a bundle be split across cores — or servers — and recombined.

    Args:
        name: The concept's name.
        dim: Vector width in bits.

    Returns:
        A ``dim``-bit integer.
    """
    out = 0
    filled = 0
    counter = 0
    while filled < dim:
        block = hashlib.blake2b(
            f"{name}|{counter}".encode("utf-8"), digest_size=64
        ).digest()
        out |= int.from_bytes(block, "big") << filled
        filled += 512
        counter += 1
    return out & ((1 << dim) - 1)


def random_vector(rng, dim: int = DIM) -> int:
    """An unnamed vector from a caller-supplied generator."""
    return rng.getrandbits(dim)


def bind(a: int, b: int) -> int:
    """Attach ``b`` to ``a``. Self-inverse, so binding twice undoes it."""
    return a ^ b


def unbind(composite: int, key: int) -> int:
    """Recover what was bound to ``key``.

    XOR is its own inverse, so this is :func:`bind` again. It has a separate
    name because the *intent* differs, and confusing the two is the error that
    silently destroys a superposition.
    """
    return composite ^ key


def permute(vector: int, times: int = 1, dim: int = DIM) -> int:
    """Rotate the bits, giving order and nesting depth.

    Binding is commutative, so ``bind(A, B) == bind(B, A)`` and a bare bundle is
    a set with no structure. Rotating one operand breaks that symmetry, which is
    what lets a sequence be distinguished from its reverse and a nested level be
    distinguished from its parent.

    Args:
        vector: The vector to rotate.
        times: How far to rotate; negative rotates the other way.
        dim: Vector width.

    Returns:
        The rotated vector.
    """
    shift = times % dim
    if shift == 0:
        return vector
    mask = (1 << dim) - 1
    return ((vector << shift) | (vector >> (dim - shift))) & mask


def hamming(a: int, b: int) -> int:
    """Number of differing bits."""
    return (a ^ b).bit_count()


def similarity(a: int, b: int, dim: int = DIM) -> float:
    """Agreement in ``[-1, 1]``: 1 identical, 0 unrelated, -1 opposite."""
    return 1.0 - 2.0 * hamming(a, b) / dim


# --------------------------------------------------------------------------- #
# bundling
# --------------------------------------------------------------------------- #
#
# Superposition is a per-bit majority vote. Counting each bit position in a
# Python loop would be dim * n interpreted operations -- 480,000 for a modest
# bundle -- so the counters are held *bit-sliced*: plane ``i`` holds bit ``i`` of
# every position's count at once. Adding one vector is then a ripple-carry add
# using only AND and XOR on whole integers, which costs O(log n) machine-speed
# operations instead of O(dim) interpreted ones.


def bundle_counts(vectors: list[int], planes: list[int] | None = None) -> list[int]:
    """Accumulate ``vectors`` into bit-sliced counter planes.

    Exposed separately from :func:`bundle` because it is the piece that
    parallelises: counts from any partition of the input can be merged with
    :func:`merge_counts` and thresholded once, and addition is associative, so
    the answer does not depend on how the work was divided.

    Args:
        vectors: Vectors to count.
        planes: Existing counters to add to.

    Returns:
        Counter planes, least significant first.
    """
    planes = list(planes or [])
    for vector in vectors:
        carry = vector
        index = 0
        while carry:
            if index == len(planes):
                planes.append(carry)
                break
            carry, planes[index] = planes[index] & carry, planes[index] ^ carry
            index += 1
    return planes


def merge_counts(left: list[int], right: list[int]) -> list[int]:
    """Add two sets of counter planes.

    This is what makes the parallel path exact rather than merely close: it is
    ordinary binary addition, so merging partial counts from any number of
    workers in any order yields the same planes as counting single-threaded.
    """
    out: list[int] = []
    carry = 0
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else 0
        b = right[index] if index < len(right) else 0
        # full adder, one bit position of the count, all dim lanes at once
        total = a ^ b ^ carry
        carry = (a & b) | (carry & (a ^ b))
        out.append(total)
    if carry:
        out.append(carry)
    return out


def _compare(planes: list[int], threshold: int, mask: int) -> tuple[int, int]:
    """Bit-parallel ``(count > threshold, count == threshold)``.

    Walks the planes from most significant down, carrying "still equal so far"
    as a mask. Every lane is decided simultaneously.
    """
    greater = 0
    equal = mask
    for index in range(max(len(planes), threshold.bit_length()) - 1, -1, -1):
        plane = planes[index] if index < len(planes) else 0
        if (threshold >> index) & 1:
            equal &= plane
        else:
            greater |= equal & plane
            equal &= ~plane & mask
    return greater, equal


def resolve(planes: list[int], count: int, dim: int = DIM,
            tiebreak: int = 0) -> int:
    """Threshold counter planes into a majority vector.

    Args:
        planes: Counters from :func:`bundle_counts`.
        count: How many vectors were counted.
        dim: Vector width.
        tiebreak: Used where exactly half the inputs voted 1, which can only
            happen for an even ``count``. Defaults to a mask derived from the
            counter planes themselves, which is deterministic, independent of
            the order the vectors were counted in, and — unlike the obvious
            choice — actually balanced at the positions where it is consulted.

    Returns:
        The bundled vector.

    Note:
        The obvious tiebreak, the XOR of the inputs, is **degenerate**. A tie
        means exactly ``count / 2`` of the bits are 1, so when ``count / 2`` is
        even their XOR is *always* 0 and every tie resolves to zero. Measured at
        ``count = 16`` that left 4,053 of 10,000 bits set instead of ~5,000, and
        the resulting density bias made every unbound probe resemble the key it
        was unbound with — costing 3 of 16 retrievals. Deriving the mask by
        hashing the planes has none of that correlation.
    """
    mask = (1 << dim) - 1
    if count <= 0:
        return 0
    greater, equal = _compare(planes, count // 2, mask)
    if count % 2:
        return greater & mask
    if tiebreak == 0:
        tiebreak = _plane_tiebreak(planes, dim)
    return (greater | (equal & tiebreak)) & mask


def _plane_tiebreak(planes: list[int], dim: int) -> int:
    """A balanced, deterministic mask derived from the counters themselves.

    The planes are order-independent — addition is associative — so any split of
    the work produces the same digest and therefore the same tiebreak. Hashing
    decorrelates it from the tie positions, which is the whole point.
    """
    width = (dim + 7) // 8
    digest = hashlib.blake2b(
        b"".join(plane.to_bytes(width, "big") for plane in planes),
        digest_size=32,
    ).hexdigest()
    return seed_vector(f"tiebreak|{digest}", dim)


def bundle(vectors: list[int], dim: int = DIM) -> int:
    """Superpose ``vectors`` by per-bit majority.

    The result is similar to every input and identical to none — which is what
    makes it a set that can still be queried. Ties on an even count are resolved
    by :func:`resolve`, deterministically and without reference to how the work
    was divided.

    Args:
        vectors: The vectors to superpose.
        dim: Vector width.

    Returns:
        The bundled vector; ``0`` for an empty input.
    """
    if not vectors:
        return 0
    if len(vectors) == 1:
        return vectors[0]
    return resolve(bundle_counts(vectors), len(vectors), dim)


class ItemMemory:
    """Cleanup memory: snap a noisy vector back to the concept it resembles.

    Unbinding a bundle does not return a stored vector — it returns that vector
    plus the crosstalk of everything else in the bundle. Without a step that
    maps the result to the nearest known item, a hypervector system degrades
    into noise after one operation. This is that step.

    The shard library's signature screen is the same idea at library scale; this
    is the small, in-memory version for the vectors a single computation is
    working with.

    Args:
        dim: Vector width.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = int(dim)
        self._items: dict[str, int] = {}

    def add(self, name: str, vector: int | None = None) -> int:
        """Store ``name``, minting its vector deterministically if not given."""
        if vector is None:
            vector = seed_vector(name, self.dim)
        self._items[name] = vector
        return vector

    def get(self, name: str) -> int:
        """The vector stored for ``name``."""
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        """Every stored name, sorted."""
        return sorted(self._items)

    def nearest(self, vector: int, top_k: int = 1) -> list[tuple[str, float]]:
        """The stored items most like ``vector``, best first.

        Ties break alphabetically so cleanup is deterministic.
        """
        scored = [
            (name, similarity(vector, stored, self.dim))
            for name, stored in self._items.items()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    def cleanup(self, vector: int, floor: float = 0.0) -> tuple[str, float] | None:
        """Snap to the nearest item, or ``None`` if nothing is close enough."""
        best = self.nearest(vector, 1)
        if not best or best[0][1] <= floor:
            return None
        return best[0]
