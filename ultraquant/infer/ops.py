"""The operations a transformer needs and an MLP does not.

Pure standard library, lists of floats, no dependencies - the same
rule the rest of the system keeps. Every function here has a
textbook definition, and the gate beside this module checks each one
against that definition computed independently at 50 significant
digits rather than against itself.

**Why that oracle and not a property test.** Properties catch some
errors and miss the ones that matter here. A softmax with the wrong
denominator still sums to one; a LayerNorm dividing by `n` instead
of `n-1` still centres; a GELU using the tanh approximation where
the model was trained with erf still looks like a GELU. Each of
those is a silent few-percent error that compounds over thirty
blocks into nonsense, and none of them is visible to a property.
Agreement with the definition at 50 digits sees all three.

**The stabilisations are the point, not an optimisation.** Softmax
subtracts its maximum before exponentiating and LayerNorm computes
variance about the measured mean. Both are algebraically no-ops and
numerically the difference between an answer and an OverflowError.
Criterion 2 of the gate exists to demonstrate that on inputs where
the naive form actually fails, rather than asserting it in a
comment.
"""

from __future__ import annotations

import math

__all__ = ["softmax", "layer_norm", "rms_norm", "gelu", "gelu_tanh",
           "silu", "linear", "add", "scale", "split_heads",
           "merge_heads", "attention", "attention_weights",
           "multi_head_attention"]


def softmax(values: list[float]) -> list[float]:
    """``exp(x_i) / sum(exp(x_j))``, computed so it cannot overflow.

    The maximum is subtracted first. That changes nothing
    algebraically - the same constant leaves numerator and
    denominator - and it is the difference between a distribution
    and an OverflowError on attention scores, which routinely reach
    the tens before scaling.
    """
    if not values:
        return []
    top = max(values)
    if top == -math.inf:
        # Every position masked. The subtraction below would compute
        # -inf minus -inf, which is NaN, and a NaN row propagates
        # silently through the whole block - the gate's criterion 4
        # caught this by asking a fully masked row to be zero and
        # getting 24 rows of NaN. A uniform answer would be worse
        # still: it invents attention where the mask said there is
        # none. Zero is the honest weight for a position nothing may
        # attend to.
        return [0.0] * len(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    if total == 0.0:
        return [0.0] * len(values)
    return [weight / total for weight in weights]


def layer_norm(values: list[float], weight: list[float] | None = None,
               bias: list[float] | None = None,
               eps: float = 1e-5) -> list[float]:
    """``(x - mean) / sqrt(var + eps) * weight + bias``.

    The variance is the BIASED one - divided by ``n``, not ``n-1``.
    Every transformer implementation uses the biased form, and the
    unbiased version is the error that still centres, still scales,
    and is silently wrong by a factor of ``sqrt(n/(n-1))``.
    """
    count = len(values)
    if count == 0:
        return []
    mean = sum(values) / count
    centred = [value - mean for value in values]
    variance = sum(value * value for value in centred) / count
    inv = 1.0 / math.sqrt(variance + eps)
    out = [value * inv for value in centred]
    if weight is not None:
        out = [value * w for value, w in zip(out, weight)]
    if bias is not None:
        out = [value + b for value, b in zip(out, bias)]
    return out


def rms_norm(values: list[float], weight: list[float] | None = None,
             eps: float = 1e-6) -> list[float]:
    """``x / sqrt(mean(x^2) + eps) * weight``.

    Not a LayerNorm with the mean left out by accident: RMSNorm
    deliberately does not centre, which is why it has no bias either.
    Text models use it; the vision encoder here uses LayerNorm, and
    picking the wrong one is a bug that produces plausible garbage.
    """
    count = len(values)
    if count == 0:
        return []
    mean_square = sum(value * value for value in values) / count
    inv = 1.0 / math.sqrt(mean_square + eps)
    out = [value * inv for value in values]
    if weight is not None:
        out = [value * w for value, w in zip(out, weight)]
    return out


def gelu(values: list[float]) -> list[float]:
    """Exact GELU: ``x * 0.5 * (1 + erf(x / sqrt(2)))``.

    The erf form, which is what "gelu" means in GGUF unless the file
    says otherwise. :func:`gelu_tanh` is the other one, kept separate
    because they differ by up to **0.000473 absolute, at x = -2.698**
    (measured, not estimated), and a model trained with one and run
    with the other is being run slightly wrong, everywhere, in a way
    no single output makes visible.
    """
    return [value * 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
            for value in values]


def gelu_tanh(values: list[float]) -> list[float]:
    """The tanh approximation to GELU, as some checkpoints train with."""
    root = math.sqrt(2.0 / math.pi)
    return [0.5 * value
            * (1.0 + math.tanh(root * (value + 0.044715 * value ** 3)))
            for value in values]


def silu(values: list[float]) -> list[float]:
    """``x * sigmoid(x)``, the gate half of SwiGLU.

    Written as the branch that cannot overflow: ``exp(-x)`` blows up
    for very negative x, so the algebraically identical form using
    ``exp(x)`` is used there instead.
    """
    out = []
    for value in values:
        if value >= 0.0:
            out.append(value / (1.0 + math.exp(-value)))
        else:
            grown = math.exp(value)
            out.append(value * grown / (1.0 + grown))
    return out


def linear(rows: list[list[float]], bias: list[float] | None,
           x: list[float]) -> list[float]:
    """``W x + b`` with ``W`` as ``[out][in]`` - the GGUF row order."""
    out = [sum(w * v for w, v in zip(row, x)) for row in rows]
    if bias is not None:
        out = [value + b for value, b in zip(out, bias)]
    return out


def add(left: list[float], right: list[float]) -> list[float]:
    """Residual addition."""
    return [a + b for a, b in zip(left, right)]


def scale(values: list[float], factor: float) -> list[float]:
    """Elementwise multiply by a constant."""
    return [value * factor for value in values]


def split_heads(sequence: list[list[float]],
                heads: int) -> list[list[list[float]]]:
    """``[seq][dim]`` to ``[head][seq][head_dim]``.

    The split is CONTIGUOUS - head h owns columns
    ``[h*head_dim, (h+1)*head_dim)`` - which is how every checkpoint
    lays out a fused QKV projection. Interleaving instead would
    scramble each head's subspace while leaving every shape correct
    and every norm plausible, which is the worst kind of wrong.
    """
    if heads <= 0:
        raise ValueError("heads must be positive")
    width = len(sequence[0]) if sequence else 0
    if width % heads:
        raise ValueError(f"width {width} does not divide into {heads} heads")
    size = width // heads
    return [[row[h * size:(h + 1) * size] for row in sequence]
            for h in range(heads)]


def merge_heads(per_head: list[list[list[float]]]) -> list[list[float]]:
    """``[head][seq][head_dim]`` back to ``[seq][dim]``."""
    if not per_head:
        return []
    length = len(per_head[0])
    return [[value for head in per_head for value in head[position]]
            for position in range(length)]


def attention_weights(queries: list[list[float]], keys: list[list[float]],
                      causal: bool = False,
                      mask: list[list[bool]] | None = None
                      ) -> list[list[float]]:
    """The attention matrix: ``softmax(Q K^T / sqrt(d_k))`` per row.

    Separated from :func:`attention` because it is the part worth
    inspecting on its own - every row is a probability distribution
    over positions, and a mask that does not mask shows up here
    before it shows up in an output that merely looks wrong.

    A masked position is scored ``-inf`` rather than dropped, so the
    softmax denominator is right by construction. Zeroing the weight
    afterwards instead would leave the surviving weights summing to
    less than one - attention quietly scaled down by however much was
    masked, which is a bug that gets worse with sequence length.
    """
    if not queries or not keys:
        return []
    size = len(queries[0])
    inv = 1.0 / math.sqrt(size) if size else 1.0
    out: list[list[float]] = []
    for i, query in enumerate(queries):
        scores = []
        for j, key in enumerate(keys):
            if causal and j > i:
                scores.append(-math.inf)
                continue
            if mask is not None and not mask[i][j]:
                scores.append(-math.inf)
                continue
            scores.append(sum(q * k for q, k in zip(query, key)) * inv)
        out.append(softmax(scores))
    return out


def attention(queries: list[list[float]], keys: list[list[float]],
              values: list[list[float]], causal: bool = False,
              mask: list[list[bool]] | None = None) -> list[list[float]]:
    """Single-head scaled dot-product attention."""
    weights = attention_weights(queries, keys, causal=causal, mask=mask)
    if not weights:
        return []
    width = len(values[0]) if values else 0
    out: list[list[float]] = []
    for row in weights:
        blended = [0.0] * width
        for weight, value in zip(row, values):
            if weight == 0.0:
                continue
            for index in range(width):
                blended[index] += weight * value[index]
        out.append(blended)
    return out


def multi_head_attention(queries: list[list[float]],
                         keys: list[list[float]],
                         values: list[list[float]], heads: int,
                         causal: bool = False) -> list[list[float]]:
    """Attention run per head and re-joined.

    Each head attends in its own subspace; the scale is ``1/sqrt(head
    dim)``, not ``1/sqrt(model dim)``, which :func:`attention` gets
    right automatically because it is handed the split vectors.
    """
    q_heads = split_heads(queries, heads)
    k_heads = split_heads(keys, heads)
    v_heads = split_heads(values, heads)
    return merge_heads([attention(q, k, v, causal=causal)
                        for q, k, v in zip(q_heads, k_heads, v_heads)])
