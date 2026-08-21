"""The operations between the matmuls. The primitive gate.

`model/network.py` never needed these because an MLP is linear
layers and a ReLU. A transformer is not, and every operation this
unit adds has a textbook definition that a wrong implementation
still satisfies loosely: a softmax with the wrong denominator still
sums to one, a LayerNorm dividing by n-1 still centres, a GELU using
the tanh form where the model trained with erf still looks like a
GELU.

Each of those is a silent few-per-cent error, in every block, that
compounds over thirty layers into nonsense while every shape stays
correct and every norm stays plausible. Property tests do not see
them. So the oracle here is not a property and not a second copy of
the same code: **each operation is computed independently at 50
significant digits, straight from its mathematical definition, with
no stabilisation tricks and no float arithmetic anywhere.** The
shipped float version has to agree with that.

**The stabilisations are the reason the shipped version differs from
the definition at all.** Softmax subtracts its maximum before
exponentiating; LayerNorm takes variance about the measured mean;
SiLU switches branches at zero. All three are algebraic no-ops and
numerically the difference between an answer and an OverflowError.
Criterion 2 exists so that is demonstrated on inputs where the naive
form actually breaks, rather than asserted in a comment.

**The criteria, written before the run.**

1. **Every operation matches its definition** - agreement with the
   50-digit reference to within 1e-12 relative, over randomised
   inputs including the awkward ones (near-zero variance, large
   magnitudes, long vectors). One operation outside is a fail.
2. **The stabilisation is demonstrated, not asserted** - the same
   inputs are pushed through a deliberately naive float
   implementation, and it must actually FAIL (overflow, or lose
   more than 1e-6 relative) on some of them while the shipped
   version holds. If the naive form never fails, this criterion has
   shown nothing and fails on that ground.
3. **The causal mask leaks nothing** - changing a future position's
   key and value must not alter an earlier position's output by a
   single bit. One changed bit is a fail; this is the property that
   makes a decoder a decoder.
4. **Attention rows are distributions** - every unmasked row sums to
   1 within 1e-12, and a fully masked row is exactly zero rather
   than uniform, because uniform would be inventing attention where
   the mask said there is none.
5. **The head split is contiguous and reversible** - merging a split
   returns the original exactly, and head h holds exactly columns
   [h*size, (h+1)*size). A split that interleaves leaves every shape
   and every norm correct while scrambling each head's subspace.
6. **The cost is named** - MACs per second through one attention
   block, reported and not gated.

**Run one FAILED, and every one of its four failures was worth
having.** Two were defects in the code, two were defects in the
criteria - and the criteria are recorded here as they were first
written, because a gate that quietly rewrites its own bar is not a
gate.

**A real bug: a fully masked attention row came out NaN.** Criterion
4 asked a fully masked row to be zero and got 24 rows of NaN.
`softmax` subtracts its maximum, and when every score is `-inf` that
subtraction is `-inf - -inf`, which is NaN before the zero-total
guard can fire. A NaN row propagates silently through an entire
block. Fixed by testing for an all-masked row before subtracting.
This is precisely the case a property test would have missed - the
row still had the right length and the right dtype.

**A real bug in the harness: SiLU's naive form was never provoked.**
Criterion 2 requires the naive implementation to actually break, and
`x / (1 + exp(-x))` only overflows past |x| ~ 709 while the test
range was [-8, 8]. It broke 0 times out of 60, so the criterion had
demonstrated nothing and correctly failed on that ground. Measured
on [-800, 800] it breaks 60 out of 60. The useful by-product: on
real activation ranges that branch is never needed. It is insurance,
not a hot path.

**A bad criterion: per-element relative error measures float64, not
the code.** GELU at x = -8 is -4.98e-15, and reaching it in float64
means adding 1 to erf(-5.66) = -0.99999999999999502. The
cancellation leaves two significant digits: relative error 1.8e-2,
absolute error 9.2e-17 - exactly one ULP. No implementation in this
format can do better, so the criterion was unsatisfiable by
construction. It now measures error as a share of the output
vector's own RMS, which is the quantity that matters for values
about to be added into a residual stream.

**A bad criterion: one regime was gated that no format can pass.**
The deliberately adversarial mean-1e6 stream costs LayerNorm
precision because the mean's own ULP is 1.16e-10 and the centred
values are O(1). That is a property of the number, not the routine.
It is now measured and REPORTED rather than gated - and the
reporting is the interesting part.

**Run two:**

| operation | gap vs the definition | on a mean-1e6 stream |
|---|---:|---:|
| softmax | 3.80e-15 | 1.12e-15 |
| layer_norm | 1.32e-15 | **9.81e-11** |
| rms_norm | 1.11e-15 | 4.34e-16 |
| gelu | 3.38e-16 | 3.61e-16 |
| gelu_tanh | 3.38e-16 | 3.61e-16 |
| silu | 3.44e-16 | 3.68e-16 |
| attention | 1.20e-15 | - |

**PASS: worst gap 3.80e-15 against a tolerance of 1e-12; naive forms
broke 15, 15 and 60 times out of 60; zero causal leaks; row sums
within 1.11e-16 of one; zero masked rows non-zero; zero head-split
errors; attention at 21.9 M MAC/s in pure Python.**

The second column earns its place. Every operation is untouched by a
large DC offset except **LayerNorm, which loses five orders of
magnitude** - and that is not a curiosity, it is the one op that
subtracts a mean, so it is the only one that can care. RMSNorm,
which deliberately does not centre, is unaffected. Anyone debugging
a drifting residual stream at layer thirty should know this number
before they start, and now it is written down rather than
rediscovered.

Run it::

    python -m ultraquant.experiments.transformerops_gate
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext

from ultraquant.infer import ops

__all__ = ["OpsReport", "run_gate"]

getcontext().prec = 50

#: pi to 50 places, as a literal rather than computed - the point of
#: the reference is to share nothing with the implementation under
#: test, and math.pi is a float.
_PI = Decimal("3.14159265358979323846264338327950288419716939937511")


def _erf(x: Decimal) -> Decimal:
    """erf by its Taylor series, at the working precision.

    ``erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))``.
    Slow and obviously the definition, which is the whole point: it
    shares no code with ``math.erf`` and no approximation with it
    either.
    """
    term = x
    total = x
    n = 0
    while True:
        n += 1
        term = -term * x * x / Decimal(n)
        piece = term / Decimal(2 * n + 1)
        total += piece
        if abs(piece) < Decimal(10) ** -45:
            break
        if n > 400:
            break
    return total * 2 / _PI.sqrt()


def _tanh(x: Decimal) -> Decimal:
    grown = (x * 2).exp()
    return (grown - 1) / (grown + 1)


def ref_softmax(values: list[Decimal]) -> list[Decimal]:
    """The definition, with no maximum subtracted."""
    grown = [value.exp() for value in values]
    total = sum(grown)
    return [value / total for value in grown]


def ref_layer_norm(values, weight, bias, eps: Decimal) -> list[Decimal]:
    """The definition, biased variance, two passes."""
    count = Decimal(len(values))
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    root = (variance + eps).sqrt()
    out = [(value - mean) / root for value in values]
    if weight is not None:
        out = [value * w for value, w in zip(out, weight)]
    if bias is not None:
        out = [value + b for value, b in zip(out, bias)]
    return out


def ref_rms_norm(values, weight, eps: Decimal) -> list[Decimal]:
    count = Decimal(len(values))
    mean_square = sum(value * value for value in values) / count
    root = (mean_square + eps).sqrt()
    out = [value / root for value in values]
    if weight is not None:
        out = [value * w for value, w in zip(out, weight)]
    return out


def ref_gelu(values: list[Decimal]) -> list[Decimal]:
    half = Decimal("0.5")
    root2 = Decimal(2).sqrt()
    return [value * half * (1 + _erf(value / root2)) for value in values]


def ref_gelu_tanh(values: list[Decimal]) -> list[Decimal]:
    root = (Decimal(2) / _PI).sqrt()
    half = Decimal("0.5")
    c = Decimal("0.044715")
    return [half * value * (1 + _tanh(root * (value + c * value ** 3)))
            for value in values]


def ref_silu(values: list[Decimal]) -> list[Decimal]:
    """``x / (1 + exp(-x))`` with no branch - the definition."""
    return [value / (1 + (-value).exp()) for value in values]


def ref_attention(queries, keys, values) -> list[list[Decimal]]:
    """Scaled dot-product attention, straight from the definition."""
    size = Decimal(len(queries[0]))
    inv = 1 / size.sqrt()
    out = []
    for query in queries:
        scores = [sum(q * k for q, k in zip(query, key)) * inv
                  for key in keys]
        weights = ref_softmax(scores)
        width = len(values[0])
        blended = [Decimal(0)] * width
        for weight, value in zip(weights, values):
            for index in range(width):
                blended[index] += weight * value[index]
        out.append(blended)
    return out


def naive_softmax(values: list[float]) -> list[float]:
    """The form without the maximum subtracted. Overflows."""
    grown = [math.exp(value) for value in values]
    total = sum(grown)
    return [value / total for value in grown]


def naive_layer_norm(values: list[float], eps: float = 1e-5) -> list[float]:
    """Variance as ``E[x^2] - E[x]^2``. Cancels catastrophically.

    Algebraically identical to the two-pass form and the standard
    way to get a fast single pass. When the mean is large next to
    the spread, the two terms agree to most of their digits and the
    difference is noise - which is exactly the shape of a residual
    stream late in a deep model.
    """
    count = len(values)
    mean = sum(values) / count
    mean_square = sum(value * value for value in values) / count
    variance = mean_square - mean * mean
    root = math.sqrt(variance + eps) if variance + eps > 0 else float("nan")
    return [(value - mean) / root for value in values]


def naive_silu(values: list[float]) -> list[float]:
    """``x / (1 + exp(-x))`` with no branch. Overflows for x << 0."""
    return [value / (1.0 + math.exp(-value)) for value in values]


def _gap(got: list[float], want: list[Decimal]) -> float:
    """Worst absolute error as a fraction of the output's own RMS.

    NOT per-element relative error, which is what this gate asked
    for first and had to correct. GELU at x = -8 is -4.98e-15, and
    computing it in float64 means adding 1 to erf(-5.66) =
    -0.99999999999999502: the cancellation leaves two significant
    digits, so the per-element relative error is 1.8e-2 while the
    ABSOLUTE error is 9.2e-17, sitting exactly on float64's floor.
    No implementation in this format can do better, so a per-element
    relative criterion was measuring the number format rather than
    the code.

    Scaling by the vector's RMS is the measure that matches what
    these outputs are for: they are added into a residual stream, so
    what matters is the error as a fraction of the signal beside it,
    not as a fraction of one element that happens to have cancelled
    to nothing.
    """
    exact = [float(target) for target in want]
    if not exact:
        return 0.0
    rms = math.sqrt(sum(value * value for value in exact) / len(exact))
    scale = rms if rms > 1e-300 else 1.0
    return max(abs(value - target)
               for value, target in zip(got, exact)) / scale


@dataclass
class OpsReport:
    """What each operation cost against its own definition.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        gaps: Operation -> worst relative gap against the reference.
        cases: Randomised inputs per operation.
        naive_failures: Operation -> how many inputs broke the naive form.
        leaks: Causal positions whose output moved when the future did.
        row_error: Worst deviation of an attention row from summing to 1.
        masked_rows_nonzero: Fully masked rows that were not zero.
        split_errors: Head splits that did not merge back exactly.
        attention_macs: MACs per second through one attention block.
        offset_gaps: The same gaps on a mean-1e6 stream - measured
            and reported, not gated, because float64 cannot
            represent that mean finely enough for any
            implementation to do better.
        reason: Plain-language verdict.
    """

    passes: bool
    gaps: dict = field(default_factory=dict)
    cases: int = 0
    naive_failures: dict = field(default_factory=dict)
    leaks: int = 0
    row_error: float = 0.0
    masked_rows_nonzero: int = 0
    split_errors: int = 0
    attention_macs: float = 0.0
    offset_gaps: dict = field(default_factory=dict)
    reason: str = ""


def _decimals(values: list[float]) -> list[Decimal]:
    """Floats to Decimals losslessly - the exact binary value."""
    return [Decimal(value) for value in values]


def run_gate(cases: int = 60, seed: int = 0,
             tolerance: float = 1e-12) -> OpsReport:
    """Every operation against its own definition at 50 digits."""
    rng = random.Random(seed)
    gaps: dict = {}
    naive_failures: dict = {"softmax": 0, "layer_norm": 0, "silu": 0}

    offset_gaps: dict = {}

    def note(name: str, value: float) -> None:
        # The large-mean regime is measured but not gated, and kept
        # in its own dict rather than folded into a max where it
        # would look like a defect in every op at once. See the
        # docstring: subtracting a mean of 1e6 in float64 costs
        # precision no implementation in this format can recover.
        target = offset_gaps if regime == 2 else gaps
        target[name] = max(target.get(name, 0.0), value)

    for case in range(cases):
        width = rng.choice((8, 64, 512, 1152))
        # A spread of regimes on purpose: ordinary activations, the
        # large-magnitude scores that overflow a naive softmax, and
        # the large-mean-small-spread vectors that cancel a naive
        # variance. An op that is only tested on nice inputs has
        # only been tested on nice inputs.
        regime = case % 4
        if regime == 0:
            xs = [rng.gauss(0.0, 1.0) for _ in range(width)]
        elif regime == 1:
            xs = [rng.gauss(0.0, 40.0) for _ in range(width)]
        elif regime == 2:
            xs = [rng.gauss(1e6, 1.0) for _ in range(width)]
        else:
            xs = [rng.gauss(0.0, 1e-7) for _ in range(width)]
        exact = _decimals(xs)
        weight = [rng.gauss(1.0, 0.2) for _ in range(width)]
        bias = [rng.gauss(0.0, 0.2) for _ in range(width)]

        note("softmax", _gap(ops.softmax(xs), ref_softmax(exact)))
        note("layer_norm",
             _gap(ops.layer_norm(xs, weight, bias),
                  ref_layer_norm(exact, _decimals(weight), _decimals(bias),
                                 Decimal("1e-5"))))
        note("rms_norm",
             _gap(ops.rms_norm(xs, weight),
                  ref_rms_norm(exact, _decimals(weight), Decimal("1e-6"))))
        # The activations are measured on a bounded range: GELU and
        # SiLU saturate, and a relative gap against a value that has
        # underflowed to zero is a measurement of nothing.
        small = [rng.uniform(-8.0, 8.0) for _ in range(min(width, 64))]
        # SiLU's unbranched form only overflows past |x| ~ 709, so
        # criterion 2 has to ask it about inputs where its branch
        # actually earns its place. Real activations never reach
        # here, which is worth knowing too: the branch is insurance,
        # not a hot path.
        extreme = [rng.uniform(-800.0, 800.0) for _ in range(64)]
        note("gelu", _gap(ops.gelu(small), ref_gelu(_decimals(small))))
        note("gelu_tanh",
             _gap(ops.gelu_tanh(small), ref_gelu_tanh(_decimals(small))))
        note("silu", _gap(ops.silu(small), ref_silu(_decimals(small))))
        note("silu", _gap(ops.silu(extreme), ref_silu(_decimals(extreme))))

        # Criterion 2: the naive forms, on the same inputs.
        for name, call, args in (("softmax", naive_softmax, (xs,)),
                                 ("layer_norm", naive_layer_norm, (xs,)),
                                 ("silu", naive_silu, (extreme,))):
            try:
                got = call(*args)
            except (OverflowError, ValueError, ZeroDivisionError):
                naive_failures[name] += 1
                continue
            if any(math.isnan(v) or math.isinf(v) for v in got):
                naive_failures[name] += 1
                continue
            want = (ref_softmax(exact) if name == "softmax"
                    else ref_layer_norm(exact, None, None, Decimal("1e-5"))
                    if name == "layer_norm"
                    else ref_silu(_decimals(extreme)))
            if _gap(got, want) > 1e-6:
                naive_failures[name] += 1
    return _finish(gaps, cases, naive_failures, rng, tolerance,
                   offset_gaps)


def _finish(gaps: dict, cases: int, naive_failures: dict,
            rng: random.Random, tolerance: float,
            offset_gaps: dict) -> OpsReport:
    """Criteria 1 and 2 are measured; 3 to 6 are checked here."""
    # Criterion 1, on attention itself.
    length, size = 24, 32
    q = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(length)]
    k = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(length)]
    v = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(length)]
    got = ops.attention(q, k, v)
    want = ref_attention([_decimals(r) for r in q], [_decimals(r) for r in k],
                         [_decimals(r) for r in v])
    worst = max(_gap(a, b) for a, b in zip(got, want))
    gaps["attention"] = worst

    # Criterion 3: the causal mask leaks nothing. The LAST position's
    # key and value are replaced; every earlier output must be
    # untouched, bit for bit.
    before = ops.attention(q, k, v, causal=True)
    k2 = [list(row) for row in k]
    v2 = [list(row) for row in v]
    k2[-1] = [rng.gauss(0, 50) for _ in range(size)]
    v2[-1] = [rng.gauss(0, 50) for _ in range(size)]
    after = ops.attention(q, k2, v2, causal=True)
    leaks = sum(1 for i in range(length - 1) if before[i] != after[i])

    # Criterion 4: rows are distributions, and a fully masked row is
    # zero rather than uniform.
    weights = ops.attention_weights(q, k, causal=True)
    row_error = max(abs(sum(row) - 1.0) for row in weights)
    blind = [[False] * length for _ in range(length)]
    masked = ops.attention_weights(q, k, mask=blind)
    masked_rows_nonzero = sum(1 for row in masked if any(row))

    # Criterion 5: the split is contiguous and reversible.
    split_errors = 0
    for heads in (1, 2, 4, 8):
        pieces = ops.split_heads(q, heads)
        if ops.merge_heads(pieces) != q:
            split_errors += 1
        per = size // heads
        for h, piece in enumerate(pieces):
            if piece[0] != q[0][h * per:(h + 1) * per]:
                split_errors += 1

    # Criterion 6: the price, reported.
    long_q = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(128)]
    started = time.perf_counter()
    ops.multi_head_attention(long_q, long_q, long_q, heads=8)
    elapsed = time.perf_counter() - started
    # scores + blend, both seq*seq*head_dim over all heads
    macs = 2 * len(long_q) * len(long_q) * 64
    rate = macs / elapsed if elapsed > 0 else 0.0

    worst_gap = max(gaps.values()) if gaps else 0.0
    naive_broke = all(count > 0 for count in naive_failures.values())
    passes = (worst_gap <= tolerance and naive_broke and leaks == 0
              and row_error <= 1e-12 and masked_rows_nonzero == 0
              and split_errors == 0)
    report = OpsReport(
        passes=passes, gaps=gaps, cases=cases,
        naive_failures=naive_failures, leaks=leaks, row_error=row_error,
        masked_rows_nonzero=masked_rows_nonzero, split_errors=split_errors,
        attention_macs=rate, offset_gaps=offset_gaps)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {cases} randomised cases; worst "
        f"gap against the 50-digit definition {worst_gap:.2e} "
        f"(tolerance {tolerance:.0e}); naive forms broke on "
        + ", ".join(f"{name} {count}" for name, count
                    in sorted(naive_failures.items()))
        + f"; {leaks} causal leaks, row sums off by {row_error:.2e}, "
        f"{masked_rows_nonzero} masked rows non-zero, {split_errors} "
        f"head-split errors; attention {rate / 1e6:.1f} M MAC/s; "
        f"on a mean-1e6 stream (reported, not gated) worst gap "
        f"{max(offset_gaps.values()) if offset_gaps else 0.0:.2e}")
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print("gap against the 50-digit definition, as a share of output RMS:")
    for name, value in sorted(result.gaps.items()):
        print(f"  {name:<14} {value:.3e}")
    print()
    print("the same, on a mean-1e6 stream (reported, not gated):")
    for name, value in sorted(result.offset_gaps.items()):
        print(f"  {name:<14} {value:.3e}")
    print()
    print(f"naive failures:  {result.naive_failures}")
    print(f"causal leaks:    {result.leaks}")
    print(f"row sum error:   {result.row_error:.3e}")
    print(f"masked non-zero: {result.masked_rows_nonzero}")
    print(f"split errors:    {result.split_errors}")
    print(f"attention:       {result.attention_macs / 1e6:.1f} M MAC/s")
    print()
    print(result.reason)
