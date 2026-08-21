"""The 27-block encoder, and what ternary costs across a stack.

§11.95 measured what ternary conversion costs ONE layer: 0.467
relative output error at 0.884 cosine. Every conclusion drawn from
that number since has been an extrapolation - "the error compounds
through a stack" was said in the README, in the CLI, and in this
book, and nobody had run a stack.

This runs the stack. `infer/vit.py` assembles the real 27-block
encoder from the real checkpoint, and the gate walks TWO towers
through it in lockstep - one in f16, one ternarised by the
conversion kit's own `choose_rules` - measuring how far apart they
drift, block by block. The f16 tower is the reference; nothing
external is needed, which matters because nothing external is
available.

**What cannot be claimed, stated first.** There is no reference
implementation on this machine that exposes intermediate
activations: LM Studio ships `llama-server.exe`, which returns text.
So this does NOT verify that the encoder reproduces llama.cpp's
numbers. Pre-norm ordering, the contiguous head split and the 2x2
merge are read from tensor names, shapes and metadata, and they are
assumptions.

That is exactly why criterion 3 exists. A drift measurement is
worthless if a genuinely broken tower drifts less than a quantised
one, because then the number cannot tell a bug from noise. So three
deliberate structural defects are built and measured on the same
scale.

**The criteria, written before the run.**

1. **The tower completes on real weights** - all 27 blocks, every
   activation finite, correct shapes throughout. One non-finite
   value is a fail.
2. **End-to-end fidelity decides usability** - cosine between the
   ternary tower's projected output and the f16 tower's. **At or
   above 0.90, converted weights are usable without retraining;
   below, they are not.** The bar is registered here before the run,
   and the honest expectation is that it FAILS - 0.884 per layer
   over 27 layers leaves little - so what the run really produces is
   the size of the shortfall.
3. **The measurement can tell a bug from quantisation** - three
   STRUCTURAL defects (heads split interleaved instead of
   contiguous, the attention residual dropped, SiLU in place of
   GELU) must each land further from the f16 tower than the ternary
   tower does. If quantisation damages the output more than a
   dropped residual, this harness cannot distinguish correctness
   from noise and nothing else it reports may be trusted.
4. **The compounding is reported as a curve, not a number** -
   per-block cosine and per-block residual RMS for both towers, so
   the SHAPE of the drift is visible. A single end-to-end figure
   cannot distinguish steady decay from a cliff at one block.

**Registered as reported, not gated:** the wrong-epsilon
perturbation - 1e-5, `ops.layer_norm`'s default, in place of the
1e-6 this checkpoint specifies. Its magnitude is an empirical
question, not a pass/fail: a 1e-5 change against a variance near 1
may well be smaller than quantisation, and if so that is a finding
about epsilon rather than a failure of the harness. Gating it would
be pretending to know the answer in advance.

**FAILED criteria 2 and 3. The failures are the result.**

27 blocks, 8 tokens, every activation finite. The compounding, which
had been asserted three times in this repository and never measured:

| block | cos(f16, ternary) | f16 RMS | ternary RMS |
|---:|---:|---:|---:|
| 0 | 0.9496 | 0.58 | 0.62 |
| 5 | 0.4889 | 0.29 | 0.28 |
| 10 | 0.2681 | 0.50 | 0.27 |
| 15 | 0.2441 | 0.86 | 0.44 |
| 20 | 0.2280 | 1.50 | 1.47 |
| 25 | **-0.0592** | 1.63 | 6.90 |
| 26 | 0.8749 | 71.73 | 53.32 |
| **after the projector** | **0.0185** | | |

**Criterion 2 fails, and not narrowly: 0.0185 against a bar of
0.90.** A converted encoder's output is not a degraded version of
the original, it is uncorrelated with it. The half-life is about
five blocks - 0.95 to 0.49 - and by block 25 the two towers are
mildly ANTI-correlated. Every previous statement in this repository
that "the error compounds through a stack" was an extrapolation from
one layer's 0.884. This is the stack, and the extrapolation was
right in direction and far too kind in degree.

**The ternary tower also destabilises in magnitude**, which the
single cosine hides. Its residual RMS reaches 7.21 at block 24
against the f16 tower's 1.96 - 3.7x - so it is not merely noisy, it
is growing where the original does not.

**A methodological trap worth more than the headline.** Block 26's
cosine is **0.8749**, and the projected output's is **0.0185**. The
final block's massive activations (RMS 71.7) dominate both towers,
so the two look aligned while their meaningful content does not
agree at all; `post_ln` divides that shared magnitude out and the
disagreement appears. **Anyone measuring conversion damage on the
last hidden state, before the final norm, would have read 0.87 and
called it a success.**

**Criterion 3 fails at 27 blocks: quantisation damaged the output
MORE than any structural bug did.** Interleaved heads 0.0193,
SiLU-for-GELU 0.1210, a dropped attention residual 0.2218 - all
above ternary's 0.0185. A tower with no attention residual retains
twelve times more signal than a correctly wired one whose weights
were converted.

**So the failure was localised rather than accepted.** The same
comparison at shallower depths:

| depth | ternary | heads | no-resid | silu | discriminates? |
|---:|---:|---:|---:|---:|:--|
| 1 | 0.9611 | 0.9482 | 0.8562 | -0.0055 | yes |
| 2 | 0.8457 | 0.5617 | 0.1603 | 0.0386 | yes |
| 4 | 0.7254 | 0.3544 | 0.1661 | 0.3086 | yes |
| 8 | 0.5880 | 0.1599 | 0.3314 | 0.3736 | yes |
| 27 | 0.0185 | 0.0193 | 0.2218 | 0.1210 | **no** |

The measurement discriminates at every depth up to 8 and only stops
at 27. **Criterion 3's failure is therefore a fact about
quantisation, not about the method** - and it is the same fact as
criterion 2's, seen from the other side. The wiring is validated
where the signal survives to validate it.

Two smaller findings fell out. **A wrong activation is instantly
fatal and a wrong head split is not**: at one block, SiLU-for-GELU
already gives -0.0055 while interleaved heads still reads 0.9482 and
takes until block 8 to become obvious. A bug that hides for eight
blocks is the more dangerous of the two. And the **wrong epsilon -
1e-5 for the file's 1e-6 - measured 1.0000 cosine**: no effect at
four decimal places, over 27 blocks. That was registered as reported
rather than gated precisely because its size was unknown, and the
answer is that this particular default would have been harmless.
Reading it from the file is still right; the reason is that the next
checkpoint's may not be.

Run it::

    python -m ultraquant.experiments.encoder_gate
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ultraquant.convert import gguf, ternary
from ultraquant.experiments.convert_gate import checkpoint_path
from ultraquant.infer import ops, vit

__all__ = ["EncoderReport", "run_gate", "synthetic_frames"]


def _cosine(left: list, right: list) -> float:
    """Cosine between two flattened token streams."""
    a = [value for token in left for value in token]
    b = [value for token in right for value in token]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _rms(stream: list) -> float:
    count = sum(len(token) for token in stream)
    return math.sqrt(sum(v * v for token in stream for v in token) / count)


def synthetic_frames(config, seed: int = 0) -> list:
    """Two frames of already-normalised pixels.

    Synthetic because the standard library cannot decode a JPEG, and
    structured rather than pure noise because a tower fed white noise
    is being asked a question no image asks. A smooth gradient with
    grain has the low-frequency content a patch embedding is built
    to find.
    """
    rng = random.Random(seed)
    size = config.image_size
    frame = [[[math.sin((x + y) / 40.0) * 0.6 + rng.gauss(0, 0.15)
               for x in range(size)] for y in range(size)]
             for _ in range(3)]
    return [frame, frame]


def _ternarise_block(weights: vit.BlockWeights) -> vit.BlockWeights:
    """The same block, through the conversion kit's own rule choice.

    `choose_rules` is what `pack_checkpoint` uses, so this measures
    what the kit actually delivers rather than a rule invented for
    the occasion.
    """
    def convert(rows: list) -> list:
        quantised, scales, _rule = ternary.choose_rules(rows)
        return [[scale * value for value in row]
                for row, scale in zip(quantised, scales)]

    return vit.BlockWeights(
        ln1_w=weights.ln1_w, ln1_b=weights.ln1_b,
        qkv_w=convert(weights.qkv_w), qkv_b=weights.qkv_b,
        out_w=convert(weights.out_w), out_b=weights.out_b,
        ln2_w=weights.ln2_w, ln2_b=weights.ln2_b,
        up_w=convert(weights.up_w), up_b=weights.up_b,
        down_w=convert(weights.down_w), down_b=weights.down_b)


def _broken_block(stream: list, weights: vit.BlockWeights, config,
                  defect: str) -> list:
    """One block, wired wrong on purpose, one way at a time."""
    width = config.width
    eps = 1e-5 if defect == "wrong eps" else config.eps
    normed = [ops.layer_norm(t, weights.ln1_w, weights.ln1_b, eps)
              for t in stream]
    fused = [ops.linear(weights.qkv_w, weights.qkv_b, t) for t in normed]
    q = [t[:width] for t in fused]
    k = [t[width:2 * width] for t in fused]
    v = [t[2 * width:] for t in fused]
    if defect == "interleaved heads":
        # Head h takes every heads-th column instead of a contiguous
        # run. Every shape stays right; every subspace is scrambled.
        def split(seq):
            return [[row[h::config.heads] for row in seq]
                    for h in range(config.heads)]
        attended = ops.merge_heads([ops.attention(a, b, c) for a, b, c
                                    in zip(split(q), split(k), split(v))])
    else:
        attended = ops.multi_head_attention(q, k, v, config.heads)
    attended = [ops.linear(weights.out_w, weights.out_b, t)
                for t in attended]
    if defect == "no attention residual":
        stream = attended
    else:
        stream = [ops.add(t, d) for t, d in zip(stream, attended)]
    normed = [ops.layer_norm(t, weights.ln2_w, weights.ln2_b, eps)
              for t in stream]
    activate = ops.silu if defect == "silu for gelu" else ops.gelu
    hidden = [activate(ops.linear(weights.up_w, weights.up_b, t))
              for t in normed]
    hidden = [ops.linear(weights.down_w, weights.down_b, t) for t in hidden]
    return [ops.add(t, d) for t, d in zip(stream, hidden)]


#: The defects criterion 3 gates on, and the one it only reports.
_STRUCTURAL = ("interleaved heads", "no attention residual",
               "silu for gelu")
_REPORTED = ("wrong eps",)


@dataclass
class EncoderReport:
    """What the stack did to a ternary conversion.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        blocks: Blocks run.
        patches: Tokens carried through them.
        finite: Whether every activation stayed finite.
        end_cosine: Ternary vs f16, after the projector.
        block_cosine: Per-block cosine, ternary vs f16.
        f16_rms: Per-block residual RMS of the f16 tower.
        ternary_rms: The same for the ternary tower.
        defect_cosine: Defect name -> its end-to-end cosine to f16.
        bar: The pre-registered usability threshold.
        reason: Plain-language verdict.
    """

    passes: bool
    blocks: int = 0
    patches: int = 0
    finite: bool = True
    end_cosine: float = 0.0
    block_cosine: list = field(default_factory=list)
    f16_rms: list = field(default_factory=list)
    ternary_rms: list = field(default_factory=list)
    defect_cosine: dict = field(default_factory=dict)
    bar: float = 0.90
    reason: str = ""


def run_gate(patches: int = 8, blocks: int | None = None,
             seed: int = 0, bar: float = 0.90,
             defects: tuple = _STRUCTURAL + _REPORTED) -> EncoderReport:
    """Walk every tower through the encoder in one pass over the file."""
    checkpoint = checkpoint_path()
    if not checkpoint.exists():
        return EncoderReport(passes=False,
                             reason="FAIL - no checkpoint; set "
                                    "UQ_CONVERT_MODEL")
    opened = gguf.read(checkpoint)
    by = {tensor.name: tensor for tensor in opened.tensors}
    config = vit.load_config(opened.metadata)
    count = config.blocks if blocks is None else min(blocks, config.blocks)

    frames = synthetic_frames(config, seed=seed)
    embedding = vit.load_patch_embedding(opened, by, config)
    start = vit.patch_embed(frames, embedding, config, patches)
    del embedding

    # One pass over the file: a block's weights are read once and
    # every tower is advanced with them before they are dropped.
    exact = [list(token) for token in start]
    quantised = [list(token) for token in start]
    broken = {name: [list(token) for token in start] for name in defects}
    block_cosine: list = []
    f16_rms: list = []
    ternary_rms: list = []
    finite = True

    for index in range(count):
        weights = vit.load_block(opened, by, index)
        exact = vit.apply_block(exact, weights, config)
        quantised = vit.apply_block(quantised, _ternarise_block(weights),
                                    config)
        for name in defects:
            broken[name] = _broken_block(broken[name], weights, config, name)
        del weights
        block_cosine.append(_cosine(exact, quantised))
        f16_rms.append(_rms(exact))
        ternary_rms.append(_rms(quantised))
        if not all(math.isfinite(v) for token in exact for v in token):
            finite = False

    tail = vit.load_tail(opened, by)
    exact_out = vit.apply_tail(exact, tail, config)
    quantised_out = vit.apply_tail(quantised, tail, config)
    defect_cosine = {name: _cosine(exact_out,
                                   vit.apply_tail(broken[name], tail, config))
                     for name in defects}
    end_cosine = _cosine(exact_out, quantised_out)
    if not all(math.isfinite(v) for token in exact_out for v in token):
        finite = False

    structural = [name for name in defects if name in _STRUCTURAL]
    sensitive = all(defect_cosine[name] < end_cosine for name in structural)
    passes = finite and end_cosine >= bar and sensitive
    report = EncoderReport(
        passes=passes, blocks=count, patches=len(start), finite=finite,
        end_cosine=end_cosine, block_cosine=block_cosine, f16_rms=f16_rms,
        ternary_rms=ternary_rms, defect_cosine=defect_cosine, bar=bar)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {count} blocks, {len(start)} "
        f"tokens; ternary output cosine {end_cosine:.4f} against a bar of "
        f"{bar:.2f}; every activation finite: {finite}; structural defects "
        + ", ".join(f"{name} {defect_cosine[name]:.4f}"
                    for name in structural)
        + (" (all below the ternary tower, so the measurement can tell a "
           "bug from quantisation)" if sensitive
           else " - at least one defect damaged LESS than quantisation, so "
                "this measurement cannot distinguish correctness from noise")
        + "; reported: "
        + ", ".join(f"{name} {defect_cosine[name]:.4f}"
                    for name in defects if name in _REPORTED))
    return report


def discrimination_by_depth(depths: tuple = (1, 2, 4, 8),
                           patches: int = 6) -> dict:
    """Where the measurement stops telling a bug from quantisation.

    Criterion 3 failed at 27 blocks. That is either a fact about the
    METHOD - end-to-end cosine cannot separate wiring from noise - or
    a fact about QUANTISATION, that by block 27 it has destroyed more
    signal than a dropped residual does. The two have opposite
    consequences and only a measurement can tell them apart, so this
    runs the same comparison at shallower depths.

    Returns depth -> {"ternary": cosine, defect: cosine, ...}.
    """
    out: dict = {}
    for depth in depths:
        report = run_gate(patches=patches, blocks=depth)
        row = dict(report.defect_cosine)
        row["ternary"] = report.end_cosine
        row["discriminates"] = all(report.defect_cosine[name]
                                   < report.end_cosine
                                   for name in _STRUCTURAL)
        out[depth] = row
    return out


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"blocks {result.blocks}, tokens {result.patches}, "
          f"finite {result.finite}")
    print()
    print(f"{'blk':>4}{'cos(f16,ternary)':>20}{'f16 RMS':>12}"
          f"{'ternary RMS':>14}")
    for i, (c, a, b) in enumerate(zip(result.block_cosine, result.f16_rms,
                                      result.ternary_rms)):
        print(f"{i:>4}{c:>20.4f}{a:>12.4f}{b:>14.4f}")
    print()
    print(f"end-to-end cosine: {result.end_cosine:.4f} "
          f"(bar {result.bar:.2f})")
    for name, value in sorted(result.defect_cosine.items()):
        tag = "gated" if name in _STRUCTURAL else "reported"
        print(f"  {name:<24} {value:>8.4f}  [{tag}]")
    print()
    print(result.reason)
