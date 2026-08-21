"""What converting a transformer actually costs. The conversion gate.

A kit that turns a trained transformer into UltraQuant's ternary tier
is not a format converter. A model trained in bfloat16 was never told
its weights would end up in {-1, 0, +1}, so the conversion is a lossy
approximation, and the only honest kit is one that measures the loss
and says it out loud.

Three rules are compared on REAL tensors from a real checkpoint on
this machine - not on synthetic weights, because the whole question
is what a trained distribution does under ternarisation:

- **per matrix, 0.75** - `model/quantize.py`'s own rule, one scale
  for the whole tensor. Right for a network the forge trained with
  quantisation in the loop; the baseline here.
- **per row, 0.75** - the same threshold, one scale per output
  channel. Costs one float per row against a whole matrix.
- **per row, optimal** - the threshold that actually minimises
  ``||w - alpha*q||^2``, found by sorting the row once and sweeping
  the kept-set size. TWN's own closed form, no heuristic.

The measure is OUTPUT error, not weight error. A layer's job is
``W x``, so the number that matters is how far ``Ŵ x`` lands from
``W x`` for Gaussian probes; errors that cancel across a row never
reach the output, and a tensor can look badly quantised entry by
entry and behave.

**The criteria, written before the run.**

1. **Gain** - the optimal per-row rule must beat the per-matrix
   baseline on mean output error by more than the tensor-to-tensor
   sd of that contrast.
2. **The cost is stated, whatever it is** - the absolute output
   error and cosine of the best rule appear in the verdict. A kit
   that reported only a relative improvement would be selling a lie:
   "40% better than the worse option" is not "usable".
3. **Nothing is silently misread** - every tensor type the reader
   does not support is refused by name and counted; zero tensors are
   decoded on a guess.
4. **The output is ternary** - every value in {-1, 0, +1}, one scale
   per row, no NaN or infinity anywhere. One violation is a fail.
5. **One-dimensional tensors are measured separately** - biases and
   norms are small, high-leverage and a different question from a
   weight matrix. Averaging them together would hide whichever is
   worse.

**It was run, and FAILED criterion 1 - and the failure is the
finding.**

On 24 real weight tensors from a 899 MB checkpoint on this machine:

| rule | output error | cosine | non-zero |
|---|---:|---:|---:|
| per matrix, 0.75 | 0.4942 | 0.8675 | 0.520 |
| per row, 0.75 | 0.4801 | 0.8753 | 0.529 |
| per row, optimal | 0.4741 | 0.8793 | 0.490 |
| chosen per tensor | **0.4667** | **0.8837** | 0.503 |

Zero ternary violations, zero tensors misread, zero refused (this
checkpoint is entirely F32/F16).

**Criterion 1 fails**: the best rule beats the baseline by +0.027
against a tensor-to-tensor sd of 0.045. The improvement is real in
the mean and smaller than the variation between tensors, so the
honest reading is that **the rule choice is second-order and the
loss is first-order.** Ternarising a trained transformer costs
roughly half its output magnitude and lands at cosine 0.88 to the
original, and no threshold rule moves that materially.

Criterion 2 is why the gate exists, so here is the number without a
relative frame around it: **0.4667 relative output error, cosine
0.8837, per layer.** A stack of thirty such layers does not preserve
much. Conversion is a starting point for retraining, not a
substitute for it, and a kit that reported "5% better than the
naive rule" while omitting this would be selling a lie.

**Two things were learned on the way, both worth more than a pass.**

The optimal threshold - the one that provably minimises
``||w - alpha*q||^2``, found in closed form - produced a WORSE
matrix-vector product than the 0.75 heuristic on **7 of 24
tensors**. Minimising weight error is not minimising output error:
the optimal rule keeps fewer, larger weights (density 0.490 against
0.529), and the small weights it discards were carrying signal that
cancelled less than the Frobenius norm suggested. That is exactly
why the measure here is ``W x`` and not ``||W - Ŵ||``, and it is why
the kit now chooses per tensor - on one set of probes, scored on
another, because choosing and reporting on the same probes is
picking the winner after seeing the coin.

And a harness error, caught before it was believed: the first
measurement of 1-D tensors reported **2.41** relative error and an
obvious conclusion ("biases must stay float"). It was measuring a
bias as though it were a matrix - putting a random vector through a
1xN dot product - which is not what a bias does. Measured as what it
is, the vector's own relative error, a 1-D tensor takes **0.527**,
in line with the matrices. The conclusion that looked so clean was
an artifact of the wrong question.

Run it::

    python -m ultraquant.experiments.convert_gate
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import gguf, ternary

__all__ = ["ConversionReport", "run_gate", "checkpoint_path"]

#: A real checkpoint on this machine. Overridable, because the gate
#: is about the method rather than about one model.
_DEFAULT = Path(os.environ.get(
    "UQ_CONVERT_MODEL",
    r"C:\Users\Stephen Hawking\.lmstudio\models\HauhauCS"
    r"\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
    r"\mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"))


def checkpoint_path() -> Path:
    """The checkpoint under test, so tests can skip without it."""
    return _DEFAULT


@dataclass
class ConversionReport:
    """What each rule cost, per tensor class, on a real checkpoint.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        checkpoint: Which file was measured.
        tensors: How many matrices were converted.
        errors: Rule -> mean relative output error.
        cosines: Rule -> mean cosine between exact and approximate.
        density: Rule -> share of weights left non-zero.
        delta: Baseline error minus the best rule's error.
        tensor_sd: Tensor-to-tensor sd of that contrast.
        one_d_error: The same measure over 1-D tensors, kept apart.
        refused: Tensor type -> count refused by name.
        violations: Non-ternary values, bad scales, NaNs.
        reason: Plain-language verdict.
    """

    passes: bool
    checkpoint: str = ""
    tensors: int = 0
    errors: dict = field(default_factory=dict)
    cosines: dict = field(default_factory=dict)
    density: dict = field(default_factory=dict)
    delta: float = 0.0
    tensor_sd: float = 0.0
    one_d_error: float = 0.0
    refused: dict = field(default_factory=dict)
    violations: int = 0
    chosen: dict = field(default_factory=dict)
    reason: str = ""


_RULES = {
    "per matrix, 0.75": dict(per_row=False, optimal=False),
    "per row, 0.75": dict(per_row=True, optimal=False),
    "per row, optimal": dict(per_row=True, optimal=True),
}


def _check(quantised: list, scales: list) -> int:
    """Ternary output, or a count of what was not."""
    bad = 0
    for index, row in enumerate(quantised):
        scale = scales[index]
        if not math.isfinite(scale) or scale <= 0.0:
            bad += 1
        for value in row:
            if value not in (-1, 0, 1):
                bad += 1
    return bad


def run_gate(matrices: int = 24, rows_each: int = 96,
             probes: int = 4, path: Path | None = None,
             seed: int = 0) -> ConversionReport:
    """Convert real tensors three ways and report what each cost."""
    checkpoint = Path(path) if path is not None else _DEFAULT
    if not checkpoint.exists():
        return ConversionReport(
            passes=False, checkpoint=str(checkpoint),
            reason="FAIL - no checkpoint to convert; set "
                   "UQ_CONVERT_MODEL to a GGUF file")
    opened = gguf.read(checkpoint)

    refused: dict = {}
    weights: list = []
    ones: list = []
    for info in opened.tensors:
        if not info.readable:
            refused[info.type_name] = refused.get(info.type_name, 0) + 1
            continue
        if len(info.dims) >= 2 and info.rows > 1:
            weights.append(info)
        else:
            ones.append(info)

    reported = list(_RULES) + ["chosen per tensor"]
    per_rule: dict = {name: [] for name in reported}
    cosines: dict = {name: [] for name in reported}
    density: dict = {name: [] for name in reported}
    chosen: dict = {}
    violations = 0
    used = 0
    for info in weights[:matrices]:
        rows = opened.rows_of(info, 0, min(rows_each, info.rows))
        if not rows or not rows[0]:
            continue
        used += 1
        for name, options in _RULES.items():
            quantised, scales = ternary.ternarise_rows(rows, **options)
            violations += _check(quantised, scales)
            error, cosine = ternary.output_error(rows, quantised, scales,
                                                 probes=probes, seed=seed)
            per_rule[name].append(error)
            cosines[name].append(cosine)
            kept = sum(1 for row in quantised for value in row if value)
            total = sum(len(row) for row in quantised)
            density[name].append(kept / total if total else 0.0)
        # Chosen per tensor on one set of probes and SCORED on
        # another: choosing and reporting on the same probes would be
        # picking the winner after seeing the coin.
        quantised, scales, picked = ternary.choose_rules(rows, seed=seed)
        violations += _check(quantised, scales)
        error, cosine = ternary.output_error(rows, quantised, scales,
                                             probes=probes,
                                             seed=seed + 977)
        per_rule["chosen per tensor"].append(error)
        cosines["chosen per tensor"].append(cosine)
        kept = sum(1 for row in quantised for value in row if value)
        total = sum(len(row) for row in quantised)
        density["chosen per tensor"].append(kept / total if total else 0.0)
        chosen[picked] = chosen.get(picked, 0) + 1

    # One-dimensional tensors, measured apart: a bias is a vector and
    # ternarising it is a different question from ternarising a
    # matrix. Averaged together, whichever is worse would hide.
    one_d: list = []
    for info in ones[:matrices]:
        row = opened.rows_of(info, 0, 1)
        if not row or not row[0]:
            continue
        quantised, scales = ternary.ternarise_rows(row, per_row=True,
                                                   optimal=True)
        # For a bias or a norm the vector IS the thing used, so its
        # own relative error is the honest measure - not W*x, which
        # would be asking how a bias behaves as a matrix.
        one_d.append(ternary.weight_error(row, quantised, scales))

    errors = {name: statistics.mean(values) if values else 0.0
              for name, values in per_rule.items()}
    mean_cosines = {name: statistics.mean(values) if values else 0.0
                    for name, values in cosines.items()}
    mean_density = {name: statistics.mean(values) if values else 0.0
                    for name, values in density.items()}
    contrasts = [base - best for base, best
                 in zip(per_rule["per matrix, 0.75"],
                        per_rule["chosen per tensor"])]
    delta = statistics.mean(contrasts) if contrasts else 0.0
    tensor_sd = (statistics.stdev(contrasts) if len(contrasts) > 1
                 else 0.0)

    passes = (used > 0 and violations == 0 and tensor_sd > 0.0
              and delta > tensor_sd)
    best = errors.get("chosen per tensor", 0.0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {used} real tensors "
        f"converted; best rule (chosen per tensor) leaves "
        f"{best:.3f} relative output error at cosine "
        f"{mean_cosines.get('chosen per tensor', 0.0):.3f}, against the "
        f"per-matrix baseline's "
        f"{errors.get('per matrix, 0.75', 0.0):.3f} "
        f"({delta:+.3f} at {tensor_sd:.3f} tensor sd); "
        f"{violations} ternary violations; "
        f"{sum(refused.values())} tensors refused by name")
    return ConversionReport(
        passes=passes, checkpoint=checkpoint.name, tensors=used,
        errors=errors, cosines=mean_cosines, density=mean_density,
        delta=delta, tensor_sd=tensor_sd,
        one_d_error=statistics.mean(one_d) if one_d else 0.0,
        refused=refused, violations=violations, chosen=chosen,
        reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"checkpoint: {report.checkpoint}")
    print(f"matrices converted: {report.tensors}")
    print()
    print("rule                 out.error   cosine   nonzero")
    for name in list(_RULES) + ["chosen per tensor"]:
        print(f"{name:<20} {report.errors.get(name, 0):8.4f} "
              f"{report.cosines.get(name, 0):8.4f} "
              f"{report.density.get(name, 0):9.3f}")
    print()
    print(f"1-D tensors (biases, norms): {report.one_d_error:.4f} "
          f"relative error in the vector itself")
    print(f"ternary violations: {report.violations}")
    print(f"refused by name: {report.refused}")
    print(f"rule chosen per tensor: {report.chosen}")
    print()
    print(report.reason)
