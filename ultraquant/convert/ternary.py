"""Ternarise a trained tensor, and measure what that destroyed.

UltraQuant's own quantiser (`model/quantize.py`) takes a matrix and
returns one ternary matrix plus ONE scale for the whole thing. That
is the right shape for a network trained with quantisation in the
loop, which is what the forge builds. A transformer trained in
bfloat16 was never told its weights would end up in {-1, 0, +1}, so
converting one is not a format change - it is a lossy approximation,
and the only honest kit is one that reports the loss.

Two things are measured here rather than assumed:

- **Where the scale lives.** One scale per MATRIX has to cover every
  output channel at once, and a transformer's channels differ in
  magnitude by a lot. One scale per ROW costs an extra float per
  output channel - nothing, against a matrix - and should be much
  better. "Should be" is why there is a gate.
- **What the error means.** Weight error is the wrong measure: a
  layer's job is ``W x``, so the number that matters is how far
  ``Ŵ x`` lands from ``W x`` for real-shaped inputs. A tensor can
  look badly quantised entry-by-entry and behave, or look fine and
  not.
"""

from __future__ import annotations

import math
import random

__all__ = ["ternarise_row", "ternarise_row_optimal",
           "ternarise_rows", "reconstruct", "output_error",
           "weight_error"]


def ternarise_row(row: list) -> tuple:
    """One row to ``(ternary, alpha)`` by the TWN rule.

    The threshold is 0.75 * mean(|w|) over the ROW and the scale is
    the mean of the survivors, which is `model/quantize.py`'s rule
    applied per output channel instead of per matrix.
    """
    magnitudes = [abs(value) for value in row]
    if not magnitudes:
        return [], 1.0
    delta = 0.75 * (sum(magnitudes) / len(magnitudes))
    if delta == 0.0:
        return [0] * len(row), 1.0
    quantised: list = []
    survivors: list = []
    for value in row:
        if abs(value) > delta:
            quantised.append(1 if value > 0 else -1)
            survivors.append(abs(value))
        else:
            quantised.append(0)
    alpha = (sum(survivors) / len(survivors)) if survivors else 1.0
    return quantised, alpha


def ternarise_row_optimal(row: list) -> tuple:
    """One row to ``(ternary, alpha)`` at the BEST threshold, exactly.

    TWN's 0.75 * mean(|w|) is a heuristic, and for a network trained
    with quantisation in the loop it is a fine one. For a tensor
    trained in bfloat16 there is no reason to accept a heuristic when
    the optimum is available in closed form.

    Minimising ``||w - alpha*q||^2`` over alpha and the threshold is
    the same as MAXIMISING ``(sum of |w| over the kept set)^2 / (size
    of the kept set)``, and the kept set is always the k largest
    magnitudes - so sorting once and sweeping k finds the true
    optimum in n log n. The scale that goes with it is the mean of
    the kept magnitudes.
    """
    magnitudes = sorted((abs(value) for value in row), reverse=True)
    if not magnitudes or magnitudes[0] == 0.0:
        return [0] * len(row), 1.0
    best_objective = -1.0
    best_k = 1
    running = 0.0
    for k, magnitude in enumerate(magnitudes, start=1):
        running += magnitude
        objective = (running * running) / k
        if objective > best_objective:
            best_objective = objective
            best_k = k
    threshold = magnitudes[best_k - 1]
    kept = 0
    total = 0.0
    quantised: list = []
    for value in row:
        if abs(value) >= threshold and kept < best_k:
            quantised.append(1 if value > 0 else -1)
            total += abs(value)
            kept += 1
        else:
            quantised.append(0)
    alpha = (total / kept) if kept else 1.0
    return quantised, alpha


def ternarise_rows(rows: list, per_row: bool = True,
                   optimal: bool = False) -> tuple:
    """A matrix to ``(ternary rows, scales)``.

    With ``per_row`` false the whole matrix shares one scale, which
    is what UltraQuant's own quantiser does - kept here so the two
    can be measured against each other rather than argued about.
    """
    if per_row:
        rule = ternarise_row_optimal if optimal else ternarise_row
        quantised, scales = [], []
        for row in rows:
            q, alpha = rule(row)
            quantised.append(q)
            scales.append(alpha)
        return quantised, scales
    flat = [value for row in rows for value in row]
    _q, alpha = ternarise_row(flat)
    magnitudes = [abs(value) for value in flat]
    delta = 0.75 * (sum(magnitudes) / len(magnitudes)) if magnitudes else 0.0
    quantised = [[(0 if abs(v) <= delta else (1 if v > 0 else -1))
                  for v in row] for row in rows]
    return quantised, [alpha] * len(rows)


def reconstruct(quantised: list, scales: list) -> list:
    """The float matrix the ternary tier would actually use."""
    return [[scales[index] * value for value in row]
            for index, row in enumerate(quantised)]


def weight_error(rows: list, quantised: list, scales: list) -> float:
    """Relative Frobenius error - reported, never the headline.

    Entry-by-entry error is the intuitive measure and the wrong one:
    a layer's job is a matrix-vector product, and errors that cancel
    across a row do not reach the output.
    """
    numerator = 0.0
    denominator = 0.0
    for index, row in enumerate(rows):
        scale = scales[index]
        for column, value in enumerate(row):
            difference = value - scale * quantised[index][column]
            numerator += difference * difference
            denominator += value * value
    if denominator == 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def output_error(rows: list, quantised: list, scales: list,
                 probes: int = 8, seed: int = 0) -> tuple:
    """``(relative error, cosine)`` of ``Ŵx`` against ``Wx``.

    Gaussian probes, because that is roughly what a normalised
    activation looks like arriving at a linear layer, and because the
    measure has to be independent of any one input.
    """
    if not rows or not rows[0]:
        return 0.0, 1.0
    rng = random.Random(seed)
    width = len(rows[0])
    error_total = 0.0
    cosine_total = 0.0
    for probe in range(probes):
        x = [rng.gauss(0.0, 1.0) for _ in range(width)]
        exact, approximate = [], []
        for index, row in enumerate(rows):
            scale = scales[index]
            q = quantised[index]
            true_sum = 0.0
            near_sum = 0.0
            for column, value in enumerate(row):
                true_sum += value * x[column]
                near_sum += scale * q[column] * x[column]
            exact.append(true_sum)
            approximate.append(near_sum)
        difference = math.sqrt(sum((a - b) ** 2
                                   for a, b in zip(exact, approximate)))
        magnitude = math.sqrt(sum(a * a for a in exact))
        error_total += (difference / magnitude) if magnitude else 0.0
        dot = sum(a * b for a, b in zip(exact, approximate))
        near_magnitude = math.sqrt(sum(b * b for b in approximate))
        if magnitude and near_magnitude:
            cosine_total += dot / (magnitude * near_magnitude)
        else:
            cosine_total += 1.0
    return error_total / probes, cosine_total / probes


def choose_rules(rows: list, seed: int = 0) -> tuple:
    """Pick the rule that suits THIS tensor, and say which.

    The gate that measured the three rules found the honest reason
    to do this: the threshold that provably minimises
    ``||w - alpha*q||^2`` produced a WORSE matrix-vector product on
    seven of twenty-four real tensors. Minimising weight error is not
    minimising output error, and no single rule wins everywhere - so
    the rule is chosen per tensor by measuring, which costs a few
    probes and nothing else.

    The choice is made on one set of probes and must be scored on
    another. Choosing and reporting on the same probes would be
    picking the winner after seeing the coin.
    """
    candidates = {
        "per row, 0.75": dict(per_row=True, optimal=False),
        "per row, optimal": dict(per_row=True, optimal=True),
    }
    best_name = ""
    best_pair = None
    best_error = None
    for name, options in candidates.items():
        quantised, scales = ternarise_rows(rows, **options)
        error, _cosine = output_error(rows, quantised, scales,
                                      probes=3, seed=seed)
        if best_error is None or error < best_error:
            best_error = error
            best_name = name
            best_pair = (quantised, scales)
    return best_pair[0], best_pair[1], best_name
