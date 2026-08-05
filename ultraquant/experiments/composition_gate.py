"""Stage 2's gate: does composition solve what a single expert cannot?

[ARCHITECTURE.md §11.3] states it before the work:

    Correct answers on compound inputs that **no** single expert classifies
    correctly. Composition that only reproduces single-expert results has not
    composed anything.

**There is a hollow way to pass this, and it is avoided.** A compound input is
described by a *pair* of labels, and no single expert has pairs in its output
space at all — so "no single expert gets it right" is true by definition and
proves nothing. The measurement therefore uses the honest version: a **monolithic
baseline that is given the pair output space directly**, trained on the same
images with the same budget. Both arms are asked the same question, in the same
vocabulary.

**The real test is unseen combinations.** Training covers only some of the
possible (shape, mark) pairs. Both arms then face pairs whose *parts* were both
seen but which never occurred *together*. A monolithic classifier has no training
signal for those classes and cannot be expected to produce them; a factored
reading built on a blackboard should, because each factor was learned separately.
That asymmetry is not a trick — it is the entire reason composition is worth
having, and the numbers below are only meaningful because the seen-combination
column shows the baseline is strong where it has data.

**Both experts see the whole image.** Handing the shape expert only border pixels
and the mark expert only interior pixels would perform the decomposition by hand
and leave nothing to learn. Each expert sees every pixel and must attend to its
own factor while ignoring the other's.

Run it::

    python -m ultraquant.experiments.composition_gate
"""

from __future__ import annotations

import random
import statistics
import sys
import tempfile
import shutil
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.reason.blackboard import Blackboard, ExpertContributor, run_blackboard
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.vault import ShardVault

__all__ = [
    "SHAPES",
    "MARKS",
    "compound",
    "one_trial",
    "run_gate",
    "main",
]

#: Border patterns, drawn on the outer ring of a 5x5 grid.
SHAPES: dict[str, list[int]] = {
    "box": [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24],
    "rails": [0, 4, 5, 9, 10, 14, 15, 19, 20, 24],
    "caps": [0, 1, 2, 3, 4, 20, 21, 22, 23, 24],
    "corners": [0, 4, 20, 24],
}

#: Interior patterns, drawn on the inner 3x3 — disjoint from every border.
#:
#: These are **nested**: ``dot`` is a subset of ``bar``, which is a subset of
#: ``plus``. That was not deliberate, and it is left in place because it is what
#: the gate was measured on. It caps how well the mark factor can be read at all
#: — see :data:`DISTINCT_MARKS` and §11.6 — and moving the goalposts after
#: seeing the result would be worth less than reporting it.
MARKS: dict[str, list[int]] = {
    "plus": [7, 11, 12, 13, 17],
    "ex": [6, 8, 12, 16, 18],
    "bar": [11, 12, 13],
    "dot": [12],
}

#: The same idea with a well-posed vocabulary: **no containment**, and pairwise
#: Hamming distance >= 4. Containment was the actual flaw in :data:`MARKS`; a
#: distance of 6 is not merely unmet but impossible here, since the inner 3x3
#: holds only three disjoint triples and a fourth pattern must overlap one of
#: them. Used only for the secondary analysis that separates "the mark factor is
#: hard to see" from "composition does not work".
DISTINCT_MARKS: dict[str, list[int]] = {
    "plus": [7, 11, 12, 13, 17],
    "ex": [6, 8, 12, 16, 18],
    "top": [6, 7, 8],
    "bottom": [16, 17, 18],
}


def compound(shape: str, mark: str, marks: dict | None = None) -> list[float]:
    """The 25-pixel overlay of a border shape and an interior mark."""
    lit = set(SHAPES[shape]) | set((marks or MARKS)[mark])
    return [1.0 if i in lit else 0.0 for i in range(25)]


def _features(pixels: list[float]) -> list[float]:
    """The 30-feature vector the experts consume (25 pixels + 5 row means)."""
    rows = [sum(pixels[r * 5:(r + 1) * 5]) / 5.0 for r in range(5)]
    return list(pixels) + rows


def _noisy(rng: random.Random, pixels: list[float], flips: int) -> list[float]:
    """A compound with ``flips`` pixels inverted."""
    out = list(pixels)
    for position in rng.sample(range(25), flips):
        out[position] = 1.0 - out[position]
    return _features(out)


def split_combinations(
    rng: random.Random, held_out: int = 6, marks: dict | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split the 16 pairs into seen and unseen, keeping every part seen.

    A held-out pair whose shape or mark appears nowhere in training would test
    recognition of something never shown, not composition of things that were.

    Returns:
        ``(seen, unseen)`` pairs.
    """
    marks = marks or MARKS
    every = [(s, m) for s in SHAPES for m in marks]
    for _attempt in range(200):
        rng.shuffle(every)
        unseen, seen = every[:held_out], every[held_out:]
        if {s for s, _ in seen} == set(SHAPES) and {m for _, m in seen} == set(marks):
            return seen, unseen
    raise RuntimeError("could not split while keeping every part in training")


def _samples(
    rng: random.Random, pairs: list[tuple[str, str]], per_pair: int, flips: int,
    marks: dict | None = None,
) -> tuple[list[list[float]], list[tuple[str, str]]]:
    """Noisy examples of each pair, with their (shape, mark) truth."""
    xs: list[list[float]] = []
    ys: list[tuple[str, str]] = []
    for shape, mark in pairs:
        base = compound(shape, mark, marks)
        for _ in range(per_pair):
            xs.append(_noisy(rng, base, flips))
            ys.append((shape, mark))
    return xs, ys


def one_trial(
    seed: int,
    per_pair: int = 24,
    flips: int = 2,
    epochs: int = 40,
    held_out: int = 6,
    marks: dict | None = None,
) -> dict:
    """Train both arms once and score them on seen and unseen pairs.

    Returns:
        Accuracies for each arm on each condition, plus the split sizes.
    """
    marks = marks or MARKS
    rng = random.Random(seed)
    seen, unseen = split_combinations(rng, held_out=held_out, marks=marks)

    train_x, train_y = _samples(rng, seen, per_pair, flips, marks)
    seen_x, seen_y = _samples(rng, seen, 12, flips, marks)
    unseen_x, unseen_y = _samples(rng, unseen, 12, flips, marks)

    workdir = Path(tempfile.mkdtemp(prefix="uq_compose_"))
    try:
        vault = ShardVault(workdir)
        pool = ExpertPool(vault, ShardCache(8 * 1024 * 1024),
                          input_dim=30, hidden=(24,), seed=seed)

        # -- composed arm: one expert per factor, both seeing whole images ----
        shape_labels = sorted(SHAPES)
        mark_labels = sorted(marks)
        pool.train_expert("shape", train_x, [shape_labels.index(s) for s, _ in train_y],
                          shape_labels, epochs=epochs, lr=0.05)
        pool.train_expert("mark", train_x, [mark_labels.index(m) for _, m in train_y],
                          mark_labels, epochs=epochs, lr=0.05)
        contributors = [
            ExpertContributor(pool, "shape", "shape"),
            ExpertContributor(pool, "mark", "mark"),
        ]

        def composed(x: list[float]) -> tuple[str, str]:
            board = run_blackboard(Blackboard(x), contributors)
            reading = board.reading()
            return reading.get("shape", ""), reading.get("mark", "")

        # -- monolithic arm: the same images, the pair output space -----------
        # Given every one of the 16 classes, not only the ones it will see, so
        # it is not prevented from naming an unseen pair by construction.
        pairs = [(s, m) for s in shape_labels for m in mark_labels]
        pool.train_expert(
            "joint", train_x, [pairs.index(y) for y in train_y],
            [f"{s}|{m}" for s, m in pairs], epochs=epochs, lr=0.05,
        )

        def monolithic(x: list[float]) -> tuple[str, str]:
            label, _confidence = pool.predict("joint", x)
            shape, _, mark = label.partition("|")
            return shape, mark

        def score(predict, xs, ys) -> float:
            return sum(predict(x) == y for x, y in zip(xs, ys)) / len(ys)

        return {
            "composed_seen": score(composed, seen_x, seen_y),
            "composed_unseen": score(composed, unseen_x, unseen_y),
            "monolithic_seen": score(monolithic, seen_x, seen_y),
            "monolithic_unseen": score(monolithic, unseen_x, unseen_y),
            "seen_pairs": len(seen),
            "unseen_pairs": len(unseen),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_gate(seeds: int = 12, marks: dict | None = None) -> dict:
    """Run the paired experiment and summarise it."""
    trials = [one_trial(seed, marks=marks) for seed in range(seeds)]

    def stats(key: str) -> tuple[float, float]:
        values = [t[key] for t in trials]
        return statistics.fmean(values), (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )

    composed_unseen, composed_unseen_sd = stats("composed_unseen")
    mono_unseen, _ = stats("monolithic_unseen")
    composed_seen, _ = stats("composed_seen")
    mono_seen, _ = stats("monolithic_seen")

    deltas = [t["composed_unseen"] - t["monolithic_unseen"] for t in trials]
    delta = statistics.fmean(deltas)
    delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "seeds": seeds,
        "composed_seen": composed_seen,
        "composed_unseen": composed_unseen,
        "composed_unseen_sd": composed_unseen_sd,
        "monolithic_seen": mono_seen,
        "monolithic_unseen": mono_unseen,
        "delta_unseen": delta,
        "delta_sd": delta_sd,
        # Same discipline as stage 1: the margin must clear seed variance, not
        # merely clear zero.
        "passes": delta > delta_sd,
        # The baseline has to be strong where it has data, or the comparison is
        # against a broken arm and means nothing.
        "baseline_sound": mono_seen > 0.80,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a verdict."""
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds = int(argv[0]) if argv else 12

    print("Stage 2 gate: does composition solve what one expert cannot?")
    print(f"{seeds} seeds; 10 of 16 (shape, mark) pairs trained, 6 never seen together")

    result = run_gate(seeds=seeds)
    print(f"\n{'':<14}{'seen pairs':>12}{'unseen pairs':>14}")
    print(f"{'monolithic':<14}{result['monolithic_seen']:>12.3f}"
          f"{result['monolithic_unseen']:>14.3f}")
    print(f"{'composed':<14}{result['composed_seen']:>12.3f}"
          f"{result['composed_unseen']:>14.3f}")
    print(f"\n  delta on unseen  {result['delta_unseen']:+.3f}  "
          f"(seed sd {result['delta_sd']:.3f})")
    ratio = (result["delta_unseen"] / result["delta_sd"]
             if result["delta_sd"] else float("inf"))
    print(f"  delta / seed sd  {ratio:+.2f}   (gate needs > 1.00)")
    print(f"  baseline sound   {result['baseline_sound']} "
          f"(monolithic must be strong where it has data)")

    print("\nverdict")
    if result["passes"] and result["baseline_sound"]:
        print("  PASS - composition answers compound inputs the monolithic")
        print("  classifier cannot, and the baseline is strong where it has data.")
    elif not result["baseline_sound"]:
        print("  INVALID - the monolithic baseline is weak even on pairs it was")
        print("  trained on, so it is not a fair comparison. Fix it before reading")
        print("  anything into the unseen column.")
    else:
        print("  FAIL - composition does not beat the monolithic baseline by more")
        print("  than seed variance. Stage 2 is not met.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
