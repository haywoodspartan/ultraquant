"""Stage 3's gate: are the concepts it proposes for itself real ones?

[ARCHITECTURE.md §11.3] states it before the work:

    Discovered clusters align with held-out labels above chance, measured with
    an assignment-invariant score.

Pre-registered: **adjusted Rand index >= 0.50**, over **12 seeds**, with the
margin exceeding seed variance — the same bar Stage 1 failed and Stage 2 cleared.

**The number of groups is never supplied.** Handing the clusterer the true class
count would leak the answer into the result; the whole claim is that the system
decides for itself what is worth calling a concept. ``discover()`` searches
k in 2..8 and picks by silhouette, which uses only geometry and never a label.
How often it lands on the true k is reported separately, because getting the
count wrong while still grouping sensibly is a different kind of partial success
than grouping badly.

**ARI, not accuracy.** Cluster 0 here and cluster 3 there may be the same set, so
accuracy is undefined. The adjusted Rand index counts agreeing pairs and
subtracts the agreement expected by chance, which is what makes "above chance" a
statement about zero instead of about a baseline that has to be argued for.

**Two controls, because a metric can flatter as easily as a method.**

* *Structureless data* — observations drawn from one blob, with labels assigned
  at random. Any honest pipeline scores ~0 here. If it does not, the measurement
  is broken and the headline number means nothing.
* *Shuffled labels* — the real observations, with their labels permuted. Same
  expectation, and it isolates the metric from the clustering.

Run it::

    python -m ultraquant.experiments.discovery_gate
"""

from __future__ import annotations

import random
import statistics
import sys

from ultraquant.reason.discovery import adjusted_rand_index, discover

__all__ = ["TARGET", "SEEDS", "make_families", "one_trial", "run_gate", "main"]

#: Pre-registered gate parameters.
TARGET = 0.50
SEEDS = 12

#: Reusable strokes, as in the Stage 1 gate: families differ, parts are shared.
STROKES: dict[str, list[int]] = {
    "top": [0, 1, 2, 3, 4],
    "middle": [10, 11, 12, 13, 14],
    "bottom": [20, 21, 22, 23, 24],
    "left": [0, 5, 10, 15, 20],
    "centre": [2, 7, 12, 17, 22],
    "right": [4, 9, 14, 19, 24],
    "slash": [4, 8, 12, 16, 20],
    "backslash": [0, 6, 12, 18, 24],
}


def _features(pixels: list[float]) -> list[float]:
    """The 30-feature vector the experts consume."""
    rows = [sum(pixels[r * 5:(r + 1) * 5]) / 5.0 for r in range(5)]
    return list(pixels) + rows


def make_families(
    rng: random.Random, classes: int = 4, per_class: int = 18, flips: int = 3,
    structured: bool = True,
) -> tuple[list[list[float]], list[int]]:
    """Observations and their true labels.

    Args:
        rng: Seeded randomness.
        classes: How many true groups exist.
        per_class: Observations per group.
        flips: Pixels inverted per observation.
        structured: When False, every observation comes from one blob and the
            labels are meaningless — the control.

    Returns:
        ``(vectors, labels)``.
    """
    names = list(STROKES)
    rng.shuffle(names)
    prototypes = []
    for index in range(classes):
        first, second = names[(2 * index) % len(names)], names[(2 * index + 1) % len(names)]
        lit = set(STROKES[first]) | set(STROKES[second])
        prototypes.append([1.0 if i in lit else 0.0 for i in range(25)])

    if not structured:
        # One blob: every observation is drawn from the same distribution, so no
        # grouping exists for any method to find.
        base = prototypes[0]
        prototypes = [list(base) for _ in range(classes)]

    vectors: list[list[float]] = []
    labels: list[int] = []
    for label, prototype in enumerate(prototypes):
        for _ in range(per_class):
            pixels = list(prototype)
            for position in rng.sample(range(25), flips):
                pixels[position] = 1.0 - pixels[position]
            vectors.append(_features(pixels))
            labels.append(label)
    return vectors, labels


def one_trial(seed: int, classes: int = 4, structured: bool = True) -> dict:
    """Discover groups in one dataset and score them against the truth."""
    rng = random.Random(seed)
    vectors, labels = make_families(rng, classes=classes, structured=structured)
    result = discover(vectors, k_range=(2, 8), seed=seed)
    found = result["assignment"]

    shuffled = list(labels)
    rng.shuffle(shuffled)
    return {
        "ari": adjusted_rand_index(labels, found),
        "ari_shuffled": adjusted_rand_index(shuffled, found),
        "k_found": result["k"],
        "k_true": classes,
        "k_correct": result["k"] == classes,
        "silhouette": result["silhouette"],
    }


def run_gate(seeds: int = SEEDS, classes: int = 4, structured: bool = True) -> dict:
    """Run the gate over ``seeds`` datasets."""
    trials = [one_trial(seed, classes, structured) for seed in range(seeds)]

    def mean(key: str) -> float:
        return statistics.fmean(float(t[key]) for t in trials)

    def sd(key: str) -> float:
        values = [float(t[key]) for t in trials]
        return statistics.stdev(values) if len(values) > 1 else 0.0

    ari = mean("ari")
    ari_sd = sd("ari")
    return {
        "seeds": seeds,
        "structured": structured,
        "ari": ari,
        "ari_sd": ari_sd,
        "ari_shuffled": mean("ari_shuffled"),
        "k_correct": mean("k_correct"),
        "k_found": [t["k_found"] for t in trials],
        "silhouette": mean("silhouette"),
        "passes": ari >= TARGET and ari > ari_sd,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a verdict."""
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds = int(argv[0]) if argv else SEEDS

    print("Stage 3 gate: are self-proposed concepts real concepts?")
    print(f"pre-registered: ARI >= {TARGET:.2f}, {seeds} seeds, "
          f"k chosen by the system (never supplied)")

    result = run_gate(seeds=seeds)
    print(f"\n  adjusted Rand index   {result['ari']:.3f} "
          f"(sd {result['ari_sd']:.3f})   target {TARGET:.2f}")
    print(f"  ARI / seed sd         {result['ari'] / result['ari_sd']:+.2f}"
          if result["ari_sd"] else "  ARI / seed sd         inf")
    print(f"  chose the true k      {result['k_correct']:.0%}  "
          f"(k found: {result['k_found']})")
    print(f"  mean silhouette       {result['silhouette']:.3f}")

    print("\ncontrols (both should sit at ~0)")
    print(f"  labels shuffled       {result['ari_shuffled']:+.3f}")
    control = run_gate(seeds=seeds, structured=False)
    print(f"  no structure at all   {control['ari']:+.3f} "
          f"(sd {control['ari_sd']:.3f})")

    print("\nsensitivity to the number of true classes:")
    print(f"  {'classes':>8} {'ARI':>8} {'k found':>9} {'right k':>9}")
    for classes in (2, 3, 4, 5, 6):
        run = run_gate(seeds=6, classes=classes)
        print(f"  {classes:>8} {run['ari']:>8.3f} "
              f"{statistics.fmean(run['k_found']):>9.1f} {run['k_correct']:>8.0%}")

    print("\nverdict")
    controls_clean = abs(control["ari"]) < 0.10 and abs(result["ari_shuffled"]) < 0.10
    if result["passes"] and controls_clean:
        print("  PASS - the groups it proposes for itself line up with the real")
        print("  classes well above chance, with the class count never supplied,")
        print("  and both controls sit at zero.")
    elif not controls_clean:
        print("  INVALID - a control did not sit at zero, so the measurement")
        print("  cannot be trusted. Fix that before reading the headline number.")
    else:
        print("  FAIL - below the pre-registered target. Stage 3 is not met.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
