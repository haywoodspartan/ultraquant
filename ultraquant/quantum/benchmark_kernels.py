"""Does a classically-hard feature map ever beat a cheap one? An experiment.

Section 7.1 measured the entangling ZZ map generalising *worse* than a plain
angle encoding on the glyph task, and left the open question: is there a task
where the expensive map wins? The glyph task is a poor test — its classes are
linearly separable, so no amount of expressive power can help.

The candidate is a problem whose label depends on *interactions between features*
rather than on the features themselves. Parity is the canonical example: with
±1 features, ``y = x_a · x_b`` cannot be separated by any function of the
individual coordinates, but is trivial once the *product* is available. The ZZ
map puts exactly those products into the phase — ``exp(i (pi - x_i)(pi - x_j))``
— so if entanglement ever earns its cost, it should earn it here.

Run ``python -m ultraquant.quantum.benchmark_kernels`` for the comparison.
Whatever it prints is the answer, including if the answer is "it doesn't help".
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

from ultraquant.quantum.encodings import AngleEncoder, ZZFeatureMap
from ultraquant.quantum.kernel import inversion_test

__all__ = ["parity_task", "linear_kernel", "kernel_accuracy", "compare", "main"]


@dataclass
class Task:
    """A labelled dataset plus a description of where its signal lives."""

    name: str
    xs: list[list[float]]
    ys: list[int]
    order: int
    hidden_features: tuple[int, ...]


def parity_task(
    num_features: int = 6, order: int = 2, samples: int = 40, seed: int = 0
) -> Task:
    """Labels depending on the parity of a hidden subset of features.

    Args:
        num_features: Feature-vector width.
        order: How many features participate in the label. 1 is linearly
            separable; 2 and above are not.
        samples: Number of examples.
        seed: RNG seed.

    Returns:
        A :class:`Task` whose label is the parity of ``order`` hidden features.
    """
    rng = random.Random(seed)
    hidden = tuple(rng.sample(range(num_features), order))
    xs: list[list[float]] = []
    ys: list[int] = []
    for _sample in range(samples):
        bits = [rng.choice((0, 1)) for _ in range(num_features)]
        label = 0
        for index in hidden:
            label ^= bits[index]
        # Map to angles away from the extremes, where the ZZ phase terms vanish.
        xs.append([0.25 * math.pi + 0.5 * math.pi * b for b in bits])
        ys.append(label)
    return Task(f"parity-{order}", xs, ys, order, hidden)


def linear_kernel(a: list[float], b: list[float]) -> float:
    """Normalised dot product — the cheapest classical baseline."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def quadratic_kernel(a: list[float], b: list[float]) -> float:
    """Squared dot product — a classical kernel that *does* see pairwise terms.

    Included deliberately: if the quantum map only beats a linear kernel, that
    proves nothing, because this classical kernel captures the same second-order
    structure for a fraction of the cost.
    """
    return linear_kernel(a, b) ** 2


def kernel_accuracy(task: Task, kernel, holdout: float = 0.4, seed: int = 0) -> float:
    """Nearest-centroid accuracy in the kernel's induced space.

    The classifier is deliberately trivial so that any difference in accuracy is
    attributable to the kernel rather than to the learner.
    """
    order = list(range(len(task.xs)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * (1 - holdout))
    train, test = order[:cut], order[cut:]

    correct = 0
    for i in test:
        scores: dict[int, list[float]] = {}
        for j in train:
            scores.setdefault(task.ys[j], []).append(kernel(task.xs[i], task.xs[j]))
        if not scores:
            continue
        predicted = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
        correct += int(predicted == task.ys[i])
    return correct / max(len(test), 1)


def _quantum_kernel(encoder):
    """Wrap an encoder into a kernel function, caching prepared circuits."""
    cache: dict[tuple, object] = {}

    def circuit_for(values: list[float]):
        key = tuple(values)
        if key not in cache:
            cache[key] = encoder.encode(list(values))
        return cache[key]

    def kernel(a: list[float], b: list[float]) -> float:
        return inversion_test(circuit_for(a), circuit_for(b))

    return kernel


def compare(
    num_features: int = 6, samples: int = 40, orders: tuple[int, ...] = (1, 2, 3),
    seed: int = 0,
) -> list[dict]:
    """Compare kernels across tasks of increasing interaction order.

    Returns:
        One row per order with each kernel's accuracy.
    """
    rows = []
    for order in orders:
        task = parity_task(num_features, order, samples, seed)
        angle = AngleEncoder(num_features, reuploads=1, entangle=False)
        angle_ent = AngleEncoder(num_features, reuploads=1, entangle=True)
        zz_linear = ZZFeatureMap(num_features, repetitions=2, entanglement="linear")
        zz_full = ZZFeatureMap(num_features, repetitions=2, entanglement="full")

        rows.append({
            "order": order,
            "hidden": task.hidden_features,
            "linear": kernel_accuracy(task, linear_kernel, seed=seed),
            "quadratic": kernel_accuracy(task, quadratic_kernel, seed=seed),
            "angle": kernel_accuracy(task, _quantum_kernel(angle), seed=seed),
            "angle+ent": kernel_accuracy(task, _quantum_kernel(angle_ent), seed=seed),
            "zz_linear": kernel_accuracy(task, _quantum_kernel(zz_linear), seed=seed),
            "zz_full": kernel_accuracy(task, _quantum_kernel(zz_full), seed=seed),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and print a verdict.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Does an entangling feature map beat a cheap one?"
    )
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=5,
                        help="average over this many dataset draws")
    args = parser.parse_args(argv)

    names = ["linear", "quadratic", "angle", "angle+ent", "zz_linear", "zz_full"]
    print("Kernel accuracy by interaction order "
          f"({args.features} features, {args.samples} samples, "
          f"{args.seeds} seeds averaged)")
    print("  order 1 is linearly separable; higher orders are not.\n")
    header = f"{'order':>6}" + "".join(f"{n:>12}" for n in names)
    print(header)

    totals: dict[int, dict[str, float]] = {}
    for order in args.orders:
        accumulated = {n: 0.0 for n in names}
        for seed in range(args.seeds):
            for row in compare(args.features, args.samples, (order,), seed=seed):
                for n in names:
                    accumulated[n] += row[n]
        for n in names:
            accumulated[n] /= args.seeds
        totals[order] = accumulated
        print(f"{order:>6}" + "".join(f"{accumulated[n]:>12.3f}" for n in names))

    print()
    for order in args.orders:
        best_quantum = max(totals[order]["zz_linear"], totals[order]["zz_full"])
        best_classical = max(totals[order]["linear"], totals[order]["quadratic"])
        margin = best_quantum - best_classical
        verdict = (
            "entangling map WINS" if margin > 0.05
            else "no advantage" if margin > -0.05
            else "entangling map LOSES"
        )
        print(f"  order {order}: quantum {best_quantum:.3f} vs classical "
              f"{best_classical:.3f}  ->  {verdict}")

    print("\n  The quadratic kernel is the honest comparison: it is classical and")
    print("  it also sees pairwise products. Beating only the linear kernel would")
    print("  show nothing about quantum advantage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
