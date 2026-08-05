"""Stage 1's gate: does a shared encoder actually buy transfer?

[ARCHITECTURE.md §11.3] states the gate before the work, so the answer cannot be
adjusted afterwards:

    A held-out category trained on 5 examples through the encoder beats the same
    category trained on 5 raw-pixel examples, by a margin larger than seed
    variance. If it does not, the encoder is decoration and the stage is
    abandoned.

**The experiment is paired.** For each seed, the same data, the same held-out
family, the same classifier architecture and the same hyperparameters are used in
both arms; only the input representation differs. What is reported is the
per-seed difference, which removes the between-seed variation that would
otherwise swamp the effect.

**The encoder never sees the held-out family, and never sees a label.** It is
trained by masked reconstruction on the pretraining families only. Both are
asserted at runtime rather than left to reviewer trust.

**There is a control, and it is the important part.** The same measurement is run
on unstructured data, where the labels are arbitrary pixel patterns sharing no
parts. Transfer *should* fail there. A method that helps on both is not
transferring structure — it is adding capacity, and would pass a gate that only
tested the favourable case. Reporting the structured result alone would be the
easy way to produce a positive answer.

Run it::

    python -m ultraquant.experiments.transfer_gate
"""

from __future__ import annotations

import random
import statistics
import sys

from ultraquant.model.encoder import PatternEncoder
from ultraquant.model.network import UltraQuantNet

__all__ = [
    "STROKES",
    "one_trial",
    "one_trial_supervised",
    "compositional_families",
    "unstructured_families",
    "run_gate",
    "main",
]

#: Reusable parts. Compositional families are built from these, so a family the
#: encoder never saw is still made of pieces it has seen.
STROKES: dict[str, list[int]] = {
    "row0": [0, 1, 2, 3, 4],
    "row2": [10, 11, 12, 13, 14],
    "row4": [20, 21, 22, 23, 24],
    "col0": [0, 5, 10, 15, 20],
    "col2": [2, 7, 12, 17, 22],
    "col4": [4, 9, 14, 19, 24],
    "diag": [0, 6, 12, 18, 24],
    "anti": [4, 8, 12, 16, 20],
    "topleft": [0, 1, 5, 6],
    "botright": [18, 19, 23, 24],
    "centre": [6, 7, 8, 11, 12, 13, 16, 17, 18],
}


#: Pixels inverted per generated example. Calibrated **on the baseline arm
#: alone**, before the encoder was run at all: at the original 2 flips a 5-shot
#: raw classifier scored 0.992, so neither arm had room to differ and the
#: measurement could not have detected transfer in either direction. 6 flips puts
#: the baseline near 0.67, which is where a difference is visible if one exists.
#: Choosing the operating point from baseline difficulty is legitimate; choosing
#: it from the treatment's result would not be.
DEFAULT_FLIPS = 6


def _pixels(indices: set[int]) -> list[float]:
    """A 25-pixel vector with ``indices`` lit."""
    return [1.0 if i in indices else 0.0 for i in range(25)]


def _features(pixels: list[float]) -> list[float]:
    """The 30-feature vector the experts actually consume (25 + 5 row means)."""
    rows = [sum(pixels[r * 5:(r + 1) * 5]) / 5.0 for r in range(5)]
    return list(pixels) + rows


def compositional_families(
    rng: random.Random, families: int = 7, labels: int = 4
) -> list[list[list[float]]]:
    """Families whose labels are pairs of shared strokes.

    Every stroke pair is globally unique, so a held-out family cannot be
    recognised by memorising a combination seen during pretraining — but its
    *parts* were all seen. That is the structure a shared encoder should be able
    to reuse, and the whole hypothesis under test.

    Args:
        rng: Seeded source of randomness.
        families: How many families to build.
        labels: Labels per family.

    Returns:
        ``families`` lists of ``labels`` prototype pixel vectors.
    """
    names = list(STROKES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rng.shuffle(pairs)
    needed = families * labels
    if len(pairs) < needed:  # pragma: no cover - guards a bad configuration
        raise ValueError(f"need {needed} distinct stroke pairs, have {len(pairs)}")
    out = []
    for f in range(families):
        family = []
        for l in range(labels):
            first, second = pairs[f * labels + l]
            family.append(_pixels(set(STROKES[first]) | set(STROKES[second])))
        out.append(family)
    return out


def unstructured_families(
    rng: random.Random, families: int = 7, labels: int = 4
) -> list[list[list[float]]]:
    """The control: labels are arbitrary patterns sharing no parts.

    There is nothing here for an encoder to carry from one family to another, so
    a method that still shows a gain is adding capacity rather than transferring
    structure.
    """
    out = []
    for _ in range(families):
        family = []
        for _ in range(labels):
            lit = set(rng.sample(range(25), 10))
            family.append(_pixels(lit))
        out.append(family)
    return out


def _noisy(rng: random.Random, prototype: list[float], flips: int) -> list[float]:
    """A prototype with ``flips`` pixels inverted, as a feature vector."""
    pixels = list(prototype)
    for position in rng.sample(range(25), flips):
        pixels[position] = 1.0 - pixels[position]
    return _features(pixels)


def _samples(
    rng: random.Random, family: list[list[float]], per_label: int, flips: int
) -> tuple[list[list[float]], list[int]]:
    """``per_label`` noisy examples of each label in ``family``."""
    xs: list[list[float]] = []
    ys: list[int] = []
    for index, prototype in enumerate(family):
        for _ in range(per_label):
            xs.append(_noisy(rng, prototype, flips))
            ys.append(index)
    return xs, ys


def _train_head(
    xs: list[list[float]], ys: list[int], classes: int, seed: int,
    epochs: int, lr: float,
) -> UltraQuantNet:
    """Train a classifier — identical in both arms except its input width."""
    net = UltraQuantNet(
        input_dim=len(xs[0]), hidden_dims=[16], num_classes=classes, seed=seed
    )
    rng = random.Random(seed)
    order = list(range(len(xs)))
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 8):
            chunk = order[start:start + 8]
            net.train_batch([xs[i] for i in chunk], [ys[i] for i in chunk], lr)
    return net


def _accuracy(net: UltraQuantNet, xs: list[list[float]], ys: list[int]) -> float:
    """Fraction correct."""
    correct = sum(net.predict(x)[0] == y for x, y in zip(xs, ys))
    return correct / len(ys)


def one_trial(
    seed: int,
    structured: bool = True,
    shots: int = 5,
    embed_dim: int = 16,
    pretrain_per_label: int = 30,
    encoder_epochs: int = 40,
    head_epochs: int = 60,
    flips: int = DEFAULT_FLIPS,
) -> dict:
    """Run both arms once and return their held-out accuracies.

    Args:
        seed: Controls data, encoder init and head init.
        structured: Compositional data when True, the control when False.
        shots: Labelled examples per held-out label.
        embed_dim: Encoder bottleneck width.
        pretrain_per_label: Unlabelled examples per pretraining label.
        encoder_epochs: Encoder training passes.
        head_epochs: Classifier training passes, identical in both arms.
        flips: Pixels inverted per generated example.

    Returns:
        ``{"raw": acc, "encoded": acc, "delta": encoded - raw}``.
    """
    rng = random.Random(seed)
    builder = compositional_families if structured else unstructured_families
    families = builder(rng)
    *pretrain, holdout = families

    # -- unsupervised pretraining, on the pretraining families only ---------
    unlabelled: list[list[float]] = []
    for family in pretrain:
        xs, _ys = _samples(rng, family, pretrain_per_label, flips)
        unlabelled.extend(xs)
    encoder = PatternEncoder(input_dim=30, embed_dim=embed_dim, seed=seed)
    encoder.fit(unlabelled, epochs=encoder_epochs, lr=0.05, seed=seed)

    # -- the few-shot task, identical for both arms -------------------------
    train_x, train_y = _samples(rng, holdout, shots, flips)
    test_x, test_y = _samples(rng, holdout, 40, flips)
    classes = len(holdout)

    raw_net = _train_head(train_x, train_y, classes, seed, head_epochs, 0.05)
    raw = _accuracy(raw_net, test_x, test_y)

    enc_train = [encoder.encode(x) for x in train_x]
    enc_test = [encoder.encode(x) for x in test_x]
    enc_net = _train_head(enc_train, train_y, classes, seed, head_epochs, 0.05)
    encoded = _accuracy(enc_net, enc_test, test_y)

    return {"raw": raw, "encoded": encoded, "delta": encoded - raw}


def one_trial_supervised(
    seed: int,
    structured: bool = True,
    shots: int = 5,
    embed_dim: int = 32,
    pretrain_per_label: int = 30,
    pretrain_epochs: int = 40,
    head_epochs: int = 60,
    flips: int = DEFAULT_FLIPS,
) -> dict:
    """The second mechanism: a representation learned *discriminatively*.

    Masked reconstruction was refuted — it learns to predict missing pixels,
    which raw pixels already support, and compression is not the same thing as
    abstraction. This tests the other candidate, and the one the library
    actually has material for: it is full of categories that already carry
    labels. Train one net across every pretraining family's labels at once, then
    use its last hidden layer as the representation for a held-out family.

    The held-out family contributes nothing but its ``shots`` labelled examples,
    exactly as in :func:`one_trial`.

    Returns:
        ``{"raw": acc, "encoded": acc, "delta": encoded - raw}``.
    """
    rng = random.Random(seed)
    builder = compositional_families if structured else unstructured_families
    families = builder(rng)
    *pretrain, holdout = families

    # -- one classifier over every pretraining label --------------------------
    pre_x: list[list[float]] = []
    pre_y: list[int] = []
    offset = 0
    for family in pretrain:
        xs, ys = _samples(rng, family, pretrain_per_label, flips)
        pre_x.extend(xs)
        pre_y.extend(y + offset for y in ys)
        offset += len(family)
    trunk = UltraQuantNet(
        input_dim=30, hidden_dims=[embed_dim], num_classes=offset, seed=seed
    )
    order = list(range(len(pre_x)))
    shuffler = random.Random(seed)
    for _ in range(pretrain_epochs):
        shuffler.shuffle(order)
        for start in range(0, len(order), 8):
            chunk = order[start:start + 8]
            trunk.train_batch([pre_x[i] for i in chunk], [pre_y[i] for i in chunk], 0.05)

    train_x, train_y = _samples(rng, holdout, shots, flips)
    test_x, test_y = _samples(rng, holdout, 40, flips)
    classes = len(holdout)

    raw_net = _train_head(train_x, train_y, classes, seed, head_epochs, 0.05)
    raw = _accuracy(raw_net, test_x, test_y)

    enc_train = [trunk.embed(x) for x in train_x]
    enc_test = [trunk.embed(x) for x in test_x]
    enc_net = _train_head(enc_train, train_y, classes, seed, head_epochs, 0.05)
    encoded = _accuracy(enc_net, enc_test, test_y)

    return {"raw": raw, "encoded": encoded, "delta": encoded - raw}


def run_gate(
    seeds: int = 20, shots: int = 5, structured: bool = True,
    flips: int = DEFAULT_FLIPS, mechanism=None,
) -> dict:
    """Run the paired experiment over ``seeds`` seeds.

    Returns:
        Summary statistics including the mean paired difference, its 95%
        confidence interval, and how often the encoder arm won.
    """
    trial = mechanism or one_trial
    trials = [trial(seed, structured=structured, shots=shots, flips=flips)
              for seed in range(seeds)]
    deltas = [t["delta"] for t in trials]
    raw = [t["raw"] for t in trials]
    encoded = [t["encoded"] for t in trials]
    mean = statistics.fmean(deltas)
    stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    # Normal approximation is adequate here; the point is whether the interval
    # clears zero by a visible margin, not a precise p-value.
    half = 1.96 * stdev / (len(deltas) ** 0.5) if deltas else 0.0
    return {
        "seeds": seeds,
        "shots": shots,
        "structured": structured,
        "flips": flips,
        "raw_mean": statistics.fmean(raw),
        "encoded_mean": statistics.fmean(encoded),
        "delta_mean": mean,
        "delta_stdev": stdev,
        "ci95": (mean - half, mean + half),
        "wins": sum(1 for d in deltas if d > 0),
        "ties": sum(1 for d in deltas if d == 0),
        # Two different questions, kept apart on purpose.
        #
        # `nonzero` asks whether the mean difference is distinguishable from
        # zero. It is a statement about the precision of an average, and enough
        # seeds will make it true of an arbitrarily small effect: at 100 seeds
        # the discriminative trunk reached +0.014 with a lower bound of +0.0002.
        #
        # `passes` is the gate as written in ARCHITECTURE.md §11.3 -- "by a
        # margin larger than seed variance". That is a statement about whether
        # the effect matters next to the noise a practitioner would actually
        # see. Reporting `nonzero` as the verdict would be answering an easier
        # question than the one committed to.
        "nonzero": (mean - half) > 0.0,
        "passes": mean > stdev,
    }


def _report(title: str, result: dict) -> None:
    """Print one condition's result."""
    low, high = result["ci95"]
    print(f"\n{title}")
    print(f"  raw features     {result['raw_mean']:.3f}")
    print(f"  through encoder  {result['encoded_mean']:.3f}")
    print(f"  paired delta     {result['delta_mean']:+.3f}  "
          f"(sd {result['delta_stdev']:.3f}, 95% CI {low:+.3f} to {high:+.3f})")
    print(f"  encoder wins     {result['wins']}/{result['seeds']}"
          f"  (ties {result['ties']})")
    ratio = (result["delta_mean"] / result["delta_stdev"]
             if result["delta_stdev"] else float("inf"))
    print(f"  delta / seed sd  {ratio:+.2f}   "
          f"(gate needs > 1.00; distinguishable from zero: {result['nonzero']})")


#: The two candidate mechanisms for a shared representation.
MECHANISMS = (
    ("masked reconstruction (unsupervised)", one_trial),
    ("discriminative trunk (supervised across pretraining families)",
     one_trial_supervised),
)


def main(argv: list[str] | None = None) -> int:
    """Run the gate for every mechanism and print a verdict."""
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds = int(argv[0]) if argv else 24
    shots = int(argv[1]) if len(argv) > 1 else 5

    print("Stage 1 gate: does a shared representation buy transfer?")
    print(f"paired over {seeds} seeds, {shots}-shot on a held-out family, "
          f"{DEFAULT_FLIPS} pixels of noise")

    verdicts = []
    for name, mechanism in MECHANISMS:
        print(f"\n{'=' * 66}\n{name}")
        structured = run_gate(seeds=seeds, shots=shots, structured=True,
                              mechanism=mechanism)
        _report("compositional data (shared strokes -- transfer is possible)",
                structured)
        control = run_gate(seeds=seeds, shots=shots, structured=False,
                           mechanism=mechanism)
        _report("CONTROL: unstructured data (no shared parts -- should NOT help)",
                control)
        verdicts.append((name, structured, control))

    print(f"\n{'=' * 66}\nverdict")
    passed = False
    for name, structured, control in verdicts:
        if structured["passes"] and not control["passes"]:
            print(f"  PASS  {name}")
            print("        the gain appears only where structure exists to transfer.")
            passed = True
        elif structured["passes"] and control["passes"]:
            print(f"  AMBIGUOUS  {name}")
            print("        it also helps where there is nothing to transfer, so this")
            print("        is added capacity rather than transfer.")
        else:
            print(f"  FAIL  {name}  "
                  f"(delta {structured['delta_mean']:+.3f}, CI includes zero)")
    if not passed:
        print("\n  Stage 1 is not met by either mechanism. Both representations cost")
        print("  more information than they recover: the control arm is strongly")
        print("  negative in each case, which is what a lossy bottleneck with")
        print("  nothing to compress should look like. On compositional data the")
        print("  learned structure roughly cancels that cost and no more.")
        print()
        print("  At 100 seeds the discriminative trunk does reach +0.014 with a")
        print("  confidence interval clearing zero -- but that is 0.2x the")
        print("  seed-to-seed standard deviation, and the gate asked for a margin")
        print("  larger than seed variance. An effect a practitioner cannot see")
        print("  above run-to-run noise does not make a new category cheap, which")
        print("  is what the stage was for.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
