"""Pattern recognition: the built-in glyph dataset and the PatternRecognizer facade.

Ties every UltraQuant subsystem together: 5x5 glyphs are flattened to floats,
optionally passed through the :class:`~ultraquant.quantum.qlayer.QuantumFeatureMap`
(hybrid mode) or a plain row-mean summary (classical mode), classified by an
:class:`~ultraquant.model.network.UltraQuantNet`, and every recognition is logged
into :class:`~ultraquant.memory.systematic.SystematicMemory`.

Pure Python stdlib only.
"""

from __future__ import annotations

import random

from ultraquant.memory.systematic import SystematicMemory
from ultraquant.model.network import UltraQuantNet
from ultraquant.quantum.backend import Backend, auto_backend
from ultraquant.quantum.qlayer import QuantumFeatureMap

__all__ = [
    "PATTERNS",
    "LABELS",
    "render",
    "make_dataset",
    "row_means",
    "PatternRecognizer",
]

#: The 8 built-in 5x5 glyph classes (``#`` = 1, ``.`` = 0).  Class index is
#: insertion order.
PATTERNS: dict[str, list[str]] = {
    "plus":      ["..#..", "..#..", "#####", "..#..", "..#.."],
    "cross":     ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "square":    ["#####", "#...#", "#...#", "#...#", "#####"],
    "diamond":   ["..#..", ".#.#.", "#...#", ".#.#.", "..#.."],
    "arrow_up":  ["..#..", ".###.", "#####", "..#..", "..#.."],
    "arrow_down": ["..#..", "..#..", "#####", ".###.", "..#.."],
    "stripes_h": ["#####", ".....", "#####", ".....", "#####"],
    "stripes_v": ["#.#.#", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
}

#: Class names in index order (derived from ``PATTERNS`` insertion order).
LABELS: list[str] = list(PATTERNS)


def render(rows: list[str]) -> list[float]:
    """Flatten glyph rows to floats, row-major (``#`` -> 1.0, anything else -> 0.0).

    Args:
        rows: Glyph rows, e.g. one of the ``PATTERNS`` values (5 strings of 5).

    Returns:
        Flat list of 25 floats for a 5x5 glyph.
    """
    return [1.0 if ch == "#" else 0.0 for row in rows for ch in row]


def make_dataset(
    n_per_class: int, flips: int, seed: int
) -> tuple[list[list[float]], list[int]]:
    """Build a noisy glyph dataset.

    For each class (in ``LABELS`` order) ``n_per_class`` samples are produced
    by copying the base glyph and flipping ``flips`` distinct random pixels.
    A single ``random.Random(seed)`` is shared across the whole build, so the
    dataset is fully determined by its arguments.

    Args:
        n_per_class: Samples per class.
        flips: Number of distinct pixels to flip in each sample.
        seed: Seed for the shared dataset RNG.

    Returns:
        Tuple ``(xs, ys)`` — flat 25-float samples and integer class labels.
    """
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for label_idx, name in enumerate(LABELS):
        base = render(PATTERNS[name])
        for _ in range(n_per_class):
            sample = list(base)
            for pix in rng.sample(range(len(sample)), flips):
                sample[pix] = 1.0 - sample[pix]
            xs.append(sample)
            ys.append(label_idx)
    return xs, ys


def row_means(x: list[float]) -> list[float]:
    """Mean of each of the 5 rows of a flat row-major 5x5 glyph.

    Args:
        x: Flat glyph of 25 floats.

    Returns:
        List of 5 per-row means.
    """
    return [sum(x[r * 5:(r + 1) * 5]) / 5.0 for r in range(5)]


class PatternRecognizer:
    """Hybrid quantum/classical glyph recognizer with systematic memory.

    In ``"hybrid"`` mode the 5 per-row means are run through a 5-qubit
    :class:`QuantumFeatureMap` and the resulting Z expectations are appended
    to the raw pixels; in ``"classical"`` mode the row means are appended
    as-is.  Either way the classifier input dimension is 30.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        backend: Backend | None = None,
        shots: int | None = None,
        seed: int = 0,
        hidden: tuple[int, ...] = (32,),
        memory_path: object = None,
        extractor: object = None,
    ) -> None:
        """Create a recognizer.

        Args:
            mode: ``"hybrid"`` (quantum feature map) or ``"classical"``.
            backend: Quantum backend for hybrid mode; defaults to
                ``auto_backend(seed=seed)`` (QPU if reachable, else simulator).
            shots: Shots for the quantum layer (``None`` = exact expectations).
            seed: Seed for network initialization (and the default backend).
            hidden: Hidden-layer widths of the ternary MLP.
            memory_path: Optional persistence path for the systematic memory.
            extractor: Optional batch feature extractor with an
                ``extract_batch(list[list[float]]) -> list[list[float]]``
                method (e.g. ``HeterogeneousFeatureExtractor``). When set,
                :meth:`train` and :meth:`evaluate` compute the 5 extra
                features for the whole dataset in one call instead of one
                quantum run per sample; :meth:`features` / :meth:`recognize`
                are unaffected.

        Raises:
            ValueError: If ``mode`` is not ``"hybrid"`` or ``"classical"``.
        """
        if mode not in ("hybrid", "classical"):
            raise ValueError(f"mode must be 'hybrid' or 'classical', got {mode!r}")
        self.mode = mode
        self.shots = shots
        self.seed = seed
        self.extractor = extractor
        self.hidden: tuple[int, ...] = tuple(int(h) for h in hidden)

        self.backend: Backend | None
        self.qmap: QuantumFeatureMap | None
        if mode == "hybrid":
            self.backend = backend or auto_backend(seed=seed)
            self.qmap = QuantumFeatureMap(5, self.backend, shots)
        else:
            self.backend = backend
            self.qmap = None

        self.net = UltraQuantNet(30, list(self.hidden), len(LABELS), seed=seed)
        self.memory = SystematicMemory(memory_path)

    # -- feature extraction --------------------------------------------------

    def features(self, x: list[float]) -> list[float]:
        """Build the 30-dim classifier input for one flat glyph.

        The 25 raw pixels are extended with 5 extra features: the quantum
        layer's Z expectations of the row means (hybrid) or the row means
        themselves (classical).

        Args:
            x: Flat glyph of 25 floats.

        Returns:
            Feature vector of length 30.
        """
        means = row_means(x)
        if self.qmap is not None:
            extra = self.qmap.extract(means)
        else:
            extra = means
        return list(x) + list(extra)

    def _batch_features(self, xs: list[list[float]]) -> list[list[float]]:
        """Feature vectors for a whole dataset.

        Uses ``extractor.extract_batch`` on all row means in one call when a
        batch extractor was configured, otherwise falls back to per-sample
        :meth:`features`.
        """
        if self.extractor is None:
            return [self.features(x) for x in xs]
        extras = self.extractor.extract_batch([row_means(x) for x in xs])
        return [list(x) + list(e) for x, e in zip(xs, extras)]

    # -- training / evaluation ----------------------------------------------

    def train(
        self,
        epochs: int = 40,
        lr: float = 0.05,
        n_per_class: int = 40,
        flips: int = 2,
        seed: int = 1,
    ) -> dict:
        """Train the classifier on a freshly built noisy glyph dataset.

        Features are precomputed once (one quantum-layer run per sample in
        hybrid mode, or a single ``extractor.extract_batch`` call when a batch
        extractor is set), then the net is trained with shuffled minibatches
        of 16.

        Args:
            epochs: Number of passes over the dataset.
            lr: Learning rate.
            n_per_class: Training samples per class.
            flips: Noise pixels flipped per sample.
            seed: Seed for the dataset build and the shuffle RNG.

        Returns:
            ``{"final_loss": float, "train_accuracy": float, "epochs": int}``.
        """
        xs, ys = make_dataset(n_per_class, flips, seed)
        feats = self._batch_features(xs)
        shuffle_rng = random.Random(seed + 20250)
        order = list(range(len(feats)))
        final_loss = 0.0
        for _ in range(epochs):
            shuffle_rng.shuffle(order)
            batch_losses: list[float] = []
            for start in range(0, len(order), 16):
                batch = order[start:start + 16]
                batch_losses.append(
                    self.net.train_batch(
                        [feats[i] for i in batch], [ys[i] for i in batch], lr
                    )
                )
            final_loss = sum(batch_losses) / len(batch_losses)
        correct = sum(
            1 for f, y in zip(feats, ys) if self.net.predict(f)[0] == y
        )
        return {
            "final_loss": final_loss,
            "train_accuracy": correct / len(ys) if ys else 0.0,
            "epochs": epochs,
        }

    def evaluate(
        self, n_per_class: int = 15, flips: int = 3, seed: int = 99
    ) -> float:
        """Accuracy on a freshly generated dataset.

        Args:
            n_per_class: Evaluation samples per class.
            flips: Noise pixels flipped per sample.
            seed: Seed for the evaluation dataset build.

        Returns:
            Fraction of samples classified correctly.
        """
        xs, ys = make_dataset(n_per_class, flips, seed)
        feats = self._batch_features(xs)
        correct = sum(
            1 for f, y in zip(feats, ys) if self.net.predict(f)[0] == y
        )
        return correct / len(ys) if ys else 0.0

    # -- recognition ---------------------------------------------------------

    @staticmethod
    def _as_flat(rows_or_flat: list[str] | list[float]) -> list[float]:
        """Normalize an input glyph to a flat list of 25 floats."""
        seq = list(rows_or_flat)
        if seq and all(isinstance(item, str) for item in seq):
            return render(seq)  # type: ignore[arg-type]
        return [float(v) for v in seq]  # type: ignore[arg-type]

    def recognize(
        self, rows_or_flat: list[str] | list[float]
    ) -> tuple[str, float]:
        """Classify one glyph and record the event in memory.

        Accepts either 5 row-strings (``#``/``.``) or 25 floats.  Logs a
        ``"recognition"`` episode (content: label, confidence, mode) and
        stores the input's bit signature under the predicted label.

        Args:
            rows_or_flat: The glyph to classify.

        Returns:
            Tuple ``(label_name, confidence)``.
        """
        flat = self._as_flat(rows_or_flat)
        label_idx, confidence = self.net.predict(self.features(flat))
        label = LABELS[label_idx]
        self.memory.remember_episode(
            "recognition",
            {"label": label, "confidence": confidence, "mode": self.mode},
            tags=["recognition", label],
        )
        bits = [1 if v >= 0.5 else 0 for v in flat]
        self.memory.store_signature(label, bits)
        return label, confidence

    # -- snapshots -----------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-safe snapshot of the recognizer.

        Contains the recognizer configuration, the full network state dict,
        and the current memory statistics.
        """
        return {
            "config": {
                "mode": self.mode,
                "shots": self.shots,
                "seed": self.seed,
                "hidden": list(self.hidden),
                "num_classes": len(LABELS),
                "backend": self.backend.name if self.backend is not None else None,
            },
            "net": self.net.state_dict(),
            "memory_stats": self.memory.stats(),
        }

    def restore_snapshot(self, payload: dict) -> None:
        """Rebuild the network from a :meth:`snapshot` payload.

        The network is reconstructed from the snapshot's own architecture
        description and its weights loaded, so predictions afterwards are
        identical to those of the recognizer that produced the snapshot
        (given the same mode / feature pipeline).

        Args:
            payload: Dictionary previously produced by :meth:`snapshot`.
        """
        net_state = payload["net"]
        cfg = net_state["config"]
        net = UltraQuantNet(
            int(cfg["input_dim"]),
            [int(h) for h in cfg["hidden_dims"]],
            int(cfg["num_classes"]),
            seed=int(cfg.get("seed", 0)),
            quantize_head=bool(cfg.get("quantize_head", False)),
            act_bits=int(cfg.get("act_bits", 8)),
        )
        net.load_state_dict(net_state)
        self.net = net
        self.hidden = tuple(int(h) for h in cfg["hidden_dims"])
