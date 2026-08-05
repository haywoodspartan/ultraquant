"""A shared representation learned without labels.

Every expert in the library reads raw features, so nothing learned for one
category is available to any other and the hundredth category costs exactly what
the first did. That is the difference between a lookup table and a learner, and
closing it is stage 1 of [ARCHITECTURE.md §11.3].

:class:`PatternEncoder` is a denoising autoencoder: part of the input is masked
out, and the network must reconstruct the whole thing from what is left. No
labels are involved, so it can be trained on every pattern the library has ever
seen. To fill in a masked pixel the network has to learn what tends to go with
what — strokes, symmetry, the parts patterns are actually made of — and that
structure is what a new category gets to reuse.

The encoder deliberately uses plain float weights rather than the ternary scheme
of :class:`~ultraquant.model.network.UltraQuantNet`. It is being asked whether a
shared representation buys transfer at all; quantizing it is a later question,
and one worth asking only if the answer here is yes. Ternary weights would
confound the measurement with a capacity change.

Whether this pays for itself is an empirical question with a falsifiable answer,
and the answer is not assumed anywhere in this module — see
``ultraquant/experiments/transfer_gate.py`` for the measurement.

Pure Python standard library.
"""

from __future__ import annotations

import math
import random

__all__ = ["PatternEncoder"]


class PatternEncoder:
    """One hidden layer trained to reconstruct partially masked inputs.

    Args:
        input_dim: Width of the feature vectors to encode.
        embed_dim: Width of the learned representation. Smaller than
            ``input_dim`` on purpose — a bottleneck forces the layer to keep
            what is shared and discard what is incidental.
        seed: Weight-initialisation seed.
    """

    def __init__(self, input_dim: int, embed_dim: int = 16, seed: int = 0) -> None:
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.seed = int(seed)
        rng = random.Random(seed)
        # Xavier-ish: keeps activations from saturating tanh at this depth.
        scale_in = math.sqrt(6.0 / (self.input_dim + self.embed_dim))
        scale_out = math.sqrt(6.0 / (self.embed_dim + self.input_dim))
        self.w_enc = [
            [rng.uniform(-scale_in, scale_in) for _ in range(self.input_dim)]
            for _ in range(self.embed_dim)
        ]
        self.b_enc = [0.0] * self.embed_dim
        self.w_dec = [
            [rng.uniform(-scale_out, scale_out) for _ in range(self.embed_dim)]
            for _ in range(self.input_dim)
        ]
        self.b_dec = [0.0] * self.input_dim

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def encode(self, x: list[float]) -> list[float]:
        """Return the representation of ``x``.

        Args:
            x: Feature vector of length ``input_dim``.

        Returns:
            The ``embed_dim``-wide representation.

        Raises:
            ValueError: If ``x`` is the wrong width.
        """
        if len(x) != self.input_dim:
            raise ValueError(
                f"encoder takes {self.input_dim} features, got {len(x)}"
            )
        out = []
        for row, bias in zip(self.w_enc, self.b_enc):
            total = bias
            for weight, value in zip(row, x):
                total += weight * value
            out.append(math.tanh(total))
        return out

    def _decode(self, h: list[float]) -> list[float]:
        """Reconstruct an input from its representation."""
        out = []
        for row, bias in zip(self.w_dec, self.b_dec):
            total = bias
            for weight, value in zip(row, h):
                total += weight * value
            out.append(total)
        return out

    def reconstruct(self, x: list[float]) -> list[float]:
        """Encode then decode ``x`` — useful for inspecting what was kept."""
        return self._decode(self.encode(x))

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def _step(self, x: list[float], corrupted: list[float], lr: float) -> float:
        """One SGD step: reconstruct the clean ``x`` from ``corrupted``.

        Returns:
            The mean squared reconstruction error before the update.
        """
        # Forward.
        pre = []
        for row, bias in zip(self.w_enc, self.b_enc):
            total = bias
            for weight, value in zip(row, corrupted):
                total += weight * value
            pre.append(total)
        hidden = [math.tanh(v) for v in pre]
        output = self._decode(hidden)

        # dL/doutput for L = mean squared error.
        n = len(output)
        d_out = [2.0 * (o - t) / n for o, t in zip(output, x)]
        loss = sum((o - t) ** 2 for o, t in zip(output, x)) / n

        # Decoder gradients, accumulating the hidden-layer error as we go.
        d_hidden = [0.0] * self.embed_dim
        for i in range(self.input_dim):
            gradient = d_out[i]
            row = self.w_dec[i]
            for j in range(self.embed_dim):
                d_hidden[j] += gradient * row[j]
                row[j] -= lr * gradient * hidden[j]
            self.b_dec[i] -= lr * gradient

        # Encoder gradients through tanh'(z) = 1 - tanh(z)^2.
        for j in range(self.embed_dim):
            gradient = d_hidden[j] * (1.0 - hidden[j] * hidden[j])
            row = self.w_enc[j]
            for k in range(self.input_dim):
                row[k] -= lr * gradient * corrupted[k]
            self.b_enc[j] -= lr * gradient
        return loss

    def fit(
        self,
        xs: list[list[float]],
        epochs: int = 60,
        lr: float = 0.05,
        mask_rate: float = 0.25,
        seed: int | None = None,
    ) -> float:
        """Train on unlabelled vectors by masked reconstruction.

        Args:
            xs: Unlabelled feature vectors.
            epochs: Passes over the data.
            lr: SGD learning rate.
            mask_rate: Fraction of each input zeroed before reconstruction.
                Zero would make this an ordinary autoencoder, which can settle
                for copying its input; masking forces it to model the
                relationships between features instead.
            seed: Shuffle/mask seed; defaults to the construction seed.

        Returns:
            Mean reconstruction loss of the final epoch.
        """
        if not xs:
            return 0.0
        rng = random.Random(self.seed if seed is None else seed)
        width = len(xs[0])
        masked_count = max(1, int(width * mask_rate))
        order = list(range(len(xs)))
        final = 0.0
        for _ in range(max(0, epochs)):
            rng.shuffle(order)
            total = 0.0
            for index in order:
                clean = xs[index]
                corrupted = list(clean)
                for position in rng.sample(range(width), masked_count):
                    corrupted[position] = 0.0
                total += self._step(clean, corrupted, lr)
            final = total / len(order)
        return final

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        """JSON-safe weights, so an encoder can live in the shard library."""
        return {
            "config": {
                "input_dim": self.input_dim,
                "embed_dim": self.embed_dim,
                "seed": self.seed,
            },
            "w_enc": [list(row) for row in self.w_enc],
            "b_enc": list(self.b_enc),
            "w_dec": [list(row) for row in self.w_dec],
            "b_dec": list(self.b_dec),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PatternEncoder":
        """Rebuild an encoder saved by :meth:`state_dict`."""
        config = state["config"]
        encoder = cls(
            input_dim=int(config["input_dim"]),
            embed_dim=int(config["embed_dim"]),
            seed=int(config.get("seed", 0)),
        )
        encoder.w_enc = [[float(v) for v in row] for row in state["w_enc"]]
        encoder.b_enc = [float(v) for v in state["b_enc"]]
        encoder.w_dec = [[float(v) for v in row] for row in state["w_dec"]]
        encoder.b_dec = [float(v) for v in state["b_dec"]]
        return encoder
