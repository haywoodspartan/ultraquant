"""Ternary Weight Network-style quantization primitives for UltraQuant.

Implements:

- :func:`quantize_ternary` -- project a float matrix onto {-1, 0, +1} with a
  single full-precision scale ``alpha``.
- :func:`dequantize` -- reconstruct the float matrix ``alpha * q``.
- :func:`fake_quantize_activation` -- uniform fake-quantization of activations.
- :class:`TernaryLinear` -- a linear layer trained with the straight-through
  estimator over its ternarized weights.

Pure Python stdlib only.
"""

from __future__ import annotations

import random

__all__ = [
    "quantize_ternary",
    "dequantize",
    "fake_quantize_activation",
    "TernaryLinear",
]


def quantize_ternary(weights: list[list[float]]) -> tuple[list[list[int]], float]:
    """Quantize a weight matrix to ternary values {-1, 0, +1} plus one scale.

    The threshold is ``delta = 0.75 * mean(|w|)`` over all entries.  Entries
    with ``|w_ij| > delta`` become ``sign(w_ij)``; the rest become 0.  The
    scale ``alpha`` is the mean of ``|w_ij|`` over the entries that survived
    (non-zero ``q_ij``), or 1.0 if none survived.

    Edge case: if ``delta == 0`` (e.g. all-zero weights) the quantized matrix
    is all zeros and the scale is 1.0.

    Args:
        weights: Matrix as ``[rows][cols]`` of floats.

    Returns:
        Tuple ``(q, alpha)`` where ``q`` has the same shape as ``weights``
        with entries in {-1, 0, +1}, and ``alpha`` is the float scale.
    """
    flat_abs = [abs(v) for row in weights for v in row]
    if not flat_abs:
        return ([[] for _ in weights], 1.0)

    delta = 0.75 * (sum(flat_abs) / len(flat_abs))
    if delta == 0.0:
        return ([[0 for _ in row] for row in weights], 1.0)

    q: list[list[int]] = []
    surviving_abs: list[float] = []
    for row in weights:
        q_row: list[int] = []
        for v in row:
            if abs(v) > delta:
                q_row.append(1 if v > 0 else -1)
                surviving_abs.append(abs(v))
            else:
                q_row.append(0)
        q.append(q_row)

    alpha = (sum(surviving_abs) / len(surviving_abs)) if surviving_abs else 1.0
    return q, alpha


def dequantize(q: list[list[int]], alpha: float) -> list[list[float]]:
    """Reconstruct the float matrix ``alpha * q`` from a ternary matrix.

    Args:
        q: Ternary matrix with entries in {-1, 0, +1}.
        alpha: Full-precision scale.

    Returns:
        Matrix of floats with the same shape as ``q``.
    """
    return [[alpha * v for v in row] for row in q]


def fake_quantize_activation(
    x: list[float], bits: int = 8, clip: float = 4.0
) -> list[float]:
    """Fake-quantize an activation vector to a uniform grid.

    Each value is clamped to ``[-clip, clip]``, snapped to the nearest of
    ``2**bits - 1`` uniformly spaced levels spanning that range, and returned
    as a dequantized float.

    Args:
        x: Activation values.
        bits: Bit width; the grid has ``2**bits - 1`` levels.
        clip: Symmetric clipping bound.

    Returns:
        List of fake-quantized floats, same length as ``x``.
    """
    levels = (2 ** bits) - 1
    if levels < 2:
        return [0.0 for _ in x]
    step = (2.0 * clip) / (levels - 1)
    out: list[float] = []
    for v in x:
        clamped = min(max(v, -clip), clip)
        idx = round((clamped + clip) / step)
        out.append(idx * step - clip)
    return out


class TernaryLinear:
    """Linear layer with ternary-quantized weights and a full-precision bias.

    Master weights are kept in full precision; the forward pass (when
    ``quantized``) uses ``alpha * q`` recomputed from the masters each call.
    Gradients flow to the master weights via the straight-through estimator.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rng: random.Random,
        quantized: bool = True,
    ) -> None:
        """Initialize the layer.

        Args:
            in_dim: Input dimensionality.
            out_dim: Output dimensionality.
            rng: Seeded random source used for weight initialization.
            quantized: If True, forward/backward use ternarized weights.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantized = quantized
        r = (6.0 / (in_dim + out_dim)) ** 0.5
        self.w: list[list[float]] = [
            [rng.uniform(-r, r) for _ in range(in_dim)] for _ in range(out_dim)
        ]
        self.b: list[float] = [0.0 for _ in range(out_dim)]
        self.gw: list[list[float]] = [
            [0.0 for _ in range(in_dim)] for _ in range(out_dim)
        ]
        self.gb: list[float] = [0.0 for _ in range(out_dim)]
        self._last_x: list[float] | None = None
        self._last_weights: list[list[float]] | None = None

    def _effective_weights(self) -> list[list[float]]:
        """Weights actually used in the forward pass."""
        if self.quantized:
            q, alpha = quantize_ternary(self.w)
            return dequantize(q, alpha)
        return [list(row) for row in self.w]

    def forward(self, x: list[float]) -> list[float]:
        """Compute ``W x + b`` using (possibly quantized) weights.

        Caches the input and the weights used, for :meth:`backward`.

        Args:
            x: Input vector of length ``in_dim``.

        Returns:
            Output vector of length ``out_dim``.
        """
        weights = self._effective_weights()
        self._last_x = list(x)
        self._last_weights = weights
        out: list[float] = []
        for o in range(self.out_dim):
            row = weights[o]
            acc = self.b[o]
            for i in range(self.in_dim):
                acc += row[i] * x[i]
            out.append(acc)
        return out

    def backward(self, grad_out: list[float]) -> list[float]:
        """Backpropagate through the layer.

        Uses the same (quantized) weights as the last :meth:`forward` call for
        the input gradient, and accumulates master-weight gradients via the
        straight-through estimator (the gradient of the quantized weight is
        applied to the master weight unchanged).

        Args:
            grad_out: Gradient w.r.t. the layer output, length ``out_dim``.

        Returns:
            Gradient w.r.t. the layer input, length ``in_dim``.
        """
        if self._last_x is None or self._last_weights is None:
            raise RuntimeError("backward() called before forward()")
        x = self._last_x
        weights = self._last_weights
        grad_in = [0.0 for _ in range(self.in_dim)]
        for o in range(self.out_dim):
            g = grad_out[o]
            self.gb[o] += g
            w_row = weights[o]
            gw_row = self.gw[o]
            for i in range(self.in_dim):
                grad_in[i] += g * w_row[i]
                gw_row[i] += g * x[i]
        return grad_in

    def step(self, lr: float) -> None:
        """Apply one SGD update to master weights and bias, then reset grads.

        Master weights are clamped to ``[-1, 1]`` after the update.

        Args:
            lr: Learning rate.
        """
        for o in range(self.out_dim):
            w_row = self.w[o]
            gw_row = self.gw[o]
            for i in range(self.in_dim):
                v = w_row[i] - lr * gw_row[i]
                w_row[i] = min(max(v, -1.0), 1.0)
                gw_row[i] = 0.0
            self.b[o] -= lr * self.gb[o]
            self.gb[o] = 0.0

    def state_dict(self) -> dict:
        """Return a JSON-safe snapshot of the layer parameters."""
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "quantized": self.quantized,
            "w": [list(row) for row in self.w],
            "b": list(self.b),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore parameters from a :meth:`state_dict` snapshot.

        Args:
            d: Dictionary previously produced by :meth:`state_dict`.
        """
        self.in_dim = int(d["in_dim"])
        self.out_dim = int(d["out_dim"])
        self.quantized = bool(d["quantized"])
        self.w = [[float(v) for v in row] for row in d["w"]]
        self.b = [float(v) for v in d["b"]]
        self.gw = [[0.0 for _ in range(self.in_dim)] for _ in range(self.out_dim)]
        self.gb = [0.0 for _ in range(self.out_dim)]
        self._last_x = None
        self._last_weights = None
