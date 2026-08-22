"""UltraQuantNet -- a small MLP with ternary hidden layers.

Hidden layers are :class:`~ultraquant.model.quantize.TernaryLinear` followed by
ReLU and fake-quantized activations; the head is full precision by default.
Trained with softmax cross-entropy via plain SGD.  Pure Python stdlib only.
"""

from __future__ import annotations

import math
import random

from ultraquant.model.quantize import TernaryLinear, fake_quantize_activation

__all__ = ["UltraQuantNet"]

_EPS = 1e-12


class UltraQuantNet:
    """MLP with ternary hidden layers and a (by default) full-precision head.

    Architecture: for each hidden dimension a ``TernaryLinear`` layer followed
    by ReLU and uniform fake-quantization of the activations, then a final
    ``TernaryLinear`` head producing logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        seed: int = 0,
        quantize_head: bool = False,
        act_bits: int = 8,
    ) -> None:
        """Build the network.

        Args:
            input_dim: Number of input features.
            hidden_dims: Sizes of the hidden layers (may be empty).
            num_classes: Number of output classes (logit dimension).
            seed: Seed for the weight-initialization RNG.
            quantize_head: If True, the head layer is ternary-quantized too.
            act_bits: Bit width for activation fake-quantization.
        """
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.seed = seed
        self.quantize_head = quantize_head
        self.act_bits = act_bits

        rng = random.Random(seed)
        self.hidden: list[TernaryLinear] = []
        prev = input_dim
        for h in self.hidden_dims:
            self.hidden.append(TernaryLinear(prev, h, rng, quantized=True))
            prev = h
        self.head = TernaryLinear(prev, num_classes, rng, quantized=quantize_head)

        self._relu_masks: list[list[float]] = []

    def _layers(self) -> list[TernaryLinear]:
        """All parameterized layers, hidden first, head last."""
        return [*self.hidden, self.head]

    def forward(self, x: list[float]) -> list[float]:
        """Compute logits for one input vector.

        Args:
            x: Input vector of length ``input_dim``.

        Returns:
            Logits of length ``num_classes``.
        """
        self._relu_masks = []
        a = list(x)
        for layer in self.hidden:
            z = layer.forward(a)
            mask = [1.0 if v > 0.0 else 0.0 for v in z]
            self._relu_masks.append(mask)
            a = [v if v > 0.0 else 0.0 for v in z]
            a = fake_quantize_activation(a, bits=self.act_bits)
        return self.head.forward(a)

    def forward_vram(self, x: list[float], cache,
                     key_prefix: str) -> list[float] | None:
        """:meth:`forward` with matmuls on resident device layers.

        Inference-only, for FROZEN experts: the device copy is uploaded
        from the current quantisation and stays valid until the caller
        drops the keys (which anything that retrains an expert must do).
        ReLU and activation fake-quantisation stay in Python, identical
        to :meth:`forward`; the matmul itself is bit-exact on the device
        (§11.45, parity 3.6e-15). Returns None when any layer cannot run
        resident - the caller falls back to :meth:`forward`, and the two
        paths agree to the house tolerance by construction.
        """
        from ultraquant.model.quantize import quantize_ternary

        def _provider(layer):
            def build():
                # Callers guard: only quantized layers reach a provider.
                q, alpha = quantize_ternary(layer.w)
                return q, alpha, list(layer.b)
            return build

        a = list(x)
        for index, layer in enumerate(self.hidden):
            if not layer.quantized:
                return None
            out = cache.forward_lazy(f"{key_prefix}:h{index}",
                                     _provider(layer), [a])
            if out is None:
                return None
            z = out[0]
            a = [v if v > 0.0 else 0.0 for v in z]
            a = fake_quantize_activation(a, bits=self.act_bits)
        if not self.head.quantized:
            # A full-precision head runs in Python on the device-computed
            # hidden activation; the resident tier covers the ternary part.
            return self.head.forward(a)
        out = cache.forward_lazy(f"{key_prefix}:head",
                                 _provider(self.head), [a])
        if out is None:
            return None
        return out[0]

    def embed(self, x: list[float]) -> list[float]:
        """Return the last hidden activation instead of the logits.

        The penultimate layer of a net trained across many categories is the
        usual place to look for a representation that transfers to a new one.
        Exposing it costs nothing — it is the same computation as
        :meth:`forward` stopping one layer early — and lets that hypothesis be
        measured rather than assumed
        (``ultraquant/experiments/transfer_gate.py``).

        Args:
            x: Input vector of length ``input_dim``.

        Returns:
            The final hidden activation; ``x`` itself if the net has no hidden
            layers.
        """
        a = list(x)
        for layer in self.hidden:
            z = layer.forward(a)
            a = [v if v > 0.0 else 0.0 for v in z]
            a = fake_quantize_activation(a, bits=self.act_bits)
        return a

    def _backward(self, grad_logits: list[float]) -> None:
        """Backpropagate a logits gradient, accumulating layer grads."""
        g = self.head.backward(grad_logits)
        for layer, mask in zip(reversed(self.hidden), reversed(self._relu_masks)):
            # Straight-through the activation fake-quantization, then ReLU.
            g = [gi * mi for gi, mi in zip(g, mask)]
            g = layer.backward(g)

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        """Numerically safe softmax (max-subtracted)."""
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def predict(self, x: list[float]) -> tuple[int, float]:
        """Predict the class of one input.

        Args:
            x: Input vector of length ``input_dim``.

        Returns:
            Tuple ``(label, confidence)`` -- the argmax class index and its
            softmax probability.
        """
        probs = self._softmax(self.forward(x))
        best = max(range(len(probs)), key=lambda k: probs[k])
        return best, probs[best]

    def predict_distribution(self, x: list[float]) -> list[float]:
        """The whole softmax, not just its tallest bar.

        :meth:`predict` discards everything except the argmax and its
        probability, which is the standard thing to report and the
        standard thing to be misled by - a network that has learned
        nothing still produces both. See
        :mod:`ultraquant.model.uncertainty` for what the rest of the
        distribution says.
        """
        return self._softmax(self.forward(x))

    def predict_with_uncertainty(self, x: list[float],
                                 floor: float = 0.0,
                                 margin_floor: float = 0.0):
        """A prediction that carries its own entropy, in bits.

        Args:
            x: Input vector.
            floor: Bits of evidence below which the layer declines to
                name a class at all. Zero accepts everything, which
                is what :meth:`predict` does silently.
            margin_floor: Gap to the runner-up below which it
                declines. The two catch different failures; see
                :mod:`ultraquant.model.uncertainty`.

        Returns:
            An :class:`~ultraquant.model.uncertainty.Prediction`.
        """
        from ultraquant.model.uncertainty import describe

        return describe(self.predict_distribution(x), floor, margin_floor)

    def predict_vram(self, x: list[float], cache,
                     key_prefix: str) -> tuple[int, float] | None:
        """:meth:`predict` over :meth:`forward_vram`, or None to fall back."""
        logits = self.forward_vram(x, cache, key_prefix)
        if logits is None:
            return None
        probs = self._softmax(logits)
        best = max(range(len(probs)), key=lambda k: probs[k])
        return best, probs[best]

    def train_batch(self, xs: list[list[float]], ys: list[int], lr: float) -> float:
        """Run one SGD step over a batch.

        Each sample is forwarded and backpropagated (softmax cross-entropy;
        the logits gradient is ``softmax - onehot``); gradients accumulate
        across the batch and a single update with ``lr / len(xs)`` is applied,
        which is equivalent to stepping with the mean gradient.

        Args:
            xs: Batch of input vectors.
            ys: Integer class labels, same length as ``xs``.
            lr: Learning rate.

        Returns:
            Mean cross-entropy loss over the batch.
        """
        if not xs:
            return 0.0
        total_loss = 0.0
        for x, y in zip(xs, ys):
            logits = self.forward(x)
            probs = self._softmax(logits)
            total_loss += -math.log(max(probs[y], _EPS))
            grad = [p - (1.0 if k == y else 0.0) for k, p in enumerate(probs)]
            self._backward(grad)
        step_lr = lr / len(xs)
        for layer in self._layers():
            layer.step(step_lr)
        return total_loss / len(xs)

    def state_dict(self) -> dict:
        """Return a JSON-safe snapshot of the full network."""
        return {
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": list(self.hidden_dims),
                "num_classes": self.num_classes,
                "seed": self.seed,
                "quantize_head": self.quantize_head,
                "act_bits": self.act_bits,
            },
            "hidden": [layer.state_dict() for layer in self.hidden],
            "head": self.head.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore the network from a :meth:`state_dict` snapshot.

        The architecture described in ``d["config"]`` must match this
        network's architecture.  After loading, predictions are identical to
        those of the network that produced the snapshot.

        Args:
            d: Dictionary previously produced by :meth:`state_dict`.

        Raises:
            ValueError: If the snapshot architecture does not match.
        """
        cfg = d["config"]
        if (
            int(cfg["input_dim"]) != self.input_dim
            or [int(h) for h in cfg["hidden_dims"]] != self.hidden_dims
            or int(cfg["num_classes"]) != self.num_classes
        ):
            raise ValueError("state_dict architecture mismatch")
        self.quantize_head = bool(cfg["quantize_head"])
        self.act_bits = int(cfg["act_bits"])
        for layer, layer_state in zip(self.hidden, d["hidden"]):
            layer.load_state_dict(layer_state)
        self.head.load_state_dict(d["head"])
