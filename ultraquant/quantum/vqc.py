"""A variational quantum classifier for the glyph task.

This is the circuit from :mod:`ultraquant.quantum.ansatz` put to work: a glyph is
amplitude-encoded into five qubits, a trainable ansatz acts on it, per-qubit ⟨Z⟩
are measured, and a small classical head turns those into class scores. The
quantum parameters are trained by the parameter-shift rule and the head by plain
gradient descent, so the whole thing learns end to end.

Why this is worth having, given the honest verdict elsewhere that the fixed
feature map buys nothing: this circuit is *trained*, and it sees the entire
25-pixel pattern in superposition rather than five row means. Whether that beats
the classical baseline is a question to be measured, not assumed —
:func:`compare_with_classical` measures it.

Cost is not hidden. Each training step needs ``2 · P`` circuit evaluations for
``P`` quantum parameters, because that is what hardware-compatible gradients
cost. Against a classical network that computes its gradient in one backward
pass, the quantum path is orders of magnitude slower per step here. That is the
real trade, and it is reported rather than buried.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ultraquant.quantum.ansatz import (
    QuantumCircuitModel,
    circuit_stats,
    parameter_shift_gradient,
)

__all__ = ["VariationalClassifier", "compare_with_classical"]


@dataclass
class VariationalClassifier:
    """Amplitude-encoded, parameter-shift-trained quantum classifier."""

    num_classes: int
    num_qubits: int = 5
    layers: int = 2
    seed: int = 0
    quantum: QuantumCircuitModel = field(init=False)
    weights: list[list[float]] = field(init=False)
    bias: list[float] = field(init=False)

    def __post_init__(self) -> None:
        """Build the circuit and the classical head."""
        self.quantum = QuantumCircuitModel(
            num_qubits=self.num_qubits, layers=self.layers, seed=self.seed
        )
        rng = random.Random(self.seed + 1)
        scale = 1.0 / math.sqrt(self.num_qubits)
        self.weights = [
            [rng.uniform(-scale, scale) for _ in range(self.num_qubits)]
            for _ in range(self.num_classes)
        ]
        self.bias = [0.0] * self.num_classes
        self.circuit_evaluations = 0

    # -- forward -----------------------------------------------------------

    def features(self, pixels: list[float], parameters: list[float] | None = None) -> list[float]:
        """Per-qubit ⟨Z⟩ for one glyph."""
        self.circuit_evaluations += 1
        return self.quantum.forward(pixels, parameters)

    def logits(self, features: list[float]) -> list[float]:
        """Class scores from the measured features."""
        return [
            sum(w * f for w, f in zip(row, features)) + b
            for row, b in zip(self.weights, self.bias)
        ]

    @staticmethod
    def softmax(logits: list[float]) -> list[float]:
        """Numerically safe softmax."""
        peak = max(logits)
        exps = [math.exp(v - peak) for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def predict(self, pixels: list[float]) -> tuple[int, float]:
        """Predicted class and its probability."""
        probabilities = self.softmax(self.logits(self.features(pixels)))
        best = max(range(len(probabilities)), key=lambda k: probabilities[k])
        return best, probabilities[best]

    # -- training ----------------------------------------------------------

    def train_step(
        self, pixels: list[float], label: int, lr: float = 0.1, lr_quantum: float = 0.05
    ) -> float:
        """One example: update the head classically, the circuit by parameter shift.

        Returns:
            The cross-entropy loss before the update.
        """
        features = self.features(pixels)
        probabilities = self.softmax(self.logits(features))
        loss = -math.log(max(probabilities[label], 1e-12))
        grad_logits = [p - (1.0 if k == label else 0.0) for k, p in enumerate(probabilities)]

        # The head's gradient is the usual outer product.
        for k, g in enumerate(grad_logits):
            for j in range(self.num_qubits):
                self.weights[k][j] -= lr * g * features[j]
            self.bias[k] -= lr * g

        # d(loss)/d(feature_j) = sum_k grad_logits[k] * weights[k][j], which is
        # exactly the weight vector the parameter-shift rule needs.
        feature_weights = [
            sum(grad_logits[k] * self.weights[k][j] for k in range(self.num_classes))
            for j in range(self.num_qubits)
        ]

        def scalar(params: list[float]) -> float:
            return sum(
                w * z for w, z in zip(feature_weights, self.features(pixels, params))
            )

        gradient = parameter_shift_gradient(scalar, self.quantum.parameters)
        self.quantum.parameters = [
            p - lr_quantum * g for p, g in zip(self.quantum.parameters, gradient)
        ]
        return loss

    def fit(
        self,
        xs: list[list[float]],
        ys: list[int],
        epochs: int = 8,
        lr: float = 0.2,
        lr_quantum: float = 0.1,
        seed: int = 0,
    ) -> dict:
        """Train on a dataset.

        Returns:
            ``{"loss", "accuracy", "circuit_evaluations", "epochs"}``.
        """
        rng = random.Random(seed)
        order = list(range(len(xs)))
        loss = 0.0
        for _epoch in range(epochs):
            rng.shuffle(order)
            total = 0.0
            for index in order:
                total += self.train_step(xs[index], ys[index], lr, lr_quantum)
            loss = total / max(len(order), 1)
        return {
            "loss": loss,
            "accuracy": self.score(xs, ys),
            "circuit_evaluations": self.circuit_evaluations,
            "epochs": epochs,
        }

    def score(self, xs: list[list[float]], ys: list[int]) -> float:
        """Accuracy on a dataset."""
        correct = sum(1 for x, y in zip(xs, ys) if self.predict(x)[0] == y)
        return correct / max(len(xs), 1)

    def describe(self) -> dict:
        """Circuit shape and cost, for reporting."""
        stats = circuit_stats(self.quantum.circuit([0.0] * (1 << self.num_qubits)))
        return {
            "num_qubits": self.num_qubits,
            "layers": self.layers,
            "quantum_parameters": self.quantum.ansatz.num_parameters,
            "classical_parameters": self.num_classes * (self.num_qubits + 1),
            "circuit": stats,
            "evaluations_per_training_step": 2 * self.quantum.ansatz.num_parameters + 1,
        }


def compare_with_classical(
    n_per_class: int = 6,
    flips: int = 1,
    epochs: int = 6,
    seed: int = 0,
    classes: tuple[str, ...] = ("plus", "square", "stripes_h", "diamond"),
) -> dict:
    """Train the quantum classifier and a classical one on the same data.

    Deliberately small: every training step costs ``2P + 1`` circuit evaluations,
    so a fair comparison at full dataset size would take hours. The point is a
    like-for-like accuracy reading and an honest cost ratio, not a benchmark win.

    Returns:
        Both models' accuracy, timings and parameter counts.
    """
    import time

    from ultraquant.model.network import UltraQuantNet
    from ultraquant.pattern.recognition import PATTERNS, render

    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for label, name in enumerate(classes):
        base = render(PATTERNS[name])
        for _ in range(n_per_class):
            variant = list(base)
            for pixel in rng.sample(range(len(variant)), flips):
                variant[pixel] = 1.0 - variant[pixel]
            xs.append(variant)
            ys.append(label)

    started = time.perf_counter()
    quantum = VariationalClassifier(num_classes=len(classes), seed=seed)
    quantum_report = quantum.fit(xs, ys, epochs=epochs, seed=seed)
    quantum_seconds = time.perf_counter() - started

    started = time.perf_counter()
    classical = UltraQuantNet(25, [8], len(classes), seed=seed)
    for _ in range(epochs * 8):
        classical.train_batch(xs, ys, lr=0.15)
    classical_accuracy = sum(
        1 for x, y in zip(xs, ys) if classical.predict(x)[0] == y
    ) / len(xs)
    classical_seconds = time.perf_counter() - started

    return {
        "samples": len(xs),
        "classes": list(classes),
        "quantum": {
            "accuracy": quantum_report["accuracy"],
            "loss": quantum_report["loss"],
            "seconds": round(quantum_seconds, 2),
            "circuit_evaluations": quantum_report["circuit_evaluations"],
            "parameters": quantum.quantum.ansatz.num_parameters,
            "circuit": quantum.describe()["circuit"],
        },
        "classical": {
            "accuracy": classical_accuracy,
            "seconds": round(classical_seconds, 2),
            "parameters": 25 * 8 + 8 + 8 * len(classes) + len(classes),
        },
        "cost_ratio": round(quantum_seconds / max(classical_seconds, 1e-9), 1),
    }
