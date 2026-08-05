"""A variational quantum expert that lives in the shard library.

:class:`QuantumExpert` implements the same small contract the classical experts
do — fit, predict, serialise — so the vault, the cache, the router and the
pattern working set need no special case for it. What differs is what gets
stored: a handful of rotation angles instead of a weight matrix.

That difference is the point. A classical expert for the glyph task stores
roughly 25×32 weights; this one stores 20 angles plus a small readout head. Both
have to be paged off storage on every cache miss, so the smaller one is cheaper
in exactly the dimension this architecture cares about.

The cost is on the other side of the ledger and is not hidden: training uses the
parameter-shift rule, at ``2P + 1`` circuit evaluations per example, so building
a quantum expert is far slower than fitting a classical one. Trained once, paged
many times — which is the usual shape for a model library, and the reason the
trade can be worth taking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ultraquant.quantum.vqc import VariationalClassifier

__all__ = ["QuantumExpert", "quantum_expert_payload"]


@dataclass
class QuantumExpert:
    """A trained variational circuit acting as a category expert."""

    labels: list[str]
    num_qubits: int = 5
    layers: int = 2
    seed: int = 0
    model: VariationalClassifier = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the underlying classifier."""
        self.model = VariationalClassifier(
            num_classes=len(self.labels),
            num_qubits=self.num_qubits,
            layers=self.layers,
            seed=self.seed,
        )
        self.trained_examples = 0

    # -- training ----------------------------------------------------------

    def fit(
        self,
        xs: list[list[float]],
        ys: list[int],
        epochs: int = 6,
        lr: float = 0.2,
        lr_quantum: float = 0.1,
        seed: int = 0,
    ) -> dict:
        """Train on labelled patterns.

        Returns:
            ``{"loss", "accuracy", "circuit_evaluations", "epochs"}``.
        """
        report = self.model.fit(
            xs, ys, epochs=epochs, lr=lr, lr_quantum=lr_quantum, seed=seed
        )
        self.trained_examples += len(xs)
        return report

    def predict(self, pixels: list[float]) -> tuple[str, float]:
        """Predicted label and confidence."""
        index, confidence = self.model.predict(pixels)
        return self.labels[index], confidence

    def score(self, xs: list[list[float]], ys: list[int]) -> float:
        """Accuracy over a dataset."""
        return self.model.score(xs, ys)

    # -- serialisation -----------------------------------------------------

    @property
    def parameter_count(self) -> int:
        """Total stored numbers: circuit angles plus the readout head."""
        return (
            self.model.quantum.ansatz.num_parameters
            + len(self.labels) * self.num_qubits
            + len(self.labels)
        )

    def state_dict(self) -> dict:
        """JSON-safe parameters, in the shape the vault stores."""
        return {
            "kind": "quantum-vqc",
            "labels": list(self.labels),
            "num_qubits": self.num_qubits,
            "layers": self.layers,
            "seed": self.seed,
            "theta": list(self.model.quantum.parameters),
            "head_weights": [list(row) for row in self.model.weights],
            "head_bias": list(self.model.bias),
            "trained_examples": self.trained_examples,
        }

    @classmethod
    def load(cls, state: dict) -> "QuantumExpert":
        """Rebuild an expert from :meth:`state_dict`.

        Raises:
            ValueError: If the payload is not a quantum expert.
        """
        if state.get("kind") != "quantum-vqc":
            raise ValueError(f"not a quantum expert payload: {state.get('kind')!r}")
        expert = cls(
            labels=list(state["labels"]),
            num_qubits=int(state["num_qubits"]),
            layers=int(state["layers"]),
            seed=int(state.get("seed", 0)),
        )
        expert.model.quantum.parameters = [float(v) for v in state["theta"]]
        expert.model.weights = [[float(v) for v in row] for row in state["head_weights"]]
        expert.model.bias = [float(v) for v in state["head_bias"]]
        expert.trained_examples = int(state.get("trained_examples", 0))
        return expert

    def describe(self) -> dict:
        """Shape, cost and circuit statistics."""
        info = self.model.describe()
        info.update({
            "kind": "quantum-vqc",
            "labels": list(self.labels),
            "total_parameters": self.parameter_count,
            "trained_examples": self.trained_examples,
        })
        return info


def quantum_expert_payload(expert: QuantumExpert, prototypes: dict | None = None) -> dict:
    """Wrap an expert in the payload shape the vault stores for any expert."""
    return {
        "labels": list(expert.labels),
        "net": expert.state_dict(),
        "trained_examples": expert.trained_examples,
        "prototypes": prototypes or {},
    }
