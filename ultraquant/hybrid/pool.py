"""Choose the expert per category by measuring both kinds.

Neither expert wins everywhere, so neither is assumed to. For each category both
a classical ternary net and a quantum variational circuit are trained on the same
data, both are scored on the same held-out split, and the one that is kept is
whichever wins on a stated rule:

* ``accuracy`` — pick the more accurate expert, ties going to the smaller one.
* ``size`` — pick the smaller expert whenever it is within ``tolerance`` of the
  better accuracy. This is the rule that suits a paged library: an expert two
  percentage points worse but ten times smaller is usually the better deal when
  the constraint is what fits in RAM.
* ``quantum`` / ``classical`` — force one kind, for comparison runs.

The chosen expert is written into the vault in the ordinary payload shape, so
everything downstream — paging, budgets, the working set, consolidation — treats
it like any other shard.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ultraquant.experts.moe import ExpertPool
from ultraquant.hybrid.expert import QuantumExpert, quantum_expert_payload
from ultraquant.model.network import UltraQuantNet

__all__ = ["HybridReport", "HybridExpertPool"]


@dataclass
class CategoryOutcome:
    """What both experts achieved for one category."""

    category: str
    chosen: str
    quantum_accuracy: float
    classical_accuracy: float
    quantum_parameters: int
    classical_parameters: int
    quantum_bytes: int
    classical_bytes: int
    quantum_seconds: float
    classical_seconds: float

    @property
    def size_ratio(self) -> float:
        """How many times smaller the quantum expert is on disk."""
        return self.classical_bytes / max(self.quantum_bytes, 1)


@dataclass
class HybridReport:
    """Outcome of building a hybrid library."""

    outcomes: list[CategoryOutcome] = field(default_factory=list)
    rule: str = "size"

    @property
    def quantum_chosen(self) -> int:
        """How many categories ended up quantum."""
        return sum(1 for o in self.outcomes if o.chosen == "quantum")

    def summary(self) -> dict:
        """Aggregate figures across every category."""
        if not self.outcomes:
            return {"categories": 0}
        count = len(self.outcomes)
        stored = sum(
            o.quantum_bytes if o.chosen == "quantum" else o.classical_bytes
            for o in self.outcomes
        )
        return {
            "categories": count,
            "rule": self.rule,
            "quantum_chosen": self.quantum_chosen,
            "classical_chosen": count - self.quantum_chosen,
            "mean_quantum_accuracy": sum(o.quantum_accuracy for o in self.outcomes) / count,
            "mean_classical_accuracy": sum(o.classical_accuracy for o in self.outcomes) / count,
            "quantum_bytes_total": sum(o.quantum_bytes for o in self.outcomes),
            "classical_bytes_total": sum(o.classical_bytes for o in self.outcomes),
            "stored_bytes": stored,
            "mean_size_ratio": sum(o.size_ratio for o in self.outcomes) / count,
            "quantum_seconds": sum(o.quantum_seconds for o in self.outcomes),
            "classical_seconds": sum(o.classical_seconds for o in self.outcomes),
        }


class HybridExpertPool:
    """Build a shard library whose experts may be quantum or classical."""

    def __init__(
        self,
        vault,
        cache,
        input_dim: int = 25,
        hidden: int = 16,
        num_qubits: int = 5,
        layers: int = 2,
        seed: int = 0,
    ) -> None:
        """Bind to a vault and cache.

        Args:
            vault: Where experts are stored.
            cache: Resident-set manager they are paged through.
            input_dim: Feature width. Amplitude encoding needs it to fit in
                ``2**num_qubits``.
            hidden: Hidden width of the classical comparison net.
            num_qubits: Register width for quantum experts.
            layers: Ansatz depth.
            seed: Base seed.
        """
        self.vault = vault
        self.cache = cache
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.num_qubits = int(num_qubits)
        self.layers = int(layers)
        self.seed = int(seed)

    # -- training ----------------------------------------------------------

    def _split(self, xs, ys, holdout: float, seed: int):
        """Shuffle and split into train and validation sets."""
        order = list(range(len(xs)))
        random.Random(seed).shuffle(order)
        cut = max(1, int(len(order) * (1.0 - holdout)))
        train, validate = order[:cut], order[cut:] or order[:1]
        return (
            [xs[i] for i in train], [ys[i] for i in train],
            [xs[i] for i in validate], [ys[i] for i in validate],
        )

    def build_category(
        self,
        category: str,
        labels: list[str],
        xs: list[list[float]],
        ys: list[int],
        rule: str = "size",
        tolerance: float = 0.05,
        quantum_epochs: int = 6,
        classical_epochs: int = 40,
        holdout: float = 0.25,
        keywords: list[str] | None = None,
    ) -> CategoryOutcome:
        """Train both kinds of expert and store whichever the rule prefers.

        Raises:
            ValueError: If ``rule`` is unknown.
        """
        if rule not in ("size", "accuracy", "quantum", "classical"):
            raise ValueError(f"unknown selection rule {rule!r}")

        train_x, train_y, val_x, val_y = self._split(xs, ys, holdout, self.seed)

        started = time.perf_counter()
        quantum = QuantumExpert(
            labels=list(labels), num_qubits=self.num_qubits,
            layers=self.layers, seed=self.seed,
        )
        quantum.fit(train_x, train_y, epochs=quantum_epochs, seed=self.seed)
        quantum_accuracy = quantum.score(val_x, val_y)
        quantum_seconds = time.perf_counter() - started

        started = time.perf_counter()
        classical = UltraQuantNet(self.input_dim, [self.hidden], len(labels), seed=self.seed)
        for _epoch in range(classical_epochs):
            classical.train_batch(train_x, train_y, lr=0.15)
        classical_accuracy = sum(
            1 for x, y in zip(val_x, val_y) if classical.predict(x)[0] == y
        ) / max(len(val_x), 1)
        classical_seconds = time.perf_counter() - started

        quantum_payload = quantum_expert_payload(quantum)
        classical_payload = {
            "labels": list(labels),
            "net": classical.state_dict(),
            "trained_examples": len(train_x),
        }
        quantum_bytes = _payload_bytes(quantum_payload)
        classical_bytes = _payload_bytes(classical_payload)

        chosen = self._choose(
            rule, quantum_accuracy, classical_accuracy,
            quantum_bytes, classical_bytes, tolerance,
        )
        payload = quantum_payload if chosen == "quantum" else classical_payload
        self.vault.add_shard(
            ExpertPool.shard_id(category), category, payload, kind="expert-net",
            associations={word: 0.5 for word in (keywords or [category, *labels])},
        )

        return CategoryOutcome(
            category=category,
            chosen=chosen,
            quantum_accuracy=quantum_accuracy,
            classical_accuracy=classical_accuracy,
            quantum_parameters=quantum.parameter_count,
            classical_parameters=self.input_dim * self.hidden + self.hidden
            + self.hidden * len(labels) + len(labels),
            quantum_bytes=quantum_bytes,
            classical_bytes=classical_bytes,
            quantum_seconds=quantum_seconds,
            classical_seconds=classical_seconds,
        )

    @staticmethod
    def _choose(
        rule: str, quantum_accuracy: float, classical_accuracy: float,
        quantum_bytes: int, classical_bytes: int, tolerance: float,
    ) -> str:
        """Apply the stated selection rule."""
        if rule == "quantum":
            return "quantum"
        if rule == "classical":
            return "classical"
        if rule == "accuracy":
            if quantum_accuracy > classical_accuracy:
                return "quantum"
            if classical_accuracy > quantum_accuracy:
                return "classical"
            return "quantum" if quantum_bytes <= classical_bytes else "classical"
        # "size": take the smaller expert unless it costs too much accuracy.
        best = max(quantum_accuracy, classical_accuracy)
        candidates = []
        if quantum_accuracy >= best - tolerance:
            candidates.append(("quantum", quantum_bytes))
        if classical_accuracy >= best - tolerance:
            candidates.append(("classical", classical_bytes))
        return min(candidates, key=lambda pair: pair[1])[0]

    def build(
        self, corpora, rule: str = "size", **kwargs
    ) -> HybridReport:
        """Build a whole library, one decision per category.

        Args:
            corpora: Objects with ``name``, ``labels``, ``xs``, ``ys`` and
                optionally ``keywords`` — the shape
                :mod:`ultraquant.forge.corpus` produces.
            rule: Selection rule.
            **kwargs: Passed to :meth:`build_category`.

        Returns:
            A :class:`HybridReport`.
        """
        report = HybridReport(rule=rule)
        for corpus in corpora:
            report.outcomes.append(self.build_category(
                corpus.name, list(corpus.labels), corpus.xs, corpus.ys,
                rule=rule, keywords=getattr(corpus, "keywords", None), **kwargs,
            ))
        return report

    # -- serving -----------------------------------------------------------

    def predict(self, category: str, features: list[float]) -> tuple[str, float]:
        """Predict with whichever kind of expert this category holds.

        The caller does not need to know which it is — that is the point of
        keeping both behind one payload shape.

        Raises:
            KeyError: If the category has no expert.
        """
        shard_id = ExpertPool.shard_id(category)
        if not self.vault.has(shard_id):
            raise KeyError(f"no expert for category {category!r}")

        def load() -> tuple[dict, int]:
            return self.vault.get(shard_id), int(self.vault.entry(shard_id)["nbytes"])

        payload = self.cache.get(shard_id, load)
        net = payload["net"]
        if net.get("kind") == "quantum-vqc":
            return QuantumExpert.load(net).predict(features)

        model = UltraQuantNet(
            input_dim=int(net["config"]["input_dim"]),
            hidden_dims=[int(h) for h in net["config"]["hidden_dims"]],
            num_classes=int(net["config"]["num_classes"]),
            seed=int(net["config"].get("seed", 0)),
        )
        model.load_state_dict(net)
        index, confidence = model.predict(features)
        return payload["labels"][index], confidence

    def kind_of(self, category: str) -> str:
        """Whether a category is served by a quantum or classical expert."""
        payload = self.vault.get(ExpertPool.shard_id(category))
        return "quantum" if payload["net"].get("kind") == "quantum-vqc" else "classical"


def _payload_bytes(payload: dict) -> int:
    """Stored size of a payload, compressed exactly as the vault stores it."""
    import json
    import zlib

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(zlib.compress(raw, 6))
