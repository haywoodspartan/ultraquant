"""A learned dispatcher: the system deciding how to use its own hardware.

Tier selection has been a fixed preference walk — CUDA, then C++, then Python —
and §6 measured that walk making real mistakes: the GPU *loses five-fold* on
circuits small enough to live in L1, and the right thread count for the forge
depends on the workload's shape. Cores, threads and tiers are a decision
problem, so they get a decision maker that learns this machine instead of a
rule that guesses at it.

Two brains are built, because the right one is an empirical question:

* **A quantum policy** — the project's own variational circuit
  (:class:`~ultraquant.quantum.vqc.VariationalClassifier`): workload features
  amplitude-encoded into three qubits, a trained ansatz, class scores per
  compute configuration. The quantum simulator deciding how to run the quantum
  simulator.
* **A classical policy** — an :class:`~ultraquant.model.network.UltraQuantNet`
  over the same features.

Both train on the same **experience**: probed timings of real workloads on
this machine. A shootout on held-out experience picks the active brain by
accuracy first and decision latency second — a decider that costs more than
it saves is worse than the static walk. The verdict is stored with its
numbers and re-fought as experience grows.

Safety property that makes learned dispatch acceptable at all: every tier
computes identical values (§1, principle 1), so a wrong decision costs
*time*, never correctness. Cold starts probe — run the options, keep the
winner, remember the timing — so early decisions are measurements rather
than guesses.

Timings are wall-clock and machine-specific, so experience is explicitly
*this machine's* knowledge; the policies retrain deterministically from
whatever experience exists.

Pure Python standard library.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

__all__ = ["WORKLOAD_KINDS", "DispatchExperience", "LearnedDispatch"]

#: Decision spaces per workload kind. Threads are real decision axes only
#: where the native layer accepts them (the forge trainer does).
WORKLOAD_KINDS: dict[str, list[str]] = {
    "quantum": ["python", "cpp", "cuda"],
    "forge": ["python", "cpp@1", "cpp@half", "cpp@max", "cuda"],
}

#: Below this softmax confidence the policy declines and a probe runs instead.
_CONFIDENCE_FLOOR = 0.55

#: Retrain and re-fight the shootout every this many new experiences per kind.
_RETRAIN_EVERY = 8

#: Bumped whenever :func:`workload_features` changes meaning; stored records
#: from another version describe different axes and are dropped on load.
FEATURES_VERSION = 2


def _log_scale(value: float, ceiling: float) -> float:
    """``log2`` squashed into [0, 1]."""
    return min(1.0, math.log2(max(1.0, float(value)) + 1.0) / ceiling)


def workload_features(kind: str, dims: dict[str, float]) -> list[float]:
    """A workload as eight normalised features.

    Args:
        kind: One of :data:`WORKLOAD_KINDS`.
        dims: Size measures — ``qubits``, ``gates``, ``batch`` for quantum;
            ``experts``, ``samples``, ``features``, ``hidden``, ``classes``
            for forge. Missing measures are zero.

    Returns:
        Eight floats in [0, 1].
    """
    # Both a linear and a log view of each size: log alone squashed the
    # decision boundaries so hard the classical brain collapsed to one answer
    # (measured: exactly the majority-class rate, 0.60, at every capacity
    # tried; linear+log lifted it to 0.87).
    if kind == "quantum":
        return [
            1.0, 0.0,
            min(1.0, dims.get("qubits", 0) / 16.0),
            _log_scale(dims.get("qubits", 0), 5.0),
            min(1.0, dims.get("batch", 0) / 32.0),
            _log_scale(dims.get("batch", 0), 6.0),
            min(1.0, dims.get("gates", 0) / 48.0),
            0.0,
        ]
    return [
        0.0, 1.0,
        min(1.0, dims.get("experts", 0) / 32.0),
        _log_scale(dims.get("samples", 0), 14.0),
        min(1.0, dims.get("hidden", 0) / 64.0),
        min(1.0, dims.get("classes", 0) / 16.0),
        min(1.0, dims.get("features", 0) / 128.0),
        _log_scale(dims.get("epochs", 0), 8.0),
    ]


class DispatchExperience:
    """What this machine has measured about itself.

    Records are ``{"kind", "features", "config", "seconds"}`` per probed
    configuration, persisted as JSON beside whatever home owns the scheduler.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[dict] = []
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if int(payload.get("version", 1)) == FEATURES_VERSION:
                self.records = list(payload.get("records", []))
                self.verdicts: dict = dict(payload.get("verdicts", {}))
            else:
                self.verdicts = {}
        else:
            self.verdicts = {}

    def add(self, kind: str, features: list[float], timings: dict[str, float]) -> None:
        """Record one probe: every configuration's measured seconds."""
        for config, seconds in timings.items():
            self.records.append({
                "kind": kind,
                "features": [round(float(v), 5) for v in features],
                "config": config,
                "seconds": round(float(seconds), 6),
            })
        self.save()

    def winners(self, kind: str) -> list[tuple[list[float], str]]:
        """Per distinct workload, the configuration that measured fastest."""
        by_workload: dict[tuple, dict[str, float]] = {}
        for record in self.records:
            if record["kind"] != kind:
                continue
            key = tuple(record["features"])
            best = by_workload.setdefault(key, {})
            config = record["config"]
            if config not in best or record["seconds"] < best[config]:
                best[config] = record["seconds"]
        out = []
        for key, timings in sorted(by_workload.items()):
            winner = min(sorted(timings), key=lambda c: timings[c])
            out.append((list(key), winner))
        return out

    def save(self) -> None:
        """Persist, when a path was given."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"records": self.records, "verdicts": self.verdicts,
                        "version": FEATURES_VERSION},
                       sort_keys=True),
            encoding="utf-8",
        )


class _ClassicalPolicy:
    """UltraQuantNet over workload features."""

    name = "classical"

    def __init__(self, configs: list[str], seed: int = 0) -> None:
        from ultraquant.model.network import UltraQuantNet

        self.configs = list(configs)
        self.net = UltraQuantNet(input_dim=8, hidden_dims=[12],
                                 num_classes=len(configs), seed=seed)

    def fit(self, examples: list[tuple[list[float], str]], epochs: int = 80) -> None:
        import random as _random

        xs = [features for features, _ in examples]
        ys = [self.configs.index(config) for _, config in examples]
        rng = _random.Random(0)
        order = list(range(len(xs)))
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, len(order), 8):
                chunk = order[start:start + 8]
                self.net.train_batch([xs[i] for i in chunk],
                                     [ys[i] for i in chunk], 0.05)

    def decide(self, features: list[float]) -> tuple[str, float]:
        index, confidence = self.net.predict(features)
        return self.configs[index], confidence


class _QuantumPolicy:
    """The variational circuit as decision maker.

    Eight features amplitude-encoded into three qubits; the trained ansatz
    and a linear head score each configuration. Slower to train (parameter-
    shift costs 2P circuit evaluations per step) and slower to ask — which is
    exactly why the shootout weighs decision latency, not only accuracy.
    """

    name = "quantum"

    def __init__(self, configs: list[str], seed: int = 0) -> None:
        from ultraquant.quantum.vqc import VariationalClassifier

        self.configs = list(configs)
        self.circuit = VariationalClassifier(
            num_classes=len(configs), num_qubits=3, layers=2, seed=seed
        )

    def fit(self, examples: list[tuple[list[float], str]], epochs: int = 12) -> None:
        import random as _random

        rng = _random.Random(0)
        order = list(range(len(examples)))
        for _ in range(epochs):
            rng.shuffle(order)
            for index in order:
                features, config = examples[index]
                self.circuit.train_step(features,
                                        self.configs.index(config))

    def decide(self, features: list[float]) -> tuple[str, float]:
        index, confidence = self.circuit.predict(features)
        return self.configs[index], confidence


class _CommitteePolicy:
    """One single-qubit circuit per core, each thinking about something else.

    The committee is the "one qubit per core" design: member *i* is a
    single-qubit variational circuit whose input angle is its own seeded
    projection of the workload features, so every core's member attends to a
    different mixture of the workload — diversity by construction, not by
    data split. Members train independently and are therefore embarrassingly
    parallel; deciding runs them across a thread pool sized to the machine.

    Honesty about the parallelism: a single-qubit statevector is two complex
    numbers, and in pure Python the GIL serialises the threads anyway — the
    committee's value must come from *diverse judgement*, and the shootout
    holds it to that. The threading is architecture (members are genuinely
    independent), measured rather than promised.

    The combination is a reliability-weighted vote: each member's training
    accuracy becomes its voice's weight, and the ensemble answer is the
    config with the greatest weighted support. Deterministic throughout —
    projections, training order and weights all derive from seeds, and a
    threaded decision equals a serial one exactly.
    """

    name = "committee"

    def __init__(self, configs: list[str], seed: int = 0,
                 members: int | None = None, depth: int | None = None) -> None:
        import random as _random

        self.configs = list(configs)
        # The committee scales with the machine: every core seats a member,
        # and a GPU deepens each member's circuit by a second projected
        # rotation. Richer hardware buys a budget for more complex judgement
        # - the brain-like stance - but the shootout still decides whether
        # that complexity earns its place over the simpler brains.
        count = members if members is not None else min(32, os.cpu_count() or 4)
        if depth is None:
            from ultraquant.native import accel

            depth = 2 if accel.load_gpu() is not None else 1
        self.depth = max(1, int(depth))
        self.members: list[dict] = []
        for index in range(count):
            rng = _random.Random(f"{seed}|committee|{index}")
            self.members.append({
                "projections": [
                    [rng.uniform(-1.0, 1.0) for _ in range(8)]
                    for _ in range(self.depth)
                ],
                "theta": [rng.uniform(-0.3, 0.3) for _ in range(self.depth)],
                "weights": [[rng.uniform(-0.5, 0.5) for _ in range(1)]
                            for _ in self.configs],
                "bias": [0.0] * len(self.configs),
                "reliability": 1.0,
            })

    def _z(self, member: dict, features: list[float]) -> float:
        """The member's <Z>: a product of projected rotations, one per depth.

        Depth 1 is a single-qubit RY readout. Depth 2 multiplies in a second
        rotation over an independent projection - the deeper judgement richer
        hardware pays for.
        """
        z = 1.0
        for level in range(self.depth):
            angle = sum(p * f for p, f in
                        zip(member["projections"][level], features))
            z *= math.cos(angle + member["theta"][level])
        return z

    def _scores(self, member: dict, features: list[float]) -> list[float]:
        z = self._z(member, features)
        return [row[0] * z + b for row, b in zip(member["weights"], member["bias"])]

    def fit(self, examples: list[tuple[list[float], str]], epochs: int = 40) -> None:
        import random as _random

        for member in self.members:
            rng = _random.Random(0)
            order = list(range(len(examples)))
            for _ in range(epochs):
                rng.shuffle(order)
                for index in order:
                    features, config = examples[index]
                    target = self.configs.index(config)
                    scores = self._scores(member, features)
                    peak = max(scores)
                    exps = [math.exp(s - peak) for s in scores]
                    total = sum(exps)
                    probabilities = [e / total for e in exps]
                    z = self._z(member, features)
                    grad_z = 0.0
                    for k, probability in enumerate(probabilities):
                        gradient = probability - (1.0 if k == target else 0.0)
                        grad_z += gradient * member["weights"][k][0]
                        member["weights"][k][0] -= 0.1 * gradient * z
                        member["bias"][k] -= 0.1 * gradient
                    for level in range(self.depth):
                        angle = sum(p * f for p, f in
                                    zip(member["projections"][level], features))
                        rest = z / max(1e-9, math.cos(
                            angle + member["theta"][level]))
                        dz = -math.sin(angle + member["theta"][level]) * rest
                        member["theta"][level] -= 0.05 * grad_z * dz
            hits = sum(
                max(range(len(self.configs)),
                    key=lambda k: self._scores(member, f)[k])
                == self.configs.index(config)
                for f, config in examples
            )
            member["reliability"] = max(0.05, hits / max(1, len(examples)))

    def decide(self, features: list[float]) -> tuple[str, float]:
        from concurrent.futures import ThreadPoolExecutor

        def voice(member: dict) -> tuple[int, float]:
            scores = self._scores(member, features)
            best = max(range(len(scores)), key=lambda k: scores[k])
            return best, member["reliability"]

        with ThreadPoolExecutor(max_workers=len(self.members)) as pool:
            votes = list(pool.map(voice, self.members))
        support = [0.0] * len(self.configs)
        for choice, weight in votes:
            support[choice] += weight
        total = sum(support) or 1.0
        best = max(range(len(support)), key=lambda k: (support[k], -k))
        return self.configs[best], support[best] / total


class LearnedDispatch:
    """Decide tier and threads from experience; probe when unsure.

    Args:
        path: Where experience persists (``dispatch.json`` under a home).
        available: Which configurations exist on this machine, per kind —
            e.g. no ``cuda`` entries without a GPU. Defaults to everything.
        seed: Policy seeds.
    """

    def __init__(self, path: str | os.PathLike | None = None,
                 available: dict[str, list[str]] | None = None,
                 seed: int = 0) -> None:
        self.experience = DispatchExperience(path)
        self.available = {
            kind: list(configs) for kind, configs in
            (available or WORKLOAD_KINDS).items()
        }
        self.seed = seed
        self._policies: dict[str, Any] = {}
        self._since_train: dict[str, int] = {}
        for kind in self.available:
            if len(self.experience.winners(kind)) >= 4:
                self._train(kind)

    # ------------------------------------------------------------------ #

    def _train(self, kind: str) -> None:
        """Fit both brains on this kind's experience and fight the shootout."""
        winners = self.experience.winners(kind)
        if len(winners) < 4:
            return
        # Deterministic split: every third example judges, the rest teach.
        held = [w for i, w in enumerate(winners) if i % 3 == 2]
        train = [w for i, w in enumerate(winners) if i % 3 != 2]
        if not held:
            held = train

        results = {}
        for maker in (_ClassicalPolicy, _QuantumPolicy, _CommitteePolicy):
            policy = maker(self.available[kind], seed=self.seed)
            policy.fit(train)
            hits = sum(policy.decide(f)[0] == config for f, config in held)
            start = time.perf_counter()
            for features, _config in held:
                policy.decide(features)
            latency = (time.perf_counter() - start) / len(held)
            results[policy.name] = {
                "policy": policy,
                "accuracy": hits / len(held),
                "decide_seconds": latency,
            }

        # Accuracy first; latency breaks ties. A brain that decides worse than
        # the other is out regardless of speed; equal brains should be cheap.
        chosen = max(results, key=lambda name: (results[name]["accuracy"],
                                                -results[name]["decide_seconds"]))
        self._policies[kind] = results[chosen]["policy"]
        committee = results.get("committee", {}).get("policy")
        self.experience.verdicts[kind] = {
            "chosen": chosen,
            "committee": {
                "members": len(committee.members) if committee else 0,
                "depth": committee.depth if committee else 0,
            },
            "examples": len(winners),
            "accuracy": {name: round(r["accuracy"], 3)
                         for name, r in results.items()},
            "decide_us": {name: round(r["decide_seconds"] * 1e6, 1)
                          for name, r in results.items()},
        }
        self.experience.save()
        self._since_train[kind] = 0

    # ------------------------------------------------------------------ #

    def decide(self, kind: str, dims: dict[str, float]) -> tuple[str, str]:
        """The configuration to use, and why.

        Returns:
            ``(config, reason)`` where reason is ``"learned"`` when a
            confident policy chose, or ``"probe"`` when the caller should run
            :meth:`probe` because nothing confident is known yet.
        """
        features = workload_features(kind, dims)
        policy = self._policies.get(kind)
        if policy is not None:
            config, confidence = policy.decide(features)
            if confidence >= _CONFIDENCE_FLOOR and config in self.available[kind]:
                return config, "learned"
        return self.available[kind][-1], "probe"

    def probe(self, kind: str, dims: dict[str, float],
              runners: dict[str, Callable[[], Any]]) -> tuple[str, dict[str, float]]:
        """Measure every offered configuration once and learn from it.

        Args:
            kind: Workload kind.
            dims: Its size measures.
            runners: ``config -> thunk`` actually executing the workload on
                that configuration. Only offered configs are timed.

        Returns:
            ``(winner, timings)``.
        """
        features = workload_features(kind, dims)
        timings: dict[str, float] = {}
        for config, run in runners.items():
            start = time.perf_counter()
            run()
            timings[config] = time.perf_counter() - start
        winner = min(sorted(timings), key=lambda c: timings[c])
        self.experience.add(kind, features, timings)
        self._since_train[kind] = self._since_train.get(kind, 0) + 1
        if (self._since_train[kind] >= _RETRAIN_EVERY
                or kind not in self._policies):
            self._train(kind)
        return winner, timings

    def report(self) -> dict:
        """The current verdicts, for the compute surfaces to show."""
        return dict(self.experience.verdicts)
