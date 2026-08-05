"""The learned dispatcher: cores, threads and tiers as a decision problem.

Live measurement behind this module: the static preference walk (CUDA first)
cost 86.9 ms across a 12-shape grid where measured-best cost 7.0 ms — the GPU
loses 350-fold on a first-touch 2-qubit batch. Decisions are learned from
probed experience; three brains compete (classical net, 3-qubit variational
circuit, one-single-qubit-member-per-core committee) and a shootout on
held-out experience picks by accuracy first and decision latency second.

What is asserted here is machinery and honesty properties, not which brain
wins — that verdict is machine-dependent by design and stored with its
numbers. The safety property that makes learned dispatch acceptable is
asserted elsewhere by the whole tier test suite: every tier computes the same
values, so a wrong decision costs time, never correctness.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.native.scheduler import (
    DispatchExperience,
    LearnedDispatch,
    _ClassicalPolicy,
    _CommitteePolicy,
    _QuantumPolicy,
    workload_features,
)

#: A synthetic machine truth the policies must be able to learn: small
#: workloads go to python, medium to cpp, large to cuda.
def _truth(qubits: int) -> str:
    if qubits <= 3:
        return "python"
    if qubits <= 12:
        return "cpp"
    return "cuda"


def _experience_grid() -> list[tuple[list[float], str]]:
    examples = []
    for qubits in range(2, 17):
        for batch in (1, 4, 16):
            features = workload_features(
                "quantum", {"qubits": qubits, "gates": qubits * 3, "batch": batch}
            )
            examples.append((features, _truth(qubits)))
    return examples


class PolicyTests(unittest.TestCase):
    """Each brain, on the same learnable truth."""

    CONFIGS = ["python", "cpp", "cuda"]

    def _accuracy(self, policy) -> float:
        examples = _experience_grid()
        train = [e for i, e in enumerate(examples) if i % 3 != 2]
        held = [e for i, e in enumerate(examples) if i % 3 == 2]
        policy.fit(train)
        return sum(policy.decide(f)[0] == c for f, c in held) / len(held)

    def test_the_classical_brain_learns_the_machine(self) -> None:
        self.assertGreaterEqual(
            self._accuracy(_ClassicalPolicy(self.CONFIGS, seed=0)), 0.8
        )

    def test_the_quantum_brain_learns_the_machine(self) -> None:
        self.assertGreaterEqual(
            self._accuracy(_QuantumPolicy(self.CONFIGS, seed=0)), 0.6
        )

    def test_the_committee_learns_the_machine(self) -> None:
        self.assertGreaterEqual(
            self._accuracy(_CommitteePolicy(self.CONFIGS, seed=0, members=8)), 0.6
        )

    def test_a_threaded_committee_decision_equals_a_serial_one(self) -> None:
        """Parallel structure must never change the answer."""
        committee = _CommitteePolicy(self.CONFIGS, seed=0, members=8)
        committee.fit(_experience_grid()[:20])
        features = workload_features("quantum", {"qubits": 9, "gates": 27,
                                                 "batch": 4})
        threaded = committee.decide(features)

        votes = []
        for member in committee.members:
            scores = committee._scores(member, features)
            best = max(range(len(scores)), key=lambda k: scores[k])
            votes.append((best, member["reliability"]))
        support = [0.0] * len(self.CONFIGS)
        for choice, weight in votes:
            support[choice] += weight
        serial = self.CONFIGS[max(range(len(support)),
                                  key=lambda k: (support[k], -k))]
        self.assertEqual(threaded[0], serial)

    def test_committee_scales_with_the_machine(self) -> None:
        """More cores seat more members; depth is a hardware budget."""
        import os

        default = _CommitteePolicy(self.CONFIGS, seed=0)
        self.assertEqual(len(default.members),
                         min(32, os.cpu_count() or 4))
        deep = _CommitteePolicy(self.CONFIGS, seed=0, members=4, depth=2)
        shallow = _CommitteePolicy(self.CONFIGS, seed=0, members=4, depth=1)
        self.assertEqual(deep.depth, 2)
        self.assertEqual(shallow.depth, 1)

    def test_policies_are_deterministic(self) -> None:
        examples = _experience_grid()[:24]
        one = _ClassicalPolicy(self.CONFIGS, seed=3)
        two = _ClassicalPolicy(self.CONFIGS, seed=3)
        one.fit(examples)
        two.fit(examples)
        features = workload_features("quantum", {"qubits": 6, "gates": 18,
                                                 "batch": 2})
        self.assertEqual(one.decide(features), two.decide(features))


class LearnedDispatchTests(unittest.TestCase):
    """Probe, learn, decide, persist."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_dispatch_"))
        self.available = {"quantum": ["python", "cpp", "cuda"]}

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fill(self, dispatch: LearnedDispatch) -> None:
        """Probe a grid with synthetic runners whose cost follows _truth."""
        for qubits in range(2, 15, 2):
            dims = {"qubits": qubits, "gates": qubits * 3, "batch": 4}
            cost = {"python": 0.0001 if qubits <= 3 else 0.01,
                    "cpp": 0.0005 if qubits <= 12 else 0.01,
                    "cuda": 0.005 if qubits <= 12 else 0.0005}
            dispatch.probe("quantum", dims, {
                config: (lambda c=config: sum(
                    range(int(cost[c] * 1e5))))  # deterministic busy-work
                for config in self.available["quantum"]
            })

    def test_cold_start_asks_for_a_probe(self) -> None:
        dispatch = LearnedDispatch(self.dir / "d.json",
                                   available=self.available)
        _config, reason = dispatch.decide("quantum", {"qubits": 5})
        self.assertEqual(reason, "probe")

    def test_probing_teaches_and_decisions_become_learned(self) -> None:
        dispatch = LearnedDispatch(self.dir / "d.json",
                                   available=self.available)
        self._fill(dispatch)
        config, reason = dispatch.decide(
            "quantum", {"qubits": 9, "gates": 27, "batch": 4}
        )
        self.assertEqual(reason, "learned")
        self.assertIn(config, self.available["quantum"])

    def test_the_shootout_verdict_is_recorded_with_numbers(self) -> None:
        dispatch = LearnedDispatch(self.dir / "d.json",
                                   available=self.available)
        self._fill(dispatch)
        verdict = dispatch.report()["quantum"]
        self.assertIn(verdict["chosen"], ("classical", "quantum", "committee"))
        self.assertEqual(set(verdict["accuracy"]),
                         {"classical", "quantum", "committee"})
        self.assertEqual(set(verdict["decide_us"]),
                         {"classical", "quantum", "committee"})
        self.assertGreater(verdict["committee"]["members"], 0)

    def test_experience_survives_a_restart(self) -> None:
        dispatch = LearnedDispatch(self.dir / "d.json",
                                   available=self.available)
        self._fill(dispatch)
        reopened = LearnedDispatch(self.dir / "d.json",
                                   available=self.available)
        config, reason = reopened.decide(
            "quantum", {"qubits": 9, "gates": 27, "batch": 4}
        )
        self.assertEqual(reason, "learned")

    def test_unavailable_configurations_are_never_chosen(self) -> None:
        cpu_only = {"quantum": ["python", "cpp"]}
        dispatch = LearnedDispatch(self.dir / "cpu.json", available=cpu_only)
        for qubits in range(2, 15, 2):
            dims = {"qubits": qubits, "gates": qubits * 3, "batch": 4}
            dispatch.probe("quantum", dims, {
                "python": lambda: None, "cpp": lambda: sum(range(100)),
            })
        for qubits in (3, 8, 14):
            config, _reason = dispatch.decide(
                "quantum", {"qubits": qubits, "gates": qubits * 3, "batch": 4}
            )
            self.assertIn(config, cpu_only["quantum"])

    def test_winners_prefer_the_fastest_measurement(self) -> None:
        experience = DispatchExperience(self.dir / "e.json")
        features = workload_features("quantum", {"qubits": 4, "gates": 12,
                                                 "batch": 2})
        experience.add("quantum", features,
                       {"python": 0.001, "cpp": 0.0005, "cuda": 0.01})
        winners = experience.winners("quantum")
        self.assertEqual(winners[0][1], "cpp")


if __name__ == "__main__":
    unittest.main()
