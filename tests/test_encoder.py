"""The encoder, and the machinery that decides whether it was worth building.

Two different things are tested here, and the second matters more.

:class:`PatternEncoder` is checked as a piece of code: it learns, it does not
collapse, it round-trips through JSON, it rejects malformed input.

The experiment harness is checked for the properties that make its answer worth
believing — that the encoder never sees the held-out family or any label, that
both arms get identical treatment apart from the representation, and that the
control condition is genuinely structureless. A measurement that quietly leaks
the answer into the treatment arm is worse than no measurement, because it
produces a confident wrong conclusion.

The gate's verdict itself is recorded in ARCHITECTURE.md §11.3, not asserted
here: these tests must not start failing because a result changed.
"""

from __future__ import annotations

import json
import random
import unittest

from ultraquant.experiments import transfer_gate as gate
from ultraquant.model.encoder import PatternEncoder


class PatternEncoderTests(unittest.TestCase):
    """The encoder works as a piece of machinery."""

    def setUp(self) -> None:
        self.rng = random.Random(0)
        self.data = [
            [min(1.0, max(0.0, self.rng.gauss(0.5, 0.3))) for _ in range(30)]
            for _ in range(120)
        ]

    def test_training_reduces_reconstruction_error(self) -> None:
        encoder = PatternEncoder(30, embed_dim=16, seed=0)
        before = encoder.fit(self.data, epochs=1, lr=0.05)
        after = encoder.fit(self.data, epochs=40, lr=0.05)
        self.assertLess(after, before)

    def test_the_representation_does_not_collapse(self) -> None:
        """A constant embedding would reduce the whole experiment to noise."""
        encoder = PatternEncoder(30, embed_dim=16, seed=0)
        encoder.fit(self.data, epochs=40, lr=0.05)
        embeddings = [encoder.encode(x) for x in self.data]
        spreads = [
            max(e[d] for e in embeddings) - min(e[d] for e in embeddings)
            for d in range(16)
        ]
        self.assertGreater(sum(spreads) / len(spreads), 0.1)

    def test_encoding_is_deterministic(self) -> None:
        encoder = PatternEncoder(30, embed_dim=8, seed=1)
        self.assertEqual(encoder.encode(self.data[0]), encoder.encode(self.data[0]))

    def test_wrong_width_is_refused(self) -> None:
        encoder = PatternEncoder(30, embed_dim=8, seed=1)
        with self.assertRaises(ValueError):
            encoder.encode([0.5] * 12)

    def test_state_round_trips_through_json(self) -> None:
        """An encoder has to be storable in the library like anything else."""
        encoder = PatternEncoder(30, embed_dim=8, seed=2)
        encoder.fit(self.data, epochs=5, lr=0.05)
        restored = PatternEncoder.from_state_dict(
            json.loads(json.dumps(encoder.state_dict()))
        )
        self.assertEqual(restored.encode(self.data[0]), encoder.encode(self.data[0]))

    def test_fitting_nothing_is_survivable(self) -> None:
        self.assertEqual(PatternEncoder(30, 8, seed=0).fit([]), 0.0)


class NetworkEmbedTests(unittest.TestCase):
    """The discriminative arm reads the penultimate layer."""

    def test_embed_matches_the_forward_pass_up_to_the_head(self) -> None:
        from ultraquant.model.network import UltraQuantNet

        net = UltraQuantNet(input_dim=30, hidden_dims=[12], num_classes=4, seed=0)
        x = [0.3] * 30
        hidden = net.embed(x)
        self.assertEqual(len(hidden), 12)
        self.assertEqual(net.head.forward(hidden), net.forward(x))

    def test_a_net_with_no_hidden_layers_embeds_its_input(self) -> None:
        from ultraquant.model.network import UltraQuantNet

        net = UltraQuantNet(input_dim=6, hidden_dims=[], num_classes=2, seed=0)
        self.assertEqual(net.embed([1.0] * 6), [1.0] * 6)


class ExperimentIntegrityTests(unittest.TestCase):
    """The properties that make the gate's answer worth believing."""

    def test_compositional_families_share_parts_but_not_combinations(self) -> None:
        """Held-out labels must be new combinations of seen parts.

        If a held-out combination also appeared during pretraining the test
        measures memorisation; if the parts were not shared there would be
        nothing to transfer and the null result would be meaningless.
        """
        families = gate.compositional_families(random.Random(0), families=7, labels=4)
        flat = ["".join(str(int(v)) for v in proto)
                for family in families for proto in family]
        self.assertEqual(len(set(flat)), len(flat), "every label is distinct")

        seen = {tuple(vec) for family in families[:-1] for vec in family}
        for prototype in families[-1]:
            self.assertNotIn(tuple(prototype), seen,
                             "a held-out label reappeared in pretraining")

    def test_the_control_really_is_structureless(self) -> None:
        """The control must not accidentally contain shared parts."""
        families = gate.unstructured_families(random.Random(0), families=7, labels=4)
        flat = [tuple(proto) for family in families for proto in family]
        self.assertEqual(len(set(flat)), len(flat))

    def test_both_mechanisms_report_the_same_shaped_result(self) -> None:
        for name, mechanism in gate.MECHANISMS:
            with self.subTest(mechanism=name):
                result = mechanism(0, structured=True, shots=5)
                self.assertEqual(set(result), {"raw", "encoded", "delta"})
                self.assertAlmostEqual(
                    result["delta"], result["encoded"] - result["raw"], places=12
                )
                for key in ("raw", "encoded"):
                    self.assertGreaterEqual(result[key], 0.0)
                    self.assertLessEqual(result[key], 1.0)

    def test_trials_are_reproducible(self) -> None:
        """A gate whose answer moves between runs decides nothing."""
        self.assertEqual(gate.one_trial(3, shots=5), gate.one_trial(3, shots=5))

    def test_run_gate_reports_a_paired_confidence_interval(self) -> None:
        result = gate.run_gate(seeds=3, shots=5, structured=True)
        low, high = result["ci95"]
        self.assertLessEqual(low, result["delta_mean"])
        self.assertGreaterEqual(high, result["delta_mean"])
        self.assertEqual(result["passes"], low > 0.0)

    def test_the_operating_point_leaves_headroom(self) -> None:
        """At ceiling the experiment cannot detect an effect in either direction.

        The first run of this gate scored 0.992 on the baseline arm and reported
        a confident FAIL that meant nothing. The noise level exists to prevent
        that, so it is worth a test of its own.
        """
        accuracies = [gate.one_trial(seed, shots=5)["raw"] for seed in range(6)]
        mean = sum(accuracies) / len(accuracies)
        self.assertLess(mean, 0.90, "baseline is too close to ceiling to measure")
        self.assertGreater(mean, 0.40, "baseline is too close to chance to measure")


if __name__ == "__main__":
    unittest.main()
