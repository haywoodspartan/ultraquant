"""A representation this system owns, distilled from one it borrowed.

§11.110 distilled the pretrained teacher's geometry into a
from-scratch encoder so the semantic advantage costs no network call.
These pins hold the projection's determinism, the encoder's ability
to learn at all, and the run-one diagnosis that a failed distillation
was an unfinished one.
"""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.model.distilled import (DistilledEncoder, project,
                                        random_projection)


def _cos(a: list, b: list) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class ProjectionTests(unittest.TestCase):
    """Deterministic, or a stored encoder stops agreeing with itself."""

    def test_the_projection_is_reproducible(self) -> None:
        self.assertEqual(random_projection(32, 8, seed=1),
                         random_projection(32, 8, seed=1))

    def test_a_different_seed_gives_a_different_projection(self) -> None:
        self.assertNotEqual(random_projection(32, 8, seed=1),
                            random_projection(32, 8, seed=2))

    def test_the_shape_is_target_by_source(self) -> None:
        planes = random_projection(64, 16, seed=0)
        self.assertEqual(len(planes), 16)
        self.assertEqual(len(planes[0]), 64)
        self.assertEqual(len(project([0.0] * 64, planes)), 16)

    def test_wider_projections_keep_more_structure(self) -> None:
        """The measurement that chose 128 over the reflex 32."""
        rng = random.Random(7)
        vectors = [[rng.gauss(0, 1) for _ in range(256)] for _ in range(24)]
        pairs = [(a, b) for i, a in enumerate(vectors)
                 for b in vectors[i + 1:]]
        truth = [_cos(a, b) for a, b in pairs]
        errors = []
        for width in (8, 64):
            planes = random_projection(256, width, seed=0)
            got = [_cos(project(a, planes), project(b, planes))
                   for a, b in pairs]
            errors.append(sum(abs(g - t) for g, t in zip(got, truth))
                          / len(truth))
        self.assertLess(errors[1], errors[0],
                        "a wider projection must distort less")


class EncoderTests(unittest.TestCase):
    """It has to actually fit, which run one discovered it had not."""

    def _data(self, count: int = 20, width: int = 24, dim: int = 8):
        rng = random.Random(3)
        xs = [[1.0 if rng.random() < 0.25 else 0.0 for _ in range(width)]
              for _ in range(count)]
        ts = [[rng.gauss(0, 0.4) for _ in range(dim)] for _ in range(count)]
        return xs, ts

    def test_training_reduces_the_loss(self) -> None:
        xs, ts = self._data()
        encoder = DistilledEncoder(24, dim=8, hidden=16, seed=0)
        first = encoder.fit(xs, ts, epochs=1, lr=0.05)
        later = encoder.fit(xs, ts, epochs=400, lr=0.05)
        self.assertLess(later, first)

    def test_more_epochs_fit_better(self) -> None:
        """Run one FAILED because 400 epochs was not a finished copy."""
        xs, ts = self._data()
        losses = []
        for epochs in (50, 800):
            encoder = DistilledEncoder(24, dim=8, hidden=16, seed=0)
            losses.append(encoder.fit(xs, ts, epochs=epochs, lr=0.05))
        self.assertLess(losses[1], losses[0])

    def test_encoding_is_deterministic(self) -> None:
        encoder = DistilledEncoder(24, dim=8, hidden=16, seed=0)
        x = [1.0, 0.0] * 12
        self.assertEqual(encoder.encode(x), encoder.encode(x))

    def test_the_output_width_is_the_requested_dimension(self) -> None:
        encoder = DistilledEncoder(24, dim=11, hidden=16, seed=0)
        self.assertEqual(len(encoder.encode([0.0] * 24)), 11)

    def test_the_same_seed_builds_the_same_encoder(self) -> None:
        a = DistilledEncoder(16, dim=4, hidden=8, seed=5)
        b = DistilledEncoder(16, dim=4, hidden=8, seed=5)
        self.assertEqual(a.encode([1.0] * 16), b.encode([1.0] * 16))

    def test_encoding_touches_no_network(self) -> None:
        """The whole point: the teacher is paid once, offline."""
        import inspect

        source = inspect.getsource(DistilledEncoder)
        for forbidden in ("LMStudio", "urlopen", "requests", "http"):
            self.assertNotIn(forbidden, source)


class GateVerdictTests(unittest.TestCase):
    """A fail whose trajectory is the finding."""

    def setUp(self) -> None:
        from ultraquant.experiments import distill_gate
        self.doc = " ".join(distill_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Nothing touches the network at query time",
                       "It keeps at least half the borrowed advantage",
                       "The advantage is still semantic",
                       "The vocabulary boundary is measured, not avoided",
                       "Both costs are named"):
            self.assertIn(phrase, self.doc)

    def test_what_was_not_attempted_is_said_first(self) -> None:
        """Learning semantics from nothing is not claimed."""
        self.assertIn("What is not attempted, said first", self.doc)
        self.assertIn("Learning that geometry from nothing", self.doc)

    def test_the_trajectory_is_recorded(self) -> None:
        self.assertIn("-12%", self.doc)
        self.assertIn("+42%", self.doc)
        self.assertIn("+58%", self.doc)

    def test_both_constraints_are_named_as_mine(self) -> None:
        """A training budget and a projection width, not walls."""
        self.assertIn("looked like a refutation and was an unfinished "
                      "copy", self.doc)
        self.assertIn("The encoder was never the binding constraint",
                      self.doc)
        self.assertIn("Neither was a wall", self.doc)

    def test_the_bar_being_stricter_than_it_read_is_admitted(self):
        self.assertIn("which as a share of the borrowed margin is **86%**, "
                      "not half", self.doc)
        self.assertIn("it is a fail against the bar as registered",
                      self.doc)

    def test_the_vocabulary_ceiling_is_recorded(self) -> None:
        self.assertIn("0.350", self.doc)
        self.assertIn("has nothing whatever for the rest", self.doc)


if __name__ == "__main__":
    unittest.main()
