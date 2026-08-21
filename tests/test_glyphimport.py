"""Weights that come back out of the library and still recognise.

§11.98 closed the loop from shard to computing network; §11.99 gave
every imported tensor a glyph. These pins hold the two decisions that
run one of the import gate found the hard way: a full-precision layer
does not travel as trits, and neither does a bias.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.convert import glyph, load
from ultraquant.experiments import glyphimport_gate
from ultraquant.model.network import UltraQuantNet
from ultraquant.model.quantize import TernaryLinear
from ultraquant.shards.vault import ShardVault


class RoundTripTests(unittest.TestCase):
    """A network stored and paged back is the same network."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_glyphimport_"))
        self.vault = ShardVault(self.dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_ternary_layer_computes_identically(self) -> None:
        rng = random.Random(3)
        layer = TernaryLinear(12, 6, rng, quantized=True)
        load.store_layer(self.vault, layer, "t:one", "pattern")
        back = load.load_layer(self.vault, "t:one")
        x = [rng.uniform(-1, 1) for _ in range(12)]
        self.assertEqual(layer.forward(x), back.forward(x))

    def test_a_full_precision_layer_is_not_stored_as_trits(self) -> None:
        """Run one's defect: the head went through the trit codec."""
        rng = random.Random(5)
        layer = TernaryLinear(12, 6, rng, quantized=False)
        payload = load.store_layer(self.vault, layer, "t:head", "pattern")
        self.assertEqual(payload.get("packing"), "floats")
        back = load.load_layer(self.vault, "t:head")
        self.assertEqual(layer.w, back.w,
                         "an exact layer must arrive exact")
        x = [rng.uniform(-1, 1) for _ in range(12)]
        self.assertEqual(layer.forward(x), back.forward(x))

    def test_a_bias_survives_bit_for_bit(self) -> None:
        rng = random.Random(7)
        layer = TernaryLinear(8, 4, rng, quantized=True)
        layer.b = [0.1234567890123, -2.5, 0.0, 1e-9]
        load.store_layer(self.vault, layer, "t:bias", "pattern")
        self.assertEqual(load.load_layer(self.vault, "t:bias").b, layer.b)

    def test_a_whole_network_survives(self) -> None:
        net = UltraQuantNet(9, [7], 4, seed=2)
        load.store_net(self.vault, net, prefix="n", category="pattern")
        back = load.load_net(self.vault, prefix="n")
        rng = random.Random(11)
        for _ in range(5):
            x = [rng.uniform(-1, 1) for _ in range(9)]
            self.assertEqual(net.forward(x), back.forward(x))

    def test_requantisation_is_idempotent_on_what_arrives(self) -> None:
        """The property criterion 1's exactness rests on."""
        rng = random.Random(13)
        layer = TernaryLinear(20, 10, rng, quantized=True)
        load.store_layer(self.vault, layer, "t:idem", "pattern")
        self.assertTrue(load.is_idempotent(
            load.load_layer(self.vault, "t:idem")))

    def test_an_imported_layer_carries_its_glyph(self) -> None:
        rng = random.Random(17)
        layer = TernaryLinear(30, 20, rng, quantized=True)
        payload = load.store_layer(self.vault, layer, "t:glyph", "pattern")
        drawn = payload["glyph"]
        self.assertEqual(len(drawn), glyph.GLYPH_SIDE)
        self.assertTrue(all(len(row) == glyph.GLYPH_SIDE for row in drawn))
        self.assertEqual(self.vault.get("t:glyph")["glyph"], drawn)


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS, and the defect run one found."""

    def setUp(self) -> None:
        self.doc = " ".join(glyphimport_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Inference is unchanged, exactly",
                       "Recognition is unchanged",
                       "Biases survive exactly",
                       "Every layer is addressable",
                       "The cost is named"):
            self.assertIn(phrase, self.doc)

    def test_run_one_is_kept_beside_the_pass(self) -> None:
        self.assertIn("0.900 -> 0.792", self.doc)
        self.assertIn("Trits are how you store a ternary matrix", self.doc)

    def test_the_cost_is_broken_out_rather_than_averaged(self) -> None:
        self.assertIn("6.208", self.doc)
        self.assertIn("86.500", self.doc)

    def test_the_gate_still_passes(self) -> None:
        report = glyphimport_gate.run_gate(epochs=8, seed=3)
        self.assertEqual(report.logit_differences, 0, report.reason)
        self.assertEqual(report.label_differences, 0, report.reason)
        self.assertEqual(report.bias_differences, 0, report.reason)
        self.assertEqual(report.unreachable, 0, report.reason)
        self.assertEqual(report.accuracy_before, report.accuracy_after,
                         report.reason)


if __name__ == "__main__":
    unittest.main()
