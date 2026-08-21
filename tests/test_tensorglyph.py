"""A glyph for every imported tensor, and the negative it measured.

§11.99 gave each packed tensor a 5x5 signature the existing
recognizer can read. It FAILED its own separation criterion, and the
library's 64-bit sketch beat it on the same measure - so these pins
hold the properties that passed, and the recorded verdict that the
glyph is not a routing metric.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.convert import glyph, library
from ultraquant.experiments import tensorglyph_gate


class RenderTests(unittest.TestCase):
    """The properties the gate confirmed."""

    def test_the_glyph_is_the_shape_the_recognizer_reads(self) -> None:
        rows = [[(-1 if (r + c) % 3 == 0 else 0) for c in range(40)]
                for r in range(40)]
        drawn = glyph.glyph_of(rows)
        self.assertEqual(len(drawn), 5)
        for row in drawn:
            self.assertEqual(len(row), 5)
            self.assertTrue(set(row) <= {"#", "."})

    def test_rendering_is_deterministic(self) -> None:
        rows = [[((r * 7 + c * 3) % 5) - 2 for c in range(31)]
                for r in range(29)]
        first = glyph.glyph_of(rows)
        self.assertEqual(first, glyph.glyph_of(rows))
        self.assertEqual(glyph.glyph_bits(first),
                         glyph.glyph_bits(glyph.glyph_of(rows)))

    def test_a_tensor_with_no_structure_renders_blank(self) -> None:
        """The honest glyph for nothing is nothing, not a tie-break."""
        rows = [[1 for _ in range(20)] for _ in range(20)]
        self.assertEqual(glyph.glyph_of(rows), ["....."] * 5)

    def test_an_empty_tensor_does_not_raise(self) -> None:
        self.assertEqual(glyph.glyph_of([]), ["....."] * 5)
        self.assertEqual(len(glyph.densities_of([])), 5)

    def test_structure_shows_up_where_it_is(self) -> None:
        """A dense top-left corner sets the top-left cells."""
        rows = [[(1 if (r < 10 and c < 10) else 0) for c in range(50)]
                for r in range(50)]
        drawn = glyph.glyph_of(rows)
        self.assertEqual(drawn[0][0], "#")
        self.assertEqual(drawn[4][4], ".")

    def test_the_densities_are_what_both_spellings_summarise(self) -> None:
        """The fairness fix: one summary, two spellings."""
        rows = [[((r + c) % 4) - 1 for c in range(25)] for r in range(25)]
        grid = glyph.densities_of(rows)
        flat = [v for line in grid for v in line]
        mean = sum(flat) / len(flat)
        expected = ["".join("#" if v > mean else "." for v in line)
                    for line in grid]
        self.assertEqual(glyph.glyph_of(rows), expected)


class CatalogTests(unittest.TestCase):
    """The glyph is an address, and the address works."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_tensorglyph_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_shape_label_needs_a_recognizer_and_says_so(self) -> None:
        self.assertEqual(library.shape_label(["....."] * 5, None), "")

    def test_kinds_ignore_the_layer_number(self) -> None:
        self.assertEqual(tensorglyph_gate.kind_of("blk.0.attn_q.weight"),
                         tensorglyph_gate.kind_of("blk.17.attn_q.weight"))
        self.assertNotEqual(tensorglyph_gate.kind_of("blk.0.attn_q.weight"),
                            tensorglyph_gate.kind_of("blk.0.attn_k.weight"))


class GateVerdictTests(unittest.TestCase):
    """A failure recorded as a failure, with its cause."""

    def setUp(self) -> None:
        self.doc = " ".join(tensorglyph_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Deterministic",
                       "The glyph carries tensor identity",
                       "The standard option is measured beside it",
                       "The name retrieves the shard",
                       "The label is not noise"):
            self.assertIn(phrase, self.doc)

    def test_the_failure_is_recorded_as_a_failure(self) -> None:
        self.assertIn("FAILED criterion 2", self.doc)
        self.assertIn("Neither signature separates tensor kind", self.doc)

    def test_both_superseded_numbers_are_kept_beside_the_correction(self):
        """Two artifacts, both left visible rather than corrected away."""
        # Pass one: the sketch was handed truncated input.
        self.assertIn("rigged the comparison", self.doc)
        self.assertIn("+0.117", self.doc)
        # Pass two: kind was confounded with dimensionality.
        self.assertIn("confounded by dimensionality", self.doc)
        self.assertIn("+0.179 at sd 0.211", self.doc)

    def test_the_one_d_pattern_is_named_across_units(self) -> None:
        """Third time in this module; the lesson is written down."""
        self.assertIn("third time 1-D tensors have contaminated", self.doc)

    def test_the_conclusion_is_written_where_the_code_is(self) -> None:
        """So nobody builds glyph-distance routing later."""
        note = " ".join(glyph.__doc__.split())
        self.assertIn("What this is NOT for", note)
        self.assertIn("nothing routes by glyph distance", note)


if __name__ == "__main__":
    unittest.main()
