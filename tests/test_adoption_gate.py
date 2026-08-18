"""The adoption gate: §11.22's pass stops at the shape it was measured on.

Hidden 64 lifted the 8-way monolith, so the tempting next line was one changed
constant in the forge. The gate re-asked at the deployed shape — four family
experts, two classes each — and measured the candidate width *worse* there,
at nearly double the forged bytes. These tests pin the machinery and the
verdict so the deployed default cannot drift on the monolith's evidence.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments.adoption_gate import (_FAMILY_OF, _WIDTHS,
                                                  AdoptionReport, run_gate)
from ultraquant.forge.corpus import BUILTIN_FAMILIES
from ultraquant.pattern.recognition import PATTERNS


class GateMachineryTests(unittest.TestCase):
    """The parts that must hold for the numbers to mean anything."""

    def test_no_variants_file_is_a_skip(self):
        report = run_gate("Z:/nowhere/variants.json")
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_every_pattern_label_belongs_to_exactly_one_family(self):
        """A variant that cannot find its family expert would silently
        vanish from the evaluation."""
        self.assertEqual(sorted(_FAMILY_OF), sorted(PATTERNS))
        for label, family in _FAMILY_OF.items():
            self.assertIn(label, BUILTIN_FAMILIES[family])

    def test_the_arms_cross_the_deployed_width_with_the_candidate(self):
        self.assertEqual(_WIDTHS, (32, 64))

    def test_the_report_text_carries_the_byte_cost(self):
        report = AdoptionReport(
            held={(32, False): 0.7, (32, True): 0.7,
                  (64, False): 0.7, (64, True): 0.7},
            canon={(32, False): 1.0, (32, True): 1.0,
                   (64, False): 1.0, (64, True): 1.0},
            store_bytes={32: 100, 64: 200},
            reason="test")
        text = report.as_text()
        self.assertIn("hidden 32 = 100", text)
        self.assertIn("hidden 64 = 200", text)
        self.assertIn("+100", text)


class GateVerdictTests(unittest.TestCase):
    """The FAIL, its cost, and its scope, pinned."""

    def test_the_recorded_verdict_is_a_fail_at_the_deployed_shape(self):
        """If "FAILED" or the worse-at-family-shape sentence leaves the
        docstring, the story changed and the numbers need re-measuring."""
        from ultraquant.experiments import adoption_gate

        doc = adoption_gate.__doc__
        self.assertIn("FAILED", doc)
        self.assertIn("worse at the deployed shape", doc)
        self.assertIn("survives measured", doc)

    def test_the_byte_cost_of_the_refused_adoption_is_recorded(self):
        from ultraquant.experiments import adoption_gate

        doc = adoption_gate.__doc__
        self.assertIn("82,820", doc)
        self.assertIn("42,595", doc)

    def test_the_ceilinged_control_is_acknowledged_not_hidden(self):
        """A control at 1.000 is §11.11's defect class unless it is shown
        able to fail; the docstring must carry that reasoning."""
        from ultraquant.experiments import adoption_gate

        doc = adoption_gate.__doc__
        self.assertIn("ceiling", doc)
        self.assertIn("11.11", doc)

    def test_the_scope_of_11_22_is_stated(self):
        """The monolith pass must not be citable for the family experts."""
        from ultraquant.experiments import adoption_gate

        doc = " ".join(adoption_gate.__doc__.split())
        self.assertIn("stops at the shape it was measured on", doc)


class DeployedDefaultTests(unittest.TestCase):
    """The default the gate protected, pinned where it lives."""

    def test_the_forge_default_width_is_still_32(self):
        """`build.py --hidden` defaults to 32; this gate measured that
        surviving. Changing it needs a new gate, not a constant edit."""
        import argparse

        from ultraquant.forge import build

        parser = argparse.ArgumentParser()
        original = build.argparse.ArgumentParser
        try:
            build.argparse.ArgumentParser = lambda **kw: parser
            build.main(["--help"])
        except SystemExit:
            pass
        finally:
            build.argparse.ArgumentParser = original
        default = next(a.default for a in parser._actions
                       if "--hidden" in a.option_strings)
        self.assertEqual(default, 32)


if __name__ == "__main__":
    unittest.main()
