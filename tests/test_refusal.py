"""An encoder that can refuse its own output.

§11.103 gave the tower a way to notice it had been destroyed without
the original to compare against. These pins hold the measure, and
the band-fitting flaw run one found: a band fitted on two runs is a
band the width of the gap between them.
"""

from __future__ import annotations

import math
import unittest

from ultraquant.infer import confidence


class AttentionEntropyTests(unittest.TestCase):
    """Normalised so the number means the same at any length."""

    def test_uniform_attention_is_the_maximum(self) -> None:
        self.assertAlmostEqual(
            confidence.attention_entropy([[0.25] * 4] * 3), 1.0, places=12)

    def test_decided_attention_is_zero(self) -> None:
        self.assertAlmostEqual(
            confidence.attention_entropy([[1.0, 0.0, 0.0, 0.0]] * 3),
            0.0, places=12)

    def test_the_scale_is_the_same_at_any_sequence_length(self) -> None:
        """Uniform is 1.0 whether there are four positions or forty."""
        for length in (2, 4, 16, 40):
            self.assertAlmostEqual(
                confidence.attention_entropy([[1.0 / length] * length]),
                1.0, places=12)

    def test_a_fully_masked_row_is_skipped_not_counted_as_certain(self):
        """A row with no attention has no certainty to report."""
        rows = [[0.0, 0.0, 0.0, 0.0], [0.25] * 4]
        self.assertAlmostEqual(confidence.attention_entropy(rows),
                               1.0, places=12)


class BandTests(unittest.TestCase):
    """Run one's flaw, and the refusal that replaced it."""

    def _traces(self, values: list) -> list:
        return [confidence.TowerTrace(attention=[v], rms=[70.0 + i])
                for i, v in enumerate(values)]

    def test_two_runs_are_refused_as_too_few(self) -> None:
        """A band fitted on two runs is the gap between them."""
        with self.assertRaises(ValueError):
            confidence.fit_band(self._traces([0.606, 0.623]))

    def test_a_band_is_wider_than_the_runs_that_fitted_it(self) -> None:
        traces = self._traces([0.620, 0.606, 0.615, 0.611])
        band = confidence.fit_band(traces)
        low, high = band.attention
        self.assertLess(low, 0.606)
        self.assertGreater(high, 0.620)

    def test_identical_runs_still_give_a_usable_band(self) -> None:
        """Zero observed spread must not mean zero width."""
        band = confidence.fit_band(self._traces([0.6, 0.6, 0.6, 0.6]))
        low, high = band.attention
        self.assertGreater(high - low, 0.0)

    def test_the_measured_healthy_and_broken_values_are_separated(self):
        """The numbers §11.103 actually measured on the real tower."""
        band = confidence.fit_band(
            self._traces([0.620, 0.606, 0.615, 0.611]))
        healthy, _why = confidence.judge(
            confidence.TowerTrace(attention=[0.6233], rms=[76.27]), band)
        broken, reasons = confidence.judge(
            confidence.TowerTrace(attention=[0.9790], rms=[53.67]), band)
        self.assertTrue(healthy, "the held-out healthy tower was refused")
        self.assertFalse(broken, "the destroyed tower was accepted")
        self.assertTrue(reasons, "refused without saying why")

    def test_a_refusal_says_why(self) -> None:
        band = confidence.fit_band(self._traces([0.6, 0.61, 0.605, 0.615]))
        _ok, reasons = confidence.judge(
            confidence.TowerTrace(attention=[0.98], rms=[70.0]), band)
        self.assertTrue(any("attention entropy" in r for r in reasons))

    def test_a_non_finite_activation_is_caught(self) -> None:
        band = confidence.fit_band(self._traces([0.6, 0.61, 0.605, 0.615]))
        ok, reasons = confidence.judge(
            confidence.TowerTrace(attention=[float("nan")], rms=[70.0]),
            band)
        self.assertFalse(ok)
        self.assertTrue(any("non-finite" in r for r in reasons))


class NoReferenceTests(unittest.TestCase):
    """The detector must not need the answer it is detecting."""

    def test_judging_takes_only_a_trace_and_a_band(self) -> None:
        import inspect

        names = list(inspect.signature(confidence.judge).parameters)
        self.assertEqual(names, ["trace", "band"])

    def test_a_trace_holds_no_reference(self) -> None:
        """Nothing in a trace could be a comparison to the original."""
        trace = confidence.TowerTrace(attention=[0.5], rms=[1.0])
        self.assertEqual(sorted(vars(trace)), ["attention", "rms"])


class GateVerdictTests(unittest.TestCase):
    """The PASS, the run-one flaw, and which measure actually fired."""

    def setUp(self) -> None:
        from ultraquant.experiments import refusal_gate
        self.doc = " ".join(refusal_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The refusal never consults the reference",
                       "The band accepts healthy and refuses destroyed",
                       "It beats the standard option",
                       "Each measure is judged alone",
                       "The profile is reported"):
            self.assertIn(phrase, self.doc)

    def test_run_ones_flaw_is_kept(self) -> None:
        self.assertIn("A band fitted on two runs is a band the width of "
                      "the gap between them", self.doc)

    def test_which_measure_fired_is_recorded(self) -> None:
        """Criterion 4 existed so this could not be hidden."""
        self.assertIn("Attention entropy does all the work", self.doc)
        self.assertIn("peak residual RMS contributes nothing", self.doc)

    def test_the_standard_option_is_recorded_as_missing_it(self) -> None:
        self.assertIn("notices nothing", self.doc)
        self.assertIn("0.0185", self.doc)

    def test_the_finding_is_stated_in_plain_words(self) -> None:
        self.assertIn("The converted tower never becomes selective",
                      self.doc)


if __name__ == "__main__":
    unittest.main()
