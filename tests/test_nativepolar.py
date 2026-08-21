"""The yes/no family says the same sentence in both tiers.

§11.97 ported the polar branch - claims, choices, comparisons and
derived subjects. These pins are the ones a future edit would trip
over: the shape of each sentence, the line that absence is never No,
and the scope statement staying true as items leave the unported
list.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from ultraquant.experiments import nativepolar_gate
from ultraquant.experiments.nativepolar_gate import probe_path

_ROOT = Path(__file__).resolve().parents[1]


def _say(lines: list[str]) -> list[str]:
    """Walk the native pipeline through a conversation."""
    got = subprocess.run([str(probe_path())],
                         input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return [line.partition("|")[2] for line in got.stdout.splitlines()]


@unittest.skipUnless(probe_path().exists(), "native tier not built")
class NativePolarTests(unittest.TestCase):
    """One conversation, every form in the family."""

    def test_a_matching_claim_is_yes_and_a_mismatched_one_is_no(self):
        said = _say(["the tower material is steel",
                     "is the tower material steel?",
                     "is the tower material iron?"])
        self.assertEqual(said[1],
                         "Yes - tower material is steel (confidence 0.60).")
        self.assertEqual(said[2], "No - tower material is steel, not iron "
                                  "(confidence 0.60).")

    def test_polarity_flips_the_verdict_on_both_sides(self):
        said = _say(["the tower material is steel",
                     "the mill material is not oak",
                     "is the tower material not iron?",
                     "is the mill material oak?",
                     "is the mill material iron?"])
        self.assertEqual(said[2], "Yes - tower material is steel, not iron "
                                  "(confidence 0.60).")
        self.assertEqual(said[3], "No - believed not oak (confidence 0.60).")
        # A denial of one value says nothing about another.
        self.assertEqual(said[4], "I don't know - I hold only that mill "
                                  "material is not oak (confidence 0.60).")

    def test_absence_is_never_no(self):
        """The line the whole family stands on."""
        said = _say(["the tower material is steel",
                     "is the obelisk steel?",
                     "is the obelisk steel or iron?"])
        for answer in said[1:]:
            first = answer.split(" ", 1)[0].strip(",-")
            self.assertNotIn(first, ("Yes", "No", "Neither"),
                             f"a verdict about an unheld subject: {answer}")

    def test_a_choice_elects_the_held_value_or_neither(self):
        said = _say(["the tower material is steel",
                     "the mill material is not oak",
                     "is the tower material steel or iron?",
                     "is the tower material oak or bronze?",
                     "is the mill material oak or iron?"])
        self.assertEqual(said[2],
                         "Steel - tower material is steel (confidence 0.60).")
        self.assertEqual(said[3], "Neither - tower material is steel "
                                  "(confidence 0.60).")
        # Ruling out is not picking: iron stays unelected.
        self.assertEqual(said[4], "Not oak, at least - I hold only that mill "
                                  "material is not oak (confidence 0.60).")

    def test_a_comparison_converts_units_and_names_both_values(self):
        said = _say(["the tower height is 300 meters",
                     "the bridge length is 2 kilometers",
                     "is the tower height taller than the bridge length?",
                     "is the bridge length taller than the tower height?"])
        self.assertEqual(said[2], "No - tower height is 300 meters, bridge "
                                  "length is 2 kilometers (units converted) "
                                  "(confidence 0.60).")
        self.assertTrue(said[3].startswith("Yes - bridge length"))

    def test_a_written_quantity_is_its_own_name(self):
        said = _say(["the tower height is 300 meters",
                     "is the tower height greater than 200 meters?"])
        self.assertEqual(said[1], "Yes - tower height is 300 meters, "
                                  "200 meters (confidence 0.60).")

    def test_an_expression_names_itself_and_its_value(self):
        said = _say(["is 3 * 4 greater than 10?",
                     "is 2 + 2 = 4?",
                     "is 2 + 2 = 5?"])
        self.assertEqual(said[0], "Yes - 3 * 4 is 12, 10 (confidence 1.00).")
        self.assertEqual(said[1], "Yes - 2 + 2 is 4 (confidence 1.00).")
        self.assertEqual(said[2], "No - 2 + 2 is 4, not 5 (confidence 1.00).")

    def test_comparisons_refuse_aloud_rather_than_falling_through(self):
        said = _say(["the tower height is 300 meters",
                     "the tower material is steel",
                     "is the obelisk height taller than the tower height?",
                     "is the tower material taller than the tower height?"])
        self.assertEqual(said[2], "I can't compare those: I hold nothing "
                                  "for 'obelisk height'.")
        self.assertEqual(said[3], "I can't compare those: 'tower material' "
                                  "holds no number (steel).")

    def test_a_derived_subject_is_marked_derived_with_its_trail(self):
        said = _say(["the spire material is steel",
                     "steel hardness is high",
                     "is the spire hardness high?"])
        self.assertEqual(said[2], "Yes - the spire hardness is high, via "
                                  "steel (derived, confidence 0.60).")

    def test_the_unported_gap_refuses_rather_than_guessing(self):
        """The named gap says it holds nothing, never a verdict."""
        said = _say(["the tower height is 300 meters",
                     "is the tower height * 2 greater than 500 meters?"])
        self.assertEqual(said[1], "I can't compare those: I hold nothing "
                                  "for 'tower height * 2'.")


@unittest.skipUnless(probe_path().exists(), "native tier not built")
class GeneratedSessionTests(unittest.TestCase):
    """A short walk of the gate itself, on a seed it has not seen."""

    def test_a_generated_session_agrees(self) -> None:
        report = nativepolar_gate.run_gate(turns=30, conversations=3,
                                           seed=7)
        self.assertEqual(report.real_differences, 0, report.reason)
        self.assertEqual(report.intent_differences, 0, report.reason)
        self.assertEqual(report.store_differences, 0, report.reason)

    def test_no_verdict_is_handed_out_about_an_unheld_subject(self) -> None:
        report = nativepolar_gate.run_gate(turns=30, conversations=3,
                                           seed=7)
        self.assertGreater(report.absence_checks, 0,
                           "the corpus never asked about an unheld subject")
        self.assertEqual(report.fabricated_verdicts, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS, and the gap it names."""

    def setUp(self) -> None:
        self.doc = " ".join(nativepolar_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("No unexplained difference",
                       "Absence is never No, in both tiers",
                       "The gap is bounded and reported",
                       "The store agrees at the end",
                       "Every form is provoked"):
            self.assertIn(phrase, self.doc)

    def test_the_gap_is_named_and_counted_rather_than_hidden(self) -> None:
        self.assertIn("74 (of 89 asked)", self.doc)
        self.assertIn("| unexplained differences | **0** |", self.doc)
        self.assertIn("**0** over 109 unheld subjects", self.doc)

    def test_the_gate_records_that_it_can_fail(self) -> None:
        """A gate that cannot fail has measured nothing."""
        self.assertIn("80 of 180 turns differ", self.doc)


class ScopeStatementTests(unittest.TestCase):
    """A scope claim that has gone stale is worse than none."""

    def test_the_forms_that_landed_left_the_unported_list(self):
        text = (_ROOT / "native" / "uq" / "include" / "uq"
                / "interpreter.hpp").read_text(encoding="utf-8")
        head = text.split("#ifndef", 1)[0]
        # Only the NOT-ported sentence, because the same words appear
        # in the ported list directly above it and a search over the
        # whole header would fail on its own success.
        self.assertIn("NOT ported here", head)
        unported = head.split("NOT ported here", 1)[1]
        for landed in ("polar", "choice", "comparison", "superlative",
                       "aggregate", "chain"):
            self.assertNotIn(landed, unported,
                             f"{landed!r} is ported; the header still "
                             "lists it as out of scope")

    def test_the_named_gap_is_written_down_where_the_code_is(self):
        text = (_ROOT / "native" / "uq" / "include" / "uq"
                / "polar.hpp").read_text(encoding="utf-8")
        self.assertIn("absence is never No", text)
        self.assertIn("expression over BELIEFS", text)


if __name__ == "__main__":
    unittest.main()
