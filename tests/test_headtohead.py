"""UltraQuant against a transformer, and the claim that lost.

§11.105 ran the comparison ARCHITECTURE.md §12.3 said had never been
run. It PASSED its criteria and refuted the claim it was built to
establish: a 27B transformer fabricated nothing on unheld subjects,
with or without permission to decline. These pins hold the design
that made the comparison hard to rig, the three-outcome classifier
that a real bug forced, and the recorded correction.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import headtohead_gate as gate


class ClassifierTests(unittest.TestCase):
    """The instrument, which decides the headline number."""

    def test_every_hand_labelled_case_is_classified_correctly(self):
        wrong = [(text, want, gate.classify(text))
                 for text, want in gate._CLASSIFY_CASES
                 if gate.classify(text) != want]
        wrong += [(text, want, gate.is_refusal(text))
                  for text, want in gate._DETECTOR_CASES
                  if gate.is_refusal(text) != want]
        self.assertEqual(wrong, [])

    def test_an_empty_answer_is_not_a_fabrication(self) -> None:
        """Run one's bug, and it pointed the flattering way."""
        self.assertEqual(gate.classify(""), "empty")
        self.assertEqual(gate.classify("   \n  "), "empty")
        self.assertNotEqual(gate.classify(""), "asserted")

    def test_the_detector_is_generous_to_free_text(self) -> None:
        """Under-counting refusals would inflate the opponent's rate."""
        for phrasing in ("I don't know.",
                         "That is not specified in the facts.",
                         "There is no information on that.",
                         "Unknown.",
                         "I cannot determine that.",
                         "The facts do not mention it, so N/A."):
            self.assertTrue(gate.is_refusal(phrasing), phrasing)

    def test_a_plain_assertion_is_not_read_as_a_refusal(self) -> None:
        for phrasing in ("The obelisk is 50 meters tall.",
                         "Bronze.",
                         "It is made of stone."):
            self.assertFalse(gate.is_refusal(phrasing), phrasing)


class FairnessTests(unittest.TestCase):
    """The guards that stop the comparison being rigged."""

    def test_the_battery_contains_a_family_the_opponent_should_win(self):
        families = {family for family, _q, _ok in gate.BATTERY}
        self.assertIn("world", families,
                      "a battery where the opponent never wins is not "
                      "a battery")

    def test_unheld_subjects_appear_in_no_fact(self) -> None:
        """Otherwise the fabrication family is not measuring absence."""
        facts = " ".join(gate.FACTS).lower()
        for family, question, _ok in gate.BATTERY:
            if family != "unheld":
                continue
            subject = question.lower().replace("what is the", "").split()[0]
            self.assertNotIn(subject, facts,
                             f"{subject!r} is in the facts, so asking "
                             "about it does not test absence")

    def test_held_subjects_do_appear(self) -> None:
        facts = " ".join(gate.FACTS).lower()
        held = [q for f, q, _ok in gate.BATTERY if f == "held"]
        self.assertTrue(held)
        for question in held:
            subject = question.lower().replace("what is the", "").split()[0]
            self.assertIn(subject, facts)

    def test_both_systems_are_given_the_same_facts(self) -> None:
        """Criterion 1, checked structurally rather than promised."""
        prompt = gate._prompt("what is the tower height?")
        for fact in gate.FACTS:
            self.assertIn(fact, prompt,
                          "a fact this system was told is missing from "
                          "the transformer's prompt")

    def test_the_transformer_is_told_it_may_decline(self) -> None:
        """Otherwise the fabrication rate measures the prompt."""
        self.assertIn("do not know", gate._SYSTEM)


class GateVerdictTests(unittest.TestCase):
    """The pass, and the claim it cost."""

    def setUp(self) -> None:
        self.doc = " ".join(gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("Both see the same facts",
                       "Fabrication on unheld subjects",
                       "The transformer must win somewhere",
                       "The refusal detector is validated",
                       "Held-fact accuracy is comparable"):
            self.assertIn(phrase, self.doc)

    def test_the_result_that_refutes_the_claim_is_recorded(self) -> None:
        self.assertIn("The transformer fabricated nothing", self.doc)
        self.assertIn("did not survive it", self.doc)

    def test_the_correction_names_the_conflation(self) -> None:
        self.assertIn("conflating two levels", self.doc)
        self.assertIn("survives about classifier heads", self.doc)

    def test_the_flattering_bug_is_recorded(self) -> None:
        self.assertIn("in the direction that would have favoured this "
                      "system", self.doc)

    def test_the_limits_are_stated(self) -> None:
        self.assertIn("One model, one setting, twenty-four questions",
                      self.doc)
        self.assertIn("did not discriminate", self.doc)


if __name__ == "__main__":
    unittest.main()
