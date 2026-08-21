"""Two stores that remember the same way - §11.90, pinned.

The store's observable surface is wider than "what does this key
recall to": it includes what storing DID, what the replaced belief
was, what a revision took down, and which fact retrieval picks. Each
of those is pinned, because each of them was wrong once.
"""

from __future__ import annotations

import subprocess
import unittest

from ultraquant.experiments import nativememory_gate
from ultraquant.memory.systematic import SystematicMemory

_PROBE = nativememory_gate.probe_path()
_BUILT = _PROBE.exists()


def _native(lines: list[str]) -> list[str]:
    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


@unittest.skipUnless(_BUILT, "native store not built")
class ParityTests(unittest.TestCase):
    """The five things that were wrong before they were right."""

    def test_new_reinforced_and_revised_are_distinguished(self) -> None:
        answers = _native([
            "remember 0.6 0 tower height|300 meters",
            "remember 0.6 0 tower height|300 meters",
            "remember 0.6 0 tower height|450 meters",
        ])
        self.assertEqual(answers[0], "new||")
        self.assertEqual(answers[1], "reinforced||")
        self.assertEqual(answers[2], "revised|300 meters|")

    def test_a_polarity_flip_is_a_change_of_mind(self) -> None:
        """Not a reinforcement: polarity is part of a fact's
        identity, and the store says so."""
        answers = _native([
            "remember 0.6 0 dome material|steel",
            "remember 0.6 1 dome material|steel",
            "remember 0.6 0 dome material|steel",
        ])
        self.assertEqual(answers[1], "revised|steel|")
        self.assertEqual(answers[2], "revised|not steel|")

    def test_confirmation_sets_and_counts(self) -> None:
        answers = _native([
            "remember 0.6 0 tower height|300 meters",
            "confirm 0.9 tower height",
            "recall tower height",
            "confirm 0.4 tower height",
            "recall tower height",
        ])
        self.assertEqual(answers[2], "fact|300 meters|0.90|0|1")
        self.assertEqual(answers[4], "fact|300 meters|0.40|0|2")

    def test_a_revised_premise_takes_its_conclusions_down(self) -> None:
        answers = _native([
            "remember 0.6 0 tower height|300 meters",
            "consolidate 0.5 tower area|big|tower height",
            "remember 0.6 0 tower height|450 meters",
            "recall tower area",
        ])
        self.assertEqual(answers[2], "revised|300 meters|tower area")
        self.assertEqual(answers[3], "absent")

    def test_retrieval_does_not_fold_plurals(self) -> None:
        """The fold belongs to the router above the store; folding
        here would find facts the Python store does not."""
        lines = ["remember 0.6 0 tower height|300 meters",
                 "find 5 what are towers?"]
        memory = SystematicMemory()
        memory.remember_fact("tower height", "300 meters", 0.6)
        self.assertEqual(_native(lines)[1],
                         "found|" + ";".join(
                             memory.find_facts("what are towers?", 5)))

    def test_a_generated_session_agrees(self) -> None:
        report = nativememory_gate.run_gate(steps=120, sessions=3,
                                            seed=7)
        self.assertEqual(report.mismatches, 0, report.reason)


class GateVerdictTests(unittest.TestCase):
    """The recorded PASS and the shape of what it caught."""

    def setUp(self) -> None:
        self.doc = " ".join(nativememory_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| records that differed | **0** |", self.doc)

    def test_the_failures_are_characterised_honestly(self) -> None:
        self.assertIn("every failure was the native tier being "
                      "reasonable instead of being faithful", self.doc)
        self.assertIn("None of the five was a wrong VALUE", self.doc)

    def test_the_indirect_symptom_is_recorded(self) -> None:
        """The confirm bug surfaced two steps later, as a differently
        ordered retrieval - which is why the gate scripts sessions
        rather than checking calls one at a time."""
        self.assertIn("not as a wrong confidence, but as a differently "
                      "ordered `find_facts`", self.doc)


if __name__ == "__main__":
    unittest.main()
