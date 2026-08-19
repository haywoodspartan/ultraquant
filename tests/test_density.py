"""Density: the crowded-library measurement and the two rules it forced.

The first token-colliding worlds measured 0.500 fabrication at only 100
facts, and the fixes are architecture, not patches: bridged evidence
must trace to a single origin (no pooling puns out of unrelated
subjects), and an origin key may have at most one token the question
never said (no answering for a more specific subject than asked). These
tests pin both rules at the mechanism level; the gate's verdict pins
the sweep.
"""

from __future__ import annotations

import random
import unittest

from ultraquant.experiments.density_gate import _assertive, _build_world
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.reason.inference import infer


def _memory(facts) -> SystematicMemory:
    memory = SystematicMemory()
    for key, value in facts:
        memory.remember_fact(key, value, confidence=0.6)
    return memory


class SubjectConsistencyTests(unittest.TestCase):
    """The single-origin rule: no pun assembled from fragments."""

    def test_pooled_activation_across_subjects_is_refused(self) -> None:
        """'old tower' and 'north keep' are both steel; their fragments
        must not jointly answer for a 'north tower' that has no chain."""
        memory = _memory([
            ("old tower material", "steel"),
            ("north keep material", "steel"),
            ("steel hardness", "4211 units"),
            ("north tower height", "88 meters"),
        ])
        self.assertIsNone(infer("what is the north tower hardness?",
                                memory))

    def test_a_single_origin_chain_still_converges(self) -> None:
        memory = _memory([
            ("north tower material", "steel"),
            ("steel hardness", "4211 units"),
        ])
        result = infer("what is the north tower hardness?", memory)
        self.assertIsNotNone(result)
        self.assertIn("4211", result.answer)
        self.assertEqual(result.premises[0][0], "north tower material")


class OriginCoverageTests(unittest.TestCase):
    """At most one origin token the question never said."""

    def test_a_more_specific_subject_does_not_answer(self) -> None:
        """The modifier drop must not let 'low mill' answer for a 'royal
        mill' the library does not hold - the density fabrication."""
        memory = _memory([
            ("low mill material", "iron"),
            ("iron conductivity", "7871 units"),
        ])
        self.assertIsNone(infer("what is the royal mill conductivity?",
                                memory))

    def test_the_relation_slot_is_still_free(self) -> None:
        """'tower material' has one token the question never says - the
        relation word - and that single absence must stay allowed or
        every chain in the book dies."""
        memory = _memory([
            ("tower material", "steel"),
            ("steel conductivity", "9155 units"),
        ])
        result = infer("what is the tower conductivity?", memory)
        self.assertIsNotNone(result)
        self.assertIn("9155", result.answer)

    def test_curiosity_respects_origin_coverage(self) -> None:
        """No asking for the continuation of somebody else's property."""
        from ultraquant.reason.inference import missing_premise

        memory = _memory([("low mill material", "iron")])
        gap = missing_premise("what is the royal mill conductivity?",
                              memory)
        self.assertIsNone(gap)


class WorldGeneratorTests(unittest.TestCase):
    """The colliding worlds must actually collide."""

    def test_entities_share_tokens(self) -> None:
        rng = random.Random(0)
        facts, chains, decoys, directs = _build_world(rng, 1000)
        sharers: dict[str, set] = {}
        for key, _value in facts:
            words = key.removeprefix("the ").split()
            if len(words) < 3:
                continue
            for word in words[:-1]:
                sharers.setdefault(word, set()).add(" ".join(words[:-1]))
        max_sharers = max((len(entities) for entities in sharers.values()),
                          default=0)
        self.assertGreater(max_sharers, 10,
                           "a density world without token collisions "
                           "measures nothing")

    def test_the_assertive_predicate_spares_refusals(self) -> None:
        self.assertFalse(_assertive(
            "I don't hold that exactly. Nearest I hold: x is y "
            "(confidence 0.60)."))
        self.assertTrue(_assertive(
            "the tower hardness is 4211 units, via steel - inferred, "
            "not stored: ... (confidence 0.60)"))


if __name__ == "__main__":
    unittest.main()
