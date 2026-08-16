"""The router learning to say "not mine".

Measured on the deployed library before any of this: of five queries belonging
to none of its categories, it declined **one**. The other four were claimed
without matching a single base keyword.

Three separate stores had accumulated the same defect, and each had to be fixed
on its own; the tests below are organised by which one.
"""

from __future__ import annotations

import tempfile
import unittest

from ultraquant.shards.router import (
    CategoryRouter,
    _informative,
    prune_uninformative,
)
from ultraquant.shards.vault import ShardVault

_DECOYS = ["how do I fix a leaking tap", "what time does the train leave",
           "recipe for banana bread", "who won the football match last night"]


def _router(**categories) -> CategoryRouter:
    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_abs_")))
    for name, keywords in categories.items():
        router.register(name, list(keywords))
    return router


class InformativeTests(unittest.TestCase):
    """One rule, applied at every point a token can enter the router."""

    def test_stopwords_and_short_tokens_are_uninformative(self):
        for token in ("the", "a", "is", "of", "what", "who", "an", "at"):
            with self.subTest(token):
                self.assertFalse(_informative(token))

    def test_content_words_are_informative(self):
        for token in ("diamond", "arrow", "arithmetic", "outline"):
            with self.subTest(token):
                self.assertTrue(_informative(token))


class LearningTests(unittest.TestCase):
    """The first store: the router's own learned weights."""

    def test_learning_does_not_weight_stopwords(self):
        """21% of the deployed router's learned weight sat on stopwords."""
        router = _router(frames=["frame", "box"])
        router.learn("what is the shape of a box", "frames")
        self.assertNotIn("the", router._learned["frames"])
        self.assertNotIn("what", router._learned["frames"])
        self.assertIn("shape", router._learned["frames"])

    def test_a_query_of_pure_stopwords_teaches_nothing(self):
        router = _router(frames=["frame", "box"])
        router.learn("what is the of a", "frames")
        self.assertEqual(router._learned["frames"], {})

    def test_the_old_behaviour_is_still_reachable_for_measurement(self):
        """The gate needs the before-arm; nothing else should use it."""
        router = _router(frames=["frame", "box"])
        router.learn("what is the shape", "frames", filter_tokens=False)
        self.assertIn("the", router._learned["frames"])

    def test_pruning_repairs_an_already_trained_table(self):
        pruned, removed = prune_uninformative(
            {"frames": {"the": 0.5, "diamond": 0.3, "is": 1.05}})
        self.assertEqual(pruned["frames"], {"diamond": 0.3})
        self.assertAlmostEqual(removed, 1.55)


class VaultTests(unittest.TestCase):
    """The second store, and the one that made the first fix look insufficient.

    After the router's own table was cleaned, decoys still routed on **vault
    association alone** - scoring 0.7 to 1.2 with no base hit and no learned
    weight - because learn() reinforces the vault with the same tokens.
    """

    def test_reinforcement_no_longer_carries_stopwords(self):
        router = _router(frames=["frame", "box"])
        router.vault.add_shard("s1", "frames", {"w": [1.0]})
        router.learn("what is the shape of a box", "frames")
        assoc = router.vault.catalog()[0]["associations"]
        self.assertNotIn("the", assoc)
        self.assertIn("shape", assoc)

    def test_pruning_repairs_an_already_reinforced_vault(self):
        vault = ShardVault(tempfile.mkdtemp(prefix="uq_abs_"))
        vault.add_shard("s1", "frames", {"w": [1.0]})
        vault.reinforce("s1", ["the", "is", "diamond"], delta=0.5)
        removed = vault.prune_associations()
        assoc = vault.catalog()[0]["associations"]
        self.assertEqual(set(assoc), {"diamond"})
        self.assertAlmostEqual(removed, 1.0)


class RegistrationTests(unittest.TestCase):
    """The third store: base keywords chosen at deployment.

    'world' shipped with 'what', 'who' and 'where' registered. With both other
    stores cleaned, those three alone still claimed "what time does the train
    leave" and "who won the football match".
    """

    def test_question_words_are_refused_at_registration(self):
        router = _router(world=["country", "history", "what", "who", "where"])
        self.assertEqual(router._base["world"], {"country", "history"})

    def test_refused_keywords_are_recorded_not_dropped_silently(self):
        router = _router(world=["country", "what", "who"])
        self.assertEqual(router.rejected_keywords["world"], {"what", "who"})

    def test_a_reloaded_router_is_filtered_too(self):
        """load() rebuilds _base directly, bypassing register()."""
        import json
        import tempfile as tf
        from pathlib import Path

        path = Path(tf.mkdtemp()) / "router.json"
        path.write_text(json.dumps({
            "base": {"world": ["country", "what", "who"]},
            "learned": {"world": {"the": 0.5, "capital": 0.2}},
        }), encoding="utf-8")
        router = CategoryRouter(ShardVault(tf.mkdtemp()), path=path)
        self.assertEqual(router._base["world"], {"country"})
        self.assertEqual(router._learned["world"], {"capital": 0.2})


class BehaviourTests(unittest.TestCase):
    """What all three fixes buy, and what the control costs."""

    def _trained(self, filter_tokens: bool) -> CategoryRouter:
        router = _router(frames=["frame", "box", "diamond", "outline"],
                         lines=["line", "stripe", "horizontal"],
                         arithmetic=["sum", "add", "numbers"])
        for text, category in (("what is the shape of a box", "frames"),
                               ("show me a line that is horizontal", "lines"),
                               ("what is the sum of these numbers",
                                "arithmetic")):
            router.learn(text, category, filter_tokens=filter_tokens)
        return router

    def test_decoys_are_declined_after_the_fix(self):
        fixed = self._trained(filter_tokens=True)
        claimed = [q for q in _DECOYS if fixed.route(q, top_k=1)]
        self.assertEqual(claimed, [], f"still claimed: {claimed}")

    def test_decoys_were_claimed_before_it(self):
        """The defect is pinned, so a regression is visible as a test failure."""
        plain = self._trained(filter_tokens=False)
        claimed = [q for q in _DECOYS if plain.route(q, top_k=1)]
        self.assertTrue(claimed, "the old behaviour should claim decoys")

    def test_genuine_queries_still_route(self):
        """Abstention is trivial to buy with accuracy; this is the control."""
        fixed = self._trained(filter_tokens=True)
        for query, want in (("draw a diamond outline", "frames"),
                            ("a horizontal stripe", "lines"),
                            ("add the numbers", "arithmetic")):
            with self.subTest(query):
                ranked = fixed.route(query, top_k=1)
                self.assertTrue(ranked and ranked[0][0] == want)

    def test_the_control_can_actually_fail(self):
        """A router with no vocabulary declines everything and routes nothing.

        Asserted so the control is known to discriminate rather than assumed
        to - the §11.11 failure was a control that could not move.
        """
        empty = _router(frames=["the", "a", "is"])
        self.assertEqual(empty._base["frames"], set())
        self.assertEqual(
            [q for q in _DECOYS if empty.route(q, top_k=1)], [],
            "perfect abstention")
        self.assertEqual(empty.route("draw a diamond outline", top_k=1), [],
                         "and zero genuine routing - the control catches this")


class GateTests(unittest.TestCase):
    """The pre-registered measurement."""

    def test_the_gate_passes_with_the_control_held(self):
        from ultraquant.experiments.abstention_gate import run_gate

        report = run_gate(seeds=6)
        self.assertTrue(report.passes, report.reason)
        self.assertGreater(report.margin_ratio, 1.0)
        self.assertGreaterEqual(report.route_after, report.route_before)


if __name__ == "__main__":
    unittest.main()
