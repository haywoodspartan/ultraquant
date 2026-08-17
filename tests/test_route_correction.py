"""Correcting a routing error, which nothing could do before.

§11.17 fixed the feedback loop going forward and left the deployed library's two
entrenched errors in place, and I said `:learn` was the remedy for them. It is
not, and this is the measurement that showed it: `:learn` reinforces from the
*reply* text rather than the misrouted query, and none of its eight question
kinds surfaces a misroute at all.

No gate module accompanies this one, deliberately. A gate is for a mechanism
whose benefit is statistical and could be confused with noise; a correction
either reverses the decision it names or it does not, so the claim is directly
checkable and these tests are the right instrument. What *does* need a control
is the collateral: a correction must not damage the categories it takes weight
from, and that is checked below.
"""

from __future__ import annotations

import tempfile
import unittest

from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardVault


def _router() -> CategoryRouter:
    """A router with the deployed library's shape.

    ``crosses`` is registered with ``two`` and ``mark`` on purpose: those are
    what let it win "add these two numbers" and "what shape is this glyph"
    against thinner categories, which is how the deployed library came to
    misroute both. Without them the fixture routes correctly from the start and
    the entrenchment it is meant to reproduce never happens - the first version
    of this file made exactly that mistake and three tests failed for the right
    reason.
    """
    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_corr_")))
    router.register("crosses", ["cross", "plus", "star", "intersection", "mark",
                                "two", "shape", "glyph", "number"])
    router.register("arithmetic", ["number", "sum"])
    router.register("geometry", ["shape", "glyph"])
    return router


def _entrenched() -> CategoryRouter:
    """A router where the wrong category has accumulated weight, as one had."""
    router = _router()
    for _ in range(8):
        # Exactly the old unconditional rule: reinforce whatever won.
        for query in ("add these two numbers", "what shape is this glyph"):
            ranked = router.route(query, top_k=1)
            if ranked:
                router.learn(query, ranked[0][0], delta=0.05)
    return router


class WhyLearnCannotDoIt(unittest.TestCase):
    """The claim that had to be withdrawn, pinned as a fact about the code."""

    def test_no_question_kind_surfaces_a_misroute(self):
        import inspect

        from ultraquant.interpreter import learning

        source = inspect.getsource(learning)
        kinds = {"unknown-term", "unknown-pattern", "untrained-category",
                 "weak-pattern", "shaky", "disputed", "unclassified",
                 "proposed-concept"}
        for kind in kinds:
            self.assertIn(kind, source)
        for absent in ("misroute", "wrong-route", "route-correction"):
            self.assertNotIn(absent, source,
                             "if this appears, the claim needs revisiting")

    def test_the_supervised_path_learns_from_the_reply_not_the_query(self):
        """Which is why it cannot move a query it was never shown."""
        import inspect

        from ultraquant.interpreter.learning import LearningSession

        source = inspect.getsource(LearningSession)
        self.assertIn("router.learn(reply", source)

    def test_answering_a_fact_barely_moves_an_entrenched_decision(self):
        """Measured on the deployed library: 1.2 -> 1.4 against a 2.15 winner."""
        router = _entrenched()
        query = "add these two numbers"
        before = dict(router.route(query, top_k=3))
        router.learn("arithmetic is the study of numbers", "arithmetic",
                     delta=0.2)
        after = dict(router.route(query, top_k=3))
        self.assertEqual(
            router.route(query, top_k=1)[0][0], "crosses",
            "one answered fact should not be enough to reverse it")
        self.assertLess(after.get("arithmetic", 0) - before.get("arithmetic", 0),
                        after.get("crosses", 0))


class CorrectionTests(unittest.TestCase):
    """What the correction primitive does."""

    def test_a_corrected_query_routes_correctly_afterwards(self):
        router = _entrenched()
        query = "add these two numbers"
        self.assertNotEqual(router.route(query, top_k=1)[0][0], "arithmetic")
        router.correct(query, "arithmetic")
        self.assertEqual(router.route(query, top_k=1)[0][0], "arithmetic")

    def test_it_weakens_the_category_that_was_winning(self):
        """Adding weight alone has to out-climb the accumulation.

        Removing the accumulation is what reverses the decision, which is why
        the primitive does both.
        """
        router = _entrenched()
        result = router.correct("add these two numbers", "arithmetic")
        self.assertIn("crosses", result["weakened"])
        self.assertGreater(result["weakened"]["crosses"], 0.0)

    def test_learned_weight_never_goes_negative(self):
        """A correction may remove evidence; it must not invent the opposite."""
        router = _entrenched()
        for _ in range(20):
            router.correct("add these two numbers", "arithmetic")
        for weights in router._learned.values():
            for token, weight in weights.items():
                self.assertGreaterEqual(weight, 0.0, token)

    def test_base_keywords_are_never_unregistered(self):
        """One correction is not grounds for discarding chosen vocabulary."""
        router = _entrenched()
        before = set(router._base["crosses"])
        router.correct("a cross and a plus", "geometry")
        self.assertEqual(router._base["crosses"], before)

    def test_only_content_tokens_are_learned(self):
        router = _router()
        result = router.correct("what is the shape of it", "geometry")
        self.assertEqual(result["tokens"], ["shape"])

    def test_an_all_stopword_text_corrects_nothing(self):
        router = _router()
        result = router.correct("what is the of a", "geometry")
        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["weakened"], {})

    def test_correcting_to_an_unknown_category_registers_it(self):
        """learn() already registers on the fly; correction should not differ."""
        router = _router()
        router.correct("a novel kind of thing", "newcategory")
        self.assertIn("newcategory", router._base)


class CollateralTests(unittest.TestCase):
    """The control: a correction must not damage what it takes weight from."""

    def test_the_weakened_category_keeps_its_own_queries(self):
        router = _entrenched()
        owned = ["a plus and a cross", "the star shaped intersection",
                 "a cross mark here"]
        router.correct("add these two numbers", "arithmetic")
        router.correct("what shape is this glyph", "geometry")
        for query in owned:
            with self.subTest(query):
                ranked = router.route(query, top_k=1)
                self.assertTrue(ranked and ranked[0][0] == "crosses",
                                f"{query!r} -> {ranked}")

    def test_correction_does_not_reopen_abstention(self):
        """It adds weight, and over-claiming was fixed in §11.15."""
        router = _entrenched()
        router.correct("add these two numbers", "arithmetic")
        decoys = ["how do I fix a leaking tap", "recipe for banana bread",
                  "who won the football match last night"]
        claimed = [q for q in decoys if router.route(q, top_k=1)]
        self.assertEqual(claimed, [], f"still claimed: {claimed}")

    def test_only_tokens_of_the_corrected_text_are_touched(self):
        router = _entrenched()
        before = dict(router._learned["crosses"])
        router.correct("add these two numbers", "arithmetic")
        after = router._learned["crosses"]
        changed = {t for t in before if before[t] != after.get(t, 0.0)}
        self.assertTrue(changed <= {"add", "two", "number"},
                        f"unrelated tokens changed: {changed}")


class ChatCommandTests(unittest.TestCase):
    """The command a person actually types."""

    def setUp(self) -> None:
        import io
        from pathlib import Path

        from ultraquant.interpreter.chat import ChatCLI
        from ultraquant.interpreter.thoughts import build_session

        self.dir = tempfile.mkdtemp(prefix="uq_corrcli_")
        self.session = build_session(Path(self.dir), seed=0)
        self.buffer = io.StringIO()
        self.cli = ChatCLI(self.session, out=self.buffer)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_reports_before_and_after(self):
        self.cli.handle(":correct geometry what shape is this glyph")
        output = self.buffer.getvalue()
        self.assertIn("before", output)
        self.assertIn("after", output)
        self.assertIn("corrected", output)

    def test_an_unknown_category_is_refused_with_the_known_ones(self):
        self.cli.handle(":correct nosuchcategory some text here")
        output = self.buffer.getvalue()
        self.assertIn("Unknown category", output)
        self.assertIn("geometry", output)

    def test_usage_is_shown_for_a_bare_command(self):
        self.cli.handle(":correct")
        self.assertIn("Usage:", self.buffer.getvalue())

    def test_the_correction_is_persisted(self):
        """It has to survive the session, or it is not a correction."""
        from pathlib import Path

        from ultraquant.interpreter.thoughts import build_session

        self.cli.handle(":correct geometry what shape is this glyph")
        reopened = build_session(Path(self.dir), seed=0)
        ranked = reopened.router.route("what shape is this glyph", top_k=1)
        self.assertTrue(ranked and ranked[0][0] == "geometry")


if __name__ == "__main__":
    unittest.main()
