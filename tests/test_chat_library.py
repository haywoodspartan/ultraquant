"""End-to-end: chat -> pattern recognition -> routing -> library shards.

The subsystems each pass their own tests, which is not the same as them working
together. These drive the real chat pipeline against a real forged library and
check that a pattern reaches the trained expert for its category — the path the
whole architecture exists to serve.

This is the test that caught the integration bug worth having: a glyph contains
no words, so keyword routing found nothing, every pattern fell through to a
default category, and a freshly invented untrained expert answered instead of the
library. Recognition accuracy was 0/8 while every unit test passed.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experts.moe import ExpertPool
from ultraquant.forge.corpus import BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy
from ultraquant.forge.forge import ModelForge
from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.pattern.recognition import PATTERNS, render

#: Which built-in family each glyph belongs to.
EXPECTED = {
    "arrow_up": "arrows", "arrow_down": "arrows",
    "square": "frames", "diamond": "frames",
    "plus": "crosses", "cross": "crosses",
    "stripes_h": "lines", "stripes_v": "lines",
}


class ChatAgainstLibraryTests(unittest.TestCase):
    """A pattern typed into the chat must reach its trained expert."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = Path(tempfile.mkdtemp(prefix="uq_chatlib_"))
        corpora = build_corpora(
            builtin_taxonomy(), n_per_class=24, seed=1, keywords=BUILTIN_KEYWORDS
        )
        cls.corpora = corpora
        forge = ModelForge(cls.dir / "home", seed=0, hidden=16, tier="auto",
                           part_bytes=16 * 1024)
        cls.report = forge.build(corpora, epochs=25)
        cls.forge = forge

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self) -> None:
        self.session = build_session(self.dir / "home", seed=0)

    def _ask(self, glyph: str):
        """Send a glyph through the chat; return (response, trace, category)."""
        response, trace = run_pipeline("\n".join(PATTERNS[glyph]), self.session)
        route = next(step for step in trace if step["thought"] == "Route")
        category = route["routes"][0] if route["routes"] else None
        return response, trace, category

    # -- the library must actually be reached ------------------------------

    def test_forge_stored_a_signature_for_every_category(self) -> None:
        """Without signatures a pattern cannot find its category at all."""
        signatures = self.forge.vault.signatures()
        self.assertEqual(set(signatures), set(EXPECTED.values()))
        for category, vectors in signatures.items():
            with self.subTest(category=category):
                self.assertGreaterEqual(len(vectors), 1)
                self.assertEqual(len(vectors[0]), 25)

    def test_every_glyph_routes_to_its_own_family(self) -> None:
        for glyph, category in EXPECTED.items():
            with self.subTest(glyph=glyph):
                _response, _trace, routed = self._ask(glyph)
                self.assertEqual(routed, category)

    def test_every_glyph_is_recognised_by_the_library_expert(self) -> None:
        correct = 0
        for glyph in EXPECTED:
            response, _trace, _routed = self._ask(glyph)
            correct += f"'{glyph}'" in response
        self.assertEqual(correct, len(EXPECTED),
                         "the forged library should recognise its own prototypes")

    def test_recognition_does_not_invent_a_new_expert(self) -> None:
        """Falling back to a fresh expert is the failure this test guards."""
        before = {entry["shard_id"] for entry in self.session.vault.catalog()}
        for glyph in EXPECTED:
            self._ask(glyph)
        after = {entry["shard_id"] for entry in self.session.vault.catalog()}
        self.assertEqual(before, after, "no new expert shard should be created")

    def test_routing_reads_no_shards(self) -> None:
        """Signatures live in the catalog, so routing must not page anything."""
        pixels = render(PATTERNS["plus"])
        before = self.session.cache.stats()["misses"]
        ranked = self.session.router.route_pattern(pixels, top_k=3)
        self.assertTrue(ranked)
        self.assertEqual(self.session.cache.stats()["misses"], before)

    # -- paging behaviour ---------------------------------------------------

    def test_experts_page_in_on_demand_then_hit(self) -> None:
        self._ask("plus")
        first = self.session.cache.stats()
        self.assertGreater(first["misses"], 0, "the first use must page the shard in")
        self._ask("cross")
        second = self.session.cache.stats()
        self.assertGreater(second["hits"], first["hits"],
                           "the same category's expert should be served from RAM")

    def test_only_the_needed_experts_become_resident(self) -> None:
        self._ask("plus")
        resident = self.session.cache.stats()["resident"]
        self.assertEqual(resident, [ExpertPool.shard_id("crosses")])

    def test_library_is_sharded_into_parts(self) -> None:
        parts = self.forge.library_parts()
        self.assertGreater(len(parts), 1)
        for entry in self.forge.vault.catalog():
            self.assertEqual(entry["location"], "library")

    # -- the rest of the pipeline still works around it ---------------------

    def test_trace_still_runs_every_thought(self) -> None:
        _response, trace, _routed = self._ask("square")
        self.assertEqual(
            [step["thought"] for step in trace],
            ["Perceive", "Recall", "Route", "Reason", "Respond", "Learn"],
        )

    def test_recognition_stores_a_signature_in_memory(self) -> None:
        before = self.session.memory.stats()["signatures"]
        self._ask("diamond")
        self.assertGreater(self.session.memory.stats()["signatures"], before)

    def test_recognition_reinforces_the_category(self) -> None:
        """Recall must strengthen the trace it used, brain-fashion."""
        shard_id = ExpertPool.shard_id("lines")
        before = self.session.vault.entry(shard_id)["access_count"]
        self._ask("stripes_h")
        self.assertGreater(self.session.vault.entry(shard_id)["access_count"], before)

    def test_text_input_still_routes_by_keyword(self) -> None:
        """Adding pattern routing must not break the word path."""
        _response, trace = run_pipeline("show me an arrow pointing up", self.session)
        route = next(step for step in trace if step["thought"] == "Route")
        self.assertEqual(route["routes"][0], "arrows")

    def test_chat_and_library_agree_with_the_expert_directly(self) -> None:
        """The answer through chat must match calling the expert by hand."""
        for glyph, category in list(EXPECTED.items())[:4]:
            with self.subTest(glyph=glyph):
                pixels = render(PATTERNS[glyph])
                from ultraquant.pattern.recognition import row_means

                direct, _confidence = self.session.experts.predict(
                    category, pixels + row_means(pixels)
                )
                response, _trace, _routed = self._ask(glyph)
                self.assertIn(f"'{direct}'", response)


class ShippedLibraryTests(unittest.TestCase):
    """A library must be usable on a machine that never trained it.

    This is what "the model is a library" has to mean in practice: copy the
    ``.uql`` files somewhere else, point a fresh vault at them, and patterns
    still reach their experts. It did not work — the container carried offsets,
    hashes and categories but not the content signatures, so every expert
    arrived intact and unroutable.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = Path(tempfile.mkdtemp(prefix="uq_shipped_"))
        forge = ModelForge(cls.dir / "home", seed=0, hidden=16, tier="auto")
        forge.build(
            build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                          keywords=BUILTIN_KEYWORDS),
            epochs=25,
        )
        cls.forge = forge

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _receiving_vault(self):
        """A vault that has only ever seen the library files."""
        from ultraquant.shards.vault import ShardVault

        fresh = ShardVault(self.dir / "elsewhere")
        for part in self.forge.library_parts():
            fresh.attach(part)
        return fresh

    def test_signatures_survive_the_trip(self) -> None:
        fresh = self._receiving_vault()
        self.assertEqual(set(fresh.signatures()), set(EXPECTED.values()))

    def test_patterns_still_route_on_the_receiving_side(self) -> None:
        from ultraquant.shards.router import CategoryRouter

        router = CategoryRouter(self._receiving_vault())
        for glyph, category in EXPECTED.items():
            with self.subTest(glyph=glyph):
                ranked = router.route_pattern(render(PATTERNS[glyph]), top_k=1)
                self.assertTrue(ranked, "nothing matched the shipped signatures")
                self.assertEqual(ranked[0][0], category)

    def test_the_experts_themselves_still_read_correctly(self) -> None:
        """Routing to the right shard is worth nothing if the shard is wrong."""
        from ultraquant.pattern.recognition import row_means
        from ultraquant.shards.budget import ShardCache

        fresh = self._receiving_vault()
        pool = ExpertPool(fresh, ShardCache(8 * 1024 * 1024), hidden=(16,))
        for glyph, category in EXPECTED.items():
            with self.subTest(glyph=glyph):
                pixels = render(PATTERNS[glyph])
                label, _confidence = pool.predict(category, pixels + row_means(pixels))
                self.assertEqual(label, glyph)


class GrowingLibraryDuringChatTests(unittest.TestCase):
    """A session must see categories added after it was built."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_chatgrow_"))
        self.forge = ModelForge(self.dir / "home", seed=0, hidden=16, tier="auto",
                                part_bytes=16 * 1024)
        taxonomy = builtin_taxonomy()
        self.first = {k: taxonomy[k] for k in ("lines", "arrows")}
        self.second = {k: taxonomy[k] for k in ("frames", "crosses")}
        self.forge.build(
            build_corpora(self.first, n_per_class=20, seed=1, keywords=BUILTIN_KEYWORDS),
            epochs=20,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_unknown_pattern_before_expansion_then_known_after(self) -> None:
        session = build_session(self.dir / "home", seed=0)
        _response, trace = run_pipeline("\n".join(PATTERNS["square"]), session)
        route = next(s for s in trace if s["thought"] == "Route")
        self.assertNotEqual(route["routes"][0], "frames",
                            "frames does not exist in the library yet")

        self.forge.expand(
            build_corpora(self.second, n_per_class=20, seed=2,
                          keywords=BUILTIN_KEYWORDS),
            epochs=20,
        )

        # A session opened after the expansion sees the new categories.
        grown = build_session(self.dir / "home", seed=0)
        response, trace = run_pipeline("\n".join(PATTERNS["square"]), grown)
        route = next(s for s in trace if s["thought"] == "Route")
        self.assertEqual(route["routes"][0], "frames")
        self.assertIn("'square'", response)

    def test_expansion_keeps_the_original_categories_working(self) -> None:
        self.forge.expand(
            build_corpora(self.second, n_per_class=20, seed=2,
                          keywords=BUILTIN_KEYWORDS),
            epochs=20,
        )
        session = build_session(self.dir / "home", seed=0)
        for glyph in ("stripes_h", "arrow_up", "square", "plus"):
            with self.subTest(glyph=glyph):
                response, _trace = run_pipeline("\n".join(PATTERNS[glyph]), session)
                self.assertIn(f"'{glyph}'", response)


if __name__ == "__main__":
    unittest.main()
