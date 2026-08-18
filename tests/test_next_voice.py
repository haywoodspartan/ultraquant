"""One teacher at a time, smallest first, until the catalogue is full.

The training was asked to take this shape, and each part carries a measured
reason: one at a time because several 20-35B teachers thrash one card; smallest
first because the 7B costs seconds per category where the 35B cost 65 s; and
"until we have enough" is made precise by the §11.13 ceiling - each category
absorbs at most SAFE_PHRASINGS_PER_CATEGORY distilled phrasings ever, tracked
across runs in vault/distilled.json.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ultraquant.forge.train_from_llm import (
    _catalogue_state,
    _save_catalogue_state,
    parameter_count,
)


class ParameterCountTests(unittest.TestCase):
    """Smallest-to-largest needs sizes, and ids are where they live."""

    def test_plain_sizes_parse(self):
        for model_id, want in (("mistralrp-noromaid-nsfw-mistral-7b", 7.0),
                               ("katarau-9b-ru-rp-nsfw", 9.0),
                               ("google/gemma-4-31b", 31.0),
                               ("cydonia-v1.3-magnum-v4-22b", 22.0)):
            with self.subTest(model_id):
                self.assertEqual(parameter_count(model_id), want)

    def test_moe_ids_use_the_total_not_the_active_count(self):
        """The first version took min() and put a 19.6 GB model first.

        MoE naming carries both: qwen3.6-35b-a3b is 35B total, 3B active, and
        load cost tracks the total.
        """
        self.assertEqual(
            parameter_count("qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive"),
            35.0)
        self.assertEqual(
            parameter_count("qwen3-48b-a4b-savant-commander-distill-12x"),
            48.0)

    def test_an_unnamed_size_sorts_last(self):
        """Unknown cost does not belong at the front of a cost-ordered queue."""
        self.assertEqual(parameter_count("c4ai-command-r-08-2024"),
                         float("inf"))

    def test_version_numbers_are_not_sizes(self):
        self.assertEqual(parameter_count("cydonia-v1.3-magnum-v4-22b"), 22.0)


class LedgerTests(unittest.TestCase):
    """The cross-run memory that makes the ceiling cumulative."""

    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="uq_ledger_"))
        (self.home / "vault").mkdir(parents=True)

    def test_a_missing_ledger_is_a_fresh_start(self):
        state = _catalogue_state(self.home)
        self.assertEqual(state["teachers"], [])
        self.assertEqual(state["phrasings"], {})

    def test_the_ledger_round_trips(self):
        _save_catalogue_state(self.home, {
            "teachers": ["mistral-7b"],
            "phrasings": {"geometry": ["a drawn figure", "some outline"]}})
        state = _catalogue_state(self.home)
        self.assertEqual(state["teachers"], ["mistral-7b"])
        self.assertEqual(len(state["phrasings"]["geometry"]), 2)

    def test_a_corrupt_ledger_degrades_to_fresh(self):
        (self.home / "vault" / "distilled.json").write_text("{ not json",
                                                            encoding="utf-8")
        state = _catalogue_state(self.home)
        self.assertEqual(state["teachers"], [])

    def test_the_budget_is_cumulative_across_runs(self):
        """Twelve per category EVER, not per run - repetition is how §11.16's
        second training run degraded the library."""
        from ultraquant.forge.distill import SAFE_PHRASINGS_PER_CATEGORY

        phrasings = [f"phrasing {i}" for i in range(SAFE_PHRASINGS_PER_CATEGORY)]
        _save_catalogue_state(self.home, {"teachers": ["m1"],
                                          "phrasings": {"geometry": phrasings}})
        state = _catalogue_state(self.home)
        room = SAFE_PHRASINGS_PER_CATEGORY - len(state["phrasings"]["geometry"])
        self.assertEqual(room, 0, "a full category has no remaining budget")


class QueueTests(unittest.TestCase):
    """Which model trains next, driven by a stub catalogue."""

    def _next(self, cards, used):
        from ultraquant.interpreter.llmls import independent_groups

        voices = independent_groups(cards)
        untried = []
        for group in voices:
            members = sorted(group, key=lambda c: (parameter_count(c.id), c.id))
            if any(c.id in used for c in group):
                continue
            untried.append(members[0])
        untried.sort(key=lambda c: (parameter_count(c.id), c.id))
        return untried[0].id if untried else None

    def _cards(self):
        from ultraquant.interpreter.llmls import ModelCard

        return [
            ModelCard("mistralrp-mistral-7b", "llama", "a"),
            ModelCard("cydonia-v1.3-magnum-v4-22b", "llama", "a"),
            ModelCard("katarau-9b-ru-rp", "qwen35", "b"),
            ModelCard("google/gemma-4-31b", "gemma4", "google"),
        ]

    def test_smallest_voice_representative_goes_first(self):
        self.assertEqual(self._next(self._cards(), used=set()),
                         "mistralrp-mistral-7b")

    def test_a_used_voice_is_skipped_even_via_a_sibling(self):
        """The queue is voices, not models: cydonia's lineage already taught
        through mistral, and a sibling adds near-identical phrasings."""
        self.assertEqual(self._next(self._cards(),
                                    used={"mistralrp-mistral-7b"}),
                         "katarau-9b-ru-rp")

    def test_an_exhausted_queue_returns_nothing(self):
        used = {"mistralrp-mistral-7b", "katarau-9b-ru-rp", "google/gemma-4-31b"}
        self.assertIsNone(self._next(self._cards(), used=used))


class ContextWiringTests(unittest.TestCase):
    """The §11.14 window, live in the session at last."""

    def setUp(self) -> None:
        import shutil

        self.dir = Path(tempfile.mkdtemp(prefix="uq_ctxwire_"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def _session(self):
        from ultraquant.interpreter.thoughts import build_session

        return build_session(self.dir / "home", seed=0)

    def test_every_session_carries_a_context_window(self):
        session = self._session()
        self.assertIsNotNone(session.context)

    def test_turns_are_appended_and_survive_reopening(self):
        from ultraquant.interpreter.thoughts import run_pipeline

        session = self._session()
        run_pipeline("the harbour depth is 12 fathoms", session)
        run_pipeline("unrelated chatter about coffee", session)
        reopened = self._session()
        self.assertEqual(reopened.context.stats()["turns"], 2)

    def test_an_evicted_turn_is_paged_back_by_recall(self):
        """The point of the whole design: buried is not gone."""
        from ultraquant.interpreter.thoughts import run_pipeline
        from ultraquant.memory.context import ContextWindow

        session = self._session()
        session.context = ContextWindow(self.dir / "home" / "ctx",
                                        budget_bytes=512)
        run_pipeline("the harbour depth is 12 fathoms", session)
        for index in range(40):
            run_pipeline(f"filler chatter {index} about coffee", session)
        self.assertLess(session.context.stats()["resident_turns"], 10,
                        "the fact turn must have been evicted")
        _resp, trace = run_pipeline("how deep is the harbour", session)
        recall = next(s["summary"] for s in trace if s["thought"] == "Recall")
        self.assertIn("paged back from disk", recall)

    def test_the_turn_being_processed_never_matches_itself(self):
        """Appended after the pipeline, or every query would self-match."""
        from ultraquant.interpreter.thoughts import run_pipeline

        session = self._session()
        _resp, trace = run_pipeline("a wholly novel subject here", session)
        recall = next(s["summary"] for s in trace if s["thought"] == "Recall")
        self.assertNotIn("paged back", recall)
        self.assertEqual(session.context.stats()["turns"], 1,
                         "appended once, after the thoughts ran")

    def test_commands_are_not_turns(self):
        """Only pipeline text lands in the window; colon-commands bypass it."""
        import io

        from ultraquant.interpreter.chat import ChatCLI

        session = self._session()
        cli = ChatCLI(session, out=io.StringIO())
        cli.handle(":help")
        cli.handle(":resident")
        self.assertEqual(session.context.stats()["turns"], 0)

    def test_resident_shows_the_split(self):
        """Permanent store vs working memory, visible in one place."""
        import io

        from ultraquant.interpreter.chat import ChatCLI
        from ultraquant.interpreter.thoughts import run_pipeline

        session = self._session()
        run_pipeline("some text to put a turn in the window", session)
        buffer = io.StringIO()
        ChatCLI(session, out=buffer).handle(":resident")
        output = buffer.getvalue()
        self.assertIn("context window", output)
        self.assertIn("on disk", output)


if __name__ == "__main__":
    unittest.main()
