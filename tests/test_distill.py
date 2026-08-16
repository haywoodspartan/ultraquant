"""Distilling a transformer's paraphrase knowledge into the lexical router.

No LM Studio is contacted. A stub panel supplies phrasings, so the suite runs
on a machine with no local models — the normal case for an optional tier.

The point being tested is the trade §11.11 left open: embeddings routed
paraphrases far better than lexical matching (0.833 vs 0.300) but cost 2.9 ms
against 0.015 ms, a 197x tax on every query forever. Paying it once offline and
teaching the result to the router keeps query time where it was.
"""

from __future__ import annotations

import unittest
from unittest import mock

from ultraquant.forge.distill import (
    DistillReport,
    distil_into_router,
    generate_phrasings,
    run_gate,
)
from ultraquant.interpreter.lmstudio import Answer, LMStudioUnavailable
from ultraquant.interpreter.llmls import ModelCard


class _StubClient:
    """Returns canned lines, and records what it was asked."""

    def __init__(self, replies: dict, fail: set | None = None):
        self.replies = replies
        self.fail = fail or set()
        self.prompts: list[tuple[str, str]] = []

    def complete(self, prompt, model=None, system=None, max_tokens=0,
                 temperature=0.0):
        if model in self.fail:
            raise LMStudioUnavailable(f"{model} is not loaded")
        self.prompts.append((model, prompt))
        return Answer(text=self.replies.get(model, ""), model=model,
                      prompt=prompt)


class _StubPanel:
    """Just enough TeacherPanel for generate_phrasings."""

    def __init__(self, client, model_ids):
        self.client = client
        self.cards = [ModelCard(id=name, arch="a", publisher=name)
                      for name in model_ids]


class GenerationTests(unittest.TestCase):
    """Turning model output into usable routing examples."""

    def _panel(self, replies, fail=None):
        client = _StubClient(replies, fail)
        return _StubPanel(client, list(replies)), client

    def test_every_teacher_contributes_all_its_phrasings(self):
        """Consensus is the wrong instrument: coverage is the goal here.

        Two models offering different phrasings is the desired outcome, not a
        disagreement to resolve - a phrasing is not a claim.
        """
        panel, _ = self._panel({
            "m1": "a box outline\nthe shape of a window",
            "m2": "four equal sides\na square figure",
        })
        report = generate_phrasings(panel, "geometry")
        self.assertEqual(len(report.phrasings), 4)
        self.assertEqual(len(report.models), 2)

    def test_list_markers_are_stripped(self):
        panel, _ = self._panel({"m1": "1. a box outline\n- the shape of a window\n"
                                      "* a four sided figure"})
        report = generate_phrasings(panel, "geometry")
        self.assertTrue(all(not p[0].isdigit() and p[0] not in "-*"
                            for p in report.phrasings), report.phrasings)
        self.assertIn("a box outline", report.phrasings)

    def test_duplicates_across_teachers_are_dropped(self):
        panel, _ = self._panel({"m1": "a box outline", "m2": "A Box Outline"})
        report = generate_phrasings(panel, "geometry")
        self.assertEqual(len(report.phrasings), 1)
        self.assertTrue(any("duplicate" in why for _p, why in report.rejected))

    def test_essays_and_single_words_are_refused(self):
        """A phrasing has to be usable as a query."""
        panel, _ = self._panel({"m1": "shapes\n" + " ".join(["word"] * 30)
                                      + "\na box outline"})
        report = generate_phrasings(panel, "geometry")
        self.assertEqual(report.phrasings, ["a box outline"])
        self.assertEqual(len(report.rejected), 2)

    def test_one_unavailable_teacher_does_not_sink_the_run(self):
        panel, _ = self._panel({"m1": "a box outline", "m2": "four sides here"},
                               fail={"m2"})
        report = generate_phrasings(panel, "geometry")
        self.assertEqual(report.phrasings, ["a box outline"])
        self.assertTrue(any("unavailable" in why for _p, why in report.rejected))

    def test_the_prompt_asks_for_varied_vocabulary(self):
        """Phrasings that reuse the category name teach the router nothing."""
        panel, client = self._panel({"m1": "a box outline"})
        generate_phrasings(panel, "geometry", "shapes and geometric figures")
        _model, prompt = client.prompts[0]
        self.assertIn("shapes and geometric figures", prompt)
        self.assertIn("avoid reusing", prompt)

    def test_a_silent_teacher_yields_nothing_rather_than_junk(self):
        panel, _ = self._panel({"m1": ""})
        self.assertEqual(generate_phrasings(panel, "geometry").phrasings, [])


class RouterTeachingTests(unittest.TestCase):
    """The phrasings have to actually change routing."""

    def _router(self):
        import tempfile

        from ultraquant.shards.router import CategoryRouter
        from ultraquant.shards.vault import ShardVault

        router = CategoryRouter(ShardVault(tempfile.mkdtemp()))
        router.register("geometry", ["geometry"])
        router.register("arithmetic", ["arithmetic"])
        return router

    def test_a_taught_phrasing_routes_afterwards(self):
        router = self._router()
        query = "a box outline drawn on paper"
        before = router.route(query, top_k=1)
        self.assertFalse(before and before[0][0] == "geometry",
                         "the point is that lexical routing fails here first")
        distil_into_router(router, "geometry", ["a box outline", "box outline"])
        after = router.route(query, top_k=1)
        self.assertTrue(after and after[0][0] == "geometry")

    def test_teaching_reports_how_many_were_applied(self):
        router = self._router()
        self.assertEqual(
            distil_into_router(router, "geometry", ["a box", "  ", "a frame"]),
            2, "blank phrasings are not applied")

    def test_nothing_reaches_memory_or_the_stash(self):
        """These are routing hints, not claims. A wrong fact is unrecoverable."""
        import inspect

        from ultraquant.forge import distill

        source = inspect.getsource(distill)
        for forbidden in ("remember_fact", "add_page", "promote("):
            self.assertNotIn(forbidden, source,
                             f"{forbidden} would turn a phrasing into belief")


class GateTests(unittest.TestCase):
    """The gate must be able to fail, and must skip rather than pass."""

    def test_no_teacher_is_a_skip_never_a_pass(self):
        with mock.patch("ultraquant.forge.distill.TeacherPanel",
                        side_effect=LMStudioUnavailable("no server")):
            from ultraquant.config import Settings

            with mock.patch.object(Settings, "load") as load:
                load.return_value.get.return_value = ["m1"]
                report = run_gate()
        self.assertTrue(report.skipped)
        self.assertFalse(report.passes)

    def test_an_unconfigured_panel_says_what_to_do(self):
        from ultraquant.config import Settings

        with mock.patch.object(Settings, "load") as load:
            load.return_value.get.return_value = []
            report = run_gate()
        self.assertTrue(report.skipped)
        self.assertIn("Panel tab", report.reason)

    def test_a_thin_teacher_is_a_skip(self):
        """Too few phrasings means the split would measure noise."""
        panel = _StubPanel(_StubClient({"m1": "a box outline"}), ["m1"])
        report = run_gate(panel, seeds=3)
        self.assertTrue(report.skipped)
        self.assertIn("usable phrasing", report.reason)

    def test_the_control_can_actually_fail(self):
        """§11.11's control was at ceiling and could not condemn the result.

        Here the decoys start at 1.000 (a fresh router places none of them) and
        distillation can only push that down - so the control has room to move
        in the direction that would invalidate a win.
        """
        import tempfile

        from ultraquant.shards.router import CategoryRouter
        from ultraquant.shards.vault import ShardVault

        router = CategoryRouter(ShardVault(tempfile.mkdtemp()))
        for category in ("geometry", "arithmetic", "colour", "sorting"):
            router.register(category, [category])
        decoys = ["how do I fix a leaking tap", "recipe for banana bread"]
        placed = [router.route(text, top_k=1) for text in decoys]
        self.assertTrue(all(not r or r[0][0] not in
                            {"geometry", "arithmetic", "colour", "sorting"}
                            for r in placed),
                        "decoys must start unplaced, so the control can fall")

    def test_the_report_prints_without_a_run(self):
        from ultraquant.forge.distill import GateReport

        text = GateReport(skipped=True, reason="no server").as_text()
        self.assertIn("SKIPPED", text)


class ReportTests(unittest.TestCase):
    def test_a_report_summarises_without_dumping_everything(self):
        report = DistillReport(category="geometry",
                               phrasings=[f"phrase {i}" for i in range(20)],
                               models=["m1"])
        text = report.as_text()
        self.assertIn("20 phrasing(s)", text)
        self.assertIn("and 12 more", text)


if __name__ == "__main__":
    unittest.main()
