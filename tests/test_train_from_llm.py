"""Training a deployed router from a local model.

No LM Studio is contacted; a stub panel supplies phrasings. What is pinned here
is the judgement the driver has to make correctly on a real library: which
categories are worth asking a model about, what the volume ceiling is, and that
a dry run does not quietly train the thing it claims to be measuring.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.forge.train_from_llm import (
    TrainingReport,
    _SYNTHETIC,
    _topic_for,
    train_router,
)
from ultraquant.interpreter.lmstudio import Answer
from ultraquant.interpreter.llmls import ModelCard


class _StubClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.asked: list[str] = []

    def complete(self, prompt, model=None, system=None, max_tokens=0,
                 temperature=0.0, reasoning_effort=None):
        self.asked.append(prompt)
        return Answer(text=self.reply, model=model or "stub", prompt=prompt)


class _StubPanel:
    def __init__(self, reply: str):
        self.client = _StubClient(reply)
        self.cards = [ModelCard(id="stub-teacher", arch="a", publisher="p")]


class _Session:
    """Just the router attribute train_router touches."""

    def __init__(self, router):
        self.router = router


def _router(categories: dict):
    from ultraquant.shards.router import CategoryRouter
    from ultraquant.shards.vault import ShardVault

    router = CategoryRouter(ShardVault(tempfile.mkdtemp()))
    for name, keywords in categories.items():
        router.register(name, keywords)
    return router


class CategorySelectionTests(unittest.TestCase):
    """Which categories are worth asking a model about."""

    def test_synthetic_families_are_recognised(self):
        for name in ("family_007", "family-7", "category_12", "class3"):
            with self.subTest(name):
                self.assertTrue(_SYNTHETIC.match(name))
        for name in ("geometry", "arithmetic", "family_history"):
            with self.subTest(name):
                self.assertFalse(_SYNTHETIC.match(name))

    def test_a_placeholder_vocabulary_yields_no_topic(self):
        """Asked about 'family_007 (sym, 0, 1)' a model invents a topic.

        It would then be taught to the router as though it meant something.
        """
        self.assertIsNone(_topic_for("family_007", {"sym", "0", "1", "family"}))

    def test_a_real_vocabulary_yields_a_topic(self):
        topic = _topic_for("frames", {"box", "diamond", "frame", "outline"})
        self.assertIsNotNone(topic)
        self.assertIn("frames", topic)
        self.assertIn("outline", topic)

    def test_synthetic_categories_are_skipped_with_a_reason(self):
        router = _router({"geometry": ["shape", "glyph", "pattern"],
                          "family_001": ["sym", "0", "1"]})
        panel = _StubPanel("a drawn figure\nsome outline on paper")
        report = train_router(_Session(router), panel)
        self.assertIn("family_001", report.skipped)
        self.assertIn("synthetic", report.skipped["family_001"])
        self.assertIn("geometry", report.trained)

    def test_a_thin_teacher_is_skipped_not_trained_on(self):
        router = _router({"geometry": ["shape", "glyph", "pattern"]})
        report = train_router(_Session(router), _StubPanel(""))
        self.assertIn("geometry", report.skipped)
        self.assertEqual(report.trained, {})


class CeilingTests(unittest.TestCase):
    """The volume ceiling is measured, so it is enforced rather than offered."""

    def test_a_larger_count_is_capped(self):
        from ultraquant.forge.distill import SAFE_PHRASINGS_PER_CATEGORY

        router = _router({"geometry": ["shape", "glyph", "pattern"]})
        panel = _StubPanel("\n".join(f"phrasing number {i} about shapes"
                                     for i in range(40)))
        train_router(_Session(router), panel, count=500)
        asked = panel.client.asked[0]
        self.assertIn(f"List {SAFE_PHRASINGS_PER_CATEGORY} ", asked)
        self.assertNotIn("List 500 ", asked)


class MeasurementTests(unittest.TestCase):
    """The report has to be about text the router has not been taught."""

    def test_the_holdout_is_not_taught(self):
        router = _router({"geometry": ["shape", "glyph", "pattern"]})
        lines = [f"unique phrasing {i} concerning polygons" for i in range(8)]
        panel = _StubPanel("\n".join(lines))
        report = train_router(_Session(router), panel, holdout=0.5)
        self.assertLess(report.trained["geometry"], len(lines),
                        "half the phrasings must be withheld")

    def test_decoys_are_reported_alongside_the_win(self):
        """The failure mode is willingness to answer, not ignorance."""
        router = _router({"geometry": ["shape", "glyph", "pattern"]})
        panel = _StubPanel("a drawn figure\nsome outline on paper\n"
                           "a shape with corners")
        report = train_router(_Session(router), panel)
        self.assertGreaterEqual(report.decoys_before, 0.0)
        self.assertGreaterEqual(report.decoys_after, 0.0)
        self.assertIn("decoys left alone", report.as_text())

    def test_a_fallen_decoy_score_is_called_out(self):
        report = TrainingReport(trained={"a": 1}, decoys_before=1.0,
                                decoys_after=0.4)
        self.assertIn("FELL", report.as_text())

    def test_a_held_decoy_score_is_not_alarming(self):
        report = TrainingReport(trained={"a": 1}, decoys_before=1.0,
                                decoys_after=1.0)
        self.assertIn("(held)", report.as_text())


class DryRunTests(unittest.TestCase):
    """A dry run that trains the library is worse than no dry run."""

    def test_dry_run_copies_the_home(self):
        """Skipping router.save() is not enough to be dry.

        CategoryRouter.learn() also calls vault.reinforce(), which writes the
        shard catalog immediately - so the first "dry" run on the real library
        genuinely trained it, and the next run started from the moved baseline
        (0.109 instead of a clean one).
        """
        import inspect

        from ultraquant.forge import train_from_llm

        source = inspect.getsource(train_from_llm.main)
        self.assertIn("copytree", source,
                      "a dry run must work on a copy of the home")

    def test_learning_writes_the_vault_which_is_why(self):
        """The mechanism behind it, asserted so the reason cannot rot."""
        import inspect

        from ultraquant.shards.router import CategoryRouter

        self.assertIn("reinforce", inspect.getsource(CategoryRouter.learn))


if __name__ == "__main__":
    unittest.main()
