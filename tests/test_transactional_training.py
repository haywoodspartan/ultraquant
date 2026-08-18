"""The decoy check as a gate with a rollback, and the leak that forced it.

Run 4 of the sequential training (command-r) taught the phrasing
"Sign of the times<|END_OF_TURN_TOKEN|>". The template token survived _clean,
so the router learned `end`, `turn` and `token` as evidence for `crosses` —
and `times`, which plural-folds to `time` and claimed "what time does the
train leave". Decoys fell 1.000 -> 0.800, the report said so, and the router
was saved anyway: a warning where a rollback was needed.
"""

from __future__ import annotations

import tempfile
import unittest

from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardVault


def _router() -> CategoryRouter:
    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_txn_")))
    router.register("crosses", ["cross", "plus", "star"])
    router.register("frames", ["frame", "box"])
    return router


class TemplateTokenTests(unittest.TestCase):
    """The leak itself, pinned at the point of entry."""

    def test_template_tokens_are_stripped(self):
        from ultraquant.forge.distill import _clean

        self.assertEqual(_clean("Sign of the times<|END_OF_TURN_TOKEN|>"),
                         "Sign of the times")
        self.assertEqual(_clean("<|im_start|>a box outline<|im_end|>"),
                         "a box outline")

    def test_ordinary_angle_brackets_survive(self):
        from ultraquant.forge.distill import _clean

        self.assertEqual(_clean("less <than> more"), "less <than> more")


class UnlearnTests(unittest.TestCase):
    """The rollback primitive: learn then unlearn restores the prior state."""

    def test_learn_then_unlearn_is_identity_on_the_table(self):
        router = _router()
        router.learn("a sign of the times here", "crosses", delta=0.1)
        before = {t: w for t, w in router._learned["crosses"].items()}
        router.learn("more crossing marks appear", "crosses", delta=0.1)
        router.unlearn("more crossing marks appear", "crosses", delta=0.1)
        self.assertEqual(router._learned["crosses"], before)

    def test_unlearn_reports_the_weight_it_removed(self):
        router = _router()
        router.learn("crossing marks", "crosses", delta=0.1)
        removed = router.unlearn("crossing marks", "crosses", delta=0.1)
        self.assertAlmostEqual(removed, 0.2, places=6)

    def test_unlearn_never_goes_negative(self):
        router = _router()
        router.learn("crossing marks", "crosses", delta=0.1)
        router.unlearn("crossing marks", "crosses", delta=5.0)
        for weight in router._learned["crosses"].values():
            self.assertGreaterEqual(weight, 0.0)

    def test_unlearn_reverses_the_vault_too(self):
        """learn() reinforces shard associations; the inverse must as well."""
        router = _router()
        router.vault.add_shard("s1", "crosses", {"w": [1.0]})
        router.learn("crossing marks", "crosses", delta=0.1)
        self.assertIn("crossing", router.vault.entry("s1")["associations"])
        router.unlearn("crossing marks", "crosses", delta=0.1)
        self.assertNotIn("crossing", router.vault.entry("s1")["associations"])

    def test_unlearning_an_untaught_text_removes_nothing(self):
        router = _router()
        self.assertEqual(router.unlearn("never taught words", "crosses"), 0.0)


class WeakenTests(unittest.TestCase):
    """The vault half of the inverse."""

    def test_weaken_mirrors_reinforce(self):
        vault = ShardVault(tempfile.mkdtemp(prefix="uq_txn_"))
        vault.add_shard("s1", "crosses", {"w": [1.0]})
        vault.reinforce("s1", ["hatching"], delta=0.3)
        vault.weaken("s1", ["hatching"], delta=0.3)
        self.assertNotIn("hatching", vault.entry("s1")["associations"])

    def test_weaken_ignores_absent_keywords(self):
        vault = ShardVault(tempfile.mkdtemp(prefix="uq_txn_"))
        vault.add_shard("s1", "crosses", {"w": [1.0]})
        vault.weaken("s1", ["never"], delta=0.3)
        self.assertNotIn("never", vault.entry("s1")["associations"])


class GateTests(unittest.TestCase):
    """A run that drops the decoy score is undone, not regretted."""

    def _session(self):
        class _Session:
            router = _router()

        return _Session()

    def test_a_decoy_fall_rolls_the_run_back(self):
        from ultraquant.forge.train_from_llm import (
            TrainingReport,
            _teach_and_measure,
        )

        session = self._session()
        # "time" is what the leaked token taught in the wild: with it,
        # "what time does the train leave" is claimed and decoys fall.
        generated = {"crosses": ["sign time turn token", "crossing hatch marks"]}
        report = _teach_and_measure(session, generated, TrainingReport(),
                                    holdout=0.5, seed=0)
        self.assertTrue(report.rolled_back)
        self.assertEqual(report.taught, {}, "nothing kept for the ledger")
        self.assertEqual(report.trained, {})
        self.assertEqual(report.decoys_after, report.decoys_before,
                         "the rollback must restore the decoy score")
        self.assertNotIn("time", session.router._learned["crosses"])

    def test_a_clean_run_is_kept(self):
        from ultraquant.forge.train_from_llm import (
            TrainingReport,
            _teach_and_measure,
        )

        session = self._session()
        generated = {"crosses": ["crossing hatch marks", "star shaped sign"]}
        report = _teach_and_measure(session, generated, TrainingReport(),
                                    holdout=0.5, seed=0)
        self.assertFalse(report.rolled_back)
        self.assertTrue(report.taught)

    def test_the_report_names_the_rollback(self):
        from ultraquant.forge.train_from_llm import TrainingReport

        report = TrainingReport(trained={}, rolled_back=True,
                                decoys_before=1.0, decoys_after=1.0)
        self.assertIn("ROLLED BACK", report.as_text())


if __name__ == "__main__":
    unittest.main()
