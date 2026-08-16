"""Self-reinforcement entrenches errors rather than creating them.

`thoughts.py:801` reinforces whichever category the pipeline routed to, with
nothing checking the route was right. This pins what that actually costs, which
is not what the obvious guess says.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments.reinforcement_drift import DriftReport, measure


class DriftTests(unittest.TestCase):
    """The measured behaviour, so a change to the learning rule is visible."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = measure(seeds=6, rounds=40)

    def test_accuracy_does_not_fall(self):
        """The obvious guess is wrong and worth pinning as wrong.

        Reinforcement cannot make a wrong answer wronger: the misrouted
        queries were already misrouted. Anyone looking for damage in the
        accuracy column will find none.
        """
        self.assertAlmostEqual(self.report.accuracy_self,
                               self.report.accuracy_before, places=2)

    def test_the_error_becomes_harder_to_overturn(self):
        """The actual harm, and the one that matters.

        A later correct signal has to beat a margin that self-reinforcement
        has been widening the whole time.
        """
        self.assertGreater(self.report.margin_self,
                           self.report.margin_before * 2,
                           "self-reinforcement should entrench the error")
        self.assertGreater(self.report.entrenchment, 2.0)

    def test_the_control_does_not_move(self):
        """Without reinforcement the margin is where it started."""
        self.assertAlmostEqual(self.report.margin_control,
                               self.report.margin_before, places=2)

    def test_the_control_and_treatment_differ(self):
        """If they did not, the mechanism would not be the cause."""
        self.assertGreater(self.report.margin_self,
                           self.report.margin_control * 2)

    def test_the_report_names_the_finding_not_the_guess(self):
        text = self.report.as_text()
        self.assertIn("accuracy is unchanged", text)
        self.assertIn("harder to overturn", text)


class SupervisedCallsAreUntouchedTests(unittest.TestCase):
    """Only the unsupervised reinforcement is circular."""

    def test_the_learning_loop_reinforces_from_a_user_answer(self):
        """That is independent evidence, so it is legitimate and left alone."""
        import inspect

        from ultraquant.interpreter import learning

        source = inspect.getsource(learning)
        self.assertIn("router.learn(reply", source,
                      "learning.py reinforces from the user's reply")

    def test_the_pipeline_reinforces_from_its_own_route(self):
        """The circular one, pinned so its removal is a deliberate change."""
        import inspect

        from ultraquant.interpreter import thoughts

        source = inspect.getsource(thoughts)
        self.assertIn("session.router.learn(ctx.text, category", source)


if __name__ == "__main__":
    unittest.main()
