"""Quarantined chaos: physical unpredictability that never breaks anything.

The founding idea was the user's — quantum-machine noise and timing instability
as chaos for whimsical, out-of-the-box decisions. What makes it admissible in a
codebase that hunts nondeterminism as a bug is three rules, and the tests are
mostly about the rules:

* chaos chooses only among acceptable-equals, never whether an option is valid;
* every draw is receipted and a run replays exactly from its log;
* off by default — the suite and every un-opted session stay deterministic.

Plus the trap kept as a warning: laundering a constant through a hash produces
statistically perfect-looking output, so entropy is credited by *raw* measures,
never by whitened balance. And the gate: exploration is free on a stable machine
and helpful under drift.
"""

from __future__ import annotations

import unittest

from ultraquant.reason.whimsy import EntropyWell, harvest_jitter


class DeterminismByDefaultTests(unittest.TestCase):
    """A disabled well is invisible — the property the whole suite relies on."""

    def test_a_disabled_well_returns_the_deterministic_default(self) -> None:
        well = EntropyWell(enabled=False)
        for _ in range(50):
            self.assertEqual(well.choose("t", "p", ["a", "b", "c"]), "a")

    def test_a_disabled_well_never_fires_the_pulse(self) -> None:
        well = EntropyWell(enabled=False)
        self.assertFalse(any(well.occasionally("t", "p") for _ in range(200)))

    def test_a_disabled_well_records_nothing(self) -> None:
        well = EntropyWell(enabled=False)
        [well.choose("t", "p", ["a", "b"]) for _ in range(20)]
        self.assertEqual(well.receipts, [])


class AcceptableEqualsTests(unittest.TestCase):
    """Chaos picks which valid option, never whether an option is valid."""

    def test_choose_only_ever_returns_a_supplied_option(self) -> None:
        well = EntropyWell(enabled=True)
        options = ["x", "y", "z"]
        for _ in range(100):
            self.assertIn(well.choose("t", "p", options), options)

    def test_a_single_option_needs_no_entropy(self) -> None:
        """One acceptable-equal is not a choice; no draw, no receipt."""
        well = EntropyWell(enabled=True)
        self.assertEqual(well.choose("t", "p", ["only"]), "only")
        self.assertEqual(well.receipts, [])

    def test_choosing_nothing_is_an_error_not_a_guess(self) -> None:
        with self.assertRaises(ValueError):
            EntropyWell(enabled=True).choose("t", "p", [])

    def test_an_enabled_well_actually_explores(self) -> None:
        """Over many draws it must not collapse to one option."""
        well = EntropyWell(enabled=True)
        picks = {well.choose("t", "p", ["a", "b", "c", "d"]) for _ in range(60)}
        self.assertGreater(len(picks), 1, "the well never left the default")


class ReceiptAndReplayTests(unittest.TestCase):
    """Every draw is auditable, and a whimsical run is exactly reproducible."""

    def test_each_draw_leaves_a_receipt(self) -> None:
        well = EntropyWell(enabled=True)
        well.choose("router", "tie-break", ["a", "b", "c"])
        self.assertEqual(len(well.receipts), 1)
        receipt = well.receipts[0]
        self.assertEqual(receipt["consumer"], "router")
        self.assertEqual(receipt["purpose"], "tie-break")
        self.assertIn(receipt["choice"], ("a", "b", "c"))

    def test_a_run_replays_exactly_from_its_receipts(self) -> None:
        live = EntropyWell(enabled=True)
        options = ["a", "b", "c", "d", "e"]
        first = [live.choose("t", "p", options) for _ in range(30)]

        replayed = EntropyWell(enabled=True, replay=live.receipts)
        second = [replayed.choose("t", "p", options) for _ in range(30)]
        self.assertEqual(first, second)

    def test_replay_reproduces_the_pulse_too(self) -> None:
        live = EntropyWell(enabled=True)
        first = [live.occasionally("t", "p", out_of=3) for _ in range(40)]
        replayed = EntropyWell(enabled=True, replay=live.receipts)
        second = [replayed.occasionally("t", "p", out_of=3) for _ in range(40)]
        self.assertEqual(first, second)


class EntropyQualityTests(unittest.TestCase):
    """Credit sources by raw measurement; whitening cannot fake entropy."""

    def test_jitter_harvest_is_not_a_constant(self) -> None:
        """The bug this guards: the first harvester produced all zeros, and a
        hash turned that constant into perfect-looking noise."""
        blocks = {bytes(harvest_jitter(16)) for _ in range(8)}
        self.assertGreater(len(blocks), 1,
                           "jitter collapsed to a constant - hashing it would "
                           "have laundered the collapse invisibly")

    def test_the_well_bottoms_out_at_os_entropy(self) -> None:
        """Even with no session and weak jitter, draws must still vary —
        os.urandom is always in the mix, so the well can never go constant."""
        well = EntropyWell(enabled=True)
        values = {well._draw_value(1000) for _ in range(50)}
        self.assertGreater(len(values), 10)


class ExplorationGateTests(unittest.TestCase):
    """The pre-registered gate: free when stable, helpful under drift.

    Random exploration does not *halve* drift regret, and the gate does not
    claim it does. What is measured and asserted: zero cost when nothing
    drifts, and a clear reduction when something does.
    """

    def _simulate(self, drift: bool, explore: bool, seed: int) -> float:
        import random

        from ultraquant.native.scheduler import LearnedDispatch

        def cost(config: str, era: int) -> float:
            if era == 0 or not drift:
                return {"python": 0.010, "cpp": 0.0006, "cuda": 0.0020}[config]
            return {"python": 0.010, "cpp": 0.0030, "cuda": 0.0008}[config]

        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "d.json"
        well = None
        if explore:
            well = EntropyWell(enabled=True)
            original = well.occasionally
            well.occasionally = lambda c, p, out_of=8: original(c, p, out_of=3)
        dispatch = LearnedDispatch(
            path, available={"quantum": ["python", "cpp", "cuda"]},
            seed=seed, well=well,
        )
        rng = random.Random(seed)
        regret = 0.0
        steps = 80
        for step in range(steps):
            era = 0 if step < steps // 2 else 1
            qubits = rng.choice([4, 6, 8, 10])
            dims = {"qubits": qubits, "gates": qubits * 3, "batch": 4}
            config, reason = dispatch.decide("quantum", dims)
            if reason in ("probe", "explore"):
                timings = {c: cost(c, era)
                           for c in ["python", "cpp", "cuda"]}
                dispatch.probe("quantum", dims,
                               {c: (lambda: None) for c in timings})
                for record in dispatch.experience.records[-3:]:
                    record["seconds"] = cost(record["config"], era)
                dispatch.experience.save()
                chosen = min(timings, key=timings.get)
            else:
                chosen = config
            regret += cost(chosen, era) - min(
                cost(c, era) for c in ["python", "cpp", "cuda"]
            )
        return regret * 1e3

    def test_exploration_is_free_on_a_stable_machine(self) -> None:
        exploit = [self._simulate(drift=False, explore=False, seed=s)
                   for s in range(4)]
        explore = [self._simulate(drift=False, explore=True, seed=s)
                   for s in range(4)]
        base = sum(exploit) / len(exploit)
        with_explore = sum(explore) / len(explore)
        self.assertLessEqual(with_explore, base + 1.0,
                             "re-probing a standing winner should cost ~0")

    def test_exploration_reduces_regret_under_drift(self) -> None:
        exploit = [self._simulate(drift=True, explore=False, seed=s)
                   for s in range(4)]
        explore = [self._simulate(drift=True, explore=True, seed=s)
                   for s in range(4)]
        base = sum(exploit) / len(exploit)
        with_explore = sum(explore) / len(explore)
        self.assertLess(with_explore, base * 0.75,
                        "exploration should remove >=25% of drift regret")


class WiringTests(unittest.TestCase):
    """The well reaches the loop it exists for, and only when opted in."""

    def setUp(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        from ultraquant.forge.languages import seed_knowledge
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_whimsy_"))
        self.session = build_session(self.dir, seed=0)
        seed_knowledge(self.session)
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_curiosity_stays_within_the_pending_questions(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        self.session.whimsy = EntropyWell(self.session, enabled=True)
        learner = LearningSession(self.session)
        pending = learner.survey()
        ids = {question.id for question in pending}
        for _ in range(20):
            chosen = learner.next_question()
            self.assertIn(chosen.id, ids,
                          "curiosity invented a question out of nothing")

    def test_without_a_well_next_question_is_the_ranking(self) -> None:
        from ultraquant.interpreter.learning import LearningSession

        learner = LearningSession(self.session)
        pending = learner.survey()
        self.assertIs(learner.next_question(), pending[0])

    def test_the_chat_command_toggles_and_receipts(self) -> None:
        import io

        from ultraquant.interpreter.chat import ChatCLI

        out = io.StringIO()
        cli = ChatCLI(self.session, out=out)
        cli.handle(":whimsy on")
        self.assertIsNotNone(self.session.whimsy)
        cli.handle(":whimsy off")
        self.assertIsNone(self.session.whimsy)


if __name__ == "__main__":
    unittest.main()
