"""Goal-directed action: deciding what to do, not only answering.

The pipeline runs once per input and dispatches to one handler, so it can answer
"what is the tower height?" and cannot answer "add it to the bridge length" — not
for want of the pieces, but because nothing in it ever chooses to do two things
in order.

The gate's verdict is in ARCHITECTURE.md §11.9. What is asserted here is the
machinery, and the properties that stop this being a dispatch table with extra
steps: that no sequence is written down anywhere, that the action pool is wider
than any task needs, and that a capability invented afterwards can be used
without touching the planner.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.reason.actions import (
    build_actions,
    goal_all_bound,
    goal_derived_from,
)
from ultraquant.reason.planner import Action, Plan, Planner, PlanningError


def _counter(name: str, produces: str, needs: str | None = None) -> Action:
    """A toy action: available once ``needs`` is bound, produces ``produces``."""

    def bindings(state):
        if produces in state:
            return []
        if needs is not None and needs not in state:
            return []
        return [()]

    def apply(state, args):
        return produces, (state.get(needs, 0) + 1) if needs else 1

    return Action(name, bindings, apply)


class PlannerTests(unittest.TestCase):
    """Search behaviour, independent of any session."""

    def test_a_goal_that_already_holds_needs_no_plan(self) -> None:
        planner = Planner([_counter("a", "x")])
        plan = planner.plan(goal_all_bound("x"), {"x": 1})
        self.assertEqual(len(plan), 0)

    def test_it_orders_actions_by_their_preconditions(self) -> None:
        """The order is nowhere in the code; it follows from what needs what."""
        planner = Planner([
            _counter("third", "c", needs="b"),
            _counter("first", "a"),
            _counter("second", "b", needs="a"),
        ])
        plan = planner.plan(goal_all_bound("c"))
        self.assertEqual([step.action for step in plan.steps],
                         ["first", "second", "third"])

    def test_it_finds_the_shortest_plan(self) -> None:
        planner = Planner([
            _counter("long", "mid"), _counter("longer", "goal", needs="mid"),
            _counter("direct", "goal"),
        ])
        self.assertEqual(len(planner.plan(goal_all_bound("goal"))), 1)

    def test_an_unreachable_goal_fails_rather_than_hanging(self) -> None:
        planner = Planner([_counter("a", "x")], max_depth=3)
        with self.assertRaises(PlanningError):
            planner.plan(goal_all_bound("never"))

    def test_the_depth_bound_is_honoured(self) -> None:
        chain = [_counter("s0", "v0")]
        chain += [_counter(f"s{i}", f"v{i}", needs=f"v{i-1}") for i in range(1, 6)]
        self.assertEqual(len(Planner(chain, max_depth=6).plan(goal_all_bound("v5"))), 6)
        with self.assertRaises(PlanningError):
            Planner(chain, max_depth=3).plan(goal_all_bound("v5"))

    def test_the_state_budget_stops_a_runaway_search(self) -> None:
        def bindings(state):
            return [(i,) for i in range(50)]

        def apply(state, args):
            return f"v{args[0]}_{len(state)}", args[0]

        planner = Planner([Action("spray", bindings, apply)], max_states=200)
        with self.assertRaises(PlanningError):
            planner.plan(goal_all_bound("never"))

    def test_planning_is_reproducible(self) -> None:
        planner = Planner([
            _counter("first", "a"), _counter("second", "b", needs="a"),
        ])
        one = planner.plan(goal_all_bound("b"))
        two = planner.plan(goal_all_bound("b"))
        self.assertEqual(one.describe(), two.describe())

    def test_a_registered_action_is_usable_immediately(self) -> None:
        """The property that separates a planner from a dispatch table."""
        planner = Planner([_counter("first", "a")])
        with self.assertRaises(PlanningError):
            planner.plan(goal_all_bound("b"))
        planner.register(_counter("second", "b", needs="a"))
        self.assertEqual(len(planner.plan(goal_all_bound("b"))), 2)


class SessionActionTests(unittest.TestCase):
    """The real capability pool against a real session."""

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.forge.corpus import (
            BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
        )
        from ultraquant.forge.forge import ModelForge
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        cls.dir = Path(tempfile.mkdtemp(prefix="uq_planner_"))
        forge = ModelForge(cls.dir, seed=0, hidden=16, tier="auto")
        forge.build(
            build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                          keywords=BUILTIN_KEYWORDS),
            epochs=25,
        )
        cls.session = build_session(cls.dir, seed=0)
        for statement in ("the tower height is 324", "the bridge length is 486",
                          "the pier depth is 12"):
            run_pipeline(statement, cls.session)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self) -> None:
        self.planner = Planner(build_actions(self.session))

    def test_the_pool_is_wider_than_any_task_needs(self) -> None:
        """If it were not, selection would be trivial and the search decorative."""
        self.assertGreaterEqual(len(self.planner.actions), 8)

    def test_it_combines_two_recalled_facts(self) -> None:
        plan = self.planner.plan(
            goal_derived_from("tower height", "bridge length"),
            {"_keys": ["tower height", "bridge length"]},
        )
        self.assertEqual(len(plan), 3)
        self.assertIn(810.0, plan.state.values())

    def test_it_scales_to_a_third_fact(self) -> None:
        plan = self.planner.plan(
            goal_derived_from("tower height", "bridge length", "pier depth"),
            {"_keys": ["tower height", "bridge length", "pier depth"]},
        )
        self.assertIn(822.0, plan.state.values())

    def test_it_chains_recognition_into_a_library_query(self) -> None:
        from ultraquant.pattern.recognition import PATTERNS

        plan = self.planner.plan(
            goal_all_bound("category", "shard_count"),
            {"_pattern": list(PATTERNS["square"])},
        )
        self.assertEqual([s.action for s in plan.steps],
                         ["categorise", "count_shards"])
        self.assertEqual(plan.state["category"], "frames")

    def test_the_sandbox_is_reachable_from_a_plan(self) -> None:
        plan = self.planner.plan(
            goal_all_bound("evaluated"),
            {"_expression": "sum([x*x for x in range(1, 6)])"},
        )
        self.assertEqual(plan.state["evaluated"], 55.0)

    def test_a_missing_fact_does_not_produce_a_wrong_answer(self) -> None:
        with self.assertRaises(PlanningError):
            self.planner.plan(
                goal_derived_from("tower height", "no such thing"),
                {"_keys": ["tower height", "no such thing"]},
            )

    def test_a_capability_invented_later_needs_no_planner_change(self) -> None:
        """A dispatch table cannot do this; a planner does not notice."""

        def bindings(state):
            return [(k,) for k, v in sorted(state.items())
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and f"{k}^2" not in state]

        self.planner.register(
            Action("square", bindings, lambda s, a: (f"{a[0]}^2", s[a[0]] ** 2))
        )
        plan = self.planner.plan(
            lambda state: "pier depth^2" in state, {"_keys": ["pier depth"]}
        )
        self.assertEqual(plan.state["pier depth^2"], 144.0)
        self.assertEqual([s.action for s in plan.steps], ["recall", "square"])


class GoalIntentTests(unittest.TestCase):
    """The planner reached from the chat pipeline."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session, run_pipeline

        self.dir = Path(tempfile.mkdtemp(prefix="uq_goalintent_"))
        self.session = build_session(self.dir, seed=0)
        for statement in ("the tower height is 324", "the bridge length is 486"):
            run_pipeline(statement, self.session)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _ask(self, text: str):
        from ultraquant.interpreter.thoughts import run_pipeline

        return run_pipeline(text, self.session)

    def test_the_single_step_pipeline_cannot_combine_two_facts(self) -> None:
        """The baseline this stage exists to beat, checked rather than assumed."""
        response, _trace = self._ask("add the tower height to the bridge length")
        self.assertNotIn("810", response)

    def test_a_goal_produces_a_plan_and_an_answer(self) -> None:
        response, trace = self._ask("goal: the tower height and the bridge length")
        self.assertIn("Plan (3 steps)", response)
        self.assertIn("810", response)
        reason = next(s for s in trace if s["thought"] == "Reason")
        self.assertIn("planned 3 steps", reason["summary"])

    def test_the_plan_is_available_for_inspection(self) -> None:
        self._ask("goal: the tower height and the bridge length")
        self.assertEqual(
            [step["thought"] for step in self.session.last_trace],
            ["Perceive", "Recall", "Route", "Reason", "Respond", "Learn"],
        )

    def test_a_goal_naming_nothing_known_says_so(self) -> None:
        response, _trace = self._ask("goal: the mass of the moon")
        self.assertIn("names things I hold", response)

    def test_ordinary_input_is_not_routed_into_the_planner(self) -> None:
        """Guessing at goals would send plain questions into a search."""
        _response, trace = self._ask("what is the tower height?")
        perceive = next(s for s in trace if s["thought"] == "Perceive")
        self.assertEqual(perceive["intent"], "question")


if __name__ == "__main__":
    unittest.main()
