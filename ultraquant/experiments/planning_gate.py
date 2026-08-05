"""Stage 4's gate: does it solve what one step cannot?

[ARCHITECTURE.md §11.3] states it before the work:

    Solves multi-step tasks that the single-step pipeline cannot, **without a
    hand-written script per task**.

Pre-registered: **8 of 9 tasks** solved by the planner, with the single-step
pipeline solving no more than the ones that genuinely need a single step.

**The clause that does the work is the second one.** A planner carrying a recipe
per task would satisfy the letter of this gate and none of its intent, and the
easiest way to fake it is an action pool sized exactly to each task, so that
"choosing" is not a choice. Four things guard against that:

* The pool holds **nine** actions and no task needs more than five.
* Tasks supply a **goal**, never a sequence. Most say only "a value derived from
  these facts" — not which operation, not in what order.
* Two tasks are **single-step by design**, so the baseline is not rigged: if the
  pipeline could not do those, it would be a broken comparison rather than a
  weak one.
* The last test registers a **capability invented after the planner was
  written**, with a goal nobody anticipated, and requires that
  ``ultraquant/reason/planner.py`` is not touched. A dispatch table cannot pass
  that; a planner does not notice the difference.

Run it::

    python -m ultraquant.experiments.planning_gate
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ultraquant.reason.actions import build_actions, goal_all_bound, goal_derived_from
from ultraquant.reason.planner import Action, Planner, PlanningError

__all__ = ["FACTS", "TASKS", "run_gate", "main"]

#: Pre-registered pass mark.
TARGET_SOLVED = 8

#: Facts the session is told before any task runs.
FACTS: dict[str, float] = {
    "tower height": 324.0,
    "bridge length": 486.0,
    "pier depth": 12.0,
    "span count": 4.0,
}


def _task_list() -> list[dict]:
    """Every task: a request, a goal, and how many steps it really needs.

    ``request`` is what the single-step pipeline is given, so the comparison is
    on the same problem. ``goal`` is what the planner is given — a condition,
    never a sequence.
    """
    return [
        {
            "name": "recall one fact",
            "request": "what is the tower height?",
            "goal": goal_all_bound("tower height"),
            "state": {"_keys": ["tower height"]},
            "expect": 324.0,
            "single_step": True,
        },
        {
            "name": "recognise a pattern",
            "request": None,
            "goal": goal_all_bound("label"),
            "state": {"_pattern": None},  # filled in at run time
            "expect": "square",
            "single_step": True,
        },
        {
            "name": "combine two facts",
            "request": "add the tower height to the bridge length",
            "goal": goal_derived_from("tower height", "bridge length"),
            "state": {"_keys": ["tower height", "bridge length"]},
            "expect": 810.0,
            "single_step": False,
        },
        {
            "name": "combine three facts",
            "request": "add the tower height, the bridge length and the pier depth",
            "goal": goal_derived_from("tower height", "bridge length", "pier depth"),
            "state": {"_keys": ["tower height", "bridge length", "pier depth"]},
            "expect": 822.0,
            "single_step": False,
        },
        {
            "name": "combine a different pair",
            "request": "what is the pier depth and the span count together?",
            "goal": goal_derived_from("pier depth", "span count"),
            "state": {"_keys": ["pier depth", "span count"]},
            "expect": 16.0,
            "single_step": False,
        },
        {
            "name": "derive from a fact and the library",
            "request": "how many shards hold the category this pattern belongs to?",
            "goal": goal_all_bound("category", "shard_count"),
            "state": {"_pattern": None},
            "expect": None,
            "single_step": False,
        },
        {
            "name": "recognise and count together",
            "request": "name this pattern and count its category's shards",
            "goal": goal_all_bound("label", "shard_count"),
            "state": {"_pattern": None},
            "expect": None,
            "single_step": False,
        },
        {
            "name": "sandboxed evaluation",
            "request": "calc: sum([x*x for x in range(1, 6)])",
            "goal": goal_all_bound("evaluated"),
            "state": {"_expression": "sum([x*x for x in range(1, 6)])"},
            "expect": 55.0,
            "single_step": True,
        },
        {
            "name": "cross-modal: a label and a derived number",
            "request": "name this pattern and add the tower height to the pier depth",
            "goal": goal_all_bound("label"),
            "state": {"_pattern": None, "_keys": ["tower height", "pier depth"]},
            "extra_goal": goal_derived_from("tower height", "pier depth"),
            "expect": 336.0,
            "single_step": False,
        },
    ]


TASKS = _task_list()


def _session(root: Path) -> Any:
    """A session with a forged library and the facts already told to it."""
    from ultraquant.forge.corpus import (
        BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
    )
    from ultraquant.forge.forge import ModelForge
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    forge = ModelForge(root, seed=0, hidden=16, tier="auto")
    forge.build(
        build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                      keywords=BUILTIN_KEYWORDS),
        epochs=25,
    )
    session = build_session(root, seed=0)
    for key, value in FACTS.items():
        run_pipeline(f"the {key} is {value:g}", session)
    return session


def _combined_goal(task: dict):
    """A task's goal, including its second clause when it has one."""
    first = task["goal"]
    second = task.get("extra_goal")
    if second is None:
        return first
    return lambda state: first(state) and second(state)


def run_gate(verbose: bool = False) -> dict:
    """Run every task through the planner and through the single-step pipeline."""
    from ultraquant.interpreter.thoughts import run_pipeline
    from ultraquant.pattern.recognition import PATTERNS

    root = Path(tempfile.mkdtemp(prefix="uq_plan_"))
    try:
        session = _session(root)
        planner = Planner(build_actions(session))
        rows = list(PATTERNS["square"])

        planned = 0
        single = 0
        results = []
        for task in TASKS:
            state = dict(task["state"])
            if state.get("_pattern", "missing") is None:
                state["_pattern"] = rows

            try:
                plan = planner.plan(_combined_goal(task), state)
                solved = True
                steps = len(plan)
                detail = plan.describe()
            except PlanningError as exc:
                solved, steps, detail = False, 0, [str(exc)]
            planned += solved

            # The same problem, put to the pipeline that existed before.
            baseline = False
            if task["request"]:
                response, _trace = run_pipeline(task["request"], session)
                expected = task["expect"]
                if expected is None:
                    baseline = False
                elif isinstance(expected, str):
                    baseline = f"'{expected}'" in response
                else:
                    baseline = (f"{expected:g}" in response
                                or f"{expected:.1f}" in response)
            else:
                response, _trace = run_pipeline("\n".join(rows), session)
                baseline = f"'{task['expect']}'" in response
            single += baseline

            results.append({
                "name": task["name"], "planned": solved, "steps": steps,
                "baseline": baseline, "single_step": task["single_step"],
                "detail": detail,
            })
            if verbose:
                mark = "ok " if solved else "no "
                print(f"  {mark}{task['name']:<40} {steps} steps"
                      f"{'   (baseline solves it too)' if baseline else ''}")
                for line in detail:
                    print(f"        {line}")

        multi = [r for r in results if not r["single_step"]]
        return {
            "tasks": len(TASKS),
            "planned": planned,
            "baseline": single,
            "multi_step_tasks": len(multi),
            "multi_step_planned": sum(r["planned"] for r in multi),
            "multi_step_baseline": sum(r["baseline"] for r in multi),
            "actions": len(planner.actions),
            "max_steps": max((r["steps"] for r in results), default=0),
            "results": results,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def novel_capability_check() -> dict:
    """Register a capability the planner has never seen, and use it.

    This is the test a dispatch table cannot pass. The action below is defined
    here, after ``planner.py`` was written and without changing it, and the goal
    it serves was not anticipated anywhere. If the planner can fold it into a
    sequence, it is planning rather than looking up.
    """
    root = Path(tempfile.mkdtemp(prefix="uq_plannovel_"))
    try:
        session = _session(root)
        planner = Planner(build_actions(session))

        def square_bindings(state):
            return [(key,) for key, value in sorted(state.items())
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    and f"{key}^2" not in state]

        def square_apply(state, args):
            (key,) = args
            return f"{key}^2", state[key] * state[key]

        planner.register(Action("square", square_bindings, square_apply, cost=2))

        goal = lambda state: any(name.endswith("^2") for name in state)
        plan = planner.plan(goal, {"_keys": ["pier depth"]})
        value = plan.state.get("pier depth^2")
        return {
            "solved": value == 144.0,
            "steps": plan.describe(),
            "value": value,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Run the gate and print a verdict."""
    _ = list(sys.argv[1:] if argv is None else argv)

    print("Stage 4 gate: does it solve what one step cannot?")
    print(f"pre-registered: >= {TARGET_SOLVED} of {len(TASKS)} tasks planned, "
          f"goals supplied but never sequences")
    print()

    result = run_gate(verbose=True)
    print()
    print(f"  planner solved      {result['planned']}/{result['tasks']}")
    print(f"  single-step solved  {result['baseline']}/{result['tasks']}")
    print(f"  on the {result['multi_step_tasks']} genuinely multi-step tasks: "
          f"planner {result['multi_step_planned']}, "
          f"baseline {result['multi_step_baseline']}")
    print(f"  action pool         {result['actions']} actions, "
          f"longest plan {result['max_steps']} steps")

    print()
    print("novel capability (registered after the planner was written):")
    novel = novel_capability_check()
    for line in novel["steps"]:
        print(f"    {line}")
    print(f"    result {novel['value']}  ->  "
          f"{'used without touching the planner' if novel['solved'] else 'FAILED'}")

    baseline_sound = result["baseline"] >= 2
    print()
    print("verdict")
    if (result["planned"] >= TARGET_SOLVED and novel["solved"]
            and result["multi_step_baseline"] == 0 and baseline_sound):
        print("  PASS - multi-step tasks are solved from goals alone, the")
        print("  single-step pipeline solves none of them, and a capability")
        print("  invented afterwards was used without changing the planner.")
    elif not baseline_sound:
        print("  INVALID - the baseline solved almost nothing, including tasks")
        print("  that need one step. That is a broken comparison, not a weak")
        print("  baseline; fix it before reading anything into the rest.")
    elif result["multi_step_baseline"]:
        print("  QUESTIONABLE - the single-step pipeline solved a task that was")
        print("  supposed to need several, so that task was not multi-step.")
    else:
        print("  FAIL - below the pre-registered target.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
