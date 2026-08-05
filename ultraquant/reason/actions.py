"""The capabilities a plan may be built from.

Each is a thin declaration over something the session could already do. What is
new is that they state their preconditions and effects, so the planner can order
them itself instead of the order being written down somewhere.

The pool is deliberately **wider than any single task needs**. A task that
required exactly the actions available would make selection trivial and the
search decorative, and the gate would pass without meaning anything.

Nothing here knows about any particular task. ``build_actions`` returns the same
pool whatever is asked, and the tasks in the gate differ only in their goals.
"""

from __future__ import annotations

from typing import Any

from ultraquant.reason.planner import Action, State

__all__ = ["build_actions", "goal_bound", "goal_all_bound", "goal_derived_from"]


def _numbers(state: State) -> list[str]:
    """Names currently bound to something arithmetic."""
    return sorted(
        key for key, value in state.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _wanted(state: State) -> list[str]:
    """Fact keys the request referred to but which are not yet recalled."""
    return [key for key in state.get("_keys", []) if key not in state]


def build_actions(session: Any) -> list[Action]:
    """The capability pool, bound to a session.

    Args:
        session: An interpreter session; its memory, coder, experts and vault
            are what the actions actually use.

    Returns:
        Actions ready to register with a :class:`~ultraquant.reason.planner.Planner`.
    """

    # -- retrieval ---------------------------------------------------------

    def recall_bindings(state: State) -> list[tuple]:
        return [(key,) for key in _wanted(state)]

    def recall_apply(state: State, args: tuple) -> tuple[str, Any] | None:
        (key,) = args
        fact = session.memory.recall_fact(key)
        if not fact:
            return None
        value = fact.get("value")
        try:
            return key, float(str(value).split()[0])
        except (ValueError, IndexError):
            return key, value

    # -- arithmetic --------------------------------------------------------
    #
    # One action per operation rather than a single "compute" that is handed an
    # expression: the planner is supposed to work out that two values must be
    # combined, not be told the formula.

    def _pair_bindings(state: State) -> list[tuple]:
        names = _numbers(state)
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

    def _ordered_pairs(state: State) -> list[tuple]:
        names = _numbers(state)
        return [(a, b) for a in names for b in names if a != b]

    def _arith(symbol: str, operation):
        def apply(state: State, args: tuple) -> tuple[str, Any] | None:
            left, right = args
            try:
                value = operation(state[left], state[right])
            except (TypeError, ZeroDivisionError):
                return None
            return f"{left}{symbol}{right}", value

        return apply

    # -- recognition -------------------------------------------------------

    def recognise_bindings(state: State) -> list[tuple]:
        if "_pattern" not in state or "label" in state:
            return []
        return [("_pattern",)]

    def recognise_apply(state: State, args: tuple) -> tuple[str, Any] | None:
        from ultraquant.pattern.recognition import render, row_means

        rows = state.get("_pattern")
        if not rows:
            return None
        pixels = render(list(rows))
        ranked = session.router.route_pattern(pixels, top_k=1)
        if not ranked:
            return None
        category = ranked[0][0]
        if not session.experts.has_expert(category):
            return None
        label, _confidence = session.experts.predict(
            category, pixels + row_means(pixels)
        )
        return "label", label

    def category_bindings(state: State) -> list[tuple]:
        if "_pattern" not in state or "category" in state:
            return []
        return [("_pattern",)]

    def category_apply(state: State, args: tuple) -> tuple[str, Any] | None:
        from ultraquant.pattern.recognition import render

        rows = state.get("_pattern")
        if not rows:
            return None
        ranked = session.router.route_pattern(render(list(rows)), top_k=1)
        if not ranked:
            return None
        return "category", ranked[0][0]

    # -- the library itself ------------------------------------------------

    def count_bindings(state: State) -> list[tuple]:
        if "category" not in state or "shard_count" in state:
            return []
        return [("category",)]

    def count_apply(state: State, args: tuple) -> tuple[str, Any] | None:
        category = state.get("category")
        if not isinstance(category, str):
            return None
        return "shard_count", float(len(session.vault.shards_in(category)))

    # -- sandboxed evaluation ----------------------------------------------

    def evaluate_bindings(state: State) -> list[tuple]:
        if "_expression" not in state or "evaluated" in state:
            return []
        return [("_expression",)]

    def evaluate_apply(state: State, args: tuple) -> tuple[str, Any] | None:
        from ultraquant.interpreter.codefunc import CodeError

        try:
            # run() returns {"result", "stdout", "defined"}, not a bare value.
            outcome = session.coder.run(state["_expression"])
            value = outcome.get("result") if isinstance(outcome, dict) else outcome
            if value is None:
                return None
            return "evaluated", float(value)
        except (CodeError, TypeError, ValueError):
            return None

    return [
        Action("recall", recall_bindings, recall_apply, cost=1),
        Action("add", _pair_bindings, _arith("+", lambda a, b: a + b), cost=2),
        Action("multiply", _pair_bindings, _arith("*", lambda a, b: a * b), cost=2),
        Action("subtract", _ordered_pairs, _arith("-", lambda a, b: a - b), cost=3),
        Action("divide", _ordered_pairs, _arith("/", lambda a, b: a / b), cost=3),
        Action("recognise", recognise_bindings, recognise_apply, cost=1),
        Action("categorise", category_bindings, category_apply, cost=1),
        Action("count_shards", count_bindings, count_apply, cost=2),
        Action("evaluate", evaluate_bindings, evaluate_apply, cost=2),
    ]


def goal_bound(name: str):
    """Goal: some variable holds ``name``'s value.

    Names are matched by suffix as well as exactly, because the arithmetic
    actions name their results after what they combined — ``tower+bridge`` — and
    the caller asking for a sum should not have to know which order the planner
    found.
    """

    def satisfied(state: State) -> bool:
        return name in state

    return satisfied


def goal_all_bound(*names: str):
    """Goal: every one of ``names`` is bound."""

    def satisfied(state: State) -> bool:
        return all(name in state for name in names)

    return satisfied


def goal_derived_from(*keys: str):
    """Goal: some value has been derived from all of ``keys``.

    This is the form most of the multi-step tasks take, and it is deliberately
    silent about *how*. Naming the operation would hand the planner the recipe;
    asking only that the answer depend on the right inputs leaves it to work out
    which actions get there, and lets a shorter route win if one exists.
    """

    def satisfied(state: State) -> bool:
        for name in state:
            if name.startswith("_") or name in keys:
                continue
            if all(key in name for key in keys):
                return True
        return False

    return satisfied
