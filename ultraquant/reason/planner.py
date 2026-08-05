"""Choosing what to do, rather than only answering what was said.

The thought pipeline is `Perceive -> Recall -> Route -> Reason -> Respond`. It
runs once per input and dispatches to exactly one handler, which means it can
answer "what is the tower height?" and cannot answer "add the tower height to the
bridge length and remember it as span" — not for want of the pieces, but because
nothing in it ever decides to do *two* things in order.

This is a plain forward-search planner over declared actions. Each action states
what it needs and what it produces; the planner searches sequences until the goal
holds. It contains no recipe for any particular task, which is the whole point:
a planner with a per-task script would satisfy the letter of
[ARCHITECTURE.md §11.3] and none of its intent.

Three properties keep it honest:

* **Actions are declared, not sequenced.** Nothing anywhere lists "recall, then
  recall, then add". Preconditions and effects are stated; the order is found.
* **The action pool is larger than any task needs.** If a task needed exactly
  the actions available, selection would be trivial and the search decorative.
* **Adding a capability requires no change here.** A new action is registered
  from outside and immediately becomes available to plans, including for goals
  nobody anticipated. That is the property tested in
  ``ultraquant/experiments/planning_gate.py``.

Search is breadth-first, so the shortest plan is found, with states deduplicated
by their bound variables and a depth bound as a backstop. The point is a planner
that is obviously correct rather than one that is fast; nothing here is on a hot
path.

Pure Python standard library.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "State",
    "Step",
    "Action",
    "Plan",
    "Planner",
    "PlanningError",
]

#: Bound variables produced so far. Values are whatever actions put there.
State = dict[str, Any]


class PlanningError(RuntimeError):
    """Raised when a plan cannot be found or fails while executing."""


@dataclass(frozen=True)
class Step:
    """One action with the arguments it was chosen with."""

    action: str
    args: tuple
    produces: str

    def describe(self) -> str:
        """A human-readable line for the trace."""
        joined = ", ".join(str(a) for a in self.args)
        return f"{self.action}({joined}) -> {self.produces}"


@dataclass
class Action:
    """A capability the planner may use.

    Args:
        name: Identifier used in plans and traces.
        bindings: Given a state, yields the argument tuples worth trying. This
            is where an action declares what it *could* apply to; returning
            nothing means it is not applicable yet.
        apply: Given a state and one argument tuple, returns
            ``(name, value)`` for what it produced, or ``None`` if it turned out
            not to apply after all.
        cost: Relative preference; lower is tried first at equal depth.
    """

    name: str
    bindings: Callable[[State], list[tuple]]
    apply: Callable[[State, tuple], tuple[str, Any] | None]
    cost: int = 1


@dataclass
class Plan:
    """A found sequence and the state it produced."""

    steps: list[Step] = field(default_factory=list)
    state: State = field(default_factory=dict)
    explored: int = 0

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> list[str]:
        """One line per step."""
        return [step.describe() for step in self.steps]


class Planner:
    """Finds and runs sequences of actions that satisfy a goal.

    Args:
        actions: The capabilities available. Registering more never requires
            changing this class.
        max_depth: Longest plan considered, as a backstop against a goal that
            can never be satisfied.
        max_states: Search budget, so an unsatisfiable goal fails promptly
            instead of exploring forever.
    """

    def __init__(
        self, actions: list[Action] | None = None,
        max_depth: int = 6, max_states: int = 4000,
    ) -> None:
        self.actions: list[Action] = list(actions or [])
        self.max_depth = int(max_depth)
        self.max_states = int(max_states)

    def register(self, action: Action) -> None:
        """Add a capability.

        A new action becomes usable by every goal immediately, including goals
        that were never anticipated. Nothing else has to change, which is what
        distinguishes a planner from a dispatch table.
        """
        self.actions.append(action)

    @staticmethod
    def _signature(state: State) -> tuple:
        """A hashable summary, so equivalent states are explored once."""
        out = []
        for key in sorted(state):
            value = state[key]
            out.append((key, value if isinstance(value, (int, float, str, bool))
                        else repr(value)))
        return tuple(out)

    def plan(
        self, goal: Callable[[State], bool], initial: State | None = None,
    ) -> Plan:
        """Find the shortest sequence of actions after which ``goal`` holds.

        Args:
            goal: Predicate over the state.
            initial: Starting bindings.

        Returns:
            The plan, whose ``steps`` is empty if the goal already held.

        Raises:
            PlanningError: If no plan exists within the depth and state budget.
        """
        start: State = dict(initial or {})
        if goal(start):
            return Plan(steps=[], state=start, explored=0)

        seen = {self._signature(start)}
        queue: deque[tuple[State, list[Step]]] = deque([(start, [])])
        explored = 0

        while queue:
            state, steps = queue.popleft()
            if len(steps) >= self.max_depth:
                continue
            for action in sorted(self.actions, key=lambda a: a.cost):
                for args in action.bindings(state):
                    explored += 1
                    if explored > self.max_states:
                        raise PlanningError(
                            f"no plan within {self.max_states} states "
                            f"(depth limit {self.max_depth})"
                        )
                    produced = action.apply(state, args)
                    if produced is None:
                        continue
                    name, value = produced
                    successor = dict(state)
                    successor[name] = value
                    signature = self._signature(successor)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    extended = steps + [Step(action.name, tuple(args), name)]
                    if goal(successor):
                        return Plan(steps=extended, state=successor,
                                    explored=explored)
                    queue.append((successor, extended))
        raise PlanningError(
            f"no plan reaches the goal within depth {self.max_depth}"
        )

    def achieve(
        self, goal: Callable[[State], bool], initial: State | None = None,
    ) -> Plan:
        """Plan and return the result.

        The search executes actions as it explores, so a found plan has already
        produced its state. Actions are therefore expected to be free of side
        effects *outside* the state, or to be idempotent — the two side-effecting
        actions in this project (storing a fact, training an expert) are applied
        by the caller from the finished plan rather than during the search.
        """
        return self.plan(goal, initial)
