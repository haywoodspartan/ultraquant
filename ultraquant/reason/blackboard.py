"""A place where partial results can meet.

The reasoning step has been `one input -> one route -> one expert -> one label`
since the beginning. That shape cannot represent *"an arrow inside a frame"*, not
because no expert knows about arrows or frames, but because there is nowhere for
two partial answers to be held at the same time. [ARCHITECTURE.md §11.2] calls
this out as one of the two things actually missing; this module is the combining
surface it lacked.

The design is the classical blackboard: a shared workspace, a set of independent
contributors that read it and write to it, and a controller that runs them until
nothing new is being said. Nothing here is clever. Its value is structural —
several experts can speak about one input, each about the part it knows, and a
later contributor can revise an earlier claim in light of what else was written.

Three properties are worth stating because they are what make it more than a
loop over experts:

* **Slots.** A contribution is filed against an *aspect* of the input, not the
  input as a whole. Two contributors that disagree about the same slot are in
  competition; two that fill different slots are composing.
* **Rounds.** A contributor sees what earlier ones wrote. That is what lets a
  constraint revise a weak reading rather than merely veto it.
* **Opting out.** ``applies()`` lets a contributor decline, so the workspace does
  not have to consult everything the library holds to answer one question.

Whether composition actually buys anything is an empirical question with a gate
attached, and this module does not assume the answer — see
``ultraquant/experiments/composition_gate.py``.

Pure Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "Contribution",
    "Blackboard",
    "Contributor",
    "ExpertContributor",
    "ConstraintContributor",
    "run_blackboard",
    "compose",
]


@dataclass(frozen=True)
class Contribution:
    """One claim about one aspect of the input.

    Args:
        source: Who wrote it — an expert category, a constraint, anything.
        slot: The aspect being described. Same slot means competition;
            different slots mean composition.
        value: The claimed label.
        confidence: How strongly it is claimed, in ``[0, 1]``.
        round: Which pass wrote it.
        evidence: Free-form detail, carried into the trace.
    """

    source: str
    slot: str
    value: str
    confidence: float
    round: int = 0
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-safe form, for traces and the archive."""
        return {
            "source": self.source,
            "slot": self.slot,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "round": self.round,
            "evidence": dict(self.evidence),
        }


class Blackboard:
    """The shared workspace for one input.

    Args:
        features: The feature vector every contributor reads.
        meta: Anything contributors need that is not a feature (raw rows, the
            originating text, a session handle).
    """

    def __init__(self, features: list[float], meta: dict | None = None) -> None:
        self.features: list[float] = list(features)
        self.meta: dict = dict(meta or {})
        self.round: int = 0
        self._entries: list[Contribution] = []

    # -- writing -----------------------------------------------------------

    def post(
        self, source: str, slot: str, value: str, confidence: float,
        evidence: dict | None = None,
    ) -> Contribution:
        """File a claim against ``slot``.

        Returns:
            The stored :class:`Contribution`.
        """
        entry = Contribution(
            source=source, slot=slot, value=value,
            confidence=float(confidence), round=self.round,
            evidence=dict(evidence or {}),
        )
        self._entries.append(entry)
        return entry

    # -- reading -----------------------------------------------------------

    def contributions(self, slot: str | None = None) -> list[Contribution]:
        """Every claim, or every claim about one slot, in the order written."""
        if slot is None:
            return list(self._entries)
        return [e for e in self._entries if e.slot == slot]

    def best(self, slot: str) -> Contribution | None:
        """The strongest claim about ``slot``.

        Later rounds win ties, because a revision that merely matches the
        confidence of what it revises is still the more informed claim.
        """
        candidates = self.contributions(slot)
        if not candidates:
            return None
        return max(candidates, key=lambda e: (e.confidence, e.round))

    def slots(self) -> list[str]:
        """Slots that have been written to, in first-written order."""
        out: list[str] = []
        for entry in self._entries:
            if entry.slot not in out:
                out.append(entry.slot)
        return out

    def reading(self) -> dict[str, str]:
        """The current best value per slot — the composed answer so far."""
        return {slot: self.best(slot).value for slot in self.slots()}

    def confidence(self) -> float:
        """Joint confidence: the weakest link in the composed reading.

        A composition is only as good as its least certain part, so the minimum
        is the honest summary. Averaging would let a confident reading of one
        slot disguise a coin-flip on another.
        """
        bests = [self.best(slot) for slot in self.slots()]
        if not bests:
            return 0.0
        return min(entry.confidence for entry in bests if entry is not None)

    def snapshot(self) -> dict:
        """JSON-safe view of the whole workspace."""
        return {
            "rounds": self.round,
            "reading": self.reading(),
            "confidence": round(self.confidence(), 4),
            "contributions": [e.as_dict() for e in self._entries],
        }


@runtime_checkable
class Contributor(Protocol):
    """Anything that can read the blackboard and write to it."""

    name: str

    def applies(self, board: Blackboard) -> bool:
        """Whether this contributor has anything to say about ``board``."""

    def contribute(self, board: Blackboard) -> bool:
        """Write to ``board``; return True if anything new was written."""


class ExpertContributor:
    """A library expert speaking about the slot it owns.

    The expert reads the **whole** feature vector, not a pre-separated slice.
    Handing each expert its own region would do the decomposition by hand and
    leave nothing interesting to learn; making every expert see the whole input
    and attend to its own factor is the actual problem.

    Args:
        pool: The :class:`~ultraquant.experts.moe.ExpertPool` to predict with.
        category: Which expert to consult.
        slot: The aspect it describes.
        min_confidence: Below this the expert declines to write, leaving the
            slot to someone better placed.
        weight: Scales the posted confidence. When experts compete for one slot
            this carries how well the input matched each category in the first
            place, so a confident expert that the input barely resembles does
            not outvote a slightly less confident one that it matches exactly.
    """

    def __init__(
        self, pool: Any, category: str, slot: str, min_confidence: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        self.pool = pool
        self.category = category
        self.slot = slot
        self.min_confidence = float(min_confidence)
        self.weight = float(weight)
        self.name = f"expert:{category}"

    def applies(self, board: Blackboard) -> bool:
        """True when the expert exists and has not already spoken."""
        if not self.pool.has_expert(self.category):
            return False
        return not any(e.source == self.name for e in board.contributions(self.slot))

    def contribute(self, board: Blackboard) -> bool:
        """Predict and file the result."""
        label, confidence = self.pool.predict(self.category, board.features)
        if confidence < self.min_confidence:
            return False
        board.post(self.name, self.slot, label, confidence * self.weight,
                   {"category": self.category, "expert_confidence": round(confidence, 4),
                    "weight": round(self.weight, 4)})
        return True


class ConstraintContributor:
    """A second-round reviser that uses what the others wrote.

    This is the part a plain loop over experts cannot do. Given a rule about
    which slot combinations are possible, it revises the *least* confident slot
    of an impossible reading rather than rejecting the whole thing — a weak
    reading corrected by a strong one is the cheapest kind of composition there
    is, and it needs the other results to already be on the board.

    Args:
        name: Identifier written into contributions.
        allowed: Returns True if a full reading is possible.
        options: ``slot -> candidate values`` to try when repairing.
    """

    def __init__(
        self,
        name: str,
        allowed: Callable[[dict[str, str]], bool],
        options: dict[str, list[str]],
    ) -> None:
        self.name = name
        self.allowed = allowed
        self.options = {k: list(v) for k, v in options.items()}

    def applies(self, board: Blackboard) -> bool:
        """Only once every constrained slot has been filled, and only if wrong."""
        reading = board.reading()
        if any(slot not in reading for slot in self.options):
            return False
        if self.allowed(reading):
            return False
        return not any(e.source == self.name for e in board.contributions())

    def contribute(self, board: Blackboard) -> bool:
        """Repair the weakest slot, if a repair exists."""
        reading = board.reading()
        weakest = min(
            (slot for slot in self.options if board.best(slot)),
            key=lambda slot: board.best(slot).confidence,
            default=None,
        )
        if weakest is None:
            return False
        current = board.best(weakest)
        for candidate in self.options[weakest]:
            if candidate == current.value:
                continue
            trial = dict(reading)
            trial[weakest] = candidate
            if self.allowed(trial):
                board.post(
                    self.name, weakest, candidate,
                    # Just above what it replaces: the repair is better
                    # supported than the reading it corrects, but it is an
                    # inference and should not outrank direct evidence.
                    min(1.0, current.confidence + 1e-3),
                    {"repaired": current.value, "rule": self.name},
                )
                return True
        return False


def run_blackboard(
    board: Blackboard, contributors: list[Any], max_rounds: int = 3
) -> Blackboard:
    """Run ``contributors`` over ``board`` until nothing new is written.

    Args:
        board: The workspace.
        contributors: Anything satisfying :class:`Contributor`.
        max_rounds: Hard stop, so a pair of contributors that keep revising each
            other cannot spin forever.

    Returns:
        The same board, for chaining.
    """
    for _ in range(max(1, max_rounds)):
        board.round += 1
        wrote = False
        for contributor in contributors:
            if contributor.applies(board) and contributor.contribute(board):
                wrote = True
        if not wrote:
            break
    return board


def compose(
    features: list[float], contributors: list[Any],
    meta: dict | None = None, max_rounds: int = 3,
) -> Blackboard:
    """Convenience: build a board, run it, return it."""
    return run_blackboard(Blackboard(features, meta), contributors, max_rounds)
