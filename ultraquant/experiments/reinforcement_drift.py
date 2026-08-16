"""What does reinforcing the router's own top route actually do?

`thoughts.py`'s Learn step reinforces whichever category the pipeline routed to,
unconditionally, at `delta=0.05`. Nothing checks the route was right — so a
wrong route teaches itself, and the next identical query is claimed more
decisively by the same wrong category.

That was inferred from the deployed library, where `crosses` had learned the
token ``number`` and beat `arithmetic` on "add these two numbers". This measures
it, and the measurement corrects the obvious guess about what the harm is.

**The harm is not that accuracy falls.** It does not: the queries that were
misrouted were already misrouted, and reinforcement cannot make a wrong answer
wronger. Measured over 60 rounds, accuracy is **0.400 before and 0.400 after**,
with or without reinforcement.

**The harm is that errors become resistant to correction.** The margin by which
the wrong category beats the right one grows **0.60 → 2.33** when the router
reinforces itself, and stays flat at 0.60 when it does not. Nearly four times
harder to overturn, from doing nothing but routing normally.

That is what makes it matter, and it explains something otherwise puzzling:
plural folding gave `arithmetic` a full point of base overlap on the deployed
library and *still* lost, because `crosses` had entrenched to 1.75 while
`arithmetic` reached 1.10. The fix was real and arrived after the error had
already been reinforced past it.

**Why this is not fixed here.** Correcting it needs a signal for whether a route
was right, and the pipeline has none: the router's own choice is the only thing
available at that point, which is precisely the circularity. Supervised calls
are fine and are left alone — ``learning.py`` and ``selflearn.py`` reinforce
from a *user's* answer, which is independent evidence. The unsupervised call at
``thoughts.py:801`` is the one with no ground truth behind it, and choosing what
should replace it is a design question with its own gate, not a patch.

Run it::

    python -m ultraquant.experiments.reinforcement_drift
"""

from __future__ import annotations

import random
import statistics
import tempfile
from dataclasses import dataclass

__all__ = ["DriftReport", "measure"]

#: A thin category competing with a rich one — the deployed situation, where
#: ``arithmetic`` effectively had one usable keyword for plural queries while
#: ``crosses`` had many plus accumulated weight.
_CATEGORIES = {
    "crosses": ["cross", "plus", "star", "intersection", "two", "pair", "mark"],
    "arithmetic": ["sum"],
    "frames": ["frame", "square", "box"],
}

#: Queries with a known correct category. Three of the five are ones the router
#: gets wrong from the start, which is the point: reinforcement acts on errors
#: that already exist.
_QUERIES = [
    ("add these two numbers", "arithmetic"),
    ("the total of two values", "arithmetic"),
    ("subtract two amounts", "arithmetic"),
    ("a cross and a plus", "crosses"),
    ("a square box", "frames"),
]


@dataclass
class DriftReport:
    """What self-reinforcement did.

    Attributes:
        accuracy_before: Routing accuracy at the start.
        accuracy_self: Accuracy after reinforcing the router's own choices.
        accuracy_control: Accuracy after the same rounds with no reinforcement.
        margin_before: How decisively the top answer beat the correct one.
        margin_self: The same, after self-reinforcement. This is the finding.
        margin_control: The same, without. Should not move.
    """

    accuracy_before: float = 0.0
    accuracy_self: float = 0.0
    accuracy_control: float = 0.0
    margin_before: float = 0.0
    margin_self: float = 0.0
    margin_control: float = 0.0

    @property
    def entrenchment(self) -> float:
        """How much harder the error became to overturn."""
        return (self.margin_self / self.margin_before
                if self.margin_before else 0.0)

    def as_text(self) -> str:
        """A printable report."""
        return "\n".join([
            f"accuracy   before {self.accuracy_before:.3f}   "
            f"self-reinforcing {self.accuracy_self:.3f}   "
            f"control {self.accuracy_control:.3f}",
            f"wrong-answer margin   before {self.margin_before:.2f}   "
            f"self-reinforcing {self.margin_self:.2f}   "
            f"control {self.margin_control:.2f}",
            f"-> accuracy is unchanged; the error became "
            f"{self.entrenchment:.1f}x harder to overturn",
        ])


def _build():
    from ultraquant.shards.router import CategoryRouter
    from ultraquant.shards.vault import ShardVault

    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_drift_")))
    for name, keywords in _CATEGORIES.items():
        router.register(name, keywords)
    return router


def _accuracy(router) -> float:
    hit = 0
    for query, truth in _QUERIES:
        ranked = router.route(query, top_k=1)
        if ranked and ranked[0][0] == truth:
            hit += 1
    return hit / len(_QUERIES)


def _margin(router) -> float:
    """Mean gap between the winning score and the correct category's score."""
    gaps = []
    for query, truth in _QUERIES:
        ranked = router.route(query, top_k=len(_CATEGORIES))
        scores = dict(ranked)
        gaps.append(scores.get(ranked[0][0], 0.0) - scores.get(truth, 0.0)
                    if ranked else 0.0)
    return statistics.fmean(gaps)


def _run(seed: int, reinforce: bool, rounds: int) -> tuple[float, float]:
    rng = random.Random(seed)
    router = _build()
    for _ in range(rounds):
        query, _truth = rng.choice(_QUERIES)
        ranked = router.route(query, top_k=1)
        if ranked and reinforce:
            # Exactly what thoughts.py:801 does after every turn.
            router.learn(query, ranked[0][0], delta=0.05)
    return _accuracy(router), _margin(router)


def measure(seeds: int = 12, rounds: int = 60) -> DriftReport:
    """Measure entrenchment with and without self-reinforcement.

    Args:
        seeds: Query orderings to average over.
        rounds: Turns of ordinary use per ordering.

    Returns:
        The :class:`DriftReport`.
    """
    fresh = _build()
    report = DriftReport(accuracy_before=_accuracy(fresh),
                         margin_before=_margin(fresh))
    for reinforce in (True, False):
        accuracies, margins = [], []
        for seed in range(seeds):
            accuracy, margin = _run(seed, reinforce, rounds)
            accuracies.append(accuracy)
            margins.append(margin)
        if reinforce:
            report.accuracy_self = statistics.fmean(accuracies)
            report.margin_self = statistics.fmean(margins)
        else:
            report.accuracy_control = statistics.fmean(accuracies)
            report.margin_control = statistics.fmean(margins)
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(measure().as_text())
