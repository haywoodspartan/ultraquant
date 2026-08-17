"""Can reinforcement stop entrenching errors without stopping learning? A gate.

§11.16 measured the harm precisely: reinforcing the router's own top route does
**not** reduce accuracy, but it grows the margin by which a wrong category beats
the right one from **0.60 to 2.33** — 3.9x harder to overturn, from nothing but
routing normally. That is why §11.15's plural fix arrived too late to rescue the
deployed library.

**The gate, written before any mechanism was tried.**

1. **Entrenchment** — the wrong-answer margin after N rounds of ordinary use must
   be materially lower than the self-reinforcing baseline, by more than the
   seed-to-seed standard deviation.
2. **Learning retained (the control, and it fails loudly)** — reinforcement has
   to still *do* something. A router that never reinforces scores perfectly on
   criterion 1 and is useless: the whole point of "recall reinforces" is that the
   catalog reorganises around what is used. So a *correct* route must still gain
   enough weight to carry a paraphrase it could not route before.

The control is the whole difficulty, exactly as it was for abstention. Not
reinforcing is trivially perfect on the metric being improved, so the metric
alone cannot distinguish a fix from an amputation.

Two quantities are reported and not gated: total learned weight, so a mechanism
that solves this by learning almost nothing is visible; and accuracy, because
§11.16 showed accuracy is *not* where this defect shows up and a reader may
expect it to be.

Run it::

    python -m ultraquant.experiments.reinforcement_gate
"""

from __future__ import annotations

import random
import statistics
import tempfile
from dataclasses import dataclass

__all__ = ["ReinforcementReport", "run_gate"]

#: A thin category against a rich one — the deployed situation.
_CATEGORIES = {
    "crosses": ["cross", "plus", "star", "intersection", "two", "pair", "mark"],
    "arithmetic": ["sum"],
    "frames": ["frame", "square", "box"],
}

#: Queries with a known correct category. Three are misrouted from the start,
#: which is what reinforcement acts on. The last two are routed correctly and
#: each carries a **distinctive token registered nowhere** (``hatching``,
#: ``bordered``) so that reinforcing them is the only way that token can come
#: to mean anything.
_QUERIES = [
    ("add these two numbers", "arithmetic"),
    ("the total of two values", "arithmetic"),
    ("subtract two amounts", "arithmetic"),
    ("a cross with hatching", "crosses"),
    ("a square with bordered edges", "frames"),
]

#: The control. Each paraphrase shares **only** the distinctive token with its
#: taught query and **no base keyword at all** with its category, so learned
#: weight is the only thing that can route it.
#:
#: The first version of this control was worthless and passed anyway. Its
#: paraphrases were "crossing marks on the page" and "a boxed square shape",
#: which contain ``mark`` and ``square`` — both *registered* keywords. Verified
#: by running a router with no learning whatsoever: both routed correctly, so
#: the control scored 1.000 for a mechanism that had learned nothing, and the
#: docstring's claim that never-reinforcing would score 0.000 here was simply
#: false. That is the §11.11 defect a third time, inside a gate written after
#: two prior encounters with it.
_PARAPHRASES = [
    ("more hatching over here", "crosses"),
    ("the bordered one please", "frames"),
]


@dataclass
class ReinforcementReport:
    """What a mechanism did to entrenchment and to learning.

    Attributes:
        passes: The verdict under the pre-registered criterion.
        margin_baseline: Wrong-answer margin under unconditional reinforcement.
        margin_fixed: The same under the mechanism being tested.
        delta: ``margin_baseline - margin_fixed`` — reduction, so higher is
            better.
        seed_sd: Seed-to-seed standard deviation of the delta.
        margin_ratio: ``delta / seed_sd``.
        transfer_baseline: Paraphrase accuracy under unconditional
            reinforcement. The control's reference point.
        transfer_fixed: The same under the mechanism. Must not collapse.
        weight_baseline: Total learned weight, baseline. Reported, not gated.
        weight_fixed: The same. A mechanism that learns nothing is visible here.
        accuracy_fixed: Reported because §11.16 showed accuracy is not where
            this defect appears, and a reader will look for it.
        reason: Plain-language verdict.
    """

    passes: bool = False
    margin_baseline: float = 0.0
    margin_fixed: float = 0.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    transfer_baseline: float = 0.0
    transfer_fixed: float = 0.0
    weight_baseline: float = 0.0
    weight_fixed: float = 0.0
    accuracy_fixed: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """A printable report, with the control beside the win."""
        return "\n".join([
            f"wrong-answer margin   {self.margin_baseline:.2f} -> "
            f"{self.margin_fixed:.2f}   reduction {self.delta:+.2f}",
            f"paraphrase transfer (CTRL) {self.transfer_baseline:.3f} -> "
            f"{self.transfer_fixed:.3f}"
            "   <- never reinforcing scores perfectly above and 0.000 here",
            f"seed sd {self.seed_sd:.3f}   margin ratio "
            f"{self.margin_ratio:.2f}x  (gate needs > 1.0)",
            f"learned weight  {self.weight_baseline:.2f} -> "
            f"{self.weight_fixed:.2f}   accuracy {self.accuracy_fixed:.3f}"
            "   (both reported, not gated)",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ])


def _build(**kwargs):
    from ultraquant.shards.router import CategoryRouter
    from ultraquant.shards.vault import ShardVault

    router = CategoryRouter(ShardVault(tempfile.mkdtemp(prefix="uq_rg_")),
                            **kwargs)
    for name, keywords in _CATEGORIES.items():
        router.register(name, keywords)
    return router


def _margin(router) -> float:
    """Mean gap between the winning score and the correct category's."""
    gaps = []
    for query, truth in _QUERIES:
        ranked = router.route(query, top_k=len(_CATEGORIES))
        scores = dict(ranked)
        gaps.append(scores.get(ranked[0][0], 0.0) - scores.get(truth, 0.0)
                    if ranked else 0.0)
    return statistics.fmean(gaps)


def _accuracy(router) -> float:
    hit = sum(1 for q, truth in _QUERIES
              if (router.route(q, top_k=1) or [("", 0)])[0][0] == truth)
    return hit / len(_QUERIES)


def _transfer(router) -> float:
    """Paraphrase accuracy — whether reinforcement still carries anything."""
    hit = sum(1 for q, truth in _PARAPHRASES
              if (router.route(q, top_k=1) or [("", 0)])[0][0] == truth)
    return hit / len(_PARAPHRASES)


def _weight(router) -> float:
    return sum(sum(w.values()) for w in router._learned.values())


def _run(seed: int, rounds: int, confirmed_only: bool) -> tuple:
    """One history of ordinary use.

    ``confirmed_only`` selects the mechanism: when True, a route is reinforced
    only if something downstream *confirmed* it, which here means the routed
    category was the correct one — standing in for the pipeline's
    ``ctx.data["prediction"]`` being non-None. When False this is the current
    behaviour: reinforce whatever was routed to.
    """
    rng = random.Random(seed)
    router = _build()
    for _ in range(rounds):
        query, truth = rng.choice(_QUERIES)
        ranked = router.route(query, top_k=1)
        if not ranked:
            continue
        chosen = ranked[0][0]
        if confirmed_only and chosen != truth:
            continue        # no downstream confirmation: learn nothing
        router.learn(query, chosen, delta=0.05)
    return _margin(router), _transfer(router), _weight(router), _accuracy(router)


def run_gate(seeds: int = 12, rounds: int = 60) -> ReinforcementReport:
    """Run the gate against the confirmation-gated mechanism.

    Args:
        seeds: Query orderings to average over.
        rounds: Turns of ordinary use per ordering.

    Returns:
        The :class:`ReinforcementReport`.
    """
    base_margins, base_transfer, base_weight = [], [], []
    fixed_margins, fixed_transfer, fixed_weight, fixed_acc = [], [], [], []
    deltas = []
    for seed in range(seeds):
        bm, bt, bw, _ba = _run(seed, rounds, confirmed_only=False)
        fm, ft, fw, fa = _run(seed, rounds, confirmed_only=True)
        base_margins.append(bm); base_transfer.append(bt); base_weight.append(bw)
        fixed_margins.append(fm); fixed_transfer.append(ft)
        fixed_weight.append(fw); fixed_acc.append(fa)
        deltas.append(bm - fm)

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: a sample of orderings, not the population of them.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    report = ReinforcementReport(
        margin_baseline=statistics.fmean(base_margins),
        margin_fixed=statistics.fmean(fixed_margins),
        delta=delta, seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        transfer_baseline=statistics.fmean(base_transfer),
        transfer_fixed=statistics.fmean(fixed_transfer),
        weight_baseline=statistics.fmean(base_weight),
        weight_fixed=statistics.fmean(fixed_weight),
        accuracy_fixed=statistics.fmean(fixed_acc),
    )

    # The control decides first: a fix that stops learning is an amputation.
    if report.transfer_fixed < report.transfer_baseline:
        report.reason = (
            f"paraphrase transfer fell {report.transfer_baseline:.3f} -> "
            f"{report.transfer_fixed:.3f}; entrenchment was reduced by learning "
            "less, which is not a fix")
        return report
    if delta <= 0:
        report.reason = f"entrenchment was not reduced ({delta:+.2f})"
        return report
    if seed_sd <= 0.0:
        report.reason = (
            f"every ordering gave the same reduction ({delta:+.2f}), so there "
            "is no variance to measure the margin against")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"reduction {delta:+.2f} is {report.margin_ratio:.2f}x "
                         f"the seed sd ({seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"wrong-answer margin {report.margin_baseline:.2f} -> "
        f"{report.margin_fixed:.2f} ({delta:+.2f}, {report.margin_ratio:.2f}x "
        f"seed sd) with paraphrase transfer held at {report.transfer_fixed:.3f}")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
