"""Aggregates over the family: count, total, average. The gate.

§11.56 enumerated the attribute family to rank it; this unit
aggregates the same enumeration: "how many height facts do you hold?",
"what is the total height?", "what is the average height?". The
honesty contract is inherited whole — scope named ("of the 3 height
facts I hold"), exclusions named, units converted where §11.42's table
connects them and the conversion said aloud — and one line matters
above the rest: **denials never enter arithmetic.** A denial names no
number, and a denied 900 meters must not move a total or a mean by a
millimeter. Counting tells the whole truth the other way around:
positive beliefs are counted (word-valued facts included — a fact is a
fact), and denials are reported beside them as what they are ("I hold
4 height fact(s), plus 1 denial(s)").

Both arms run the live pipeline; the baseline arm turns
`_AGGREGATES_ON` off (aggregate questions fall to the hedging
machinery), the treatment arm is current. Worlds mix, in world-varying
proportions: counts with and without denials, single-unit totals,
cross-unit totals and averages, empty families (the baseline's free
share, and the variance), and worlds holding a LARGE negated value
whose inclusion would visibly shift the arithmetic — the drift
control. The §11.30 pairwise combine ("what is the sum of the tower
height and the bridge height?") must answer identically in both arms:
the aggregate door may not shadow the named-operands door.

**The criteria, written before the run.**

1. **Gain** — aggregate correctness (exact count with its denial note,
   exact total and average with scope and conversion marks, empty
   families refused) must beat the baseline arm by more than the
   seed-to-seed sd.
2. **A denial never drifts the arithmetic** — every delivered total
   and average must equal the positive-only value exactly. One drifted
   number is a fail.
3. **The pairwise door is untouched** — named-operand combine
   questions answer correctly in BOTH arms at 1.000, and direct recall
   is 1.000.

**It was run twice — once before and once after a harness hardening —
and PASSED both times.**

Between the runs, a latent harness hazard was closed before it ever
fired: the drift control demands exact float-string equality, and the
builder's ``km + sum(meters)/1000`` could differ in the last bits from
the mechanism's per-member ``Σ mᵢ × 0.001`` on unlucky values — a
spurious drift waiting on a seed. Cross-unit worlds now draw meter
values that convert to exact binary eighths, so summation order
cannot manufacture a phantom drift. The authoritative run, on the
hardened worlds:

| arm | aggregated | pairwise | recall |
|---|---:|---:|---:|
| hedges | 0.343 | 1.000 | 1.000 |
| aggregates | **1.000** | 1.000 | 1.000 |

**+0.657 at 5.05x seed sd, zero drifted numbers** — every count exact
with its denial note, every total and average exact with its scope and
conversion mark, empty families refused, no denied value moved any
number — and the pairwise combine door and recall identical at 1.000
in both arms. The enumeration §11.56 built now ranks AND reckons, and
the same three honesty rules govern both: name the scope, name the
exclusions, and never let a denial touch a number.

Run it::

    python -m ultraquant.experiments.aggregate_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AggregateReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]


@dataclass
class AggregateReport:
    """The two arms, the drift control, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        aggregated: Arm -> aggregate correctness over all worlds.
        drifted: Delivered totals/averages that departed from the
            positive-only value (must be 0).
        pairwise: Arm -> named-operand combine correctness (1.0 both).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    aggregated: dict = field(default_factory=dict)
    drifted: int = 0
    pairwise: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'aggregated':>11} {'pairwise':>9} "
                 f"{'recall':>7}"]
        for arm in ("hedges", "aggregates"):
            lines.append(
                f"{arm:<10} {self.aggregated[arm]:>11.3f} "
                f"{self.pairwise[arm]:>9.3f} {self.recall[arm]:>7.3f}")
        lines += [
            f"drifted numbers (treatment): {self.drifted}",
            f"delta (aggregates vs hedges) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One aggregate world.

    Returns ``(statements, questions, pair_q, directs)`` where
    questions are ``(question, kind, expected)`` — kind in count/
    count-denied/sum/mean/conv-sum/empty; expected the exact string a
    correct answer must contain (None for empty).
    """
    names = []
    seen = set()
    while len(names) < 9:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    km = 0
    if rng.random() < 0.5:
        km = rng.randrange(2, 9)
        statements.append(f"the {names[3]} height is {km} kilometers")
    meters = []
    for entity in names[:3]:
        # In cross-unit worlds the meter values are multiples of 125:
        # 125 * 0.001 rounds to an exact binary eighth, so the
        # mechanism's per-member conversion and this builder's
        # sum-then-divide agree to the last bit and the exact-string
        # drift check cannot fire on float summation order.
        value = (rng.choice((125, 250, 375, 500, 625, 750)) if km
                 else rng.randrange(10, 800))
        meters.append(value)
        statements.append(f"the {entity} height is {value} meters")

    denied = rng.random() < 0.7
    if denied:
        statements.append(
            f"the {names[4]} height is not "
            f"{rng.randrange(900, 999)} meters")

    widths = []
    for entity in names[5:7]:
        value = rng.randrange(10, 800)
        widths.append(value)
        statements.append(f"the {entity} width is {value} meters")

    questions = []
    count = 3 + (1 if km else 0)
    kind = "count-denied" if denied else "count"
    expected = (f"I hold {count} height fact(s), plus 1 denial(s)"
                if denied else f"I hold {count} height fact(s)")
    questions.append(("how many height facts do you hold?", kind,
                      expected))

    if km:
        total_km = km + sum(meters) / 1000.0
        questions.append(("what is the total height?", "conv-sum",
                          f"{total_km:g} kilometers"))
        if rng.random() < 0.6:
            mean_km = total_km / count
            questions.append(("what is the average height?", "mean",
                              f"{mean_km:g} kilometers"))
    else:
        questions.append(("what is the total height?", "sum",
                          f"{sum(meters):g} meters"))
        if rng.random() < 0.6:
            questions.append(("what is the average height?", "mean",
                              f"{sum(meters) / 3:g} meters"))
    if rng.random() < 0.6:
        questions.append(("what is the total width?", "sum",
                          f"{sum(widths):g} meters"))
    if rng.random() < 0.7:
        questions.append(("what is the total depth?", "empty", None))
    if rng.random() < 0.7:
        questions.append(("how many depth facts do you hold?",
                          "empty", None))

    a, b = names[5], names[6]
    pair_q = (f"what is the sum of the {a} width and the {b} width?",
              f"{sum(widths):g}")

    directs = []
    for spot, entity in enumerate(rng.sample(names[:3], 2)):
        directs.append((f"what is the {entity} height?",
                        str(meters[names.index(entity)])))

    rng.shuffle(statements)
    return statements, questions, pair_q, directs


def run_gate(seeds: int = 12) -> AggregateReport:
    """Run both arms over generated aggregate worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`AggregateReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    agg_scores = {"hedges": [], "aggregates": []}
    pair_scores = {"hedges": [], "aggregates": []}
    recall_scores = {"hedges": [], "aggregates": []}
    drifted = 0

    real_flag = thoughts_module._AGGREGATES_ON
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, questions, pair_q, directs = _build_world(rng)

        for arm in ("hedges", "aggregates"):
            thoughts_module._AGGREGATES_ON = (arm == "aggregates")
            root = Path(tempfile.mkdtemp(prefix=f"uq_agg_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind, expected in questions:
                    response, _trace = run_pipeline(question, session)
                    if kind == "empty":
                        good = ("I hold no" in response
                                or response.startswith("I don't hold"))
                        won.append(float(good))
                        continue
                    good = expected in response
                    if kind in ("sum", "conv-sum", "mean"):
                        good = good and "facts I hold" in response
                        if (arm == "aggregates"
                                and "facts I hold" in response
                                and expected not in response):
                            drifted += 1
                    won.append(float(good))
                agg_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                question, expected = pair_q
                response, _trace = run_pipeline(question, session)
                pair_scores[arm].append(
                    float(expected in response
                          and "inferred" in response))

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._AGGREGATES_ON = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = AggregateReport(
        aggregated={arm: statistics.fmean(vals)
                    for arm, vals in agg_scores.items()},
        pairwise={arm: statistics.fmean(vals)
                  for arm, vals in pair_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        drifted=drifted,
    )

    contrasts = [agg_scores["aggregates"][i] - agg_scores["hedges"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.drifted > 0:
        report.reason = (
            f"the arithmetic drifted {report.drifted} time(s): a "
            "delivered number departed from the positive-only value")
        return report
    if min(report.pairwise.values()) < 1.0:
        report.reason = ("the aggregate door shadowed the pairwise "
                         "door: named-operand combines regressed")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"aggregating buys nothing ({report.delta:+.3f})"
        return report
    if report.seed_sd <= 0.0:
        report.reason = (f"every world gave the same contrast "
                         f"({report.delta:+.3f}); nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {report.delta:+.3f} is "
                         f"{report.margin_ratio:.2f}x the seed sd "
                         f"({report.seed_sd:.3f})")
        return report
    report.passes = True
    report.reason = (
        f"aggregates answer at {report.aggregated['aggregates']:.3f} "
        f"against the hedging arm's "
        f"{report.aggregated['hedges']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), zero drifted numbers, "
        "and the pairwise door and recall held at 1.000 in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
