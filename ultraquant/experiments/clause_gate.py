"""Two statements joined are two statements. The clause gate.

"the tower material is iron and the bridge material is steel" hit the
first " is " split and stored the ENTIRE second clause inside the
first value — 'tower material' = "iron and the bridge material is
steel" — corruption narrated as success, the second fact drowned
where no question would ever find it. §11.74 splits clauses only
where BOTH sides independently parse as statements, so the fix cannot
overreach: "the motto is live and learn" never splits, because its
right side carries no verb, and a value containing "and" survives
whole. Each clause then runs the complete statement machinery —
polarity per clause, revision notices per clause — through the same
plumbing §11.71 built.

Both arms run the live pipeline; the baseline arm turns
`_CLAUSE_CONJUNCTIONS` off (the shipped corruption), the treatment
arm is current. Worlds mix, in world-varying proportions: two- and
three-clause statements, mixed-polarity clauses, and-valued single
statements (the integrity floor, identical in both arms), and
singular statements.

**The criteria, written before the run.**

1. **Gain** — clause correctness (every clause stored under its own
   key with its own value and narrated) must beat the baseline arm by
   more than the seed-to-seed sd.
2. **No value swallows a statement** — a stored value containing
   " is " in the treatment arm is a fail, once. (The baseline's count
   is the shipped corruption, measured.)
3. **A value containing "and" survives whole** — an and-valued
   statement split into pieces, in either arm, is a fail, once.
4. **Singular statements are untouched** at 1.000 in both arms.

**It was run once, and PASSED — perfectly on the claim, zero on
both lines.**

| arm | taught | floor |
|---|---:|---:|
| swallows | 0.361 | 1.000 |
| splits | **1.000** | 1.000 |

**+0.639 at 4.59x seed sd, zero statements swallowed, zero and-values
split — while the baseline swallowed 20**, the shipped corruption
measured beside the fix. Every clause stored under its own key with
its own polarity, "live and learn" survived whole every time, and
singulars never moved. The statement grammar is complete in both
directions now: subjects conjoin (§11.71), relations distribute
(§11.72), questions mirror (§11.73), and clauses split (§11.74) —
four forms, one machinery, and no shape of "and" left that silently
loses or corrupts a fact.

Run it::

    python -m ultraquant.experiments.clause_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ClauseReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_AND_VALUES = ["live and learn", "slow and steady", "safe and sound"]


@dataclass
class ClauseReport:
    """The two arms, the two integrity lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        taught: Arm -> clause correctness over all worlds.
        swallowed: Stored values containing " is " in treatment
            (must be 0; baseline count is the measured corruption).
        baseline_swallowed: The shipped corruption's count, reported.
        split_values: And-valued statements split (must be 0, both).
        floor: Arm -> singular correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    taught: dict = field(default_factory=dict)
    swallowed: int = 0
    baseline_swallowed: int = 0
    split_values: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'taught':>7} {'floor':>7}"]
        for arm in ("swallows", "splits"):
            lines.append(f"{arm:<10} {self.taught[arm]:>7.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"statements swallowed into values (treatment): "
            f"{self.swallowed} (baseline: {self.baseline_swallowed})",
            f"and-valued statements split: {self.split_values}",
            f"delta (splits vs swallows) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One clause world with fixed name slots.

    Returns ``moves`` — ``(kind, entities, values)`` in order.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    moves = []
    count = rng.choice((2, 3))
    moves.append(("clauses", names[0:count],
                  rng.sample(_MATERIALS, count)))
    if rng.random() < 0.8:
        moves.append(("mixed", names[3:5],
                      rng.sample(_MATERIALS, 2)))
    if rng.random() < 0.8:
        moves.append(("and-value", [names[5]],
                      [rng.choice(_AND_VALUES)]))
    moves.append(("single", [names[6]], [rng.choice(_MATERIALS)]))
    return moves


def run_gate(seeds: int = 12) -> ClauseReport:
    """Run both arms over generated clause worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ClauseReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    taught_scores = {"swallows": [], "splits": []}
    floor_scores = {"swallows": [], "splits": []}
    swallowed = 0
    baseline_swallowed = 0
    split_values = 0

    real_flag = thoughts_module._CLAUSE_CONJUNCTIONS
    for seed in range(seeds):
        rng = random.Random(seed)
        moves = _build_world(rng)

        for arm in ("swallows", "splits"):
            thoughts_module._CLAUSE_CONJUNCTIONS = (arm == "splits")
            root = Path(tempfile.mkdtemp(prefix=f"uq_cl_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                floor_hits = []
                for kind, entities, values in moves:
                    if kind in ("clauses", "mixed"):
                        negate_last = kind == "mixed"
                        clauses = []
                        for spot, (e, v) in enumerate(
                                zip(entities, values)):
                            last = spot == len(entities) - 1
                            neg = "not " if negate_last and last else ""
                            clauses.append(
                                f"the {e} material is {neg}{v}")
                        response, _t = run_pipeline(
                            " and ".join(clauses), session)
                        good = True
                        for spot, (e, v) in enumerate(
                                zip(entities, values)):
                            record = memory.recall_fact(f"{e} material")
                            last = spot == len(entities) - 1
                            want_neg = negate_last and last
                            good = (good and record is not None
                                    and record["value"] == v
                                    and bool(record.get("negated"))
                                    == want_neg)
                        won.append(float(good))
                    elif kind == "and-value":
                        entity, value = entities[0], values[0]
                        run_pipeline(f"the {entity} motto is {value}",
                                     session)
                        record = memory.recall_fact(f"{entity} motto")
                        whole = (record is not None
                                 and record["value"] == value)
                        won.append(float(whole))
                        if not whole:
                            split_values += 1
                    else:
                        entity, value = entities[0], values[0]
                        response, _t = run_pipeline(
                            f"the {entity} material is {value}",
                            session)
                        record = memory.recall_fact(f"{entity} material")
                        floor_hits.append(float(
                            record is not None
                            and record["value"] == value))
                taught_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)
                floor_scores[arm].append(
                    statistics.fmean(floor_hits) if floor_hits
                    else 1.0)

                for key in memory.fact_keys():
                    value = str(memory.recall_fact(key)["value"])
                    if " is " in value:
                        if arm == "splits":
                            swallowed += 1
                        else:
                            baseline_swallowed += 1
            finally:
                thoughts_module._CLAUSE_CONJUNCTIONS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = ClauseReport(
        taught={arm: statistics.fmean(vals)
                for arm, vals in taught_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        swallowed=swallowed,
        baseline_swallowed=baseline_swallowed,
        split_values=split_values,
    )

    contrasts = [taught_scores["splits"][i]
                 - taught_scores["swallows"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.swallowed > 0:
        report.reason = (f"a value swallowed a statement "
                         f"{report.swallowed} time(s) in the "
                         "treatment arm")
        return report
    if report.split_values > 0:
        report.reason = (f"an and-valued statement was split "
                         f"{report.split_values} time(s)")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "singular statements regressed"
        return report
    if report.delta <= 0:
        report.reason = f"splitting buys nothing ({report.delta:+.3f})"
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
        f"clause statements teach at {report.taught['splits']:.3f} "
        f"against the swallowing arm's "
        f"{report.taught['swallows']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), zero statements "
        "swallowed, zero and-values split - while the baseline "
        f"swallowed {report.baseline_swallowed}, the corruption "
        "measured beside the fix - and singulars held at 1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
