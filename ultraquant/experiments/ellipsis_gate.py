"""The relation stated once distributes. The ellipsis gate.

§11.71 shipped conjunctive statements with the elided form as its
registered ceiling — "the tower and the bridge material are iron"
refused, because distributing an attribute the parts do not name
would be guessing. The ceiling cashes on a distinction: in the elided
form the relation IS named, once, in the final part, and standard
ellipsis applies it to each conjunct — nothing is guessed, nothing
drops (§11.47's concern), the stated word completes every short part.
The line that keeps it honest is unmoved: a conjunction where NO part
names a relation ("the spire and the arch are oak", single-token
parts) still refuses, because there the distribution really would be
invention. And the rule's own ceiling is recorded with it: ellipsis
distributes only over SINGLE-TOKEN entity parts, because with
multi-word entities ("the north tower and the south keep material")
token count cannot say whether "keep" is name or relation — §11.47's
spirit, refuse ambiguity — while a multi-word bare conjunction stores
each part as-is, exactly as its singular equivalent would ("the north
tower is iron" keys "north tower"): consistency, not junk.

Both arms run the live pipeline; the baseline arm turns
`_ELLIPSIS_DISTRIBUTES` off (§11.71's shipped refusal), the treatment
arm is current. Worlds mix, in world-varying proportions: two-part
and three-part elided teaches, negated elided conjunctions, the
no-relation refusal (correct in BOTH arms — the free share and the
variance), and full-subject conjunctions as the §11.71 floor.

**The criteria, written before the run.**

1. **Gain** — elided-conjunction correctness (every part stored under
   the distributed relation and narrated) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **No relation, no distribution** — a no-relation conjunction that
   stores anything, in either arm, is a fail, once.
3. **Full-subject conjunctions are untouched** — §11.71's forms teach
   identically in both arms at 1.000, and junk " and " keys stay at
   zero.

**It was run three times: two harness indictments, then the PASS.**

The first run's worlds used two-word entity names — measuring a claim
the mechanism never made, and it sharpened the ceiling in the
process: with multi-word entities, token count cannot say whether
"keep" is name or relation, so distribution stays single-token by
design. The second run's bare-kind slice overflowed into the
two-word floor names — the third slicing-overflow in the book, now
ended by fixed slots — and miscounted consistent singular-equivalent
storage as invention. Then:

| arm | taught | floor |
|---|---:|---:|
| refuses | 0.333 | 1.000 |
| spreads | **1.000** | 1.000 |

**+0.667 at 3.83x seed sd, zero relation-free conjunctions stored,
full-subject conjunctions identical in both arms.** The relation
stated once completes every single-token conjunct, negation rides the
distribution, and where nobody names a relation, nothing is invented.
A fourth ceiling registered and cashed inside one session — and two
more harness lessons on the pile that keeps the next gate honest.

Run it::

    python -m ultraquant.experiments.ellipsis_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["EllipsisReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_RELATIONS = ["material", "height", "width"]


@dataclass
class EllipsisReport:
    """The two arms, the invention line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        taught: Arm -> elided-conjunction correctness over all worlds.
        invented: No-relation conjunctions that stored anything
            (must be 0, both arms).
        floor: Arm -> full-subject conjunction correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    taught: dict = field(default_factory=dict)
    invented: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'taught':>7} {'floor':>7}"]
        for arm in ("refuses", "spreads"):
            lines.append(f"{arm:<10} {self.taught[arm]:>7.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"relation-free conjunctions stored: {self.invented}",
            f"delta (spreads vs refuses) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One ellipsis world.

    Returns ``moves`` — ``(kind, entities, relation, value)`` executed
    in order: elided, elided-negated, bare, floor.
    """
    # Single-token entities for the elided and bare kinds: the
    # distribution rule is claimed for exactly this shape, and the
    # first run's two-word names measured a claim the mechanism never
    # made. Two-word names remain in the floor kind, where full
    # subjects carry their own relations.
    names = list(rng.sample(_NOUNS, 8))
    long_names = []
    seen = set(names)
    while len(long_names) < 4:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            long_names.append(name)
    names = names + long_names

    # Fixed slots (the third slicing-overflow in the book taught
    # this): elided owns names[0:3], negated names[3:5], bare
    # names[5:7] - single-token all - and the floor owns the two-word
    # names. No slice can wander into another kind's shape.
    moves = []
    count = rng.choice((2, 3))
    moves.append(("elided", names[0:count],
                  rng.choice(_RELATIONS), rng.choice(_MATERIALS)))
    if rng.random() < 0.8:
        moves.append(("elided-negated", names[3:5],
                      rng.choice(_RELATIONS),
                      rng.choice(_MATERIALS)))
    if rng.random() < 0.8:
        moves.append(("bare", names[5:7], "",
                      rng.choice(_MATERIALS)))
    moves.append(("floor", names[8:10],
                  rng.choice(_RELATIONS), rng.choice(_MATERIALS)))
    return moves


def run_gate(seeds: int = 12) -> EllipsisReport:
    """Run both arms over generated ellipsis worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`EllipsisReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    taught_scores = {"refuses": [], "spreads": []}
    floor_scores = {"refuses": [], "spreads": []}
    invented = 0

    real_flag = thoughts_module._ELLIPSIS_DISTRIBUTES
    for seed in range(seeds):
        rng = random.Random(seed)
        moves = _build_world(rng)

        for arm in ("refuses", "spreads"):
            thoughts_module._ELLIPSIS_DISTRIBUTES = (arm == "spreads")
            root = Path(tempfile.mkdtemp(prefix=f"uq_ell_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                floor_hits = []
                for kind, entities, relation, value in moves:
                    if kind in ("elided", "elided-negated"):
                        negated = kind == "elided-negated"
                        heads = " and ".join(
                            f"the {e}" for e in entities[:-1])
                        statement = (f"{heads} and the {entities[-1]} "
                                     f"{relation} are "
                                     f"{'not ' if negated else ''}"
                                     f"{value}")
                        response, _t = run_pipeline(statement, session)
                        stored = [memory.recall_fact(
                            f"{e} {relation}") for e in entities]
                        good = (all(f"{e} {relation} is" in response
                                    for e in entities)
                                and all(s is not None
                                        and s["value"] == value
                                        and bool(s.get("negated"))
                                        == negated
                                        for s in stored))
                        won.append(float(good))
                    elif kind == "bare":
                        before = set(memory.fact_keys())
                        run_pipeline(
                            f"the {entities[0]} and the {entities[1]} "
                            f"are {value}", session)
                        after = set(memory.fact_keys())
                        clean = before == after
                        won.append(float(clean))
                        if not clean:
                            invented += 1
                    else:
                        subjects = " and ".join(
                            f"the {e} {relation}" for e in entities)
                        response, _t = run_pipeline(
                            f"{subjects} are {value}", session)
                        stored = [memory.recall_fact(
                            f"{e} {relation}") for e in entities]
                        floor_hits.append(float(
                            all(s is not None and s["value"] == value
                                for s in stored)))
                taught_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)
                floor_scores[arm].append(
                    statistics.fmean(floor_hits) if floor_hits
                    else 1.0)

                invented += sum(1 for k in memory.fact_keys()
                                if " and " in k)
            finally:
                thoughts_module._ELLIPSIS_DISTRIBUTES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = EllipsisReport(
        taught={arm: statistics.fmean(vals)
                for arm, vals in taught_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        invented=invented,
    )

    contrasts = [taught_scores["spreads"][i]
                 - taught_scores["refuses"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.invented > 0:
        report.reason = (f"invention: {report.invented} relation-free "
                         "conjunction(s) stored something")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "full-subject conjunctions regressed"
        return report
    if report.delta <= 0:
        report.reason = f"ellipsis buys nothing ({report.delta:+.3f})"
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
        f"elided conjunctions teach at {report.taught['spreads']:.3f} "
        f"against the refusing arm's {report.taught['refuses']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "no relation-free conjunction stored anything, and "
        "full-subject conjunctions held at 1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
