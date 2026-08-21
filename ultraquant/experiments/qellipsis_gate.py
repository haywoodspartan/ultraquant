"""One grammar for teaching and asking. The question-ellipsis gate.

§11.72 let the relation stated once distribute across a conjunctive
STATEMENT; the question side had the same gap — "what is the tower
and the bridge material?" hedged with a junk curiosity while its
full-subject form answered both parts. §11.73 applies the rule
verbatim to `_compound_parts`: the final part's trailing relation
completes every single-token question part, the same two lines hold
(no relation anywhere refuses; multi-word entities stay the ambiguity
ceiling), and teaching and asking finally share one grammar — a
sentence that stores two facts can be asked back in the same shape.

Building it caught a pre-existing §11.48 gap: the compound formatter
predates polarity and asserted a denied value positively ("dome
material is steel" for a stored "not steel") the moment the ellipsis
routed a denial through it. The formatter speaks polarity now, and
the gate holds it at zero tolerance.

Both arms run the live pipeline; the baseline arm turns
`_QUESTION_ELLIPSIS` off (the hedge), the treatment arm is current.
Worlds mix, in world-varying proportions: two- and three-part elided
questions, elided questions over a denial (polarity spoken), the
no-relation form (hedged in BOTH arms — the free share and the
variance), full-subject compounds and the arithmetic combine as
floors.

**The criteria, written before the run.**

1. **Gain** — elided-question correctness (every part answered beside
   its key, polarity included) must beat the baseline arm by more
   than the seed-to-seed sd.
2. **A denial never reads positive** — a compound answer carrying a
   denied value without its "not", in either arm, is a fail, once.
3. **The floors hold** — full-subject compounds and named-operand
   combines answer identically in both arms at 1.000.

**It was run once, and PASSED — perfectly on the claim, zero on the
line.**

| arm | asked | floor |
|---|---:|---:|
| hedges | 0.278 | 1.000 |
| asks | **1.000** | 1.000 |

**+0.722 at 4.04x seed sd, zero denials rendered bare, both floors at
1.000 in both arms.** Every elided question answered every part
beside its key, polarity spoken where a part was a denial, the
no-relation form hedged in both arms, and the arithmetic combine
never disturbed. Teaching and asking share one grammar now — a
sentence that stores two facts can be asked back in the same shape —
and the §11.48 gap the build surfaced (the compound formatter
asserting a denied value positively) is closed at zero tolerance
where it was found.

Run it::

    python -m ultraquant.experiments.qellipsis_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["QEllipsisReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]


@dataclass
class QEllipsisReport:
    """The two arms, the polarity line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        asked: Arm -> elided-question correctness over all worlds.
        bare_denials: Denied values rendered without their "not"
            (must be 0, both arms).
        floor: Arm -> full-subject and combine correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    asked: dict = field(default_factory=dict)
    bare_denials: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'asked':>7} {'floor':>7}"]
        for arm in ("hedges", "asks"):
            lines.append(f"{arm:<10} {self.asked[arm]:>7.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"denials rendered bare: {self.bare_denials}",
            f"delta (asks vs hedges) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One question-ellipsis world with fixed name slots.

    Returns ``moves`` — ``(kind, entities, values)`` executed in
    order: elided, elided-denial, bare, floor, combine.
    """
    names = list(rng.sample(_NOUNS, 9))

    moves = []
    count = rng.choice((2, 3))
    values = rng.sample(_MATERIALS, count)
    moves.append(("elided", names[0:count], values))
    if rng.random() < 0.8:
        moves.append(("elided-denial", names[3:5],
                      rng.sample(_MATERIALS, 2)))
    if rng.random() < 0.8:
        moves.append(("bare", names[5:7], []))
    moves.append(("floor", names[7:9], rng.sample(_MATERIALS, 2)))
    return moves


def run_gate(seeds: int = 12) -> QEllipsisReport:
    """Run both arms over generated question-ellipsis worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`QEllipsisReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    asked_scores = {"hedges": [], "asks": []}
    floor_scores = {"hedges": [], "asks": []}
    bare_denials = 0

    real_flag = thoughts_module._QUESTION_ELLIPSIS
    for seed in range(seeds):
        rng = random.Random(seed)
        moves = _build_world(rng)

        for arm in ("hedges", "asks"):
            thoughts_module._QUESTION_ELLIPSIS = (arm == "asks")
            root = Path(tempfile.mkdtemp(prefix=f"uq_qell_{seed}_"))
            try:
                session = build_session(root, seed=seed)

                won = []
                floor_hits = []
                for kind, entities, values in moves:
                    if kind == "elided":
                        for e, v in zip(entities, values):
                            run_pipeline(f"the {e} material is {v}",
                                         session)
                        heads = " and ".join(
                            f"the {e}" for e in entities[:-1])
                        response, _t = run_pipeline(
                            f"what is {heads} and the {entities[-1]} "
                            "material?", session)
                        won.append(float(all(
                            f"{e} material is {v}" in response
                            for e, v in zip(entities, values))))
                    elif kind == "elided-denial":
                        pos, neg = entities
                        pv, nv = values
                        run_pipeline(f"the {pos} material is {pv}",
                                     session)
                        run_pipeline(f"the {neg} material is not {nv}",
                                     session)
                        response, _t = run_pipeline(
                            f"what is the {pos} and the {neg} "
                            "material?", session)
                        good = (f"{pos} material is {pv}" in response
                                and f"{neg} material is not {nv}"
                                in response)
                        won.append(float(good))
                        if (f"{neg} material is {nv}" in response
                                and f"not {nv}" not in response):
                            bare_denials += 1
                    elif kind == "bare":
                        response, _t = run_pipeline(
                            f"what is the {entities[0]} and the "
                            f"{entities[1]}?", session)
                        answered = " is " in response and \
                            response.count(";") >= 1
                        won.append(float(not answered))
                    else:
                        a, b = entities
                        va, vb = values
                        run_pipeline(f"the {a} material is {va}",
                                     session)
                        run_pipeline(f"the {b} material is {vb}",
                                     session)
                        response, _t = run_pipeline(
                            f"what is the {a} material and the {b} "
                            "material?", session)
                        floor_hits.append(float(
                            f"{a} material is {va}" in response
                            and f"{b} material is {vb}" in response))
                        run_pipeline(f"the {a} height is 300 meters",
                                     session)
                        run_pipeline(f"the {b} height is 120 meters",
                                     session)
                        response, _t = run_pipeline(
                            f"what is the sum of the {a} height and "
                            f"the {b} height?", session)
                        floor_hits.append(float("420" in response))
                asked_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)
                floor_scores[arm].append(
                    statistics.fmean(floor_hits) if floor_hits
                    else 1.0)
            finally:
                thoughts_module._QUESTION_ELLIPSIS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = QEllipsisReport(
        asked={arm: statistics.fmean(vals)
               for arm, vals in asked_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        bare_denials=bare_denials,
    )

    contrasts = [asked_scores["asks"][i] - asked_scores["hedges"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.bare_denials > 0:
        report.reason = (f"a denial read positive {report.bare_denials} "
                         "time(s) in a compound answer")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = ("full-subject compounds or the combine "
                         "regressed")
        return report
    if report.delta <= 0:
        report.reason = f"the mirror buys nothing ({report.delta:+.3f})"
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
        f"elided questions answer at {report.asked['asks']:.3f} "
        f"against the hedging arm's {report.asked['hedges']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "no denial read positive, and both floors held at 1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
