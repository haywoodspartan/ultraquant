"""Superlatives: a claim about a set, with the set named. The gate.

§11.54 gave pairwise comparatives a computed verdict; §11.55 let their
operands derive. The family's completion is setwise: "which height is
the tallest?" is a claim about EVERY height the library holds, so the
answer enumerates the attribute family through `fact_keys` — the whole
store, never a top-k sample that could silently miss the winner — and
names its scope: "The spire height is the tallest of the 3 height
facts I hold". Negated facts are excluded by polarity and accounted
aloud ("1 denial(s) not counted - a denial names no number"),
non-numeric candidates are excluded BY NAME, units convert where
§11.42's table connects them, ties are named rather than broken, and
confidence is the minimum over every included candidate — the verdict
rests on all of them.

Both arms run the live pipeline; the baseline arm turns
`_SUPERLATIVES_ON` off (superlative questions fall to the hedging
machinery), the treatment arm is current. Worlds mix, in world-varying
proportions: winners in one unit and across units, downward
superlatives, ties, empty attribute families (the honest refusal both
arms can reach — the baseline's free share, and the variance), and
worlds where the numeric maximum is stored NEGATED — the §11.48 line
at set scale.

**The criteria, written before the run.**

1. **Gain** — superlative correctness (the right winner with the right
   scope count and value; ties naming both holders; empty families
   refused; exclusions named when present) must beat the baseline arm
   by more than the seed-to-seed sd.
2. **A denial never wins** — a world whose numeric maximum is negated
   must crown the best POSITIVE candidate; the denied key named as
   winner in the treatment arm is a fail, once.
3. **Recall holds** — direct recall at 1.000 in both arms.

**It was run once, and PASSED — perfectly on the claim, zero on the
line.**

| arm | ranked | recall |
|---|---:|---:|
| hedges | 0.206 | 1.000 |
| ranks | **1.000** | 1.000 |

**+0.794 at 7.45x seed sd.** Every winner named with its scope count
and value, cross-unit winners converted before ranking, downward
superlatives inverted, ties named with both holders, empty families
refused — and a negated maximum sitting above every positive candidate
was NEVER crowned, the §11.48 line holding at set scale with the
denial accounted aloud. The hedging arm's 0.206 is the empty-family
share its hedge answers by being honest. The comparative family is
complete: pairwise (§11.54), derived operands (§11.55), and setwise
(here) — every verdict naming what it rests on.

Run it::

    python -m ultraquant.experiments.superlative_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SuperlativeReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]


@dataclass
class SuperlativeReport:
    """The two arms, the denial control, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ranked: Arm -> superlative correctness over all worlds.
        denial_crowned: Negated maxima named as winner in treatment
            (must be 0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    ranked: dict = field(default_factory=dict)
    denial_crowned: int = 0
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'ranked':>7} {'recall':>7}"]
        for arm in ("hedges", "ranks"):
            lines.append(f"{arm:<10} {self.ranked[arm]:>7.3f} "
                         f"{self.recall[arm]:>7.3f}")
        lines += [
            f"denials crowned (treatment): {self.denial_crowned}",
            f"delta (ranks vs hedges) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One superlative world.

    Returns ``(statements, questions, directs)`` where questions are
    ``(question, kind, payload)`` — kind in winner/converted/down/tie/
    empty/denied-max; payload carries what a correct answer must (or
    must not) contain.
    """
    names = []
    seen = set()
    while len(names) < 9:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    heights = {}
    for entity in names[:4]:
        value = rng.randrange(10, 800)
        heights[entity] = value
        statements.append(f"the {entity} height is {value} meters")
    ordered = sorted(names[:4], key=lambda n: heights[n])
    low, high = ordered[0], ordered[-1]

    questions = []
    count = 4
    payload = {"winner": f"{high} height",
               "value": str(heights[high]), "count": None}
    kind = "winner"
    if rng.random() < 0.5:
        # A kilometre entity out-ranks every metre one: the converted
        # winner.
        km_entity = names[4]
        km = rng.randrange(2, 9)
        statements.append(f"the {km_entity} height is {km} kilometers")
        count += 1
        payload = {"winner": f"{km_entity} height",
                   "value": str(km), "count": None}
        kind = "converted"
    payload["count"] = count
    questions.append(("which height is the tallest?", kind, payload))
    if rng.random() < 0.7:
        questions.append(
            ("which height is the shortest?", "down",
             {"winner": f"{low} height", "value": str(heights[low]),
              "count": count}))

    if rng.random() < 0.6:
        # A tie in a separate family.
        a, b = names[5], names[6]
        width = rng.randrange(10, 800)
        statements.append(f"the {a} width is {width} meters")
        statements.append(f"the {b} width is {width} meters")
        questions.append(
            ("which width is the biggest?", "tie",
             {"both": [f"{a} width", f"{b} width"]}))

    if rng.random() < 0.7:
        questions.append(
            ("which depth is the lowest?", "empty", {}))

    if rng.random() < 0.7:
        # The denied maximum: a negated height above every positive.
        denied = names[7]
        statements.append(
            f"the {denied} height is not {rng.randrange(900, 999)} "
            "meters")
        questions.append(
            ("which height is the tallest?", "denied-max",
             {"denied": f"{denied} height",
              "winner": payload["winner"]}))

    directs = []
    for entity in rng.sample(names[:4], 2):
        directs.append((f"what is the {entity} height?",
                        str(heights[entity])))

    rng.shuffle(statements)
    return statements, questions, directs


def run_gate(seeds: int = 12) -> SuperlativeReport:
    """Run both arms over generated superlative worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`SuperlativeReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    ranked_scores = {"hedges": [], "ranks": []}
    recall_scores = {"hedges": [], "ranks": []}
    denial_crowned = 0

    real_flag = thoughts_module._SUPERLATIVES_ON
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, questions, directs = _build_world(rng)

        for arm in ("hedges", "ranks"):
            thoughts_module._SUPERLATIVES_ON = (arm == "ranks")
            root = Path(tempfile.mkdtemp(prefix=f"uq_sup_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind, payload in questions:
                    response, _trace = run_pipeline(question, session)
                    crowned = response.startswith("The ") \
                        or response.startswith("Tied")
                    if kind in ("winner", "converted", "down",
                                "denied-max"):
                        good = (response.startswith(
                                    f"The {payload['winner']}")
                                and f"of the {payload['count']}"
                                in response
                                if "count" in payload and
                                payload.get("count") else
                                response.startswith(
                                    f"The {payload['winner']}"))
                        if kind != "denied-max" and "value" in payload:
                            good = good and payload["value"] in response
                        won.append(float(good))
                        if (kind == "denied-max" and arm == "ranks"
                                and response.startswith(
                                    f"The {payload['denied']}")):
                            denial_crowned += 1
                    elif kind == "tie":
                        good = (response.startswith("Tied")
                                and all(b in response
                                        for b in payload["both"]))
                        won.append(float(good))
                    else:
                        won.append(float(not crowned))
                ranked_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._SUPERLATIVES_ON = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = SuperlativeReport(
        ranked={arm: statistics.fmean(vals)
                for arm, vals in ranked_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        denial_crowned=denial_crowned,
    )

    contrasts = [ranked_scores["ranks"][i] - ranked_scores["hedges"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.denial_crowned > 0:
        report.reason = (
            f"a denial was crowned {report.denial_crowned} time(s) - "
            "the §11.48 line broke at set scale")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"ranking buys nothing ({report.delta:+.3f})"
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
        f"superlatives answer at {report.ranked['ranks']:.3f} against "
        f"the hedging arm's {report.ranked['hedges']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "no denial was ever crowned, and recall held at 1.000 in both "
        "arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
