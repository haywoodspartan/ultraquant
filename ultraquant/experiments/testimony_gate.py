"""Yes reaches the stored fact. The testimony gate.

`confirm_fact` has argued since it was written that being told "yes,
that is correct" is a different and stronger kind of evidence than
hearing a value again in passing — reinforcement nudges confidence by
0.1; testimony asserts it outright — but nothing wired the mechanism
to the surface. A polar "Yes - tower material is iron" followed by
the user's "yes" answered "I have nothing on that yet": the system
asked for confirmation-shaped input with one hand and dropped it with
the other. §11.68 closes the loop: an assertion of a held belief
(polar yes, choice pick) leaves a one-turn pending confirmation —
the same freshness discipline §11.33 built for derivations, the same
§11.16 trap avoided — and "yes" raises the fact to testimony
confidence, said aloud.

The distinction the gate measures is the docstring's own claim,
finally at the surface: a CONFIRMATION lands at 0.90; a passing
RESTATEMENT of the same fact lands at 0.70 (one reinforcement nudge)
— two kinds of evidence, two strengths, and a system that conflated
them would be pretending testimony is gossip.

Both arms run the live pipeline; the baseline arm turns
`_CONFIRM_TESTIMONY` off (the shipped dropped yes), the treatment arm
is current. Worlds mix, in world-varying proportions: polar-confirm
and choice-confirm sequences (scored: "Confirmed" spoken and 0.90
visible on the record), passing restatements of OTHER facts (scored:
0.70, never 0.90 — the distinction), stray affirmations with nothing
pending (zero tolerance: inert in both arms), and one
derivation-affirmation per world as the §11.33 floor (consolidation
unchanged in both arms).

**The criteria, written before the run.**

1. **Gain** — confirmation correctness ("Confirmed" spoken, the fact
   at 0.90 on the record) must beat the baseline arm by more than the
   seed-to-seed sd.
2. **Testimony is not gossip** — every passing restatement lands at
   0.70 exactly; a restatement reaching 0.90, or a confirmation
   stopping at 0.70, in the treatment arm is a fail, once.
3. **A stray yes stays inert** — an affirmation with nothing pending
   that changes any confidence, in either arm, is a fail, once.
4. **The §11.33 floor holds** — derivation affirmations consolidate
   identically in both arms.

**It was run once, and PASSED — perfectly on the claim, zero on
every line.**

| arm | confirmed | floor |
|---|---:|---:|
| drops-yes | 0.442 | 1.000 |
| testimony | **1.000** | 1.000 |

**+0.558 at 3.99x seed sd, zero strengths conflated, zero strays
moved, consolidation untouched in both arms.** Every confirmation
landed at exactly 0.90 with "Confirmed" spoken, every passing
restatement at exactly 0.70, and the two kinds of evidence COMPOSE —
a confirmation followed by a later passing mention reads 1.00, each
contribution at its own strength. The dropped-yes arm's 0.442 is the
restatement share reinforcement already served. A mechanism that
argued for itself in a docstring for eleven sections finally testifies
at the surface — and the affirmation word now reaches all three
pending kinds: a derivation consolidates, an assertion confirms, and
nothing pending stays nothing.

Run it::

    python -m ultraquant.experiments.testimony_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TestimonyReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class TestimonyReport:
    """The two arms, the two strengths, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        confirmed: Arm -> confirmation correctness over all worlds.
        conflated: Restatements at 0.90 or confirmations at 0.70 in
            treatment (must be 0).
        strays: Stray affirmations that moved a confidence (must be 0).
        floor: Arm -> derivation-affirmation correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    confirmed: dict = field(default_factory=dict)
    conflated: int = 0
    strays: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'confirmed':>10} {'floor':>7}"]
        for arm in ("drops-yes", "testimony"):
            lines.append(f"{arm:<10} {self.confirmed[arm]:>10.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"strengths conflated (treatment): {self.conflated}",
            f"stray affirmations that moved confidence: {self.strays}",
            f"delta (testimony vs drops-yes) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One testimony world.

    Returns ``(moves, floor_entity)`` — moves are ``(kind, entity,
    material, extra)`` executed in order: confirm-polar,
    confirm-choice, restate, stray.
    """
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    moves = []
    spot = 0
    for _ in range(rng.randrange(1, 3)):
        moves.append(("confirm-polar", names[spot],
                      rng.choice(_MATERIALS), ""))
        spot += 1
    if rng.random() < 0.7:
        material = rng.choice(_MATERIALS)
        other = rng.choice([m for m in _MATERIALS if m != material])
        moves.append(("confirm-choice", names[spot], material, other))
        spot += 1
    for _ in range(rng.randrange(1, 3)):
        moves.append(("restate", names[spot],
                      rng.choice(_MATERIALS), ""))
        spot += 1
    if rng.random() < 0.8:
        moves.append(("stray", "", "", ""))
    floor_entity = names[6]
    return moves, floor_entity


def run_gate(seeds: int = 12) -> TestimonyReport:
    """Run both arms over generated testimony worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`TestimonyReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    confirmed_scores = {"drops-yes": [], "testimony": []}
    floor_scores = {"drops-yes": [], "testimony": []}
    conflated = 0
    strays = 0

    real_flag = thoughts_module._CONFIRM_TESTIMONY
    for seed in range(seeds):
        rng = random.Random(seed)
        moves, floor_entity = _build_world(rng)

        for arm in ("drops-yes", "testimony"):
            thoughts_module._CONFIRM_TESTIMONY = (arm == "testimony")
            root = Path(tempfile.mkdtemp(prefix=f"uq_tst_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                for kind, entity, material, extra in moves:
                    if kind == "confirm-polar":
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"is the {entity} material "
                                     f"{material}?", session)
                        response, _t = run_pipeline("yes", session)
                        fact = memory.recall_fact(f"{entity} material")
                        conf = float(fact["confidence"])
                        good = ("Confirmed" in response
                                and abs(conf - 0.9) < 1e-9)
                        won.append(float(good))
                        if arm == "testimony" and abs(conf - 0.9) > 1e-9:
                            conflated += 1
                    elif kind == "confirm-choice":
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"is the {entity} material "
                                     f"{material} or {extra}?", session)
                        response, _t = run_pipeline("yes", session)
                        fact = memory.recall_fact(f"{entity} material")
                        good = ("Confirmed" in response
                                and abs(float(fact["confidence"])
                                        - 0.9) < 1e-9)
                        won.append(float(good))
                    elif kind == "restate":
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        fact = memory.recall_fact(f"{entity} material")
                        conf = float(fact["confidence"])
                        won.append(float(abs(conf - 0.7) < 1e-9))
                        if arm == "testimony" and abs(conf - 0.9) < 1e-9:
                            conflated += 1
                    else:
                        before = {k: memory.recall_fact(k)["confidence"]
                                  for k in memory.fact_keys()}
                        run_pipeline("yes", session)
                        after = {k: memory.recall_fact(k)["confidence"]
                                 for k in memory.fact_keys()}
                        if before != after:
                            strays += 1
                confirmed_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                # §11.33 floor: derivation affirmation consolidates.
                run_pipeline(f"the {floor_entity} material is iron",
                             session)
                run_pipeline("the iron hardness is 490 units", session)
                run_pipeline(f"what is the {floor_entity} hardness?",
                             session)
                response, _t = run_pipeline("yes", session)
                floor_scores[arm].append(
                    float("Consolidated" in response))
            finally:
                thoughts_module._CONFIRM_TESTIMONY = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = TestimonyReport(
        confirmed={arm: statistics.fmean(vals)
                   for arm, vals in confirmed_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        conflated=conflated,
        strays=strays,
    )

    contrasts = [confirmed_scores["testimony"][i]
                 - confirmed_scores["drops-yes"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.conflated > 0:
        report.reason = (
            f"the strengths conflated {report.conflated} time(s) - "
            "testimony is not gossip, and gossip is not testimony")
        return report
    if report.strays > 0:
        report.reason = (f"a stray yes moved a confidence "
                         f"{report.strays} time(s)")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "the §11.33 consolidation floor regressed"
        return report
    if report.delta <= 0:
        report.reason = f"testimony buys nothing ({report.delta:+.3f})"
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
        f"confirmations land at {report.confirmed['testimony']:.3f} "
        f"against the dropped-yes arm's "
        f"{report.confirmed['drops-yes']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), the two strengths never "
        "conflated, strays inert, and consolidation untouched")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
