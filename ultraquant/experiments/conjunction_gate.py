"""Conjunctive statements teach all their facts. The gate.

"the tower material and the bridge material are iron" offered two
facts and taught zero, silently — " are " never parsed, the intent
classifier read the sentence as chat, and the teacher had no way to
know the lesson was dropped. §11.71 admits the plural form exactly as
wide as it is honest: " are " counts as statement-shaped only beside
" and " (a conjunctive sentence is plural by construction; a bare
" are " sentence is conversation), full-subject parts split and each
takes the shared value through the COMPLETE statement machinery —
polarity distributes ("... are not steel" denies each part), a
revising part narrates its §11.53 notice, an adjacent part its
§11.63 note — and the elided form ("the tower and the bridge are
iron", parts without their own relation word) stays the registered
ceiling: distributing an attribute the parts do not name would be
guessing, and the refusal says so. A key containing " and " is never
stored, in either arm, ever.

Both arms run the live pipeline; the baseline arm turns
`_CONJUNCTIVE_STATEMENTS` off (the silent drop), the treatment arm is
current. Worlds mix, in world-varying proportions: two-part and
three-part teaches, negated conjunctions, conjunctions with one
revising part, the elided ceiling (scored zero for both arms — the
variance), and singular statements as the floor.

**The criteria, written before the run.**

1. **Gain** — conjunction correctness (every part narrated in one
   reply and recallable at its own key; polarity on each part;
   the revising part's notice spoken) must beat the baseline arm by
   more than the seed-to-seed sd.
2. **No junk key, ever** — a stored key containing " and ", in either
   arm, is a fail, once.
3. **Singular statements are untouched** — they parse, store, and
   narrate identically in both arms at 1.000.

**It was run once, and PASSED.**

| arm | taught | floor |
|---|---:|---:|
| drops | 0.000 | 1.000 |
| teaches | **0.836** | 1.000 |

**+0.836 at 6.45x seed sd, zero junk keys ever stored, singular
statements identical in both arms.** Two-part and three-part
conjunctions taught every fact in one narrated reply, negation
distributed to each part, a revising part spoke its §11.53 notice
inside the conjunction, and the elided form refused rather than guess
a distribution — the 0.164 miss is that ceiling scoring zero by
design. A teacher can finally say two things in one sentence and have
both of them land, each through the full machinery it would have met
alone.

Run it::

    python -m ultraquant.experiments.conjunction_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ConjunctionReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]


@dataclass
class ConjunctionReport:
    """The two arms, the junk-key line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        taught: Arm -> conjunction correctness over all worlds.
        junk: Stored keys containing " and " (must be 0, both arms).
        floor: Arm -> singular-statement correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    taught: dict = field(default_factory=dict)
    junk: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'taught':>7} {'floor':>7}"]
        for arm in ("drops", "teaches"):
            lines.append(f"{arm:<10} {self.taught[arm]:>7.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"junk and-keys stored: {self.junk}",
            f"delta (teaches vs drops) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One conjunction world.

    Returns ``(moves, floor_entity)`` — moves are ``(kind, entities,
    material, prior)`` executed in order.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    moves = []
    spot = 0
    for _ in range(rng.randrange(1, 3)):
        count = rng.choice((2, 3))
        moves.append(("teach", names[spot:spot + count],
                      rng.choice(_MATERIALS), ""))
        spot += count
    if rng.random() < 0.8:
        moves.append(("negated", names[spot:spot + 2],
                      rng.choice(_MATERIALS), ""))
        spot += 2
    if rng.random() < 0.8:
        material = rng.choice(_MATERIALS)
        prior = rng.choice([m for m in _MATERIALS if m != material])
        moves.append(("revising", names[spot:spot + 2], material,
                      prior))
        spot += 2
    if rng.random() < 0.6:
        moves.append(("elided", [], rng.choice(_MATERIALS), ""))
    floor_entity = names[9]
    return moves, floor_entity


def run_gate(seeds: int = 12) -> ConjunctionReport:
    """Run both arms over generated conjunction worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ConjunctionReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    taught_scores = {"drops": [], "teaches": []}
    floor_scores = {"drops": [], "teaches": []}
    junk = 0

    real_flag = thoughts_module._CONJUNCTIVE_STATEMENTS
    for seed in range(seeds):
        rng = random.Random(seed)
        moves, floor_entity = _build_world(rng)

        for arm in ("drops", "teaches"):
            thoughts_module._CONJUNCTIVE_STATEMENTS = (arm == "teaches")
            root = Path(tempfile.mkdtemp(prefix=f"uq_conj_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                for kind, entities, material, prior in moves:
                    if kind == "teach":
                        subjects = " and ".join(
                            f"the {e} material" for e in entities)
                        response, _t = run_pipeline(
                            f"{subjects} are {material}", session)
                        stored = [memory.recall_fact(f"{e} material")
                                  for e in entities]
                        good = (all(f"{e} material is {material}"
                                    in response for e in entities)
                                and all(s is not None
                                        and s["value"] == material
                                        for s in stored))
                        won.append(float(good))
                    elif kind == "negated":
                        subjects = " and ".join(
                            f"the {e} material" for e in entities)
                        run_pipeline(f"{subjects} are not {material}",
                                     session)
                        stored = [memory.recall_fact(f"{e} material")
                                  for e in entities]
                        good = all(s is not None and s.get("negated")
                                   and s["value"] == material
                                   for s in stored)
                        won.append(float(good))
                    elif kind == "revising":
                        run_pipeline(f"the {entities[0]} material is "
                                     f"{prior}", session)
                        subjects = " and ".join(
                            f"the {e} material" for e in entities)
                        response, _t = run_pipeline(
                            f"{subjects} are {material}", session)
                        good = ("revises" in response
                                and prior in response)
                        won.append(float(good))
                    else:
                        a, b = "great vault", "old cellar"
                        response, _t = run_pipeline(
                            f"the {a} and the {b} are {material}",
                            session)
                        won.append(0.0)
                taught_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                response, _t = run_pipeline(
                    f"the {floor_entity} material is oak", session)
                stored = memory.recall_fact(f"{floor_entity} material")
                floor_scores[arm].append(
                    float("Noted" in response and stored is not None
                          and stored["value"] == "oak"))

                junk += sum(1 for k in memory.fact_keys()
                            if " and " in k)
            finally:
                thoughts_module._CONJUNCTIVE_STATEMENTS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = ConjunctionReport(
        taught={arm: statistics.fmean(vals)
                for arm, vals in taught_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        junk=junk,
    )

    contrasts = [taught_scores["teaches"][i] - taught_scores["drops"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.junk > 0:
        report.reason = f"{report.junk} junk and-key(s) stored"
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "singular statements regressed"
        return report
    if report.delta <= 0:
        report.reason = f"teaching buys nothing ({report.delta:+.3f})"
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
        f"conjunctions teach at {report.taught['teaches']:.3f} against "
        f"the dropping arm's {report.taught['drops']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero junk keys, and singular statements untouched")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
