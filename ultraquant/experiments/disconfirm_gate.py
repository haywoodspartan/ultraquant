"""No is testimony too. The disconfirmation gate.

§11.68 wired "yes" to the stored fact; its mirror was still mush — a
polar "Yes - tower material is iron" answered with the user's "no"
fell to the chat fallback, and testimony AGAINST a belief vanished.
The mirror closes with an honest asymmetry: a contest drops the
belief to doubt (0.30) but does NOT delete it, because a bare "no"
names no replacement and deleting on it would invent an absence — the
drop is said aloud, the correction is asked for ("tell me what it
is"), and a following statement closes into §11.53's narrated
revision: doubt, ask, revise, each step spoken.

Both arms run the live pipeline; the baseline arm turns
`_DISCONFIRM_TESTIMONY` off (the shipped mush), the treatment arm is
current. Worlds mix, in world-varying proportions: contests (scored:
the drop spoken and 0.30 on the record, the fact still held),
contest-then-correct loops (scored: the revision narrated with the
old value named, the new value answering at stated confidence), stray
"no"s with nothing pending (zero tolerance: inert in both arms), and
one confirmation per world as the §11.68 floor ("yes" still lands at
0.90 in both arms).

**The criteria, written before the run.**

1. **Gain** — contest correctness (the drop spoken, 0.30 on record,
   the belief still held) and loop correctness (revision narrated,
   new value at 0.60) must beat the baseline arm by more than the
   seed-to-seed sd.
2. **A contest never deletes** — a contested fact missing from the
   store in the treatment arm is a fail, once: doubt is not absence.
3. **A stray no stays inert** — a disagreement with nothing pending
   that changes any confidence, in either arm, is a fail, once.
4. **The §11.68 floor holds** — confirmations land at 0.90 in both
   arms.

**It was run once, and PASSED — perfectly on the claim, zero on
every line.**

| arm | contested | floor |
|---|---:|---:|
| mush | 0.278 | 1.000 |
| doubts | **1.000** | 1.000 |

**+0.722 at 3.33x seed sd, zero deletions, zero moved strays,
confirmations untouched in both arms.** Every contest dropped its
belief to exactly 0.30 with the drop spoken and the correction asked,
every contest-then-correct loop closed into §11.53's narrated
revision with the new value at stated confidence, and doubt never
became absence. The testimony pair is complete: yes raises to 0.90,
no lowers to 0.30, both said aloud, neither deletes, and a bare word
with nothing pending moves nothing — the user's agreement and
disagreement finally weigh exactly what the epistemics always said
they should.

Run it::

    python -m ultraquant.experiments.disconfirm_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DisconfirmReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]


@dataclass
class DisconfirmReport:
    """The two arms, the doubt-not-absence line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        contested: Arm -> contest/loop correctness over all worlds.
        deleted: Contested facts missing from the store in treatment
            (must be 0).
        strays: Stray disagreements that moved a confidence (must be 0).
        floor: Arm -> §11.68 confirmation correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    contested: dict = field(default_factory=dict)
    deleted: int = 0
    strays: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'contested':>10} {'floor':>7}"]
        for arm in ("mush", "doubts"):
            lines.append(f"{arm:<10} {self.contested[arm]:>10.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"contested facts deleted (treatment): {self.deleted}",
            f"stray disagreements that moved confidence: {self.strays}",
            f"delta (doubts vs mush) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One disconfirmation world.

    Returns ``(moves, floor_entity)`` — moves are ``(kind, entity,
    material, correction)`` executed in order: contest, loop, stray.
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
        moves.append(("contest", names[spot],
                      rng.choice(_MATERIALS), ""))
        spot += 1
    if rng.random() < 0.8:
        material = rng.choice(_MATERIALS)
        correction = rng.choice(
            [m for m in _MATERIALS if m != material])
        moves.append(("loop", names[spot], material, correction))
        spot += 1
    if rng.random() < 0.8:
        moves.append(("stray", "", "", ""))
    floor_entity = names[6]
    return moves, floor_entity


def run_gate(seeds: int = 12) -> DisconfirmReport:
    """Run both arms over generated disconfirmation worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`DisconfirmReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    contested_scores = {"mush": [], "doubts": []}
    floor_scores = {"mush": [], "doubts": []}
    deleted = 0
    strays = 0

    real_flag = thoughts_module._DISCONFIRM_TESTIMONY
    for seed in range(seeds):
        rng = random.Random(seed)
        moves, floor_entity = _build_world(rng)

        for arm in ("mush", "doubts"):
            thoughts_module._DISCONFIRM_TESTIMONY = (arm == "doubts")
            root = Path(tempfile.mkdtemp(prefix=f"uq_dis_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                for kind, entity, material, correction in moves:
                    if kind == "contest":
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"is the {entity} material "
                                     f"{material}?", session)
                        response, _t = run_pipeline("no", session)
                        fact = memory.recall_fact(f"{entity} material")
                        if fact is None:
                            if arm == "doubts":
                                deleted += 1
                            won.append(0.0)
                            continue
                        good = ("you say that is wrong" in response
                                and abs(float(fact["confidence"])
                                        - 0.3) < 1e-9)
                        won.append(float(good))
                    elif kind == "loop":
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"is the {entity} material "
                                     f"{material}?", session)
                        run_pipeline("no", session)
                        response, _t = run_pipeline(
                            f"the {entity} material is {correction}",
                            session)
                        fact = memory.recall_fact(f"{entity} material")
                        good = ("revises" in response
                                and material in response
                                and fact is not None
                                and fact["value"] == correction
                                and abs(float(fact["confidence"])
                                        - 0.6) < 1e-9)
                        won.append(float(good))
                    else:
                        before = {k: memory.recall_fact(k)["confidence"]
                                  for k in memory.fact_keys()}
                        run_pipeline("no", session)
                        after = {k: memory.recall_fact(k)["confidence"]
                                 for k in memory.fact_keys()}
                        if before != after:
                            strays += 1
                contested_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                run_pipeline(f"the {floor_entity} material is iron",
                             session)
                run_pipeline(f"is the {floor_entity} material iron?",
                             session)
                run_pipeline("yes", session)
                fact = memory.recall_fact(f"{floor_entity} material")
                floor_scores[arm].append(
                    float(fact is not None
                          and abs(float(fact["confidence"])
                                  - 0.9) < 1e-9))
            finally:
                thoughts_module._DISCONFIRM_TESTIMONY = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = DisconfirmReport(
        contested={arm: statistics.fmean(vals)
                   for arm, vals in contested_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        deleted=deleted,
        strays=strays,
    )

    contrasts = [contested_scores["doubts"][i]
                 - contested_scores["mush"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.deleted > 0:
        report.reason = (f"a contest deleted {report.deleted} "
                         "fact(s) - doubt is not absence")
        return report
    if report.strays > 0:
        report.reason = (f"a stray no moved a confidence "
                         f"{report.strays} time(s)")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "the §11.68 confirmation floor regressed"
        return report
    if report.delta <= 0:
        report.reason = f"doubting buys nothing ({report.delta:+.3f})"
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
        f"contests land at {report.contested['doubts']:.3f} against "
        f"the mush arm's {report.contested['mush']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "nothing deleted, strays inert, and confirmations untouched")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
