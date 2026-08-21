"""What happened: the retraction, askable. The gate.

§11.64 opened the revision record's read side; this unit opens the
other ending. A consolidated fact that truth maintenance takes down
(§11.30: a conclusion does not outlive its premise) simply vanished —
the user who watched it consolidate found a hedge where it stood,
though the retraction episode records exactly what happened and why.
The data had the same gap §11.64's polarity fix closed for revisions:
retraction episodes carried no tags and could not be found by key.
They are tagged now, and a past-tense question over a subject no
longer held answers from the record: "I no longer hold tower
hardness: retracted - premise 'tower material' was revised."

The zero-invention line extends unchanged: only a RECORDED ending
answers. A subject that never existed, or was never consolidated,
gets the hedge it always got — an ending invented is a history
invented.

Both arms run the live pipeline under `_HISTORY_ANSWERS` (the
takedown branch lives inside §11.64's flag; the baseline arm answers
past-tense questions with the present machinery). Worlds mix, in
world-varying proportions: consolidate-then-revise sequences built
live through the affirm flow (scored: the takedown named with its
cause), never-existed subjects (the hedge, zero tolerance on invented
endings), and one ordinary revised fact per world as the §11.64
regression floor.

**The criteria, written before the run.**

1. **Gain** — retracted-subject questions answered with the takedown
   ("no longer hold" plus the premise named in the cause) must beat
   the baseline arm by more than the seed-to-seed sd.
2. **No ending is invented** — a never-existed subject answered with
   "no longer hold" in the treatment arm is a fail, once.
3. **§11.64 holds** — ordinary revision histories answer at 1.000 in
   the treatment arm, and present-tense questions are identical in
   both arms.

**It was run once, and PASSED — perfectly on the claim, zero on the
line.**

| arm | endings | floors |
|---|---:|---:|
| present-only | 0.369 | 1.000 |
| remembers | **1.000** | 1.000 |

**+0.631 at 7.82x seed sd, zero endings invented** — every retracted
subject answered with its takedown and the premise that caused it,
every never-existed subject hedged, the §11.64 histories and the
present untouched in both arms. The baseline's 0.369 is the
never-kind share its hedge answers by being honest. The tense family
is complete: what is (recall and derivation), what was (§11.64), what
happened (here), and why (§11.51) — one record, four doors, nothing
invented through any of them.

Run it::

    python -m ultraquant.experiments.retraction_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RetractionReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class RetractionReport:
    """The two arms, the invention line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        endings: Arm -> retracted-subject correctness over all worlds.
        invented: "no longer hold" answers over never-existed subjects
            in treatment (must be 0).
        floors: Arm -> §11.64 history + present correctness (1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    endings: dict = field(default_factory=dict)
    invented: int = 0
    floors: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<12} {'endings':>8} {'floors':>7}"]
        for arm in ("present-only", "remembers"):
            lines.append(f"{arm:<12} {self.endings[arm]:>8.3f} "
                         f"{self.floors[arm]:>7.3f}")
        lines += [
            f"endings invented (treatment): {self.invented}",
            f"delta (remembers vs present-only) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One retraction world.

    Returns ``(setup, consolidations, revisions, questions,
    floor_qs)`` — setup then consolidations (asked and affirmed) then
    revisions run in ORDER (§11.64's lesson: sequence is semantics);
    questions are ``(question, kind, payload)`` with kind in
    retracted/never; floor_qs are ``(question, expected)``.
    """
    names = []
    seen = set()
    while len(names) < 7:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    setup = []
    consolidations = []
    revisions = []
    questions = []
    floor_qs = []

    # Consolidate-then-revise, one or two per world.
    for entity in names[:1 + rng.randrange(1, 3)]:
        material = rng.choice(_MATERIALS)
        prop = rng.choice(_PROPERTIES)
        setup.append(f"the {entity} material is {material}")
        setup.append(
            f"the {material} {prop} is {rng.randrange(100, 900)} units")
        consolidations.append(f"what is the {entity} {prop}?")
        replacement = rng.choice(
            [m for m in _MATERIALS if m != material])
        revisions.append(f"the {entity} material is {replacement}")
        questions.append(
            (f"what was the {entity} {prop}?", "retracted",
             ["no longer hold",
              f"premise '{entity} material' was revised"]))
    # Never-existed subjects.
    for _ in range(rng.randrange(1, 3)):
        questions.append(
            (f"what was the {names[4]} {rng.choice(_PROPERTIES)}?",
             "never", []))
    # The §11.64 floor: an ordinary revised fact.
    entity = names[5]
    first, second = rng.sample(_MATERIALS, 2)
    setup.append(f"the {entity} material is {first}")
    revisions.append(f"the {entity} material is {second}")
    floor_qs.append((f"what was the {entity} material?",
                     f"It was {first}"))
    floor_qs.append((f"what is the {entity} material?", second))

    return setup, consolidations, revisions, questions, floor_qs


def run_gate(seeds: int = 12) -> RetractionReport:
    """Run both arms over generated retraction worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`RetractionReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    ending_scores = {"present-only": [], "remembers": []}
    floor_scores = {"present-only": [], "remembers": []}
    invented = 0

    real_flag = thoughts_module._HISTORY_ANSWERS
    for seed in range(seeds):
        rng = random.Random(seed)
        (setup, consolidations, revisions,
         questions, floor_qs) = _build_world(rng)

        for arm in ("present-only", "remembers"):
            thoughts_module._HISTORY_ANSWERS = (arm == "remembers")
            root = Path(tempfile.mkdtemp(prefix=f"uq_ret_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in setup:
                    run_pipeline(statement, session)
                for question in consolidations:
                    run_pipeline(question, session)
                    run_pipeline("yes", session)
                for statement in revisions:
                    run_pipeline(statement, session)

                won = []
                for question, kind, payload in questions:
                    response, _trace = run_pipeline(question, session)
                    told_ending = "no longer hold" in response
                    if kind == "retracted":
                        won.append(float(
                            told_ending
                            and all(p in response for p in payload)))
                    else:
                        won.append(float(not told_ending))
                        if arm == "remembers" and told_ending:
                            invented += 1
                ending_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, expected in floor_qs:
                    response, _trace = run_pipeline(question, session)
                    if expected.startswith("It was"):
                        hits.append(float(
                            expected in response
                            if arm == "remembers" else 1.0))
                    else:
                        hits.append(float(expected in response))
                floor_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._HISTORY_ANSWERS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = RetractionReport(
        endings={arm: statistics.fmean(vals)
                 for arm, vals in ending_scores.items()},
        floors={arm: statistics.fmean(vals)
                for arm, vals in floor_scores.items()},
        invented=invented,
    )

    contrasts = [ending_scores["remembers"][i]
                 - ending_scores["present-only"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.invented > 0:
        report.reason = (
            f"an ending was invented {report.invented} time(s) - an "
            "ending invented is a history invented")
        return report
    if min(report.floors.values()) < 1.0:
        report.reason = "the §11.64 floor or the present regressed"
        return report
    if report.delta <= 0:
        report.reason = f"the record buys nothing ({report.delta:+.3f})"
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
        f"retracted subjects answer at {report.endings['remembers']:.3f} "
        f"against the baseline's {report.endings['present-only']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero endings invented, and the §11.64 floor and present held")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
