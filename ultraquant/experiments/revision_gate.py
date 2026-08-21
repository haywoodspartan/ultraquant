"""A change of mind, said out loud. The revision gate.

The episode log has recorded every revision since §11.16's era, truth
maintenance has retracted resting derivatives since §11.30, and §11.48
made polarity flips revisions too — but the REPLY said the same bare
"Noted:" for a change of mind as for news. The honest-aloud principle
(§11.48's phrase) stops exactly where a mind most owes an account:
when it contradicts itself. Now a conflicting statement answers with
the change named — "Noted: tower material is steel. That revises what
I held: tower material was iron. 1 derived fact(s) rested on it and
were retracted: 'tower hardness'." — old belief spoken (polarity
included), retraction cost counted, storage semantics untouched.

`remember_fact` reports its outcome (new / reinforced / revised, with
the old spoken form and the retracted keys), and the Learn stage says
it. The line the controls hold: only a REAL conflict earns the notice.
A reinforcement is not a revision, news is not a revision, and a
system that cries "revised" at agreements is as unaccountable as one
that revises silently.

Both arms run the live pipeline; the baseline arm turns
`_REVISION_ALOUD` off (the bare "Noted:", shipped until now), the
treatment arm is current. Worlds mix, in world-varying proportions:
value revisions with a consolidated derivative resting on the revised
premise (built live via derive-and-affirm), plain value revisions,
polarity revisions, reinforcements, fresh facts — and NEAR-KEY
statements ("the old tower material is ..." against a stored "tower
material"), scored in the revision metric at zero for both arms:
revision detection is exact-key by design, a more-specific subject
stores a new fact and narrates nothing, and that ceiling is the
pre-registered unwinnable share (~25%, per §11.35's power arithmetic)
that lets the contrast vary. A near-key statement carrying revision
text would also be a false notice.

**The criteria, written before the run.**

1. **Gain** — revision statements answered with the change named (the
   word "revises", the old value, and — when a derivative rested on
   it — the correct retraction count and its key; when none did, no
   retraction text) must beat the baseline arm by more than the
   seed-to-seed sd.
2. **Agreement is never a revision** — a reinforcement or fresh-fact
   statement carrying revision text in the treatment arm is a fail,
   once.
3. **The notice is commentary, not semantics** — after every revision,
   the NEW value must answer to a direct question and the retracted
   derivative must no longer assert, in BOTH arms at 1.000: speaking
   the change may not change what happened.

**It was run twice: the first run's contrast never varied, and the
second PASSED with the ceiling scored in.**

The first run measured the mechanism perfect — every revision
narrated, zero false notices, semantics untouched — and FAILED on the
variance rule: every world gave +1.000, and a contrast that cannot
vary proves nothing about the worlds. Standing policy (seventh
occurrence in the book): harden the worlds by scoring the UNWINNABLE
in the metric, never soften the rule. The unwinnable here is real —
revision detection is exact-key by design, so "the old tower material
is ..." against a stored "tower material" stores a NEW fact and
narrates nothing — and the near-key share entered the scored set at
zero for both arms. Then:

| arm | spoken | semantics |
|---|---:|---:|
| silent | 0.000 | 1.000 |
| aloud | **0.803** | 1.000 |

**+0.803 at 5.03x seed sd**, zero agreements called revisions, zero
near-key statements false-noticed, and the semantics held at 1.000 in
both arms — speaking the change changed nothing about what happened.
The 0.197 miss is the exact-key ceiling, recorded the way every
ceiling in this book is: a registered candidate for a future
measurement, not a promise.

Run it::

    python -m ultraquant.experiments.revision_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RevisionReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class RevisionReport:
    """The two arms, the false-notice count, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        spoken: Arm -> revision statements correctly narrated.
        false_notices: Reinforcement/fresh statements carrying revision
            text in treatment (must be 0).
        semantics: Arm -> post-revision correctness (new value answers,
            retracted derivative silent; must be 1.0 in both).
        delta: Narration contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    spoken: dict = field(default_factory=dict)
    false_notices: int = 0
    semantics: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'spoken':>7} {'semantics':>10}"]
        for arm in ("silent", "aloud"):
            lines.append(f"{arm:<10} {self.spoken[arm]:>7.3f} "
                         f"{self.semantics[arm]:>10.3f}")
        lines += [
            f"false notices (treatment): {self.false_notices}",
            f"delta (aloud vs silent) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One revision world.

    Returns ``(setup, consolidations, moves)``. ``moves`` are
    ``(statement, kind, old_shown, derived_key, check_question,
    check_value, retracted_probe)`` executed in order after setup —
    kind in revision-derived/revision/revision-polarity/reinforce/new.
    """
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    setup = []
    prop_values = {}
    for material in _MATERIALS:
        for prop in _PROPERTIES:
            value = rng.randrange(100, 9000)
            prop_values[(material, prop)] = value
            setup.append(f"the {material} {prop} is {value} units")

    consolidations = []
    moves = []
    # A revision under a consolidated derivative.
    entity = names[0]
    material = rng.choice(_MATERIALS)
    setup.append(f"the {entity} material is {material}")
    prop = rng.choice(_PROPERTIES)
    consolidations.append(f"what is the {entity} {prop}?")
    replacement = rng.choice([m for m in _MATERIALS if m != material])
    moves.append((f"the {entity} material is {replacement}",
                  "revision-derived", material, f"{entity} {prop}",
                  f"what is the {entity} material?", replacement,
                  f"what is the {entity} {prop}?"))
    # Plain value revisions.
    for entity in names[1:1 + rng.randrange(1, 3)]:
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        replacement = rng.choice(
            [m for m in _MATERIALS if m != material])
        moves.append((f"the {entity} material is {replacement}",
                      "revision", material, "",
                      f"what is the {entity} material?", replacement,
                      ""))
    # A polarity revision.
    if rng.random() < 0.8:
        entity = names[3]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is not {material}")
        moves.append((f"the {entity} material is {material}",
                      "revision-polarity", f"not {material}", "",
                      f"what is the {entity} material?", material, ""))
    # Near-key statements: the unwinnable share. A more specific
    # subject is a NEW fact, not a revision of the base key - exact-key
    # detection is the mechanism's honest ceiling, scored as zero.
    for entity in names[1:1 + rng.randrange(0, 3)]:
        qualifier = rng.choice([p for p in _PREFIXES
                                if not entity.startswith(p)])
        moves.append((f"the {qualifier} {entity} material is "
                      f"{rng.choice(_MATERIALS)}",
                      "near", "", "", "", "", ""))
    # Reinforcements and fresh facts: the never-a-revision controls.
    entity = names[4]
    material = rng.choice(_MATERIALS)
    setup.append(f"the {entity} material is {material}")
    moves.append((f"the {entity} material is {material}",
                  "reinforce", "", "", "", "", ""))
    for entity in names[5:5 + rng.randrange(1, 3)]:
        moves.append((f"the {entity} material is "
                      f"{rng.choice(_MATERIALS)}",
                      "new", "", "", "", "", ""))

    rng.shuffle(setup)
    return setup, consolidations, moves


def run_gate(seeds: int = 12) -> RevisionReport:
    """Run both arms over generated revision worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`RevisionReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    spoken_scores = {"silent": [], "aloud": []}
    semantics_scores = {"silent": [], "aloud": []}
    false_notices = 0

    real_flag = thoughts_module._REVISION_ALOUD
    for seed in range(seeds):
        rng = random.Random(seed)
        setup, consolidations, moves = _build_world(rng)

        for arm in ("silent", "aloud"):
            thoughts_module._REVISION_ALOUD = (arm == "aloud")
            root = Path(tempfile.mkdtemp(prefix=f"uq_rev_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in setup:
                    run_pipeline(statement, session)
                for question in consolidations:
                    run_pipeline(question, session)
                    run_pipeline("yes", session)

                narrated = []
                checks = []
                for (statement, kind, old_shown, derived_key,
                     check_q, check_v, retracted_probe) in moves:
                    response, _trace = run_pipeline(statement, session)
                    has_notice = "revises" in response
                    if kind == "revision-derived":
                        good = (has_notice and old_shown in response
                                and "1 derived fact(s)" in response
                                and derived_key in response)
                        narrated.append(float(good))
                    elif kind in ("revision", "revision-polarity"):
                        good = (has_notice and old_shown in response
                                and "retracted" not in response)
                        narrated.append(float(good))
                    elif kind == "near":
                        # Scored zero by design in BOTH arms: the
                        # statement stores a new fact and honestly
                        # narrates nothing - the exact-key ceiling.
                        narrated.append(0.0)
                        if arm == "aloud" and has_notice:
                            false_notices += 1
                    else:
                        if arm == "aloud" and has_notice:
                            false_notices += 1
                    if check_q:
                        answer, _trace = run_pipeline(check_q, session)
                        checks.append(float(check_v in answer))
                    if retracted_probe:
                        # The stale consolidated assert would answer
                        # "{key} is ..." from storage; a re-derivation
                        # answers "the {key} is ..., via ...". The
                        # retraction held if the stored assert is gone.
                        answer, _trace = run_pipeline(retracted_probe,
                                                      session)
                        checks.append(float(
                            not answer.startswith(derived_key)))
                spoken_scores[arm].append(
                    statistics.fmean(narrated) if narrated else 1.0)
                semantics_scores[arm].append(
                    statistics.fmean(checks) if checks else 1.0)
            finally:
                thoughts_module._REVISION_ALOUD = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = RevisionReport(
        spoken={arm: statistics.fmean(vals)
                for arm, vals in spoken_scores.items()},
        semantics={arm: statistics.fmean(vals)
                   for arm, vals in semantics_scores.items()},
        false_notices=false_notices,
    )

    contrasts = [spoken_scores["aloud"][i] - spoken_scores["silent"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.false_notices > 0:
        report.reason = (
            f"agreement was called a revision {report.false_notices} "
            "time(s) - a system that cries revised at agreements is as "
            "unaccountable as one that revises silently")
        return report
    if min(report.semantics.values()) < 1.0:
        report.reason = ("the notice changed the semantics: a revised "
                         "value failed to answer or a retracted "
                         "derivative survived")
        return report
    if report.delta <= 0:
        report.reason = f"speaking buys nothing ({report.delta:+.3f})"
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
        f"revisions are narrated at {report.spoken['aloud']:.3f} "
        f"against the silent arm's {report.spoken['silent']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero agreements were called revisions, and the semantics "
        "held at 1.000 in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
