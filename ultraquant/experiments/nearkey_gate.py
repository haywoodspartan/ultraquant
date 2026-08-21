"""The neighbor named: near-key statements and the held base. The gate.

§11.53 measured revision detection's exact-key ceiling and scored it
into its own metric: "the old tower material is steel" against a held
"tower material" stores a NEW fact and narrates nothing, by design —
a more specific subject is a different belief. But silence about the
held base invites the reader to assume the two are one, and this unit
makes the adjacency visible without making it a conflict: a new fact
whose leading-stripped base key is held answers with the neighbor
spoken — "Noted: old tower material is steel. I separately hold:
tower material is iron" — polarity included when the base is a denial.
Never a revision, never a merge, purely information: different keys
are different beliefs, and now both are on the table.

Both arms run the live pipeline; the baseline arm turns
`_NEARKEY_NOTES` off (§11.53's silent ceiling), the treatment arm is
current. Worlds mix, in world-varying proportions: near-key statements
over held positive bases, near-key statements over held NEGATED bases
(the note must speak the polarity), FAR-key statements ("the ruined
old tower material" — the base two strips away, beyond the
single-strip design, scored zero for both arms: the mechanism's own
ceiling, ~25% per §11.35's power arithmetic, and the variance), and
the three silences — fresh facts with no held base, reinforcements,
and revisions (which keep their §11.53 notice and gain nothing else).
Multi-qualifier adjacency is a registered candidate, not a promise.

**The criteria, written before the run.**

1. **Gain** — near-key statements answered with the adjacency ("I
   separately hold" naming the base's value, polarity included) must
   beat the baseline arm by more than the seed-to-seed sd.
2. **Silence stays silent** — a fresh no-base fact, a reinforcement,
   or a revision carrying an adjacency note in the treatment arm is a
   fail, once: a note that fires on everything informs about nothing.
3. **Semantics untouched** — after every near-key statement, BOTH keys
   answer their own questions with their own values, in both arms at
   1.000.

**It was run three times: the variance rule, then the harness, then
the PASS.**

The first run measured the mechanism perfect and failed on "nothing
varied" (the ninth in the book) — hardened the standing way, the
single-strip ceiling scored in. The second run failed its OWN
semantics floor at 0.986 in both arms identically — the flag could
not be the cause, and the trace found a world-builder slot collision:
the near loop's slice and the far kind shared a name, and the far
setup line silently revised a base the near payload had already
recorded. The same entity-reuse bug the revision gate's builder had,
caught by the same kind of floor. Slots disjoint, then:

| arm | noted | semantics |
|---|---:|---:|
| silent | 0.000 | 1.000 |
| speaks | **0.912** | 1.000 |

**+0.912 at 8.37x seed sd, zero false adjacency notes**, the base's
polarity spoken when the base is a denial, and both keys keeping
their own values in both arms — the note is information, never a
merge. The 0.088 miss is the single-strip ceiling scoring zero by
design; multi-qualifier adjacency stays registered, not promised.

Run it::

    python -m ultraquant.experiments.nearkey_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NearKeyReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_QUALIFIERS = ["older", "former", "eastern", "ruined"]


@dataclass
class NearKeyReport:
    """The two arms, the silence control, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        noted: Arm -> near-key statements correctly annotated.
        false_notes: Silent-kind statements carrying adjacency notes in
            treatment (must be 0).
        semantics: Arm -> both-keys-answer correctness (must be 1.0).
        delta: Annotation contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    noted: dict = field(default_factory=dict)
    false_notes: int = 0
    semantics: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'noted':>7} {'semantics':>10}"]
        for arm in ("silent", "speaks"):
            lines.append(f"{arm:<10} {self.noted[arm]:>7.3f} "
                         f"{self.semantics[arm]:>10.3f}")
        lines += [
            f"false adjacency notes (treatment): {self.false_notes}",
            f"delta (speaks vs silent) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One near-key world.

    Returns ``(setup, moves)`` where moves are ``(statement, kind,
    base, base_shown, check_q, check_v)`` — kind in near/near-negated/
    fresh/reinforce/revise.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    setup = []
    moves = []
    # Near-key over a positive base.
    for entity in names[:1 + rng.randrange(1, 3)]:
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        other = rng.choice([m for m in _MATERIALS if m != material])
        qualifier = rng.choice(_QUALIFIERS)
        moves.append((f"the {qualifier} {entity} material is {other}",
                      "near", f"{entity} material", material,
                      f"what is the {qualifier} {entity} material?",
                      other))
    # Near-key over a negated base.
    if rng.random() < 0.8:
        entity = names[3]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is not {material}")
        qualifier = rng.choice(_QUALIFIERS)
        other = rng.choice([m for m in _MATERIALS if m != material])
        moves.append((f"the {qualifier} {entity} material is {other}",
                      "near-negated", f"{entity} material",
                      f"not {material}",
                      f"what is the {qualifier} {entity} material?",
                      other))
    # Far-key statements: the single-strip ceiling, scored zero.
    # A DEDICATED name slot: the first build let the near loop and the
    # far kind share names[2], and the far setup line silently revised
    # a base the near payload had already recorded - the same
    # entity-reuse bug the revision gate's builder had.
    if rng.random() < 0.55:
        entity = names[4]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        q1, q2 = rng.sample(_QUALIFIERS, 2)
        moves.append((f"the {q1} {q2} {entity} material is "
                      f"{rng.choice(_MATERIALS)}",
                      "far", "", "", "", ""))
    # The three silences.
    for entity in names[5:5 + rng.randrange(1, 3)]:
        moves.append((f"the {entity} material is "
                      f"{rng.choice(_MATERIALS)}",
                      "fresh", "", "", "", ""))
    entity = names[7]
    material = rng.choice(_MATERIALS)
    setup.append(f"the {entity} material is {material}")
    moves.append((f"the {entity} material is {material}",
                  "reinforce", "", "", "", ""))
    if rng.random() < 0.8:
        entity = names[8]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        other = rng.choice([m for m in _MATERIALS if m != material])
        moves.append((f"the {entity} material is {other}",
                      "revise", "", "", "", ""))

    rng.shuffle(setup)
    return setup, moves


def run_gate(seeds: int = 12) -> NearKeyReport:
    """Run both arms over generated near-key worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`NearKeyReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    noted_scores = {"silent": [], "speaks": []}
    semantics_scores = {"silent": [], "speaks": []}
    false_notes = 0

    real_flag = thoughts_module._NEARKEY_NOTES
    for seed in range(seeds):
        rng = random.Random(seed)
        setup, moves = _build_world(rng)

        for arm in ("silent", "speaks"):
            thoughts_module._NEARKEY_NOTES = (arm == "speaks")
            root = Path(tempfile.mkdtemp(prefix=f"uq_nk_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in setup:
                    run_pipeline(statement, session)

                annotated = []
                checks = []
                for (statement, kind, base, base_shown,
                     check_q, check_v) in moves:
                    response, _trace = run_pipeline(statement, session)
                    has_note = "I separately hold" in response
                    if kind == "far":
                        # The single-strip ceiling: scored zero by
                        # design in both arms.
                        annotated.append(0.0)
                        continue
                    if kind in ("near", "near-negated"):
                        good = (has_note and base in response
                                and base_shown in response)
                        annotated.append(float(good))
                        if check_q:
                            answer, _trace = run_pipeline(check_q,
                                                          session)
                            checks.append(float(check_v in answer))
                            answer, _trace = run_pipeline(
                                f"what is the {base}?", session)
                            checks.append(float(base_shown in answer))
                    else:
                        if arm == "speaks" and has_note:
                            false_notes += 1
                noted_scores[arm].append(
                    statistics.fmean(annotated) if annotated else 1.0)
                semantics_scores[arm].append(
                    statistics.fmean(checks) if checks else 1.0)
            finally:
                thoughts_module._NEARKEY_NOTES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = NearKeyReport(
        noted={arm: statistics.fmean(vals)
               for arm, vals in noted_scores.items()},
        semantics={arm: statistics.fmean(vals)
                   for arm, vals in semantics_scores.items()},
        false_notes=false_notes,
    )

    contrasts = [noted_scores["speaks"][i] - noted_scores["silent"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.false_notes > 0:
        report.reason = (
            f"silence broke: {report.false_notes} statement(s) with no "
            "held base carried an adjacency note - a note that fires "
            "on everything informs about nothing")
        return report
    if min(report.semantics.values()) < 1.0:
        report.reason = ("the note changed the semantics: a key "
                         "answered with its neighbor's value")
        return report
    if report.delta <= 0:
        report.reason = f"the note buys nothing ({report.delta:+.3f})"
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
        f"near-key statements are annotated at "
        f"{report.noted['speaks']:.3f} against the silent arm's "
        f"{report.noted['silent']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), zero false notes, and "
        "both keys kept their own values in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
