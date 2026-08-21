"""The trail is the answer — askable. The why gate.

Since §11.31 the architecture's thesis has been that an answer IS its
evidence trail: every inference names the facts it crossed, every
consolidated fact records the premises it rests on (`derived_from`,
load-bearing since §11.30), every stated fact carries its confidence
and reinforcement history. But nothing could ASK for any of it. "why
is X Y?" now answers from provenance: a consolidated fact cites its
premises ("Because tower material is iron; iron hardness is 490 units
- consolidated from a confirmed derivation"), a stated fact cites its
statement, a derivable-but-unstored one derives and shows the trail —
and the control this gate exists to hold is the anti-rationalisation
line: **never explain what is not believed.** "why is the tower
material steel?" when iron is held answers "It isn't - tower material
is iron", because inventing a justification for a false premise is the
transformer failure mode this architecture defines itself against, and
an explanation machine that will explain anything explains nothing.

Both arms run the live pipeline; the baseline arm turns `_WHY_ANSWERS`
off, the treatment arm is current. Worlds mix why-questions of five
kinds in world-varying proportions: stated (the fact was told
directly), consolidated (a chain was derived and affirmed through the
live protocol during world build), derived (the chain exists,
unconfirmed), corrected (the claim contradicts held belief), and
unheld (nothing to explain). Direct recall is the floor.

**The criteria, written before the run.**

1. **Gain** — why-question correctness (stated cites its statement,
   consolidated and derived cite BOTH premises with their provenance
   marker, corrected answers "It isn't" with the actual value, unheld
   stays open) must beat the baseline arm by more than the seed sd.
2. **Never explain the unbelieved** — a corrected-kind question
   answered "Because ..." in the treatment arm is a rationalisation:
   one is a fail.
3. **Never explain the unheld** — an unheld-kind question answered
   "Because ..." in the treatment arm is a fabricated explanation:
   one is a fail.
4. **Recall holds** — direct recall at 1.000 in both arms.

**It was run once, and PASSED — perfect on every kind, zero on both
lines.**

| arm | why | recall |
|---|---:|---:|
| silent | 0.170 | 1.000 |
| answers | **1.000** | 1.000 |

**+0.830 at 19.88x seed sd.** Every stated fact cited its statement,
every consolidated fact named both premises with its provenance
marker, every unconfirmed derivation named its trail and said so,
every false premise was corrected ("It isn't - tower material is
iron") with ZERO rationalised, every unheld subject stayed
unexplained with ZERO invented — and recall held at 1.000 in both
arms. The silent arm's 0.170 is the unheld share its hedge gets for
free. The thesis the architecture was built on — the trail IS the
answer — is now a question the library can be asked, and the two
zero-tolerance lines are what make the answers worth having: an
explanation machine that will explain anything explains nothing.

Run it::

    python -m ultraquant.experiments.why_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["WhyReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class WhyReport:
    """The two arms, the two zero-tolerance lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        why: Arm -> why-question correctness over all worlds.
        rationalised: Corrected-kind "Because" answers in treatment
            (must be 0).
        fabricated: Unheld-kind "Because" answers in treatment
            (must be 0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Why-correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    why: dict = field(default_factory=dict)
    rationalised: int = 0
    fabricated: int = 0
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'why':>7} {'recall':>7}"]
        for arm in ("silent", "answers"):
            lines.append(f"{arm:<10} {self.why[arm]:>7.3f} "
                         f"{self.recall[arm]:>7.3f}")
        lines += [
            f"false premises rationalised (treatment): "
            f"{self.rationalised}",
            f"unheld subjects explained (treatment): {self.fabricated}",
            f"delta (why on vs off) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One provenance world.

    Returns ``(statements, consolidations, whys, directs)``:
    consolidations are chain questions asked-and-affirmed during world
    build (both arms identically); whys are ``(question, kind,
    premise_words)`` with kind in stated/consolidated/derived/
    corrected/unheld and premise_words the tokens a correct
    explanation must contain.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    prop_values = {}
    for material in _MATERIALS:
        for prop in _PROPERTIES:
            value = rng.randrange(100, 9000)
            prop_values[(material, prop)] = value
            statements.append(f"the {material} {prop} is {value} units")

    whys = []
    consolidations = []
    # Stated facts, plus the corrected kind against them.
    for entity in names[:2]:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is {material}")
        if rng.random() < 0.8:
            whys.append((f"why is the {entity} material {material}?",
                         "stated", []))
        other = rng.choice([m for m in _MATERIALS if m != material])
        whys.append((f"why is the {entity} material {other}?",
                     "corrected", [material]))
    # Consolidated: derive and affirm during build.
    for entity in names[2:4]:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is {material}")
        prop = rng.choice(_PROPERTIES)
        consolidations.append(f"what is the {entity} {prop}?")
        whys.append(
            (f"why is the {entity} {prop} "
             f"{prop_values[(material, prop)]} units?",
             "consolidated", [material, prop]))
    # Derived, unconfirmed.
    for entity in names[4:6]:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is {material}")
        if rng.random() < 0.8:
            prop = rng.choice(_PROPERTIES)
            whys.append(
                (f"why is the {entity} {prop} "
                 f"{prop_values[(material, prop)]} units?",
                 "derived", [material, prop]))
    # Unheld.
    for entity in names[6:8]:
        statements.append(
            f"the {entity} height is {rng.randrange(3, 900)} meters")
        if rng.random() < 0.8:
            whys.append(
                (f"why is the {entity} "
                 f"{rng.choice(_PROPERTIES)} 500 units?",
                 "unheld", []))

    directs = []
    for material in rng.sample(_MATERIALS, 2):
        prop = rng.choice(_PROPERTIES)
        directs.append((f"what is the {material} {prop}?",
                        str(prop_values[(material, prop)])))

    rng.shuffle(statements)
    return statements, consolidations, whys, directs


def run_gate(seeds: int = 12) -> WhyReport:
    """Run both arms over generated provenance worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`WhyReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    why_scores = {"silent": [], "answers": []}
    recall_scores = {"silent": [], "answers": []}
    rationalised = 0
    fabricated = 0

    real_flag = thoughts_module._WHY_ANSWERS
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, consolidations, whys, directs = _build_world(rng)

        for arm in ("silent", "answers"):
            thoughts_module._WHY_ANSWERS = (arm == "answers")
            root = Path(tempfile.mkdtemp(prefix=f"uq_why_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)
                for question in consolidations:
                    run_pipeline(question, session)
                    run_pipeline("yes", session)

                won = []
                for question, kind, premise_words in whys:
                    response, _trace = run_pipeline(question, session)
                    explains = response.startswith("Because")
                    corrects = response.startswith("It isn't")
                    if kind == "stated":
                        good = explains and "stated directly" in response
                    elif kind == "consolidated":
                        good = (explains and "consolidated" in response
                                and all(w in response
                                        for w in premise_words))
                    elif kind == "derived":
                        good = (explains and "derived just now" in response
                                and all(w in response
                                        for w in premise_words))
                    elif kind == "corrected":
                        good = (corrects
                                and all(w in response
                                        for w in premise_words))
                        if arm == "answers" and explains:
                            rationalised += 1
                    else:
                        good = not explains and not corrects
                        if arm == "answers" and explains:
                            fabricated += 1
                    won.append(float(good))
                why_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._WHY_ANSWERS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = WhyReport(
        why={arm: statistics.fmean(vals)
             for arm, vals in why_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        rationalised=rationalised,
        fabricated=fabricated,
    )

    contrasts = [why_scores["answers"][i] - why_scores["silent"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.rationalised > 0:
        report.reason = (
            f"the anti-rationalisation line failed: "
            f"{report.rationalised} false premise(s) explained - an "
            "explanation machine that explains anything explains "
            "nothing")
        return report
    if report.fabricated > 0:
        report.reason = (f"{report.fabricated} unheld subject(s) got an "
                         "explanation invented for them")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"why buys nothing ({report.delta:+.3f})"
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
        f"why-questions answer at {report.why['answers']:.3f} against "
        f"the silent arm's {report.why['silent']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero false premises rationalised, zero unheld subjects "
        "explained, and recall held at 1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
