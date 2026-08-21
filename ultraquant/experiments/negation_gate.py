"""Belief-of-absence against absence-of-belief. The negation gate.

The library had no concept of polarity, and the probe that opened this
unit caught a live fabrication: "the dome material is not steel" stored
"not steel" as a value, the fold dropped "not" as a stopword, and the
chain machinery derived the dome's melting point "via not steel" — a
denial bridged as an assertion, the premise trail displaying the exact
statement it was contradicting. Negation is now a first-class polarity:
stored as a flag on the record (part of the fact's identity — a
positive restatement is a revision, never a reinforcement), inhibitory
everywhere the network reasons (negated facts neither seed nor relay
the spread, never join arithmetic, never mint curiosity premises), and
surfaced in speech ("believed not steel").

With polarity comes the question form that needs it: "is X Y?" answers
yes against matching belief, no against contrary belief — a different
stored value, or a stored negation — and an honest don't-know
otherwise. The line this gate exists to hold: **absence is never no.**
A subject the library holds nothing about must not be denied; "no" is
reserved for actual contrary belief. The distinction between not
believing X and believing not-X is the whole point of having polarity.

Both arms run the live pipeline over the same worlds; the baseline arm
turns `_NEGATION_AWARE` off (shipped behavior: "not steel" stored raw,
no polar branch), the treatment arm runs the new machinery. Worlds mix
polar questions of four kinds in world-varying proportions (the
variance), entities whose only material knowledge is negative (the
chain control), unheld subjects (the absence control), and genuine
positive chains plus direct recalls (the regression floor).

**The criteria, written before the run.**

1. **Gain** — polar-question correctness (yes-kind answered yes,
   no-kind answered no with the actual value, negated-kind answered no,
   unknown-kind answered without a yes or a no) must beat the baseline
   arm by more than the seed-to-seed sd.
2. **A denial never derives** — chain questions whose only premise is
   negative must be asserted ZERO times in the treatment arm. The
   baseline count is reported beside it: it is today's fabrication,
   measured.
3. **Absence is never no** — an unheld subject answered "yes" or "no"
   in the treatment arm is a fail, once, sd or no sd.
4. **Nothing regresses** — genuine positive chains answer in the
   treatment arm at no less than the baseline rate, and direct recall
   is 1.000 in both arms.

**It was run once, and PASSED — with the old fabrication measured
beside the fix.**

| arm | polar | chains | recall | denial-derived |
|---|---:|---:|---:|---:|
| baseline | 0.273 | 1.000 | 1.000 | **36** |
| polarity | **1.000** | 1.000 | 1.000 | **0** |

**+0.727 at 28.33x seed sd.** The baseline arm derived a chain
straight through a denial 36 times — every negative-premise question
in every world, the fabrication this unit exists to close, measured at
its full rate. The polarity arm derived zero, denied zero unheld
subjects (the hedge answered every one — absence stayed open), scored
every polar kind perfectly, and moved neither chains nor recall. The
margin is wide because the baseline's polar score is nearly constant
(the ~0.27 it gets is the unknown-kind share its hedge answers by
being honest) while the treatment answers everything — a capability
the baseline simply does not have, measured against the arm that
proves the worlds allowed nothing else to claim it.

Negative chain ANSWERS — "the dome climate is believed not temperate"
derived through a chain — stay a registered candidate: phase one made
denials inert everywhere the network reasons; making them speak at
depth is its own claim or none.

Run it::

    python -m ultraquant.experiments.negation_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NegationReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class NegationReport:
    """The two arms, three controls, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        polar: Arm -> polar-question correctness over all worlds.
        derived_from_denial: Arm -> negative-premise chain assertions
            (treatment must be 0; baseline is the measured fabrication).
        absence_denied: Unheld subjects answered yes/no in treatment
            (must be 0).
        chains: Arm -> genuine positive-chain answer rate.
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Polar-correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    polar: dict = field(default_factory=dict)
    derived_from_denial: dict = field(default_factory=dict)
    absence_denied: int = 0
    chains: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'polar':>7} {'chains':>7} {'recall':>7} "
                 f"{'denial-derived':>15}"]
        for arm in ("baseline", "polarity"):
            lines.append(
                f"{arm:<10} {self.polar[arm]:>7.3f} "
                f"{self.chains[arm]:>7.3f} {self.recall[arm]:>7.3f} "
                f"{self.derived_from_denial[arm]:>15}")
        lines += [
            f"unheld subjects denied (treatment): {self.absence_denied}",
            f"delta (polar, with vs without) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _classify(response: str) -> str:
    lowered = response.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return "open"


def _build_world(rng: random.Random):
    """One polarity world.

    Returns ``(facts, polar, denial_chains, chains, directs)`` where
    ``facts`` are ``(key, value)`` statement texts fed through the
    pipeline (so each arm parses them its own way), ``polar`` entries
    are ``(question, kind)`` with kind in yes/no/negated/unknown,
    ``denial_chains`` are chain questions whose only premise is
    negative, ``chains`` are ``(question, expected)`` genuine chains,
    and ``directs`` are ``(question, expected)``.
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    prop_values = {}
    statements = []
    for material in _MATERIALS:
        for prop in _PROPERTIES:
            value = rng.randrange(100, 9000)
            prop_values[(material, prop)] = value
            statements.append(
                f"the {material} {prop} is {value} units")

    polar = []
    denial_chains = []
    chains = []
    # Positive entities: material stored, polar yes/no questions and a
    # genuine chain each.
    positives = names[:3]
    for entity in positives:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is {material}")
        other = rng.choice([m for m in _MATERIALS if m != material])
        kinds = ["yes", "no"]
        rng.shuffle(kinds)
        for kind in kinds[:rng.randrange(1, 3)]:
            if kind == "yes":
                polar.append(
                    (f"is the {entity} material {material}?", "yes"))
            else:
                polar.append(
                    (f"is the {entity} material {other}?", "no"))
        prop = rng.choice(_PROPERTIES)
        chains.append((f"what is the {entity} {prop}?",
                       str(prop_values[(material, prop)])))

    # Negative entities: ONLY a negation is held.
    negatives = names[3:6]
    for entity in negatives:
        denied = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is not {denied}")
        if rng.random() < 0.7:
            polar.append(
                (f"is the {entity} material {denied}?", "negated"))
        prop = rng.choice(_PROPERTIES)
        denial_chains.append(f"what is the {entity} {prop}?")

    # Unheld subjects: nothing stored about their material.
    unknowns = names[6:9]
    for entity in unknowns:
        for attr in ("height", "width"):
            statements.append(
                f"the {entity} {attr} is {rng.randrange(3, 900)} meters")
        if rng.random() < 0.8:
            polar.append(
                (f"is the {entity} material {rng.choice(_MATERIALS)}?",
                 "unknown"))

    rng.shuffle(statements)
    directs = []
    for material in rng.sample(_MATERIALS, 3):
        prop = rng.choice(_PROPERTIES)
        directs.append((f"what is the {material} {prop}?",
                        str(prop_values[(material, prop)])))
    return statements, polar, denial_chains, chains, directs


def _assertive(response: str) -> bool:
    lowered = response.lower()
    if lowered.startswith(("i don't hold", "nothing believed",
                           "i can't plan", "i don't know")):
        return False
    return "(confidence" in lowered or "inferred" in lowered


def run_gate(seeds: int = 12) -> NegationReport:
    """Run both arms over generated polarity worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`NegationReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    polar_scores = {"baseline": [], "polarity": []}
    chain_scores = {"baseline": [], "polarity": []}
    recall_scores = {"baseline": [], "polarity": []}
    denial_derived = {"baseline": 0, "polarity": 0}
    absence_denied = 0

    real_flag = thoughts_module._NEGATION_AWARE
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, polar, denial_chains, chains, directs = \
            _build_world(rng)

        for arm in ("baseline", "polarity"):
            thoughts_module._NEGATION_AWARE = (arm == "polarity")
            root = Path(tempfile.mkdtemp(prefix=f"uq_neg_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind in polar:
                    response, _trace = run_pipeline(question, session)
                    shape = _classify(response)
                    if kind == "yes":
                        won.append(float(shape == "yes"))
                    elif kind in ("no", "negated"):
                        won.append(float(shape == "no"))
                    else:
                        won.append(float(shape == "open"))
                        if arm == "polarity" and shape != "open":
                            absence_denied += 1
                polar_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                for question in denial_chains:
                    response, _trace = run_pipeline(question, session)
                    if _assertive(response) and "inferred" in response:
                        denial_derived[arm] += 1

                hits = []
                for question, expected in chains:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response
                                      and "inferred" in response))
                chain_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._NEGATION_AWARE = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = NegationReport(
        polar={arm: statistics.fmean(vals)
               for arm, vals in polar_scores.items()},
        chains={arm: statistics.fmean(vals)
                for arm, vals in chain_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        derived_from_denial=dict(denial_derived),
        absence_denied=absence_denied,
    )

    contrasts = [polar_scores["polarity"][i] - polar_scores["baseline"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.derived_from_denial["polarity"] > 0:
        report.reason = (
            f"a denial derived: "
            f"{report.derived_from_denial['polarity']} chain(s) "
            "asserted from a negative premise in the polarity arm")
        return report
    if report.absence_denied > 0:
        report.reason = (
            f"absence was denied: {report.absence_denied} unheld "
            "subject(s) answered yes or no - not believing X is not "
            "believing not-X")
        return report
    if report.chains["polarity"] < report.chains["baseline"]:
        report.reason = (
            f"positive chains regressed: "
            f"{report.chains['polarity']:.3f} against the baseline's "
            f"{report.chains['baseline']:.3f}")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"polarity buys nothing ({report.delta:+.3f})"
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
        f"polar questions answer at {report.polar['polarity']:.3f} "
        f"against the baseline's {report.polar['baseline']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "no denial derived, no unheld subject was denied, and chains "
        "and recall held in both arms — while the baseline arm derived "
        f"{report.derived_from_denial['baseline']} chain(s) straight "
        "through a denial, which is the fabrication this unit closes")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
