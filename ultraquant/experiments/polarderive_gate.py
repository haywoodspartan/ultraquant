"""Polar questions answered through chains. The polar-derive gate.

§11.48 gave polar questions their matrix over STORED facts; §11.49 gave
denials a voice at chain terminals. The seam between them: "is the dome
city climate temperate?" when no fact holds the key "dome city climate"
— the verdict is derivable ("dome city is york"; "york climate is not
temperate" entails No) but the polar branch only consulted recall. This
unit closes the §11.48 family: a polar question whose subject is not
stored DERIVES it through the chain machinery and meets the claim under
the same matrix — polarity included, the reply marked derived with its
trail, and the affirmation flow offered ("yes" consolidates the derived
conclusion, §11.33's protocol unchanged).

Building it caught two split bugs before the gate ran, both recorded in
the pins: the subject ladder let a stored SHORT prefix ("dome city")
claim a question about a longer derivable subject ("dome city
climate"), answering about the city; and a wrong split survived by
modifier tolerance ("tower hardness 490" deriving with '490' read as a
modifier) — a derivation that reads part of its own subject away means
the split was wrong, and refuses. A trailing "not" may never end a
subject: the polarity word belongs to the claim.

Both arms run the live pipeline; the baseline arm turns
`_POLAR_DERIVES` off (§11.48/49 shipped state), the treatment arm is
current. Worlds hold derivable-subject polar questions of four kinds in
world-varying mixes (yes / no / negated-terminal / unknown), entities
whose only material knowledge is negative (polar questions against
their properties must stay open — the inhibition line crossing into the
interrogative form), direct-fact polar questions and recalls as floors.

**The criteria, written before the run.**

1. **Gain** — derived-polar correctness (yes-kind yes, no-kind no with
   the actual derived value, negated-kind no, unknown-kind neither)
   must beat the baseline arm by more than the seed-to-seed sd.
2. **Absence is never no** — an unknown-kind question answered yes or
   no in the treatment arm is a fail, once.
3. **Inhibition crosses the form** — polar questions over a negated
   MIDDLE must answer yes/no ZERO times in both arms: the interrogative
   door must not re-admit the denial bridge.
4. **Honesty** — every derived verdict names "derived" and its "via"
   trail. One bare verdict is a fail.
5. **Nothing regresses** — direct-fact polar questions answer at 1.000
   in BOTH arms (the flag gates only derivation), and direct recall is
   1.000.

**It was run once, and PASSED — perfectly on the claim, zero on every
control.**

| arm | derived | direct | recall | mid-verdicts |
|---|---:|---:|---:|---:|
| recall-only | 0.688 | 1.000 | 1.000 | 0 |
| derives | **1.000** | 1.000 | 1.000 | 0 |

**+0.312 at 5.11x seed sd.** The treatment arm answered every derived
kind — yes, no with the actual value, no through a negated terminal,
open for the unheld — every verdict named "derived" and its "via"
trail, no unknown subject was denied, negated middles got zero
verdicts in either arm (the inhibition line holds across the
interrogative form), and both floors stayed at 1.000. The baseline's
0.688 is the unknown-kind share its hedge answers by being honest —
the same asymmetry §11.48's gate measured, one rung up. The §11.48
family is closed: statements parse polarity, storage carries it,
reasoning is inhibited by it, terminals speak it, and questions —
recalled or derived — meet it with the same matrix.

Run it::

    python -m ultraquant.experiments.polarderive_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PolarDeriveReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_PLACES = ["york", "delft", "siena", "bruges", "cadiz", "trento",
           "arles", "leipzig"]
_ATTRS = ["climate", "terrain", "air"]
_ATTR_VALUES = ["temperate", "humid", "arid", "rocky", "flat", "misty"]
_MATERIALS = ["steel", "iron", "copper", "granite"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class PolarDeriveReport:
    """The two arms, three controls, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        derived: Arm -> derived-polar correctness over all worlds.
        absence_denied: Unknown-kind yes/no answers in treatment
            (must be 0).
        middle_verdicts: Arm -> yes/no answers over negated middles
            (must be 0 in both).
        honesty: Derived verdicts naming "derived" and "via" (1.0).
        direct: Arm -> direct-fact polar correctness (must be 1.0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Derived-polar contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    derived: dict = field(default_factory=dict)
    absence_denied: int = 0
    middle_verdicts: dict = field(default_factory=dict)
    honesty: float = 1.0
    direct: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'derived':>8} {'direct':>7} "
                 f"{'recall':>7} {'mid-verdicts':>13}"]
        for arm in ("recall-only", "derives"):
            lines.append(
                f"{arm:<10} {self.derived[arm]:>8.3f} "
                f"{self.direct[arm]:>7.3f} {self.recall[arm]:>7.3f} "
                f"{self.middle_verdicts[arm]:>13}")
        lines += [
            f"unknown subjects denied (treatment): {self.absence_denied}",
            f"derived verdicts naming trail: {self.honesty:.3f}",
            f"delta (derive on vs off) {self.delta:+.3f}   "
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
    """One polar-derivation world.

    Returns ``(statements, derived_polar, middles, direct_polar,
    directs)``: derived_polar entries are ``(question, kind)`` with kind
    in yes/no/negated/unknown; middles are polar questions over a
    negated middle; direct_polar are ``(question, kind)`` against
    stored facts; directs are ``(question, expected)``.
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

    derived_polar = []
    used_places = rng.sample(_PLACES, 6)
    # Positive chains: entity -> place -> attribute value.
    for spot, entity in enumerate(names[:3]):
        place = used_places[spot]
        statements.append(f"the {entity} city is {place}")
        attr = rng.choice(_ATTRS)
        value = rng.choice(_ATTR_VALUES)
        statements.append(f"the {place} {attr} is {value}")
        kinds = ["yes", "no"]
        rng.shuffle(kinds)
        for kind in kinds[:rng.randrange(1, 3)]:
            if kind == "yes":
                derived_polar.append(
                    (f"is the {entity} city {attr} {value}?", "yes"))
            else:
                other = rng.choice(
                    [v for v in _ATTR_VALUES if v != value])
                derived_polar.append(
                    (f"is the {entity} city {attr} {other}?", "no"))
    # Negated terminals: the claim names the denied value.
    for spot, entity in enumerate(names[3:5]):
        place = used_places[3 + spot]
        statements.append(f"the {entity} city is {place}")
        attr = rng.choice(_ATTRS)
        denied = rng.choice(_ATTR_VALUES)
        statements.append(f"the {place} {attr} is not {denied}")
        if rng.random() < 0.8:
            derived_polar.append(
                (f"is the {entity} city {attr} {denied}?", "negated"))
    # Unknown subjects: no chain exists.
    for entity in names[5:7]:
        statements.append(
            f"the {entity} height is {rng.randrange(3, 900)} meters")
        if rng.random() < 0.8:
            derived_polar.append(
                (f"is the {entity} city {rng.choice(_ATTRS)} "
                 f"{rng.choice(_ATTR_VALUES)}?", "unknown"))

    middles = []
    for entity in names[7:9]:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is not {material}")
        prop = rng.choice(_PROPERTIES)
        middles.append(
            f"is the {entity} {prop} "
            f"{prop_values[(material, prop)]} units?")

    direct_polar = []
    entity = names[9]
    material = rng.choice(_MATERIALS)
    statements.append(f"the {entity} material is {material}")
    direct_polar.append((f"is the {entity} material {material}?", "yes"))
    other = rng.choice([m for m in _MATERIALS if m != material])
    direct_polar.append((f"is the {entity} material {other}?", "no"))

    directs = []
    for material in rng.sample(_MATERIALS, 2):
        prop = rng.choice(_PROPERTIES)
        directs.append((f"what is the {material} {prop}?",
                        str(prop_values[(material, prop)])))

    rng.shuffle(statements)
    return statements, derived_polar, middles, direct_polar, directs


def run_gate(seeds: int = 12) -> PolarDeriveReport:
    """Run both arms over generated polar-derivation worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`PolarDeriveReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    derived_scores = {"recall-only": [], "derives": []}
    direct_scores = {"recall-only": [], "derives": []}
    recall_scores = {"recall-only": [], "derives": []}
    middle_verdicts = {"recall-only": 0, "derives": 0}
    absence_denied = 0
    honest_hits, honest_total = 0, 0

    real_flag = thoughts_module._POLAR_DERIVES
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, derived_polar, middles, direct_polar, directs = \
            _build_world(rng)

        for arm in ("recall-only", "derives"):
            thoughts_module._POLAR_DERIVES = (arm == "derives")
            root = Path(tempfile.mkdtemp(prefix=f"uq_pold_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind in derived_polar:
                    response, _trace = run_pipeline(question, session)
                    shape = _classify(response)
                    if kind == "yes":
                        good = shape == "yes"
                    elif kind in ("no", "negated"):
                        good = shape == "no"
                    else:
                        good = shape == "open"
                        if arm == "derives" and shape != "open":
                            absence_denied += 1
                    won.append(float(good))
                    if arm == "derives" and shape in ("yes", "no") \
                            and kind != "unknown":
                        honest_total += 1
                        honest_hits += int("derived" in response
                                           and "via" in response)
                derived_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                for question in middles:
                    response, _trace = run_pipeline(question, session)
                    if _classify(response) != "open":
                        middle_verdicts[arm] += 1

                hits = []
                for question, kind in direct_polar:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(_classify(response) == kind))
                direct_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._POLAR_DERIVES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = PolarDeriveReport(
        derived={arm: statistics.fmean(vals)
                 for arm, vals in derived_scores.items()},
        direct={arm: statistics.fmean(vals)
                for arm, vals in direct_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        middle_verdicts=dict(middle_verdicts),
        absence_denied=absence_denied,
        honesty=(honest_hits / honest_total) if honest_total else 1.0,
    )

    contrasts = [derived_scores["derives"][i]
                 - derived_scores["recall-only"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.absence_denied > 0:
        report.reason = (
            f"absence was denied: {report.absence_denied} unknown "
            "subject(s) answered yes or no")
        return report
    if max(report.middle_verdicts.values()) > 0:
        report.reason = (
            f"the interrogative door re-admitted the bridge: negated "
            f"middles got verdicts (recall-only "
            f"{report.middle_verdicts['recall-only']}, derives "
            f"{report.middle_verdicts['derives']})")
        return report
    if report.honesty < 1.0:
        report.reason = (
            f"honesty failed: only {report.honesty:.3f} of derived "
            "verdicts named their trail")
        return report
    if min(report.direct.values()) < 1.0:
        report.reason = ("direct-fact polar questions regressed: the "
                         "flag must gate only derivation")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"derivation buys nothing ({report.delta:+.3f})"
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
        f"derived polar questions answer at "
        f"{report.derived['derives']:.3f} against the recall-only "
        f"arm's {report.derived['recall-only']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), every verdict named its "
        "trail, no unknown subject was denied, negated middles got zero "
        "verdicts in either arm, and direct polar and recall held at "
        "1.000")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
