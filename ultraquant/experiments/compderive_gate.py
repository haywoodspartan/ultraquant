"""Derived operands compare. The compderive gate.

§11.54 built the comparative owner and registered its ceiling in the
same breath: the comparator compares RECALLED operands, and "is the
mill hardness bigger than the wall hardness?" — both sides derivable
through material chains, neither stored — refused honestly. This unit
cashes that ceiling the way §11.52 cashed §11.47's: an operand the
store does not hold may be DERIVED through the chain machinery, under
the full discipline the question family already carries — §11.50's
split rules (a modifier-rescued derivation is a wrong split and
refuses), §11.48's polarity line (a derived denial holds no number to
compare), and §11.54's own honesty (the verdict names both values, and
now marks WHICH side was derived and its trail: "mill hardness is 500
units (derived via steel)").

Both arms run the live pipeline; the baseline arm turns
`_COMPARE_DERIVES` off (§11.54's shipped refusal), the treatment arm
is current. Worlds mix, in world-varying proportions: both-derived
comparatives in both directions, mixed comparatives (one stored
operand, one derived), ~25% broken-chain comparatives — the entity's
material missing, or the material's property never stored — refused by
both arms and scored zero (the unwinnable share, per §11.35), and
negated-material comparatives (the §11.48 inhibition crossing into
this door). Stored-operand comparatives and direct recalls are the
floors.

**The criteria, written before the run.**

1. **Gain** — derived-operand comparative correctness (right verdict,
   both values named, every derived side marked "(derived via ...)")
   must beat the baseline arm by more than the seed-to-seed sd.
   Broken-chain comparatives score zero for both arms.
2. **No verdict through a broken or denied chain** — a broken-chain or
   negated-material comparative answered Yes or No in the treatment
   arm is a fail, once.
3. **The trail is named** — every derived side in a delivered verdict
   carries its "(derived via ...)" mark. One bare derived verdict is a
   fail.
4. **Nothing regresses** — stored-operand comparatives answer at 1.000
   in BOTH arms (the flag gates only derivation), and direct recall is
   1.000.

**It was run once, and PASSED.**

| arm | derived | stored | recall |
|---|---:|---:|---:|
| refuses | 0.000 | 1.000 | 1.000 |
| derives | **0.674** | 1.000 | 1.000 |

**+0.674 at 5.84x seed sd.** Every delivered verdict named both values
and marked every derived side with its trail ("mill hardness is 500
units (derived via steel)"), zero verdicts crossed a broken or denied
chain — the §11.48 inhibition holding through yet another door — and
stored-operand comparatives and recall sat at 1.000 in both arms: the
flag gated only derivation, exactly as claimed. The 0.326 miss is the
broken-and-denied share scoring zero by design. Three families now
compose in one answer: the chain machinery supplies the operands, the
§11.42 table converts them, and the §11.54 comparator delivers a
verdict that can be checked at a glance.

Run it::

    python -m ultraquant.experiments.compderive_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CompDeriveReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite"]
_COMPS_UP = ["bigger", "higher", "larger"]


@dataclass
class CompDeriveReport:
    """The two arms, the zero-tolerance lines, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        derived: Arm -> derived-comparative correctness over all worlds.
        fabricated: Broken/negated comparatives answered in treatment
            (must be 0).
        trailed: Delivered derived verdicts naming every derived
            side's trail (must be 1.0).
        stored: Arm -> stored-operand comparative correctness
            (must be 1.0 in both).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Derived-correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    derived: dict = field(default_factory=dict)
    fabricated: int = 0
    trailed: float = 1.0
    stored: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'derived':>8} {'stored':>7} "
                 f"{'recall':>7}"]
        for arm in ("refuses", "derives"):
            lines.append(
                f"{arm:<10} {self.derived[arm]:>8.3f} "
                f"{self.stored[arm]:>7.3f} {self.recall[arm]:>7.3f}")
        lines += [
            f"broken/denied chains answered (treatment): "
            f"{self.fabricated}",
            f"derived sides trailed: {self.trailed:.3f}",
            f"delta (derives vs refuses) {self.delta:+.3f}   "
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
    """One derived-comparative world.

    Returns ``(statements, questions, stored_qs, directs)`` where
    questions are ``(question, kind, values, derived_sides)`` — kind
    in yes/no/mixed-yes/mixed-no/broken/negated; derived_sides the
    count of sides a correct verdict must mark "(derived via".
    """
    names = []
    seen = set()
    while len(names) < 10:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    statements = []
    hardness = {}
    for material in _MATERIALS:
        value = rng.randrange(100, 900)
        hardness[material] = value
        statements.append(f"the {material} hardness is {value} units")

    # Chain entities: hardness derivable, never stored.
    chain_mats = {}
    for entity in names[:3]:
        material = rng.choice(_MATERIALS)
        chain_mats[entity] = material
        statements.append(f"the {entity} material is {material}")
    ordered = sorted(names[:3],
                     key=lambda n: hardness[chain_mats[n]])
    low, high = ordered[0], ordered[-1]
    if hardness[chain_mats[low]] == hardness[chain_mats[high]]:
        # Same material drawn everywhere: reroll the high entity to a
        # strictly harder material so yes/no kinds stay determinate.
        harder = max(_MATERIALS, key=lambda m: hardness[m])
        if chain_mats[high] != harder:
            chain_mats[high] = harder
            statements.append(f"the {high} material is {harder}")

    # A stored-operand entity for mixed kinds and the floor.
    stored_entity = names[3]
    stored_value = rng.randrange(100, 900)
    statements.append(
        f"the {stored_entity} hardness is {stored_value} units")
    stored_entity2 = names[4]
    stored_value2 = rng.randrange(100, 900)
    while stored_value2 == stored_value:
        stored_value2 = rng.randrange(100, 900)
    statements.append(
        f"the {stored_entity2} hardness is {stored_value2} units")

    # Broken chains and the negated material.
    no_prop_entity = names[5]
    statements.append(f"the {no_prop_entity} material is oak")
    no_material_entity = names[6]
    statements.append(
        f"the {no_material_entity} height is "
        f"{rng.randrange(10, 900)} meters")
    negated_entity = names[7]
    statements.append(
        f"the {negated_entity} material is not "
        f"{rng.choice(_MATERIALS)}")

    questions = []
    for _ in range(rng.randrange(1, 3)):
        up = rng.choice(_COMPS_UP)
        questions.append(
            (f"is the {high} hardness {up} than the {low} hardness?",
             "yes",
             [str(hardness[chain_mats[high]]),
              str(hardness[chain_mats[low]])], 2))
    if rng.random() < 0.8:
        up = rng.choice(_COMPS_UP)
        questions.append(
            (f"is the {low} hardness {up} than the {high} hardness?",
             "no",
             [str(hardness[chain_mats[low]]),
              str(hardness[chain_mats[high]])], 2))
    # Mixed: derived vs stored.
    entity = rng.choice([low, high])
    derived_value = hardness[chain_mats[entity]]
    kind = ("mixed-yes" if derived_value > stored_value
            else "mixed-no")
    if derived_value != stored_value:
        questions.append(
            (f"is the {entity} hardness "
             f"{rng.choice(_COMPS_UP)} than the {stored_entity} "
             "hardness?", kind,
             [str(derived_value), str(stored_value)], 1))
    # The unwinnable share and the denial.
    for broken in (no_prop_entity, no_material_entity):
        if rng.random() < 0.6:
            questions.append(
                (f"is the {broken} hardness "
                 f"{rng.choice(_COMPS_UP)} than the {low} hardness?",
                 "broken", [], 0))
    if rng.random() < 0.8:
        questions.append(
            (f"is the {negated_entity} hardness "
             f"{rng.choice(_COMPS_UP)} than the {low} hardness?",
             "negated", [], 0))

    stored_qs = []
    kind = ("yes" if stored_value > stored_value2 else "no")
    stored_qs.append(
        (f"is the {stored_entity} hardness "
         f"{rng.choice(_COMPS_UP)} than the {stored_entity2} "
         "hardness?", kind,
         [str(stored_value), str(stored_value2)]))

    directs = []
    for material in rng.sample(_MATERIALS, 2):
        directs.append((f"what is the {material} hardness?",
                        str(hardness[material])))

    rng.shuffle(statements)
    return statements, questions, stored_qs, directs


def run_gate(seeds: int = 12) -> CompDeriveReport:
    """Run both arms over generated derived-comparative worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`CompDeriveReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    derived_scores = {"refuses": [], "derives": []}
    stored_scores = {"refuses": [], "derives": []}
    recall_scores = {"refuses": [], "derives": []}
    fabricated = 0
    trailed_hits, trailed_total = 0, 0

    real_flag = thoughts_module._COMPARE_DERIVES
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, questions, stored_qs, directs = _build_world(rng)

        for arm in ("refuses", "derives"):
            thoughts_module._COMPARE_DERIVES = (arm == "derives")
            root = Path(tempfile.mkdtemp(prefix=f"uq_cd_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, kind, values, d_sides in questions:
                    response, _trace = run_pipeline(question, session)
                    shape = _classify(response)
                    named = all(v in response for v in values)
                    marks = response.count("(derived via")
                    if kind in ("yes", "mixed-yes"):
                        good = (shape == "yes" and named
                                and marks == d_sides)
                        won.append(float(good))
                    elif kind in ("no", "mixed-no"):
                        good = (shape == "no" and named
                                and marks == d_sides)
                        won.append(float(good))
                    else:
                        won.append(0.0)
                        if arm == "derives" and shape != "open":
                            fabricated += 1
                    if (arm == "derives" and kind not in
                            ("broken", "negated")
                            and shape in ("yes", "no")):
                        trailed_total += 1
                        trailed_hits += int(marks == d_sides)
                derived_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, kind, values in stored_qs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(_classify(response) == kind
                                      and all(v in response
                                              for v in values)))
                stored_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._COMPARE_DERIVES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = CompDeriveReport(
        derived={arm: statistics.fmean(vals)
                 for arm, vals in derived_scores.items()},
        stored={arm: statistics.fmean(vals)
                for arm, vals in stored_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        fabricated=fabricated,
        trailed=(trailed_hits / trailed_total) if trailed_total
        else 1.0,
    )

    contrasts = [derived_scores["derives"][i]
                 - derived_scores["refuses"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.fabricated > 0:
        report.reason = (
            f"a verdict through a broken or denied chain: "
            f"{report.fabricated} in the treatment arm")
        return report
    if report.trailed < 1.0:
        report.reason = (f"the trail went unnamed: only "
                         f"{report.trailed:.3f} of derived verdicts "
                         "marked every derived side")
        return report
    if min(report.stored.values()) < 1.0:
        report.reason = ("stored-operand comparatives regressed: the "
                         "flag must gate only derivation")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"deriving buys nothing ({report.delta:+.3f})"
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
        f"derived-operand comparatives answer at "
        f"{report.derived['derives']:.3f} against the refusing arm's "
        f"{report.derived['refuses']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), every derived side "
        "named its trail, zero verdicts crossed a broken or denied "
        "chain, and stored comparatives and recall held at 1.000 in "
        "both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
