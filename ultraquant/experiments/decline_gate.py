"""Declining a derivation names its premises. The decline gate.

The testimony family's last door: "no" after a DERIVED answer. §11.69
gave contests of stored beliefs their protocol (doubt, ask, revise);
a contested derivation is different — nothing was stored, so nothing
lowers, and a bare "no" cannot say WHICH premise it blames, so no
premise confidence moves either: blaming an unnamed premise would
invent the accusation. What the refusal earns is the trail — §11.31's
thesis, serving correction: every premise named, so the wrong one can
be restated, §11.53 narrates the revision, the chain re-derives from
what is NOW believed, and "yes" consolidates the corrected
conclusion. Contest, correct, re-derive, confirm — the full cycle,
every step spoken.

Both arms run the live pipeline; the baseline arm turns
`_DECLINE_DERIVATIONS` off (the shipped silent clear — "no" after a
derivation fell to chat mush), the treatment arm is current. Worlds
mix, in world-varying proportions: declines (scored: premises named,
nothing consolidated, premise confidences unmoved), full
contest-correct-rederive-confirm loops (scored: the revision
narrated, the re-derived value correct, the corrected conclusion
consolidated), stray "no"s (zero tolerance: inert), and one plain
affirmation per world as the §11.33 floor.

**The criteria, written before the run.**

1. **Gain** — decline correctness and loop correctness must beat the
   baseline arm by more than the seed-to-seed sd.
2. **An unnamed premise is never blamed** — any premise confidence
   moved by a bare decline, in the treatment arm, is a fail, once.
3. **A decline never consolidates** — the declined conclusion present
   in the store is a fail, once.
4. **The §11.33 floor holds** — plain affirmations consolidate in both
   arms.

**It was run once, and PASSED — perfectly on the claim, zero on
every line.**

| arm | declined | floor |
|---|---:|---:|
| mush | 0.472 | 1.000 |
| names | **1.000** | 1.000 |

**+0.528 at 3.07x seed sd, zero unnamed premises blamed, zero
declined conclusions stored, consolidation untouched in both arms.**
Every decline named its whole trail with nothing lowered and nothing
kept; every contest-correct-rederive-confirm loop closed with the
corrected conclusion consolidated and the wrong one gone. The
testimony family is whole: yes consolidates a derivation or confirms
an assertion, no declines a derivation naming its premises or
contests an assertion dropping it to doubt — four responses to two
words, chosen by what was actually pending, and a bare word with
nothing pending still moves nothing anywhere.

Run it::

    python -m ultraquant.experiments.decline_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DeclineReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]


@dataclass
class DeclineReport:
    """The two arms, the accusation line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        declined: Arm -> decline/loop correctness over all worlds.
        blamed: Premise confidences moved by bare declines (must be 0).
        consolidated: Declined conclusions found stored (must be 0).
        floor: Arm -> plain-affirmation correctness (1.0 both).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    declined: dict = field(default_factory=dict)
    blamed: int = 0
    consolidated: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'declined':>9} {'floor':>7}"]
        for arm in ("mush", "names"):
            lines.append(f"{arm:<10} {self.declined[arm]:>9.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"unnamed premises blamed (treatment): {self.blamed}",
            f"declined conclusions consolidated: {self.consolidated}",
            f"delta (names vs mush) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One declination world.

    Returns ``(moves, floor_entity)`` — moves are ``(kind, entity,
    material, prop, correction)`` executed in order.
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
        moves.append(("decline", names[spot], rng.choice(_MATERIALS),
                      rng.choice(_PROPERTIES), ""))
        spot += 1
    if rng.random() < 0.8:
        material = rng.choice(_MATERIALS)
        correction = rng.choice(
            [m for m in _MATERIALS if m != material])
        moves.append(("loop", names[spot], material,
                      rng.choice(_PROPERTIES), correction))
        spot += 1
    if rng.random() < 0.8:
        moves.append(("stray", "", "", "", ""))
    floor_entity = names[6]
    return moves, floor_entity


def run_gate(seeds: int = 12) -> DeclineReport:
    """Run both arms over generated declination worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`DeclineReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    declined_scores = {"mush": [], "names": []}
    floor_scores = {"mush": [], "names": []}
    blamed = 0
    consolidated = 0

    real_flag = thoughts_module._DECLINE_DERIVATIONS
    for seed in range(seeds):
        rng = random.Random(seed)
        moves, floor_entity = _build_world(rng)

        for arm in ("mush", "names"):
            thoughts_module._DECLINE_DERIVATIONS = (arm == "names")
            root = Path(tempfile.mkdtemp(prefix=f"uq_dec_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                memory = session.memory

                won = []
                for kind, entity, material, prop, correction in moves:
                    if kind == "decline":
                        value = rng.randrange(100, 900)
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"the {material} {prop} is "
                                     f"{value} units", session)
                        run_pipeline(f"what is the {entity} {prop}?",
                                     session)
                        response, _t = run_pipeline("no", session)
                        stored = memory.recall_fact(f"{entity} {prop}")
                        conf_a = memory.recall_fact(
                            f"{entity} material")["confidence"]
                        conf_b = memory.recall_fact(
                            f"{material} {prop}")["confidence"]
                        good = (f"{entity} material is {material}"
                                in response
                                and "not consolidating" in response
                                and stored is None
                                and abs(conf_a - 0.6) < 1e-9
                                and abs(conf_b - 0.6) < 1e-9)
                        won.append(float(good))
                        if arm == "names":
                            if stored is not None:
                                consolidated += 1
                            if (abs(conf_a - 0.6) > 1e-9
                                    or abs(conf_b - 0.6) > 1e-9):
                                blamed += 1
                    elif kind == "loop":
                        value = rng.randrange(100, 900)
                        new_value = rng.randrange(100, 900)
                        while new_value == value:
                            new_value = rng.randrange(100, 900)
                        run_pipeline(f"the {entity} material is "
                                     f"{material}", session)
                        run_pipeline(f"the {material} {prop} is "
                                     f"{value} units", session)
                        run_pipeline(f"the {correction} {prop} is "
                                     f"{new_value} units", session)
                        run_pipeline(f"what is the {entity} {prop}?",
                                     session)
                        run_pipeline("no", session)
                        revised, _t = run_pipeline(
                            f"the {entity} material is {correction}",
                            session)
                        rederived, _t = run_pipeline(
                            f"what is the {entity} {prop}?", session)
                        confirmed, _t = run_pipeline("yes", session)
                        stored = memory.recall_fact(f"{entity} {prop}")
                        good = ("revises" in revised
                                and str(new_value) in rederived
                                and "Consolidated" in confirmed
                                and stored is not None
                                and str(new_value)
                                in str(stored["value"]))
                        won.append(float(good))
                    else:
                        before = {k: memory.recall_fact(k)["confidence"]
                                  for k in memory.fact_keys()}
                        run_pipeline("no", session)
                        after = {k: memory.recall_fact(k)["confidence"]
                                 for k in memory.fact_keys()}
                        won.append(float(before == after))
                declined_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                run_pipeline(f"the {floor_entity} material is iron",
                             session)
                run_pipeline("the iron density is 300 units", session)
                run_pipeline(f"what is the {floor_entity} density?",
                             session)
                response, _t = run_pipeline("yes", session)
                floor_scores[arm].append(
                    float("Consolidated" in response))
            finally:
                thoughts_module._DECLINE_DERIVATIONS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = DeclineReport(
        declined={arm: statistics.fmean(vals)
                  for arm, vals in declined_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        blamed=blamed,
        consolidated=consolidated,
    )

    contrasts = [declined_scores["names"][i]
                 - declined_scores["mush"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.blamed > 0:
        report.reason = (f"an unnamed premise was blamed "
                         f"{report.blamed} time(s)")
        return report
    if report.consolidated > 0:
        report.reason = (f"a declined conclusion was consolidated "
                         f"{report.consolidated} time(s)")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "the §11.33 consolidation floor regressed"
        return report
    if report.delta <= 0:
        report.reason = f"naming buys nothing ({report.delta:+.3f})"
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
        f"declines land at {report.declined['names']:.3f} against the "
        f"mush arm's {report.declined['mush']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "no unnamed premise blamed, no declined conclusion stored, "
        "and consolidation untouched")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
