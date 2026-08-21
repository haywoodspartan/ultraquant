"""Compound disjuncts compare whole. The multiword-choice gate.

§11.66 shipped choices with multi-word disjuncts as its registered
ceiling — the branch refused "reinforced steel or iron" rather than
half-parse a compound alternative. The ceiling cashes the family way:
fold-equality compares a compound alternative WHOLE or not at all, so
there was never anything to half-parse — "is the spire architect wren
or wren the younger?" picks the held compound by name, and "reinforced
steel" against a held "steel" answers Neither, because a qualified
value is a different value (§11.47's lesson, in question form). The
spoken lead names the belief AS HELD: article-stripping made "wren the
younger" parse as "wren younger", and the first probe caught the lead
speaking the mangled parse — the answer now leads with the stored
value.

Both arms run the live pipeline; the baseline arm turns
`_CHOICE_MULTIWORD` off (§11.66's shipped refusal — those questions
fall to the hedging machinery), the treatment arm is current. Worlds
mix, in world-varying proportions: compound picks in both orders
(entities whose value is "{agent} the {epithet}"), qualified-value
neithers ("reinforced {material} or {other}" against a held bare
value), plain single-word choices (the §11.66 floor, identical in
both arms), and unheld compound choices (the hedge — absence still
never picks a side, however many words the side has).

**The criteria, written before the run.**

1. **Gain** — compound-choice correctness (the held compound named
   first, as stored; qualified-value questions answering Neither with
   the actual) must beat the baseline arm by more than the
   seed-to-seed sd.
2. **Absence never picks a side** — an unheld subject's answer
   starting with any offered alternative or "Neither" in the
   treatment arm is a fail, once.
3. **Single-word choices are untouched** — §11.66's forms answer
   identically in both arms at 1.000.

**It was run once, and PASSED — perfectly on the claim, zero on the
line.**

| arm | chosen | floor |
|---|---:|---:|
| refuses | 0.175 | 1.000 |
| compares | **1.000** | 1.000 |

**+0.825 at 7.66x seed sd, zero sides elected without grounds** — the
held compound named first as stored in either order, qualified values
answering Neither with the actual, unheld compound subjects hedging,
and §11.66's single-word forms identical at 1.000 in both arms. The
refusing arm's 0.175 is its unheld free share. A ceiling registered
and cashed within one session, the third such: the family's method
carries its own momentum now.

Run it::

    python -m ultraquant.experiments.multichoice_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MultiChoiceReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_AGENTS = ["wren", "aldous", "berta", "cosimo", "delia", "edmund"]
_EPITHETS = ["younger", "elder", "lesser"]
_ROLES = ["architect", "founder", "sculptor", "keeper"]


@dataclass
class MultiChoiceReport:
    """The two arms, the election line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        chosen: Arm -> compound-choice correctness over all worlds.
        elected: Sides picked without grounds in treatment (must be 0).
        floor: Arm -> single-word choice correctness (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    chosen: dict = field(default_factory=dict)
    elected: int = 0
    floor: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'chosen':>7} {'floor':>7}"]
        for arm in ("refuses", "compares"):
            lines.append(f"{arm:<10} {self.chosen[arm]:>7.3f} "
                         f"{self.floor[arm]:>7.3f}")
        lines += [
            f"sides elected without grounds (treatment): {self.elected}",
            f"delta (compares vs refuses) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One multiword-choice world.

    Returns ``(setup, questions, floor_qs)`` where questions are
    ``(question, kind, payload)`` — kind in pick/neither/unheld.
    """
    names = []
    seen = set()
    while len(names) < 7:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    setup = []
    questions = []
    floor_qs = []

    # Compound picks, both orders.
    for entity in names[:1 + rng.randrange(1, 3)]:
        role = rng.choice(_ROLES)
        base = rng.choice(_AGENTS)
        epithet = rng.choice(_EPITHETS)
        compound = f"{base} the {epithet}"
        setup.append(f"the {entity} {role} is {compound}")
        pair = ([base, compound] if rng.random() < 0.5
                else [compound, base])
        questions.append(
            (f"is the {entity} {role} {pair[0]} or {pair[1]}?",
             "pick", {"lead": f"{compound.capitalize()}",
                      "bare": base}))
    # Qualified-value neithers.
    if rng.random() < 0.8:
        entity = names[3]
        material = rng.choice(_MATERIALS)
        other = rng.choice([m for m in _MATERIALS if m != material])
        setup.append(f"the {entity} material is {material}")
        questions.append(
            (f"is the {entity} material reinforced {material} or "
             f"{other}?", "neither", {"true": material}))
    # Unheld compound choices.
    if rng.random() < 0.8:
        base = rng.choice(_AGENTS)
        questions.append(
            (f"is the {names[4]} {rng.choice(_ROLES)} {base} or "
             f"{base} the {rng.choice(_EPITHETS)}?",
             "unheld", {"base": base}))

    # The §11.66 floor.
    entity = names[5]
    material = rng.choice(_MATERIALS)
    other = rng.choice([m for m in _MATERIALS if m != material])
    setup.append(f"the {entity} material is {material}")
    floor_qs.append(
        (f"is the {entity} material {material} or {other}?",
         material))

    return setup, questions, floor_qs


def run_gate(seeds: int = 12) -> MultiChoiceReport:
    """Run both arms over generated multiword-choice worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`MultiChoiceReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    chosen_scores = {"refuses": [], "compares": []}
    floor_scores = {"refuses": [], "compares": []}
    elected = 0

    real_flag = thoughts_module._CHOICE_MULTIWORD
    for seed in range(seeds):
        rng = random.Random(seed)
        setup, questions, floor_qs = _build_world(rng)

        for arm in ("refuses", "compares"):
            thoughts_module._CHOICE_MULTIWORD = (arm == "compares")
            root = Path(tempfile.mkdtemp(prefix=f"uq_mc_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in setup:
                    run_pipeline(statement, session)

                won = []
                for question, kind, payload in questions:
                    response, _trace = run_pipeline(question, session)
                    lowered = response.strip().lower()
                    if kind == "pick":
                        won.append(float(
                            response.startswith(payload["lead"])))
                    elif kind == "neither":
                        won.append(float(
                            response.startswith("Neither")
                            and payload["true"] in response))
                    else:
                        picked = (lowered.startswith("neither")
                                  or lowered.startswith(
                                      payload["base"]))
                        won.append(float(not picked))
                        if arm == "compares" and picked:
                            elected += 1
                chosen_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, lead in floor_qs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(response.strip().lower()
                                      .startswith(lead)))
                floor_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._CHOICE_MULTIWORD = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = MultiChoiceReport(
        chosen={arm: statistics.fmean(vals)
                for arm, vals in chosen_scores.items()},
        floor={arm: statistics.fmean(vals)
               for arm, vals in floor_scores.items()},
        elected=elected,
    )

    contrasts = [chosen_scores["compares"][i]
                 - chosen_scores["refuses"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.elected > 0:
        report.reason = (
            f"a side was elected without grounds {report.elected} "
            "time(s) - however many words the side has")
        return report
    if min(report.floor.values()) < 1.0:
        report.reason = "single-word choices regressed"
        return report
    if report.delta <= 0:
        report.reason = f"the lift buys nothing ({report.delta:+.3f})"
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
        f"compound choices answer at {report.chosen['compares']:.3f} "
        f"against the refusing arm's {report.chosen['refuses']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "zero sides elected without grounds, and single-word choices "
        "held at 1.000 in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
