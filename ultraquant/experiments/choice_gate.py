"""Choices: the held disjunct answers by name. The gate.

"is the tower material steel or iron?" reached the polar matrix as
one compound claim, matched nothing, and answered **No — to a
question whose true answer was on the table.** §11.66 gives
disjunctions an owner: the offered alternatives are VALUES, and the
held one answers by name ("Iron - tower material is iron"), "Neither"
answers with the actual, a stored denial rules out its own value
without electing another ("Not steel, at least" — ruling out is not
picking, because a denial of one value affirms nothing about
another), and an unheld subject falls through to the hedge. The
zero-invention line, one more way: **absence never picks a side.**

Multi-word disjuncts ("reinforced steel or iron") are the registered
ceiling: the branch refuses rather than half-parse a compound
alternative, and those questions score zero for both arms — the
unwinnable share (§11.35) and the variance.

Both arms run the live pipeline; the baseline arm turns
`_POLAR_CHOICES` off (the shipped wrong No), the treatment arm is
current. Worlds mix, in world-varying proportions: picks in both
orders, neithers, denial-partial and denial-unrelated choices,
unheld subjects, and the multi-word ceiling.

**The criteria, written before the run.**

1. **Gain** — choice correctness (the true disjunct named first for
   picks; "Neither" with the actual named; "Not {V}, at least" for
   denial-partials with NO other disjunct elected; don't-know for
   denial-unrelated; hedges for the unheld) must beat the baseline by
   more than the seed-to-seed sd.
2. **Absence never picks a side** — an unheld subject's answer
   starting with any offered alternative, or with "Neither", in the
   treatment arm is a fail, once. A denial-partial electing the other
   disjunct is the same fail.
3. **Plain polar is untouched** — non-disjunctive polar questions
   answer identically in both arms at 1.000.

**It was run once, and PASSED.**

| arm | chosen | polar |
|---|---:|---:|
| wrong-no | 0.235 | 1.000 |
| chooses | **0.901** | 1.000 |

**+0.667 at 5.92x seed sd, zero sides elected without grounds** — the
held disjunct named first in either order, "Neither" carrying the
actual, denials ruling out without electing, unheld subjects hedging,
and plain polar identical in both arms. The 0.099 miss is the
multi-word ceiling scoring zero by design; the baseline's 0.235 is
the share its hedges answer by being honest — and its pick-kind zero
is the shipped wrong No, measured beside the fix. The polar family
now covers assertion, denial, derivation, comparison, and choice —
one matrix, and absence never says yes, no, or picks a side through
any door of it.

Run it::

    python -m ultraquant.experiments.choice_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ChoiceReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]


@dataclass
class ChoiceReport:
    """The two arms, the election line, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        chosen: Arm -> choice correctness over all worlds.
        elected: Sides picked without grounds in treatment (must be 0).
        polar: Arm -> plain-polar correctness (must be 1.0).
        delta: Correctness contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    chosen: dict = field(default_factory=dict)
    elected: int = 0
    polar: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'chosen':>7} {'polar':>7}"]
        for arm in ("wrong-no", "chooses"):
            lines.append(f"{arm:<10} {self.chosen[arm]:>7.3f} "
                         f"{self.polar[arm]:>7.3f}")
        lines += [
            f"sides elected without grounds (treatment): {self.elected}",
            f"delta (chooses vs wrong-no) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One choice world.

    Returns ``(setup, questions, polar_qs)`` where questions are
    ``(question, kind, payload)`` — kind in pick/neither/denial-part/
    denial-far/unheld/ceiling; payload carries expectations.
    """
    names = []
    seen = set()
    while len(names) < 8:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            names.append(name)

    setup = []
    questions = []
    polar_qs = []

    # Picks, both orders.
    for entity in names[:1 + rng.randrange(1, 3)]:
        material = rng.choice(_MATERIALS)
        other = rng.choice([m for m in _MATERIALS if m != material])
        setup.append(f"the {entity} material is {material}")
        pair = ([material, other] if rng.random() < 0.5
                else [other, material])
        questions.append(
            (f"is the {entity} material {pair[0]} or {pair[1]}?",
             "pick", {"true": material, "others": [other]}))
        polar_qs.append(
            (f"is the {entity} material {material}?", "yes"))
    # Neither.
    if rng.random() < 0.8:
        entity = names[3]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        wrong = rng.sample([m for m in _MATERIALS if m != material], 2)
        questions.append(
            (f"is the {entity} material {wrong[0]} or {wrong[1]}?",
             "neither", {"true": material}))
    # Denial-partial and denial-unrelated.
    entity = names[4]
    denied = rng.choice(_MATERIALS)
    other = rng.choice([m for m in _MATERIALS if m != denied])
    setup.append(f"the {entity} material is not {denied}")
    questions.append(
        (f"is the {entity} material {denied} or {other}?",
         "denial-part", {"denied": denied, "other": other}))
    if rng.random() < 0.7:
        wrong = rng.sample([m for m in _MATERIALS if m != denied], 2)
        questions.append(
            (f"is the {entity} material {wrong[0]} or {wrong[1]}?",
             "denial-far", {"denied": denied}))
    # Unheld.
    if rng.random() < 0.8:
        pair = rng.sample(_MATERIALS, 2)
        questions.append(
            (f"is the {names[5]} material {pair[0]} or {pair[1]}?",
             "unheld", {"alts": pair}))
    # The multi-word ceiling.
    if rng.random() < 0.55:
        entity = names[6]
        material = rng.choice(_MATERIALS)
        setup.append(f"the {entity} material is {material}")
        other = rng.choice([m for m in _MATERIALS if m != material])
        questions.append(
            (f"is the {entity} material reinforced {material} or "
             f"{other}?", "ceiling", {}))

    return setup, questions, polar_qs


def run_gate(seeds: int = 12) -> ChoiceReport:
    """Run both arms over generated choice worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ChoiceReport`.
    """
    from ultraquant.interpreter import thoughts as thoughts_module
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    chosen_scores = {"wrong-no": [], "chooses": []}
    polar_scores = {"wrong-no": [], "chooses": []}
    elected = 0

    real_flag = thoughts_module._POLAR_CHOICES
    for seed in range(seeds):
        rng = random.Random(seed)
        setup, questions, polar_qs = _build_world(rng)

        for arm in ("wrong-no", "chooses"):
            thoughts_module._POLAR_CHOICES = (arm == "chooses")
            root = Path(tempfile.mkdtemp(prefix=f"uq_choice_{seed}_"))
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
                            lowered.startswith(payload["true"])
                            and all(not lowered.startswith(o)
                                    for o in payload["others"])))
                    elif kind == "neither":
                        won.append(float(
                            response.startswith("Neither")
                            and payload["true"] in response))
                    elif kind == "denial-part":
                        good = (response.startswith(
                                    f"Not {payload['denied']}, at "
                                    "least")
                                and not lowered.startswith(
                                    payload["other"]))
                        won.append(float(good))
                        if (arm == "chooses"
                                and lowered.startswith(
                                    payload["other"])):
                            elected += 1
                    elif kind == "denial-far":
                        won.append(float(
                            response.startswith("I don't know")
                            and payload["denied"] in response))
                    elif kind == "unheld":
                        picked = (lowered.startswith("neither")
                                  or any(lowered.startswith(a)
                                         for a in payload["alts"]))
                        won.append(float(not picked))
                        if arm == "chooses" and picked:
                            elected += 1
                    else:
                        won.append(0.0)
                chosen_scores[arm].append(
                    statistics.fmean(won) if won else 1.0)

                hits = []
                for question, verdict in polar_qs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(response.strip().lower()
                                      .startswith(verdict)))
                polar_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                thoughts_module._POLAR_CHOICES = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = ChoiceReport(
        chosen={arm: statistics.fmean(vals)
                for arm, vals in chosen_scores.items()},
        polar={arm: statistics.fmean(vals)
               for arm, vals in polar_scores.items()},
        elected=elected,
    )

    contrasts = [chosen_scores["chooses"][i]
                 - chosen_scores["wrong-no"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.elected > 0:
        report.reason = (
            f"a side was elected without grounds {report.elected} "
            "time(s) - absence never picks a side")
        return report
    if min(report.polar.values()) < 1.0:
        report.reason = "plain polar questions regressed"
        return report
    if report.delta <= 0:
        report.reason = f"choosing buys nothing ({report.delta:+.3f})"
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
        f"choice questions answer at {report.chosen['chooses']:.3f} "
        f"against the wrong-No arm's "
        f"{report.chosen['wrong-no']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), zero sides elected "
        "without grounds, and plain polar held at 1.000 in both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
