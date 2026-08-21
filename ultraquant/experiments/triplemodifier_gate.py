"""Three adjectives may decorate. The triple-modifier gate.

§11.36 shipped adjective tolerance at one dropped token; §11.46 moved
the cap to two behind the residual floor and recorded triple-modifier
questions refusing as its honest ceiling. This gate is that ceiling's
measurement, the §11.46 design one rung up: at most THREE
library-unknown tokens may be dropped, the floor unmoved — at least
two known informative tokens must survive, and convergence still
demands a bridge. "The old weathered crumbling tower conductivity"
reads three adjectives away and converges through ``tower material ->
steel``; a question that is mostly unknown still refuses, because the
floor, not the cap, is the safety line — §11.46's words, still true
one adjective later.

Both arms run the live pipeline; the baseline arm is the shipped
at-most-two cap and the treatment arm allows three, otherwise
identical. World mix: triple-modifier bridged questions (the claim),
quadruple-modifier questions (~25%, refused by the cap in both arms —
the variance, per §11.35's power arithmetic), and in every world two
decoys: the unknown-entity form and the thin-residual form.

**The criteria, written before the run.**

1. **Gain** — over all worlds, tolerated-answer rate (correct value,
   marked inferred, all three modifiers named) must beat the baseline
   arm by more than the seed-to-seed sd. Quadruple-modifier worlds
   score zero for both arms.
2. **Fabrication control** — "what is the <attr> of the <ghost>?" must
   be asserted in NEITHER arm, ever.
3. **Residual-floor control** — "what is the <adj> <ghost> <attr>?"
   (two unknowns, one-token residual) must be asserted in NEITHER arm,
   ever: a bigger cap must not thin a question past understanding.
4. **Modifier honesty** — every tolerated answer names every token it
   read away.

**It was run once, and PASSED — narrowly, in §11.13's tradition of
saying so, on the same dealt hand §11.46 drew.**

| arm | tolerated answers |
|---|---:|
| cap-two | 0.000 |
| cap-three | **0.583** |

**+0.583 at 1.13x seed sd** — narrow for the same honest reason as
§11.46: the RNG dealt 5/12 quadruple-modifier worlds against the
intended ~25% (the identical seed positions draw the identical hand),
each scoring zero in both arms by design. Zero decoys of either form
asserted, every tolerated answer named all three dropped words. The
floor has now held the line through three cap moves — one, two, three
adjectives — without once being touched: the cap is policy, the floor
is safety, and the distinction has been measured every time it
mattered. The quadruple rung stays registered, not promised.

Run it::

    python -m ultraquant.experiments.triplemodifier_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TripleModifierReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]
_ADJECTIVES = ["weathered", "crumbling", "ancient", "gleaming",
               "storied"]
_GHOST_ENTITIES = ["obelisk", "viaduct", "citadel"]


@dataclass
class TripleModifierReport:
    """The two arms, both decoy counts, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        tolerated: Arm -> tolerated-answer rate over all worlds.
        fabricated: Assertions for unknown-entity decoys (must be 0).
        thinned: Assertions for thin-residual decoys (must be 0).
        named: Tolerated answers that named every dropped token.
        delta: Tolerance contrast (cap-3 arm minus cap-2).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    tolerated: dict = field(default_factory=dict)
    fabricated: int = 0
    thinned: int = 0
    named: float = 1.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'tolerated answers':>17}"]
        for arm in ("cap-two", "cap-three"):
            lines.append(f"{arm:<10} {self.tolerated[arm]:>17.3f}")
        lines += [
            f"unknown-entity fabrications: {self.fabricated}",
            f"thin-residual assertions: {self.thinned}",
            f"all modifiers named in tolerated answers: {self.named:.3f}",
            f"delta (cap 3 vs cap 2) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _asserted(response: str, value: int) -> bool:
    # §11.36's operationalisation, §11.46's lesson: the demote hedge
    # names the value under its OWN key and is not an assertion.
    return (str(value) in response
            and not response.startswith("I don't hold")
            and not response.startswith("Nothing believed"))


def run_gate(seeds: int = 12) -> TripleModifierReport:
    """Run both arms over generated triple-modifier worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`TripleModifierReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    tolerated = {"cap-two": [], "cap-three": []}
    fabricated = 0
    thinned = 0
    named_hits, named_total = 0, 0

    real_cap = inference_module._MAX_MODIFIERS
    for seed in range(seeds):
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        ghost = rng.choice(_GHOST_ENTITIES)
        material = rng.choice(_MATERIALS)
        prop = rng.choice(_PROPERTIES)
        value = rng.randrange(100, 2000)
        quadruple = rng.random() < 0.25
        adjectives = rng.sample(_ADJECTIVES, 4 if quadruple else 3)

        facts = [(f"the {entity} material", material),
                 (f"the {material} {prop}", f"{value} units")]
        decoration = " ".join(adjectives)
        question = f"what is the {decoration} {entity} {prop}?"
        decoy = f"what is the {prop} of the {ghost}?"
        thin = f"what is the {adjectives[0]} {ghost} {prop}?"

        for arm in ("cap-two", "cap-three"):
            inference_module._MAX_MODIFIERS = (2 if arm == "cap-two"
                                               else 3)
            root = Path(tempfile.mkdtemp(prefix=f"uq_trimod_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)

                response, _trace = run_pipeline(question, session)
                good = (str(value) in response and "inferred" in response)
                tolerated[arm].append(float(good and not quadruple))
                if good and arm == "cap-three":
                    named_total += 1
                    named_hits += int(
                        "as modifiers" in response
                        and all(f"'{adj}'" in response
                                for adj in adjectives))

                response, _trace = run_pipeline(decoy, session)
                if _asserted(response, value):
                    fabricated += 1
                response, _trace = run_pipeline(thin, session)
                if _asserted(response, value):
                    thinned += 1
            finally:
                inference_module._MAX_MODIFIERS = real_cap
                shutil.rmtree(root, ignore_errors=True)

    report = TripleModifierReport(
        tolerated={arm: statistics.fmean(vals)
                   for arm, vals in tolerated.items()},
        fabricated=fabricated,
        thinned=thinned,
        named=(named_hits / named_total) if named_total else 1.0,
    )

    contrasts = [tolerated["cap-three"][i] - tolerated["cap-two"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.fabricated > 0:
        report.reason = (
            f"the fabrication control failed: {report.fabricated} "
            "unknown-entity decoy(s) asserted")
        return report
    if report.thinned > 0:
        report.reason = (
            f"the residual floor failed: {report.thinned} thin-residual "
            "decoy(s) asserted")
        return report
    if report.named < 1.0:
        report.reason = (
            f"modifier honesty failed: only {report.named:.3f} of "
            "tolerated answers named every dropped token")
        return report
    if report.delta <= 0:
        report.reason = f"the third drop buys nothing ({report.delta:+.3f})"
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
        f"triple-modifier questions answer at "
        f"{report.tolerated['cap-three']:.3f} against the cap-2 "
        f"baseline's {report.tolerated['cap-two']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "every tolerated answer names all three dropped words, and "
        "zero decoys of either form were asserted in either arm")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
