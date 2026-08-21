"""Two adjectives may decorate. The multi-modifier gate.

§11.36 shipped tolerance at exactly one dropped token and recorded its
honest ceiling: double-modifier questions refused at 0.333, "because
two unknown tokens is no longer a decoration". This gate registers the
next rung — at most TWO library-unknown tokens may be dropped — and
carries the safety line explicitly: at least two known informative
tokens must survive the drop (the residual floor), and convergence
still demands a bridge, so the residual can never thin below the shape
the spread already guards. "The old weathered tower conductivity"
drops both adjectives and converges through ``tower material ->
steel``; "the ancient obelisk density" drops two tokens into a
one-token residual and refuses, because a question that is mostly
unknown is not decorated — it is not understood.

Both arms run the live pipeline; the baseline arm is the shipped
at-most-one cap and the extended arm allows two, otherwise identical.
World mix: double-modifier bridged questions (the claim),
triple-modifier questions (~25%, refused by the cap in both arms — the
variance, per §11.35's power arithmetic), and in every world two
decoys: the unknown-entity form and the thin-residual form.

**The criteria, written before the run.**

1. **Gain** — over all worlds, tolerated-answer rate (correct value,
   marked inferred, both modifiers named) must beat the baseline arm
   by more than the seed-to-seed sd. Triple-modifier worlds score zero
   for both arms.
2. **Fabrication control** — "what is the <attr> of the <ghost>?" must
   be asserted in NEITHER arm, ever. One assertion is a fail, sd or no
   sd.
3. **Residual-floor control** — "what is the <adj> <ghost> <attr>?"
   (two unknowns, one-token residual) must be asserted in NEITHER arm,
   ever: the second drop must never thin a question past understanding.
4. **Modifier honesty** — every tolerated answer must name every token
   it read away, so the reader can veto the reading.

**It was run twice: the first run indicted the harness, the second
PASSED — narrowly, in §11.13's tradition of saying so.**

The first run reported 24/24 decoys "asserted" in BOTH arms — including
the baseline arm running shipped behavior that measured zero
fabrications in §11.36. That shape said harness, not mechanism: the
check counted any response containing the value, and §11.29's designed
demote hedge ("I don't hold that exactly. Nearest I hold: steel
hardness is ...") names the value under its OWN key without asserting
it for the ghost. The registered word is "asserted", §11.36 already
operationalised it (value delivered outside a refusal shape), and the
harness was corrected to the registered word — the worlds, seeds, and
criteria untouched. Then:

| arm | tolerated answers |
|---|---:|
| baseline (cap 1) | 0.000 |
| extended (cap 2) | **0.583** |

**+0.583 at 1.13x seed sd** — a narrow pass, and the margin is narrow
for an honest reason: the RNG dealt 5/12 triple-modifier worlds (42%
against the intended ~25%), and every one of them scores zero in both
arms by design. All seven double-modifier worlds answered, every
answer named both dropped words in question order ("reading
'crumbling' and 'weathered' as modifiers"), and zero decoys of either
form were asserted anywhere — the residual floor held at exactly the
shape it was built for, "the ancient obelisk density" thinning to one
token and dying. The triple-modifier ceiling is recorded the way
§11.36 recorded its own: the next rung is a registered candidate, not
a promise.

Run it::

    python -m ultraquant.experiments.multimodifier_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["MultiModifierReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]
_ADJECTIVES = ["weathered", "crumbling", "ancient", "gleaming"]
_GHOST_ENTITIES = ["obelisk", "viaduct", "citadel"]


@dataclass
class MultiModifierReport:
    """The two arms, both decoy counts, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        tolerated: Arm -> tolerated-answer rate over all worlds.
        fabricated: Assertions for unknown-entity decoys (must be 0).
        thinned: Assertions for thin-residual decoys (must be 0).
        named: Tolerated answers that named every dropped token.
        delta: Tolerance contrast (extended arm minus baseline).
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
        for arm in ("baseline", "extended"):
            lines.append(f"{arm:<10} {self.tolerated[arm]:>17.3f}")
        lines += [
            f"unknown-entity fabrications: {self.fabricated}",
            f"thin-residual assertions: {self.thinned}",
            f"all modifiers named in tolerated answers: {self.named:.3f}",
            f"delta (cap 2 vs cap 1) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _asserted(response: str, value: int) -> bool:
    # "Asserted" is §11.36's operationalisation: the value delivered
    # outside a refusal shape. The demoted hedge ("I don't hold that
    # exactly. Nearest I hold: steel hardness is ...") names the value
    # under its OWN key - that is §11.29's designed demotion, not a
    # fabrication, and this gate's first run counted it as one.
    return (str(value) in response
            and not response.startswith("I don't hold")
            and not response.startswith("Nothing believed"))


def run_gate(seeds: int = 12) -> MultiModifierReport:
    """Run both arms over generated double-modifier worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`MultiModifierReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    tolerated = {"baseline": [], "extended": []}
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
        triple = rng.random() < 0.25
        adjectives = rng.sample(_ADJECTIVES, 3 if triple else 2)

        facts = [(f"the {entity} material", material),
                 (f"the {material} {prop}", f"{value} units")]
        decoration = " ".join(adjectives)
        question = f"what is the {decoration} {entity} {prop}?"
        decoy = f"what is the {prop} of the {ghost}?"
        thin = f"what is the {adjectives[0]} {ghost} {prop}?"

        for arm in ("baseline", "extended"):
            inference_module._MAX_MODIFIERS = 1 if arm == "baseline" else 2
            root = Path(tempfile.mkdtemp(prefix=f"uq_multimod_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)

                response, _trace = run_pipeline(question, session)
                good = (str(value) in response and "inferred" in response)
                tolerated[arm].append(float(good and not triple))
                if good and arm == "extended":
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

    report = MultiModifierReport(
        tolerated={arm: statistics.fmean(vals)
                   for arm, vals in tolerated.items()},
        fabricated=fabricated,
        thinned=thinned,
        named=(named_hits / named_total) if named_total else 1.0,
    )

    contrasts = [tolerated["extended"][i] - tolerated["baseline"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.fabricated > 0:
        report.reason = (
            f"the fabrication control failed: {report.fabricated} "
            "unknown-entity decoy(s) asserted - the wrong-metal answer "
            "returned")
        return report
    if report.thinned > 0:
        report.reason = (
            f"the residual floor failed: {report.thinned} thin-residual "
            "decoy(s) asserted - the second drop read a question past "
            "understanding")
        return report
    if report.named < 1.0:
        report.reason = (
            f"modifier honesty failed: only {report.named:.3f} of "
            "tolerated answers named every token they dropped")
        return report
    if report.delta <= 0:
        report.reason = f"the second drop buys nothing ({report.delta:+.3f})"
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
        f"double-modifier questions answer at "
        f"{report.tolerated['extended']:.3f} against the cap-1 "
        f"baseline's {report.tolerated['baseline']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "every tolerated answer names both dropped words, and zero "
        "decoys of either form were asserted in either arm")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
