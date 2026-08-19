"""Adjective tolerance on the inference path. The modifier gate.

§11.35 recorded the brittleness: "the weathered tower conductivity"
defeated the coverage rules, because no fact key holds "weathered" and
full coverage is what keeps pun-rejection honest. The registered claim
is tolerance WITHOUT reopening the fabrication hole, and the design
carries the safety argument in its structure: one library-unknown token
may be dropped and the spread retried — on the INFERENCE path only —
because convergence still demands a bridge. Drop "tungsten" from "the
melting point of tungsten" and the residual's coverage is all-direct,
which the spread already refuses as plain recall's territory; drop
"weathered" from "the weathered tower conductivity" and the residual
converges through ``tower material -> steel``. The exact/keyword recall
path NEVER gets the drop: there, the same trick would assert steel's
number for tungsten, which is the §11.29 fabrication verbatim.

Both arms run the live pipeline; the baseline arm has token-unknownness
disabled (nothing ever counts as a modifier) and is otherwise identical.
World mix: single-modifier bridged questions (the claim), double-modifier
questions (~30%, refused by the at-most-one rule in both arms — the
variance, per standing policy), and unknown-ENTITY decoys in every world.

**The criteria, written before the run.**

1. **Gain** — over all worlds, tolerated-answer rate (correct value,
   marked inferred, modifier named) must beat the baseline arm by more
   than the seed-to-seed sd. Double-modifier worlds score zero for both
   arms.
2. **Fabrication control** — "what is the <attr> of <unknown entity>?"
   must be asserted in NEITHER arm, ever: one assertion is a fail, sd or
   no sd. This control is the safety claim itself — the drop must never
   convert an unknown entity into a modifier.
3. **Modifier honesty is reported** — every tolerated answer must name
   the token it read as a modifier, so the reader can veto the reading.

**It was run, and it PASSED — with the safety claim holding in both arms.**

| arm | tolerated answers |
|---|---:|
| baseline | 0.000 |
| modifier | **0.667** |

**+0.667 at 1.35x seed sd**, zero unknown-entity fabrications across
every decoy in either arm, and every tolerated answer named the token
it dropped ("reading 'weathered' as a modifier") so a reader can veto
the reading. The 0.333 miss is the double-modifier worlds: the
at-most-one rule refuses them, because two unknown tokens is no longer
a decoration — it is a question the library does not understand, and
saying so beats guessing which word mattered.

The structural safety argument measured out exactly as designed: the
drop lives on the inference path only, where convergence still demands
a bridge — so "the melting point of tungsten" minus its entity leaves
all-direct coverage the spread refuses, while the same drop on the
recall path would have asserted steel's number for tungsten, the
§11.29 fabrication verbatim. Tolerance without a new hole, because the
old guard does the guarding.

Run it::

    python -m ultraquant.experiments.modifier_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ModifierReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["conductivity", "hardness", "density"]
_ADJECTIVES = ["weathered", "crumbling", "ancient", "gleaming"]
_GHOST_ENTITIES = ["obelisk", "viaduct", "citadel"]


@dataclass
class ModifierReport:
    """The two arms, the fabrication count, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        tolerated: Arm -> tolerated-answer rate over all worlds.
        fabricated: Assertions for unknown-entity decoys (must be 0).
        named: Tolerated answers that named their modifier.
        delta: Tolerance contrast (modifier arm minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    tolerated: dict = field(default_factory=dict)
    fabricated: int = 0
    named: float = 1.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'tolerated answers':>17}"]
        for arm in ("baseline", "modifier"):
            lines.append(f"{arm:<10} {self.tolerated[arm]:>17.3f}")
        lines += [
            f"unknown-entity fabrications: {self.fabricated}",
            f"modifier named in tolerated answers: {self.named:.3f}",
            f"delta (tolerance, with vs without) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def run_gate(seeds: int = 12) -> ModifierReport:
    """Run both arms over generated modifier worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ModifierReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    tolerated = {"baseline": [], "modifier": []}
    fabricated = 0
    named_hits, named_total = 0, 0

    real_unknown = inference_module._library_unknown
    for seed in range(seeds):
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        ghost = rng.choice(_GHOST_ENTITIES)
        material = rng.choice(_MATERIALS)
        prop = rng.choice(_PROPERTIES)
        value = rng.randrange(100, 2000)
        adjectives = rng.sample(_ADJECTIVES, 2)
        double = rng.random() < 0.3

        facts = [(f"the {entity} material", material),
                 (f"the {material} {prop}", f"{value} units")]
        decoration = (" ".join(adjectives) if double else adjectives[0])
        question = f"what is the {decoration} {entity} {prop}?"
        decoy = f"what is the {prop} of the {ghost}?"

        for arm in ("baseline", "modifier"):
            inference_module._library_unknown = (
                (lambda token, memory: False) if arm == "baseline"
                else real_unknown)
            root = Path(tempfile.mkdtemp(prefix=f"uq_modifier_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)

                response, _trace = run_pipeline(question, session)
                good = (str(value) in response and "inferred" in response)
                tolerated[arm].append(float(good and not double))
                if good and arm == "modifier":
                    named_total += 1
                    named_hits += int("as a modifier" in response)

                response, _trace = run_pipeline(decoy, session)
                if (str(value) in response
                        and not response.startswith("I don't hold")
                        and not response.startswith("Nothing believed")):
                    fabricated += 1
            finally:
                inference_module._library_unknown = real_unknown
                shutil.rmtree(root, ignore_errors=True)

    report = ModifierReport(
        tolerated={arm: statistics.fmean(vals)
                   for arm, vals in tolerated.items()},
        fabricated=fabricated,
        named=(named_hits / named_total) if named_total else 1.0,
    )

    contrasts = [tolerated["modifier"][i] - tolerated["baseline"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.fabricated > 0:
        report.reason = (
            f"the fabrication control failed: {report.fabricated} "
            "unknown-entity decoy(s) asserted - the drop converted an "
            "entity into a modifier, which is the §11.29 wrong-metal "
            "answer returned")
        return report
    if report.named < 1.0:
        report.reason = (
            f"modifier honesty failed: only {report.named:.3f} of "
            "tolerated answers named the token they dropped")
        return report
    if report.delta <= 0:
        report.reason = f"tolerance buys nothing ({report.delta:+.3f})"
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
        f"one-modifier questions answer at "
        f"{report.tolerated['modifier']:.3f} against the baseline's "
        f"{report.tolerated['baseline']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd), every tolerated answer "
        "names its modifier, and zero unknown-entity decoys were "
        "asserted in either arm")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
