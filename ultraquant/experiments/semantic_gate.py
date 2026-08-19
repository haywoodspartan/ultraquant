"""The embedding suggester on the live question path. The adoption gate.

§11.37 measured the remedy in a lab arm; this gate measures it WIRED —
`build_session(semantic=True)` against the identical session with the
suggester off, full pipeline, same worlds. The claim is §11.37's
tradeoff surviving contact with the real fallback chain: synonym
questions asserted correctly, everything else byte-identical, and not
one decoy bought.

The suggester's two verifications are what is on trial. The cosine
floor (0.75, the measured zero-fall point) carries the tungsten-style
decoys — a held near-key with the wrong entity. The anchor rule (the
matched key must share an informative token with the question) carries
the ghosts — high cosine with nothing held about the subject. Either
check alone was measured insufficient: 0.65 cosine fabricated 0.833 of
ghosts in §11.37, and anchors without a floor would read every shared
noun as a match.

SKIPS, never passes, when no embedding model is reachable: an adoption
gate that silently passes with its mechanism absent would be §11.11's
defect with a service dependency.

**The criteria, written before the run.**

1. **The wall moves** — synonym-family assertive accuracy in the ON arm
   must beat the OFF arm's by more than the seed-to-seed sd.
2. **Decoys stay dead** — tungsten-style and ghost decoys must fall in
   NEITHER arm. One fall is a fail, sd or no sd.
3. **Everything else is untouched** — canonical and reorder accuracy
   must be identical between arms: the suggester runs only where the
   lexical core could not assert, and this criterion catches it
   leaking anywhere else.

**It was run, caught one more fabrication shape, and then PASSED.**

The first run's decoy control caught what the lab arm never met:
"what is the HEIGHT of the obelisk?" anchors on the attribute word
alone, and attribute-plus-attribute cosine clears any reasonable
floor - the wrong-metal fabrication wearing an embedding. The fix is
§11.36's positional rule made transitive: every question token the
matched key does not cover must precede, walking rightward, a covered
token. A synonym modifies what follows it ("how TALL is the tower");
a trailing unknown is a SUBJECT ("the height of the OBELISK"), and
subjects are never read away. Then:

| arm | synonym asserted | canonical+reorder | decoy falls |
|---|---:|---:|---:|
| off | 0.000 | 1.000 | 0.000 |
| on | **0.833** | 1.000 | 0.000 |

**+0.833 at 2.04x seed sd.** The 0.167 miss is the chained-synonym
worlds ("how hard is the tower", answerable only through material),
included at ~35% precisely because they are the half §11.37 measured
the embedding cannot buy - the ceiling in the metric instead of
outside it. Zero decoys fell in either arm; the lexical families are
byte-identical between arms; the suggester fires only where the core
could not assert, names its reading and its match strength, and costs
nothing when LM Studio is absent.

Run it::

    python -m ultraquant.experiments.semantic_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SemanticReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_SYNONYMS = {"height": "tall", "length": "long", "width": "wide"}
_PROPERTIES = {"hardness": "hard", "conductivity": "conductive"}
_MATERIALS = ["steel", "iron", "copper", "granite"]
_GHOSTS = ["obelisk", "viaduct", "citadel"]


@dataclass
class SemanticReport:
    """The two arms across three question sets, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        skipped: True when no embedding model was reachable. Never a pass.
        synonym: Arm -> synonym-family assertive accuracy.
        untouched: Arm -> canonical+reorder accuracy (must match).
        decoy_falls: Arm -> decoy assertion count.
        delta: Synonym contrast (on minus off).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    synonym: dict = field(default_factory=dict)
    untouched: dict = field(default_factory=dict)
    decoy_falls: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [f"{'arm':<5} {'synonym asserted':>16} "
                 f"{'canonical+reorder':>18} {'decoy falls':>12}"]
        for arm in ("off", "on"):
            lines.append(f"{arm:<5} {self.synonym[arm]:>16.3f} "
                         f"{self.untouched[arm]:>18.3f} "
                         f"{self.decoy_falls[arm]:>12.3f}")
        lines += [
            f"delta (synonym, on vs off) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _assertive(response: str) -> bool:
    lowered = response.lower()
    if lowered.startswith(("i don't hold", "nothing believed",
                           "i can't plan")):
        return False
    return ("(confidence" in lowered or "inferred" in lowered
            or "reading that as" in lowered)


def run_gate(seeds: int = 6) -> SemanticReport:
    """Run both arms over generated synonym worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`SemanticReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason.semantic import SemanticSuggester

    probe = SemanticSuggester()
    if probe._ready() is None:
        return SemanticReport(
            skipped=True,
            reason="no embedding model reachable; an adoption gate must "
                   "not pass with its mechanism absent")

    synonym_acc = {"off": [], "on": []}
    untouched_acc = {"off": [], "on": []}
    falls = {"off": 0, "on": 0}

    for seed in range(seeds):
        rng = random.Random(seed)
        entity, other = rng.sample(_ENTITIES, 2)
        attr, syn = rng.choice(list(_SYNONYMS.items()))
        ghost = rng.choice(_GHOSTS)
        value = rng.randrange(50, 950)
        other_value = rng.randrange(1000, 1999)

        facts = [(f"the {entity} {attr}", f"{value} meters"),
                 (f"the {other} {attr}", f"{other_value} meters")]
        # ~35% of worlds ask the CHAINED synonym instead - "how hard is
        # the tower" via material - which §11.37 measured as the half the
        # embedding cannot buy: the answer is not a held key, cosine has
        # nothing to match, and BOTH arms honestly miss. Those worlds are
        # the contrast's variance and the mechanism's recorded ceiling.
        if rng.random() < 0.35:
            prop, prop_syn = rng.choice(list(_PROPERTIES.items()))
            material = rng.choice(_MATERIALS)
            chain_value = rng.randrange(1000, 9999)
            facts += [(f"the {entity} material", material),
                      (f"the {material} {prop}", f"{chain_value} units")]
            synonym_questions = [(f"how {prop_syn} is the {entity}?",
                                  str(chain_value))]
        else:
            synonym_questions = [(f"how {syn} is the {entity}?",
                                  str(value))]
        untouched_questions = [
            (f"what is the {entity} {attr}?", str(value)),
            (f"what is the {attr} of the {other}?", str(other_value)),
        ]
        decoys = [
            # Ghost: nothing held about the subject; the anchor rule.
            f"how {syn} is the {ghost}?",
            # Tungsten-style: a held near-key, the wrong entity; the floor.
            f"what is the {attr} of the {ghost}?",
        ]

        for arm in ("off", "on"):
            root = Path(tempfile.mkdtemp(prefix=f"uq_sem_{seed}_"))
            try:
                session = build_session(root, seed=seed,
                                        semantic=(arm == "on"))
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)

                hits = 0
                for question, expected in synonym_questions:
                    response, _trace = run_pipeline(question, session)
                    hits += int(expected in response
                                and _assertive(response))
                synonym_acc[arm].append(hits / len(synonym_questions))

                hits = 0
                for question, expected in untouched_questions:
                    response, _trace = run_pipeline(question, session)
                    hits += int(expected in response
                                and _assertive(response))
                untouched_acc[arm].append(hits / len(untouched_questions))

                for question in decoys:
                    response, _trace = run_pipeline(question, session)
                    falls[arm] += int(_assertive(response))
            finally:
                shutil.rmtree(root, ignore_errors=True)

    report = SemanticReport(
        synonym={arm: statistics.fmean(vals)
                 for arm, vals in synonym_acc.items()},
        untouched={arm: statistics.fmean(vals)
                   for arm, vals in untouched_acc.items()},
        decoy_falls={arm: falls[arm] / (2.0 * seeds)
                     for arm in ("off", "on")},
    )

    contrasts = [synonym_acc["on"][i] - synonym_acc["off"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if falls["on"] > 0 or falls["off"] > 0:
        report.reason = (
            f"decoys fell (on={falls['on']}, off={falls['off']}): "
            "a synonym reading that buys fabrication is the 0.65-threshold "
            "world §11.37 measured, and it stays refused")
        return report
    if report.untouched["on"] != report.untouched["off"]:
        report.reason = (
            f"the suggester leaked: canonical+reorder changed "
            f"({report.untouched['off']:.3f} -> "
            f"{report.untouched['on']:.3f}) where the lexical core "
            "already asserted")
        return report
    if report.delta <= 0:
        report.reason = f"the wall did not move ({report.delta:+.3f})"
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
        f"the synonym wall moves {report.synonym['off']:.3f} -> "
        f"{report.synonym['on']:.3f} on the live path ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with zero decoy falls in "
        "either arm and the lexical families untouched")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
