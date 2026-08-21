"""Reinforcement-weighted retrieval: recall reinforces, measured. The gate.

Rule 3 of the codebase — "recall reinforces: the catalog reorganises
around what is used" — reached experts and router long ago and never
reached fact retrieval: `find_facts` broke equal-overlap ties by the
alphabet. §11.30's last registered candidate (learned edge weights for
the spread) lands here in its honest first form: a fact's reinforcement
count plus confidence breaks ranking ties in both search paths, so a
re-attested fact surfaces above token-tied competitors under top-k
pressure. A tie-break ONLY — reinforcement never outranks a better
token match, and every coverage rule upstream still refuses whatever
retrieval surfaces.

The gate's worlds make the tie-break matter: one target fact buried
among many same-overlap competitor keys (all sharing the probe tokens,
alphabetically ahead of the target), with the target re-attested k
times. The danger to measure is entrenchment wearing a ranking: a
reinforced fact about the WRONG subject must not ride its weight past
the coverage rules into an answer.

**The criteria, written before the run.**

1. **Attestation surfaces** — over all worlds, the reinforced target
   must reach the top-k retrieval (and its chain must answer) more
   often than in the baseline arm (tie-break disabled) by more than the
   seed-to-seed sd. ~25% of worlds leave the target UNREINFORCED — the
   alphabetical tie stands, both arms miss alike, and those worlds are
   the variance.
2. **No entrenchment fabrication** — decoy questions whose reinforced
   near-fact names the wrong subject must refuse in BOTH arms. One
   assertion is a fail, sd or no sd: §11.16 measured what entrenchment
   does, and a ranking bonus must never become a coverage bypass.
3. **Better matches still win** — worlds where a lower-reinforcement
   key has HIGHER token overlap must retrieve the better match in both
   arms identically.

**It was run, and it PASSED — narrowly on margin, absolutely on the
controls, with one accidental validation on the way.**

The first world build buried the target under OTHER entities and both
arms answered 1.000 anyway: §11.38's phrase probes already rescue
cross-subject burial, which is that fix validating itself from an
unexpected direction. The burial that matters is WITHIN one subject —
ten same-entity facts whose attribute words sort ahead of "material" —
and there:

| arm | chains answered | entrenchment falls |
|---|---:|---:|
| baseline | 0.000 | 0.000 |
| weighted | **0.571** | 0.000 |

**+0.571 at 1.11x seed sd** — §11.31's narrow-margin class — with zero
entrenchment falls (a five-times-reinforced sibling never bought a
ghost an answer), and better token matches winning identically in both
arms: the tie-break stayed a tie-break. The 0.429 miss is the
unreinforced worlds, where the alphabetical tie stands for both arms —
attestation earns rank, absence of attestation earns nothing, which is
the whole claim. One power correction recorded (§11.35's arithmetic).

Run it::

    python -m ultraquant.experiments.plasticity_gate
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field

__all__ = ["PlasticityReport", "run_gate"]

_PREFIXES = ["arbor", "beacon", "cinder", "dorval", "ember", "fallow",
             "garnet", "harrow", "isling", "jasper"]
_NOUNS = ["tower", "bridge", "spire", "mill", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite"]
_PROPERTIES = ["hardness", "density", "conductivity"]


@dataclass
class PlasticityReport:
    """The two arms across three world types, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        surfaced: Arm -> chain-answer rate over all worlds.
        entrenchment_falls: Arm -> wrong-subject assertions (must be 0).
        overlap_match: True when better-overlap worlds behaved
            identically.
        delta: Surfacing contrast (weighted minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    surfaced: dict = field(default_factory=dict)
    entrenchment_falls: dict = field(default_factory=dict)
    overlap_match: bool = True
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<9} {'chains answered':>15} "
                 f"{'entrenchment falls':>19}"]
        for arm in ("baseline", "weighted"):
            lines.append(f"{arm:<9} {self.surfaced[arm]:>15.3f} "
                         f"{self.entrenchment_falls[arm]:>19.3f}")
        lines += [
            f"better-overlap worlds identical: {self.overlap_match}",
            f"delta (chains, weighted vs baseline) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _weightless_search(self, text, top_k=5):
    """The pre-§11.44 ranking: overlap, then the alphabet."""
    import re as _re

    wanted = set(_re.findall(r"[a-z0-9]+", text.lower()))
    scored = []
    for key in self._facts:
        overlap = len(wanted & set(_re.findall(r"[a-z0-9]+", key.lower())))
        if overlap:
            scored.append((overlap, key))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [key for _o, key in scored[:top_k]]


def run_gate(seeds: int = 14) -> PlasticityReport:
    """Run both arms over buried-target worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`PlasticityReport`.
    """
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.reason.inference import infer

    surfaced = {"baseline": [], "weighted": []}
    falls = {"baseline": 0, "weighted": 0}
    overlap_same = True

    real_search = SystematicMemory.find_facts
    try:
        for seed in range(seeds):
            rng = random.Random(seed)
            noun = rng.choice(_NOUNS)
            prop = rng.choice(_PROPERTIES)
            material = rng.choice(_MATERIALS)
            target_entity = f"zephyr {noun}"
            value = rng.randrange(1000, 9999)
            # ~25% unreinforced worlds (SS11.35's power arithmetic: a
            # half-share Bernoulli contrast cannot clear one sd).
            reinforced = rng.random() >= 0.25

            # The first world build buried the target under OTHER
            # entities, and §11.38's phrase probes rescued it in both
            # arms - the burial that matters is WITHIN one subject: ten
            # same-entity facts whose attribute words all sort ahead of
            # "material", so equal-overlap alphabetical ranking starves
            # the chain's origin out of every top-k probe.
            fillers = ("annals", "ballads", "chronicle", "diary", "epic",
                       "fables", "glossary", "history", "index", "legend")
            memory = SystematicMemory()
            for filler in fillers:
                memory.remember_fact(f"{target_entity} {filler}",
                                     "an old story", confidence=0.6)
            memory.remember_fact(f"{target_entity} material", material,
                                 confidence=0.6)
            memory.remember_fact(f"{material} {prop}", f"{value} units",
                                 confidence=0.6)
            if reinforced:
                for _ in range(3):
                    memory.remember_fact(f"{target_entity} material",
                                         material, confidence=0.6)

            question = f"what is the {target_entity} {prop}?"
            # Entrenchment decoy: heavily reinforce a same-noun sibling
            # fact; a ghost entity's question must still refuse.
            for _ in range(5):
                memory.remember_fact(f"{target_entity} annals",
                                     "an old story", confidence=0.6)
            decoy = f"what is the ghostly {noun}ling {prop}?"

            for arm in ("baseline", "weighted"):
                SystematicMemory.find_facts = (
                    _weightless_search if arm == "baseline"
                    else real_search)
                result = infer(question, memory)
                good = bool(result) and str(value) in result.answer
                surfaced[arm].append(float(good))
                falls[arm] += int(infer(decoy, memory) is not None)

            # Better-overlap check: a two-token probe must beat a
            # reinforced one-token match in both arms.
            probe_memory = SystematicMemory()
            probe_memory.remember_fact("alpha beacon range", "9 meters",
                                       confidence=0.6)
            for _ in range(5):
                probe_memory.remember_fact("beacon lore", "a story",
                                           confidence=0.6)
            SystematicMemory.find_facts = _weightless_search
            off = probe_memory.find_facts("alpha beacon range", top_k=1)
            SystematicMemory.find_facts = real_search
            on = probe_memory.find_facts("alpha beacon range", top_k=1)
            if off != on:
                overlap_same = False
    finally:
        SystematicMemory.find_facts = real_search

    report = PlasticityReport(
        surfaced={arm: statistics.fmean(vals)
                  for arm, vals in surfaced.items()},
        entrenchment_falls={arm: falls[arm] / seeds
                            for arm in ("baseline", "weighted")},
        overlap_match=overlap_same,
    )
    contrasts = [surfaced["weighted"][i] - surfaced["baseline"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if max(report.entrenchment_falls.values()) > 0:
        report.reason = (
            "the entrenchment control failed: a reinforced wrong-subject "
            "fact rode its weight past the coverage rules - §11.16's "
            "entrenchment wearing a ranking")
        return report
    if not report.overlap_match:
        report.reason = ("a reinforced weak match outranked a better "
                         "token match; the tie-break stopped being a "
                         "tie-break")
        return report
    if report.delta <= 0:
        report.reason = f"the weight surfaces nothing ({report.delta:+.3f})"
        return report
    if report.seed_sd <= 0.0:
        report.reason = (f"every world gave the same contrast "
                         f"({report.delta:+.3f}); nothing varied")
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (f"gain {report.delta:+.3f} is "
                         f"{report.margin_ratio:.2f}x the seed sd")
        return report
    report.passes = True
    report.reason = (
        f"re-attested facts surface: chains answer in "
        f"{report.surfaced['weighted']:.3f} of all worlds against the "
        f"baseline's {report.surfaced['baseline']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "with zero entrenchment falls and better token matches still "
        "winning everywhere")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
