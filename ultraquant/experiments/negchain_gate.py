"""Denials speak at terminals, and only at terminals. The negchain gate.

§11.48 made negation inhibitory everywhere: a denial neither seeds nor
relays the spread, and the fabrication it used to feed ("the dome
melting point, via not steel") measured zero with the old rate — 36
derivations per run — recorded beside it. That phase deliberately left
negative knowledge silent at depth: "the dome city is york" plus "the
york climate is not temperate" refused, though the two facts together
honestly entail "believed not temperate". This unit opens exactly one
door: a negated fact may be reached as a chain TERMINAL and answer,
polarity spoken ("believed not temperate, via york"), premise trail
carrying the denial as a denial — and it still never bridges, never
seeds, never relays. The claim is the door's width: one fact-shape
gained a voice, and nothing else moved.

Both arms run the live pipeline over the same worlds; the baseline arm
closes the door (`_NEGATED_TERMINALS = False`, §11.48's phase one), the
treatment arm is the shipped state. Worlds hold entity -> place chains
whose place attribute is stored NEGATED (the claim, ~25% of them with
the terminal never stored — the unwinnable share, per §11.35's power
arithmetic), entities whose only material knowledge is negative with
the material's properties positive (the §11.48 inertness control,
re-asserted under the opened door), genuine positive chains, and
direct recalls.

**The criteria, written before the run.**

1. **Gain** — negative-terminal chain questions answered ("believed
   not" plus the via trail, marked inferred) must beat the closed-door
   baseline by more than the seed-to-seed sd. Missing-terminal worlds
   score zero for both arms.
2. **The door is exactly one door** — negated-MIDDLE chain questions
   must derive ZERO times in BOTH arms: opening the terminal must not
   have reopened the bridge, by even one assertion.
3. **Honesty** — every scored answer names its polarity ("believed
   not") and its trail ("via"). An answer delivering the denied value
   without the polarity is a fail.
4. **Nothing regresses** — positive chains answer in the treatment arm
   at no less than the baseline rate, and direct recall is 1.000 in
   both arms.

**It was run once, and PASSED — with the door measured at exactly its
stated width.**

| arm | neg-chains | chains | recall | mid-bridged |
|---|---:|---:|---:|---:|
| closed | 0.000 | 1.000 | 1.000 | 0 |
| open | **0.694** | 1.000 | 1.000 | **0** |

**+0.694 at 2.09x seed sd.** Every delivered answer named its polarity
and its trail ("the dome city climate is believed not temperate, via
york"), negated middles derived zero times in EITHER arm — the §11.48
inertness re-measured intact with the terminal door standing open —
and neither positive chains nor recall moved. The 0.306 miss is the
missing-terminal share scoring zero by design. Negative knowledge now
composes with depth (§11.47's three-fact machinery carries it:
"believed not temperate, via wren, york") and consolidates with its
polarity when affirmed, staying inert as a bridge forever after.

Run it::

    python -m ultraquant.experiments.negchain_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["NegChainReport", "run_gate"]

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
class NegChainReport:
    """The two arms, the door-width control, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        negative: Arm -> negative-terminal answer rate over all worlds.
        bridged: Arm -> negated-middle derivations (must be 0 in BOTH).
        honesty: Scored answers naming polarity and trail (must be 1.0).
        chains: Arm -> genuine positive-chain answer rate.
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Negative-terminal contrast (treatment minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    negative: dict = field(default_factory=dict)
    bridged: dict = field(default_factory=dict)
    honesty: float = 1.0
    chains: dict = field(default_factory=dict)
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'neg-chains':>11} {'chains':>7} "
                 f"{'recall':>7} {'mid-bridged':>12}"]
        for arm in ("closed", "open"):
            lines.append(
                f"{arm:<10} {self.negative[arm]:>11.3f} "
                f"{self.chains[arm]:>7.3f} {self.recall[arm]:>7.3f} "
                f"{self.bridged[arm]:>12}")
        lines += [
            f"polarity and trail named in scored answers: "
            f"{self.honesty:.3f}",
            f"delta (terminal door open vs closed) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One negative-terminal world.

    Returns ``(statements, negatives, middles, chains, directs)``:
    negatives are ``(question, denied_value|None)`` — None marks a
    missing-terminal world whose honest answer is refusal; middles are
    negated-middle questions that must never derive; chains and
    directs are ``(question, expected)``.
    """
    names = []
    seen = set()
    while len(names) < 9:
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

    negatives = []
    used_places = rng.sample(_PLACES, 6)
    for spot, entity in enumerate(names[:3]):
        place = used_places[spot]
        statements.append(f"the {entity} city is {place}")
        attr = rng.choice(_ATTRS)
        missing = rng.random() < 0.25
        denied = rng.choice(_ATTR_VALUES)
        if not missing:
            statements.append(f"the {place} {attr} is not {denied}")
        negatives.append(
            (f"what is the {entity} city {attr}?",
             None if missing else denied))

    middles = []
    for entity in names[3:5]:
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is not {material}")
        prop = rng.choice(_PROPERTIES)
        middles.append(f"what is the {entity} {prop}?")

    chains = []
    for spot, entity in enumerate(names[5:7]):
        material = rng.choice(_MATERIALS)
        statements.append(f"the {entity} material is {material}")
        prop = rng.choice(_PROPERTIES)
        chains.append((f"what is the {entity} {prop}?",
                       str(prop_values[(material, prop)])))

    directs = []
    for material in rng.sample(_MATERIALS, 2):
        prop = rng.choice(_PROPERTIES)
        directs.append((f"what is the {material} {prop}?",
                        str(prop_values[(material, prop)])))

    rng.shuffle(statements)
    return statements, negatives, middles, chains, directs


def run_gate(seeds: int = 12) -> NegChainReport:
    """Run both arms over generated negative-terminal worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`NegChainReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    negative_scores = {"closed": [], "open": []}
    chain_scores = {"closed": [], "open": []}
    recall_scores = {"closed": [], "open": []}
    bridged = {"closed": 0, "open": 0}
    honest_hits, honest_total = 0, 0

    real_flag = inference_module._NEGATED_TERMINALS
    for seed in range(seeds):
        rng = random.Random(seed)
        statements, negatives, middles, chains, directs = \
            _build_world(rng)

        for arm in ("closed", "open"):
            inference_module._NEGATED_TERMINALS = (arm == "open")
            root = Path(tempfile.mkdtemp(prefix=f"uq_negch_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for statement in statements:
                    run_pipeline(statement, session)

                won = []
                for question, denied in negatives:
                    response, _trace = run_pipeline(question, session)
                    if denied is None:
                        won.append(0.0)
                        continue
                    good = ("believed not" in response
                            and denied in response
                            and "inferred" in response
                            and "via" in response)
                    won.append(float(good))
                    if arm == "open" and denied in response \
                            and "inferred" in response:
                        honest_total += 1
                        honest_hits += int("believed not" in response
                                           and "via" in response)
                negative_scores[arm].append(
                    statistics.fmean(won) if won else 0.0)

                for question in middles:
                    response, _trace = run_pipeline(question, session)
                    if "inferred" in response:
                        bridged[arm] += 1

                hits = []
                for question, expected in chains:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response
                                      and "inferred" in response))
                chain_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                inference_module._NEGATED_TERMINALS = real_flag
                shutil.rmtree(root, ignore_errors=True)

    report = NegChainReport(
        negative={arm: statistics.fmean(vals)
                  for arm, vals in negative_scores.items()},
        chains={arm: statistics.fmean(vals)
                for arm, vals in chain_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        bridged=dict(bridged),
        honesty=(honest_hits / honest_total) if honest_total else 1.0,
    )

    contrasts = [negative_scores["open"][i] - negative_scores["closed"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if max(report.bridged.values()) > 0:
        report.reason = (
            f"the door was wider than a terminal: negated-middle "
            f"chains derived (closed {report.bridged['closed']}, "
            f"open {report.bridged['open']})")
        return report
    if report.honesty < 1.0:
        report.reason = (
            f"honesty failed: only {report.honesty:.3f} of delivered "
            "negative answers named their polarity and trail")
        return report
    if report.chains["open"] < report.chains["closed"]:
        report.reason = (
            f"positive chains regressed: {report.chains['open']:.3f} "
            f"against the baseline's {report.chains['closed']:.3f}")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = (f"the terminal door buys nothing "
                         f"({report.delta:+.3f})")
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
        f"negative terminals answer at {report.negative['open']:.3f} "
        f"against the closed door's {report.negative['closed']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd), "
        "every delivered answer named its polarity and trail, negated "
        "middles derived zero times in either arm, and chains and "
        "recall held")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
