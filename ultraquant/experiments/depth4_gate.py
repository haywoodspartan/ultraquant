"""The fourth fact: the stated limit, cashed in or confirmed. The gate.

§11.31 bounded the spread at two bridge hops with a warning ("longer
chains multiply pun risk faster than they add reach"), and §11.47
re-recorded the limit after measuring three-fact chains: "the fourth
fact stays refused by the decay floor — the architecture's stated
limit, not a promise." But the limit was set BEFORE the rules that now
carry the pun resistance existed: single-origin lineages, origin
coverage, order-as-evidence, whole-value bridging with accounted
tokens, synchronous frontiers. The floor may now be a depth cap in
disguise — and whether the warning still binds is a question for a
measurement, not a memory.

The candidate constants: ``_MAX_HOPS = 3``, ``_FLOOR = 0.1`` — a
four-fact chain's origin activation arrives at 1.0 x 0.5^3 = 0.125,
above the new floor; a fifth fact would arrive at 0.0625, below it, so
depth five stays structurally refused. The gate monkeypatches both
constants per arm; the shipped source does not change unless the gate
passes.

Worlds chain entity -> agent -> place -> region -> attribute (four
facts, three bridges), with ~25% of scored chains broken at a middle
(the unwinnable share, per §11.35), §11.47's full decoy family at the
new depth — epithet agents ("wren the younger" holds the role, only
plain wren has a hometown), ghost entities — plus three-fact and
two-fact chains and direct recalls as regression floors.

**The criteria, written before the run.**

1. **Gain** — four-fact chain questions answered (correct value,
   marked inferred, all three middles named in the trail) must beat
   the two-hop baseline by more than the seed-to-seed sd.
   Broken-middle chains score zero for both arms.
2. **The pun warning, measured** — epithet decoys and ghost decoys
   must be asserted ZERO times in the three-hop arm. §11.31's warning
   binds the moment one pun lands: one assertion is a fail, and the
   limit stays.
3. **Shallower chains hold** — three-fact and two-fact chains must
   answer in the three-hop arm at no less than the baseline rate, and
   direct recall must be 1.000 in both arms. Deeper reach may not
   cost shallower honesty.

**It was run once, and PASSED — the warning, measured, no longer
binds.**

| arm | 4-chains | 3-chains | 2-chains | recall |
|---|---:|---:|---:|---:|
| two-hop | 0.000 | 1.000 | 1.000 | 1.000 |
| three-hop | **0.708** | 1.000 | 1.000 | 1.000 |

**+0.708 at 2.37x seed sd, with ZERO decoy assertions at depth** —
epithet agents and ghosts alike refused through three bridges — and
three-chains, two-chains, and recall identical in both arms. The
0.292 miss is the broken-middle share scoring zero by design. §11.31's
warning was right when it was written: at that time the pun resistance
lived in the decay floor, and depth multiplied risk. It lives in the
structural rules now — single-origin lineages, origin coverage,
order-as-evidence, whole-value bridging with accounted tokens,
synchronous frontiers — and under them the third bridge adds reach
without adding puns. The constants shipped (`_MAX_HOPS = 3`,
`_FLOOR = 0.1`), the fifth fact arrives at 0.0625 and stays
structurally refused, and THAT limit is recorded the way this one was:
a measurement away from moving, never a promise.

Run it::

    python -m ultraquant.experiments.depth4_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Depth4Report", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_ROLES = ["architect", "founder", "sculptor", "keeper"]
_AGENTS = ["wren", "aldous", "berta", "cosimo", "delia", "edmund",
           "flora", "gideon"]
_PLACES = ["york", "delft", "siena", "bruges", "cadiz", "trento"]
_REGIONS = ["norland", "esterre", "vastmark", "sudmoor"]
_REGION_ATTRS = ["climate", "terrain"]
_ATTR_VALUES = ["temperate", "humid", "arid", "rocky", "flat", "misty"]
_EPITHETS = ["younger", "elder", "lesser"]


@dataclass
class Depth4Report:
    """The two arms, the pun count, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        four: Arm -> four-fact chain answer rate over all worlds.
        three: Arm -> three-fact chain answer rate (regression floor).
        two: Arm -> two-fact chain answer rate (regression floor).
        puns: Decoy assertions in the three-hop arm (must be 0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Four-chain contrast (three hops minus two).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    four: dict = field(default_factory=dict)
    three: dict = field(default_factory=dict)
    two: dict = field(default_factory=dict)
    puns: int = 0
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'4-chains':>9} {'3-chains':>9} "
                 f"{'2-chains':>9} {'recall':>7}"]
        for arm in ("two-hop", "three-hop"):
            lines.append(
                f"{arm:<10} {self.four[arm]:>9.3f} "
                f"{self.three[arm]:>9.3f} {self.two[arm]:>9.3f} "
                f"{self.recall[arm]:>7.3f}")
        lines += [
            f"decoy assertions at depth (three-hop arm): {self.puns}",
            f"delta (three hops vs two) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One depth-four world.

    Returns ``(facts, fours, threes, twos, decoys, directs)`` where
    fours are ``(question, expected|None, middles)`` — expected None
    marks a broken chain; middles are the three values a correct trail
    must name.
    """
    entities = []
    seen = set()
    while len(entities) < 12:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            entities.append(name)

    facts = []
    attr_values = {}
    for region in _REGIONS:
        for attr in _REGION_ATTRS:
            value = rng.choice(_ATTR_VALUES)
            attr_values[(region, attr)] = value
            facts.append((f"the {region} {attr}", value))

    place_regions = {}
    for place in _PLACES:
        region = rng.choice(_REGIONS)
        place_regions[place] = region
        facts.append((f"{place} region", region))

    agents = rng.sample(_AGENTS, 6)
    agent_places = {}
    fours = []
    agent_index = 0
    for spot, entity in enumerate(entities):
        if spot < 4:
            # A scored four-fact chain.
            role = rng.choice(_ROLES)
            agent = agents[agent_index]
            agent_index += 1
            place = rng.choice(_PLACES)
            broken = rng.random() < 0.25
            facts.append((f"the {entity} {role}", agent))
            if not broken:
                facts.append((f"{agent} hometown", place))
                agent_places[agent] = place
            attr = rng.choice(_REGION_ATTRS)
            region = place_regions[place]
            expected = (None if broken
                        else attr_values[(region, attr)])
            fours.append(
                (f"what is the {entity} {role} hometown region {attr}?",
                 expected, [agent, place, region]))
        elif spot < 6:
            # Epithet decoys: the §11.47 pun family, one level deeper.
            role = rng.choice(_ROLES)
            base = agents[rng.randrange(0, agent_index)]
            epithet = rng.choice(_EPITHETS)
            facts.append((f"the {entity} {role}",
                          f"{base} the {epithet}"))
            if base not in agent_places:
                place = rng.choice(_PLACES)
                agent_places[base] = place
                facts.append((f"{base} hometown", place))

    decoys = []
    for entity in entities[4:6]:
        role = rng.choice(_ROLES)
        decoys.append(f"what is the {entity} {role} hometown region "
                      f"{rng.choice(_REGION_ATTRS)}?")
    for _ in range(2):
        ghost = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if ghost in seen:
            continue
        decoys.append(f"what is the {ghost} {rng.choice(_ROLES)} "
                      f"hometown region {rng.choice(_REGION_ATTRS)}?")

    threes = []
    for entity in entities[6:8]:
        role = rng.choice(_ROLES)
        agent = rng.choice([a for a in agents if a in agent_places])
        facts.append((f"the {entity} {role}", agent))
        place = agent_places[agent]
        threes.append(
            (f"what is the {entity} {role} hometown region?",
             place_regions[place]))

    twos = []
    for entity in entities[8:10]:
        place = rng.choice(_PLACES)
        facts.append((f"the {entity} city", place))
        region = place_regions[place]
        twos.append((f"what is the {entity} city region?", region))

    directs = []
    pool = [f for f in facts if f[0].startswith("the ")]
    rng.shuffle(pool)
    for key, value in pool[:4]:
        directs.append((f"what is {key}?", value))

    rng.shuffle(facts)
    return facts, fours, threes, twos, decoys, directs


def run_gate(seeds: int = 12) -> Depth4Report:
    """Run both arms over generated depth-four worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`Depth4Report`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    four_scores = {"two-hop": [], "three-hop": []}
    three_scores = {"two-hop": [], "three-hop": []}
    two_scores = {"two-hop": [], "three-hop": []}
    recall_scores = {"two-hop": [], "three-hop": []}
    puns = 0

    real_hops = inference_module._MAX_HOPS
    real_floor = inference_module._FLOOR
    for seed in range(seeds):
        rng = random.Random(seed)
        facts, fours, threes, twos, decoys, directs = _build_world(rng)

        for arm in ("two-hop", "three-hop"):
            if arm == "three-hop":
                inference_module._MAX_HOPS = 3
                inference_module._FLOOR = 0.1
            else:
                inference_module._MAX_HOPS = real_hops
                inference_module._FLOOR = real_floor
            root = Path(tempfile.mkdtemp(prefix=f"uq_d4_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, value in facts:
                    session.memory.remember_fact(key, value,
                                                 confidence=0.6)

                won = []
                for question, expected, middles in fours:
                    response, _trace = run_pipeline(question, session)
                    if expected is None:
                        won.append(0.0)
                        continue
                    good = (expected in response
                            and "inferred" in response
                            and all(m in response for m in middles))
                    won.append(float(good))
                four_scores[arm].append(
                    statistics.fmean(won) if won else 0.0)

                for question in decoys:
                    response, _trace = run_pipeline(question, session)
                    if "inferred" in response:
                        if arm == "three-hop":
                            puns += 1

                hits = []
                for question, expected in threes:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response
                                      and "inferred" in response))
                three_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in twos:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response
                                      and "inferred" in response))
                two_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall_scores[arm].append(
                    statistics.fmean(hits) if hits else 1.0)
            finally:
                inference_module._MAX_HOPS = real_hops
                inference_module._FLOOR = real_floor
                shutil.rmtree(root, ignore_errors=True)

    report = Depth4Report(
        four={arm: statistics.fmean(vals)
              for arm, vals in four_scores.items()},
        three={arm: statistics.fmean(vals)
               for arm, vals in three_scores.items()},
        two={arm: statistics.fmean(vals)
             for arm, vals in two_scores.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall_scores.items()},
        puns=puns,
    )

    contrasts = [four_scores["three-hop"][i] - four_scores["two-hop"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.puns > 0:
        report.reason = (
            f"§11.31's warning binds: {report.puns} decoy(s) asserted "
            "at depth - the limit stays, and this run is the "
            "measurement that keeps it")
        return report
    if report.three["three-hop"] < report.three["two-hop"] \
            or report.two["three-hop"] < report.two["two-hop"]:
        report.reason = ("deeper reach cost shallower honesty: "
                        "3-chains or 2-chains regressed under the "
                        "extended constants")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = f"the third hop buys nothing ({report.delta:+.3f})"
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
        f"four-fact chains answer at {report.four['three-hop']:.3f} "
        f"against the two-hop baseline's {report.four['two-hop']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd) "
        "with zero decoy assertions at depth, shallower chains and "
        "recall intact - the structural rules, not the decay floor, "
        "carry the pun resistance now")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
