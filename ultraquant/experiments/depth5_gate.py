"""The fifth fact: where does the warning finally bind? The gate.

§11.52 moved the two-hop limit to three by measuring that the pun
resistance had migrated from the decay floor into the structural rules
— single-origin lineages, origin coverage, order-as-evidence,
whole-value bridging with accounted tokens, synchronous frontiers —
and recorded the next limit the same way: "a fifth fact arrives at
0.0625 and stays structurally refused — a measurement away from
moving, never a promise." This gate is that measurement. The open
question is real in both directions: either the structural rules carry
a FOURTH bridge too, or this is the depth where §11.31's warning
("longer chains multiply pun risk faster than they add reach")
finally binds — and a FAIL here is not a defeat, it is the binding
measured, recorded exactly the way a pass would be.

The candidate constants: ``_MAX_HOPS = 4``, ``_FLOOR = 0.05`` — a
five-fact chain's origin activation arrives at 1.0 x 0.5^4 = 0.0625,
above the new floor; a sixth fact would arrive at 0.03125, below it.
The gate monkeypatches both constants per arm; the shipped source does
not change unless the gate passes.

Worlds chain entity -> agent -> place -> region -> realm -> attribute
(five facts, four bridges), with ~25% broken-middle chains as the
unwinnable share (§11.35), the full decoy family at the new depth —
epithet agents, ghost entities — and four-fact, three-fact, and
two-fact chains plus direct recalls as regression floors.

**The criteria, written before the run.**

1. **Gain** — five-fact chain questions answered (correct value,
   marked inferred, all four middles named in the trail) must beat the
   three-hop baseline by more than the seed-to-seed sd. Broken-middle
   chains score zero for both arms.
2. **The warning, measured at four bridges** — epithet and ghost
   decoys must be asserted ZERO times in the four-hop arm. One
   assertion binds the warning: the limit stays, and this run is the
   measurement that keeps it.
3. **Shallower chains hold** — four-, three-, and two-fact chains must
   answer in the four-hop arm at no less than the baseline rate, and
   direct recall must be 1.000 in both arms.

**It was run once, and PASSED — the warning does not bind at four
bridges either.**

| arm | 5-chains | 4-chains | 3-chains | 2-chains | recall |
|---|---:|---:|---:|---:|---:|
| three-hop | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| four-hop | **0.896** | 1.000 | 1.000 | 1.000 | 1.000 |

**+0.896 at 6.96x seed sd, with ZERO decoy assertions at four
bridges** — epithet agents and ghosts refused through the full chain —
and every shallower floor identical in both arms. The 0.104 miss is
the broken-middle share scoring zero by design. The constants shipped
(`_MAX_HOPS = 4`, `_FLOOR = 0.05`); the sixth fact arrives at 0.03125
and stays structurally refused, a measurement away like the two
limits before it. The ladder protocol moved up with the capability a
second time — depth-5 converges without a rung, depth-6 closes with
exactly one confirmation, and a bottom revision still takes the rung
down — which is what a protocol built on rungs is for. Two limits
moved by measurement in one session; the § of record is not the
constant but the method.

Run it::

    python -m ultraquant.experiments.depth5_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Depth5Report", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_ROLES = ["architect", "founder", "sculptor", "keeper"]
_AGENTS = ["wren", "aldous", "berta", "cosimo", "delia", "edmund",
           "flora", "gideon"]
_PLACES = ["york", "delft", "siena", "bruges", "cadiz", "trento"]
_REGIONS = ["norland", "esterre", "vastmark", "sudmoor"]
_REALMS = ["aurelia", "borealis", "cormant"]
_REALM_ATTRS = ["anthem", "banner"]
_ATTR_VALUES = ["golden", "crimson", "silver", "emerald"]
_EPITHETS = ["younger", "elder", "lesser"]


@dataclass
class Depth5Report:
    """The two arms, the pun count, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        five: Arm -> five-fact chain answer rate over all worlds.
        four: Arm -> four-fact chain answer rate (regression floor).
        three: Arm -> three-fact chain answer rate (regression floor).
        two: Arm -> two-fact chain answer rate (regression floor).
        puns: Decoy assertions in the four-hop arm (must be 0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Five-chain contrast (four hops minus three).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    five: dict = field(default_factory=dict)
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
        lines = [f"{'arm':<10} {'5-chains':>9} {'4-chains':>9} "
                 f"{'3-chains':>9} {'2-chains':>9} {'recall':>7}"]
        for arm in ("three-hop", "four-hop"):
            lines.append(
                f"{arm:<10} {self.five[arm]:>9.3f} "
                f"{self.four[arm]:>9.3f} {self.three[arm]:>9.3f} "
                f"{self.two[arm]:>9.3f} {self.recall[arm]:>7.3f}")
        lines += [
            f"decoy assertions at depth (four-hop arm): {self.puns}",
            f"delta (four hops vs three) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One depth-five world.

    Returns ``(facts, fives, fours, threes, twos, decoys, directs)``
    where fives are ``(question, expected|None, middles)``.
    """
    entities = []
    seen = set()
    while len(entities) < 14:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            entities.append(name)

    facts = []
    attr_values = {}
    for realm in _REALMS:
        for attr in _REALM_ATTRS:
            value = rng.choice(_ATTR_VALUES)
            attr_values[(realm, attr)] = value
            facts.append((f"{realm} {attr}", value))

    region_realms = {}
    for region in _REGIONS:
        realm = rng.choice(_REALMS)
        region_realms[region] = realm
        facts.append((f"{region} realm", realm))
    place_regions = {}
    for place in _PLACES:
        region = rng.choice(_REGIONS)
        place_regions[place] = region
        facts.append((f"{place} region", region))

    agents = rng.sample(_AGENTS, 6)
    agent_places = {}
    fives = []
    agent_index = 0
    for spot, entity in enumerate(entities):
        if spot < 4:
            role = rng.choice(_ROLES)
            agent = agents[agent_index]
            agent_index += 1
            place = rng.choice(_PLACES)
            broken = rng.random() < 0.25
            facts.append((f"{entity} {role}", agent))
            if not broken:
                facts.append((f"{agent} hometown", place))
                agent_places[agent] = place
            attr = rng.choice(_REALM_ATTRS)
            region = place_regions[place]
            realm = region_realms[region]
            expected = (None if broken
                        else attr_values[(realm, attr)])
            fives.append(
                (f"what is the {entity} {role} hometown region "
                 f"realm {attr}?", expected,
                 [agent, place, region, realm]))
        elif spot < 6:
            role = rng.choice(_ROLES)
            base = agents[rng.randrange(0, agent_index)]
            epithet = rng.choice(_EPITHETS)
            facts.append((f"{entity} {role}",
                          f"{base} the {epithet}"))
            if base not in agent_places:
                place = rng.choice(_PLACES)
                agent_places[base] = place
                facts.append((f"{base} hometown", place))

    decoys = []
    for entity in entities[4:6]:
        role = rng.choice(_ROLES)
        decoys.append(f"what is the {entity} {role} hometown region "
                      f"realm {rng.choice(_REALM_ATTRS)}?")
    for _ in range(2):
        ghost = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if ghost in seen:
            continue
        decoys.append(f"what is the {ghost} {rng.choice(_ROLES)} "
                      f"hometown region realm "
                      f"{rng.choice(_REALM_ATTRS)}?")

    fours = []
    for entity in entities[6:8]:
        role = rng.choice(_ROLES)
        agent = rng.choice([a for a in agents if a in agent_places])
        facts.append((f"{entity} {role}", agent))
        place = agent_places[agent]
        fours.append(
            (f"what is the {entity} {role} hometown region realm?",
             region_realms[place_regions[place]]))

    threes = []
    for entity in entities[8:10]:
        place = rng.choice(_PLACES)
        facts.append((f"{entity} city", place))
        threes.append(
            (f"what is the {entity} city region realm?",
             region_realms[place_regions[place]]))

    twos = []
    for entity in entities[10:12]:
        place = rng.choice(_PLACES)
        facts.append((f"{entity} port", place))
        twos.append((f"what is the {entity} port region?",
                     place_regions[place]))

    directs = []
    pool = [f for f in facts if f[0].split()[-1] in _REALM_ATTRS]
    rng.shuffle(pool)
    for key, value in pool[:4]:
        directs.append((f"what is the {key}?", value))

    rng.shuffle(facts)
    return facts, fives, fours, threes, twos, decoys, directs


def run_gate(seeds: int = 12) -> Depth5Report:
    """Run both arms over generated depth-five worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`Depth5Report`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    five_scores = {"three-hop": [], "four-hop": []}
    four_scores = {"three-hop": [], "four-hop": []}
    three_scores = {"three-hop": [], "four-hop": []}
    two_scores = {"three-hop": [], "four-hop": []}
    recall_scores = {"three-hop": [], "four-hop": []}
    puns = 0

    real_hops = inference_module._MAX_HOPS
    real_floor = inference_module._FLOOR
    for seed in range(seeds):
        rng = random.Random(seed)
        (facts, fives, fours, threes, twos,
         decoys, directs) = _build_world(rng)

        for arm in ("three-hop", "four-hop"):
            if arm == "four-hop":
                inference_module._MAX_HOPS = 4
                inference_module._FLOOR = 0.05
            else:
                inference_module._MAX_HOPS = real_hops
                inference_module._FLOOR = real_floor
            root = Path(tempfile.mkdtemp(prefix=f"uq_d5_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, value in facts:
                    session.memory.remember_fact(key, value,
                                                 confidence=0.6)

                won = []
                for question, expected, middles in fives:
                    response, _trace = run_pipeline(question, session)
                    if expected is None:
                        won.append(0.0)
                        continue
                    good = (expected in response
                            and "inferred" in response
                            and all(m in response for m in middles))
                    won.append(float(good))
                five_scores[arm].append(
                    statistics.fmean(won) if won else 0.0)

                for question in decoys:
                    response, _trace = run_pipeline(question, session)
                    if "inferred" in response and arm == "four-hop":
                        puns += 1

                for scores, batch in ((four_scores, fours),
                                      (three_scores, threes),
                                      (two_scores, twos)):
                    hits = []
                    for question, expected in batch:
                        response, _trace = run_pipeline(question,
                                                        session)
                        hits.append(float(expected in response
                                          and "inferred" in response))
                    scores[arm].append(
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

    report = Depth5Report(
        five={arm: statistics.fmean(vals)
              for arm, vals in five_scores.items()},
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

    contrasts = [five_scores["four-hop"][i]
                 - five_scores["three-hop"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.puns > 0:
        report.reason = (
            f"§11.31's warning binds at four bridges: {report.puns} "
            "decoy(s) asserted - the limit stays, and this run is the "
            "measurement that keeps it")
        return report
    floors = ((report.four, "4-chains"), (report.three, "3-chains"),
              (report.two, "2-chains"))
    for scores, label in floors:
        if scores["four-hop"] < scores["three-hop"]:
            report.reason = (f"deeper reach cost shallower honesty: "
                             f"{label} regressed under the extended "
                             "constants")
            return report
    if min(report.recall.values()) < 1.0:
        report.reason = "direct recall regressed"
        return report
    if report.delta <= 0:
        report.reason = (f"the fourth hop buys nothing "
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
        f"five-fact chains answer at {report.five['four-hop']:.3f} "
        f"against the three-hop baseline's "
        f"{report.five['three-hop']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with zero decoy "
        "assertions at four bridges and every shallower floor intact "
        "- the structural rules carry a fourth bridge too")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
