"""Three-fact chains under the density lessons. The depth gate.

§11.31 built spreading activation and §11.37 hardened it against 10,000
colliding facts — but every density world chained exactly TWO facts
(entity -> material -> property). Three-fact chains (entity -> agent ->
place -> attribute, two bridges) turn out to be mechanically reachable
inside `_MAX_HOPS = 2`: the middle fact's direct contact merges into
the carried lineage, and decay 1.0 -> 0.25 clears the 0.2 floor. Found
is not believed: reachable in a clean room is not honest under
collision pressure, and a four-fact probe already caught the
depth-specific pun — a bridge carries lineage on ANY shared token with
the fact's value, so "spire architect = wren the younger" hops through
plain wren's birthplace and asserts york's climate for an entity whose
birthplace was never stored. The origin-coverage lesson ("low mill" may
not answer for "the royal mill") mirrored onto the VALUE side: a
qualifier silently dropped substitutes one entity for another.

Both arms run the live question surface over the same worlds; facts are
ingested directly into session memory (ingestion is not under test),
questions go through the pipeline. The baseline arm caps the spread at
one hop (two-fact chains only), the treatment arm runs the shipped two
hops. World mix: three-fact chain questions (the claim), ~25% of them
broken in the middle (the agent's relation was never stored — refused
by both arms, the variance, per §11.35's power arithmetic), epithet
decoys in every world ("wren the younger" holds the role, only plain
wren holds a birthplace), and ghost-entity decoys in the §11.36
tradition.

**The criteria, written before the run.**

1. **Gain** — over all worlds, the three-chain answer rate (correct
   value, marked inferred, full trail naming both middle values) must
   beat the one-hop baseline by more than the seed-to-seed sd.
   Broken-middle chains score zero for both arms.
2. **Partial-value control** — epithet-decoy questions must be
   asserted in NEITHER arm, ever: the answer for "wren the younger"
   must never be plain wren's. One assertion is a fail, sd or no sd.
3. **Ghost control** — unknown-entity chain questions must be asserted
   in neither arm, ever.
4. **Recall regression** — direct recall of sampled stored facts must
   answer at 1.000 in both arms: depth work may not disturb the
   retrieval below it.

**It was run three times, and each run indicted a different layer.**

Run one: 3-chains 0.000 in BOTH arms, 72/72 epithet "assertions" — but
reading actual responses showed every one was the same interception:
the question surface's exact-recall branch asserted the sub-key ("the
high mill sculptor is aldous") for the whole depth question and
returned before inference ran. Recall's candidate ladder tries
sub-n-grams, so the branch believed "hit" meant "exact"; §11.29's
coverage rule now guards it, and uncovered questions fall through to
the chain machinery that could always answer them.

Run two: chains answered — 0.667 in the ONE-HOP arm, which should be
impossible. It was iteration order: origins mutated in place during a
round's pass, so a lucky dict order cascaded origin -> middle ->
target inside one "hop". Hop depth depended on dict order — cognition
must not. Rounds are now synchronous frontiers (updates collect during
the pass, apply after), and `_MAX_HOPS` finally means what it says.
The same run measured the epithet pun genuinely: 72/72 asserted
through inference, plain wren's birthplace delivered for "wren the
younger". The whole-value rule closed it, both directions, with one
terminal relation slot allowed per hop ("bronzite BASE" traverses;
"wren the YOUNGER birthplace" refuses — an unaccounted token inside
the subject is a qualifier being dropped).

Run three, with all three layers honest:

| arm | 3-chains | recall |
|---|---:|---:|
| one-hop | 0.000 | 1.000 |
| two-hop | **0.688** | 1.000 |

**+0.688 at 2.85x seed sd**, zero assertions on either decoy form in
either arm, recall untouched. The two-hop miss is the broken-middle
share scoring zero by design, and the one-hop zero is the baseline
finally meaning depth. Three-fact chains — two bridges, decay
1.0 -> 0.25 over the 0.2 floor — are now a measured capability, and
the floor's refusal of the fourth fact stays the architecture's
stated limit.

Run it::

    python -m ultraquant.experiments.depth_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DepthReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall"]
_ROLES = ["architect", "founder", "sculptor", "keeper"]
_RELATIONS = ["birthplace", "hometown", "workshop"]
_PLACE_ATTRS = ["climate", "elevation", "population"]
_AGENTS = ["wren", "aldous", "berta", "cosimo", "delia", "edmund",
           "flora", "gideon", "hesper", "ivo", "jorah", "keziah"]
_PLACES = ["york", "delft", "siena", "bruges", "cadiz", "trento",
           "arles", "leipzig", "orense", "tartu"]
_EPITHETS = ["younger", "elder", "lesser"]
_ATTRIBUTES = ["height", "width", "length"]


@dataclass
class DepthReport:
    """The two arms, three controls, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        chains: Arm -> three-chain answer rate over all worlds.
        partial_value: Epithet-decoy assertions (must be 0).
        ghosts: Ghost-entity assertions (must be 0).
        recall: Arm -> direct-recall rate (must be 1.0).
        delta: Chain contrast (two hops minus one).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    chains: dict = field(default_factory=dict)
    partial_value: int = 0
    ghosts: int = 0
    recall: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<10} {'3-chains':>9} {'recall':>7}"]
        for arm in ("one-hop", "two-hop"):
            lines.append(f"{arm:<10} {self.chains[arm]:>9.3f} "
                         f"{self.recall[arm]:>7.3f}")
        lines += [
            f"partial-value assertions (wren the younger): "
            f"{self.partial_value}",
            f"ghost-entity assertions: {self.ghosts}",
            f"delta (two hops vs one) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _build_world(rng: random.Random):
    """One depth world: chains, decoys, and collision filler.

    Returns ``(facts, chain_questions, epithet_decoys, ghost_decoys,
    directs)`` where chain_questions are ``(question, expected|None,
    agent, place)`` - expected None marks a broken-middle chain whose
    honest answer is refusal.
    """
    entities = []
    seen = set()
    while len(entities) < 14:
        name = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if name not in seen:
            seen.add(name)
            entities.append(name)

    agents = rng.sample(_AGENTS, 10)
    places = rng.sample(_PLACES, 6)
    facts = []
    place_values = {}
    for place in places:
        for attr in _PLACE_ATTRS:
            value = rng.randrange(100, 9000)
            place_values[(place, attr)] = value
            facts.append((f"the {place} {attr}", f"{value} units"))

    chain_questions = []
    epithet_decoys = []
    agent_relations: dict[str, dict[str, str]] = {}
    #: (agent, relation) pairs a BROKEN chain depends on staying absent.
    forbidden: set[tuple[str, str]] = set()
    agent_index = 0
    for spot, entity in enumerate(entities):
        for attr in rng.sample(_ATTRIBUTES, 2):
            facts.append((f"the {entity} {attr}",
                          f"{rng.randrange(3, 900)} meters"))
        if spot < 4:
            # A scored chain: entity -> agent -> place -> attribute.
            role = rng.choice(_ROLES)
            relation = rng.choice(_RELATIONS)
            agent = agents[agent_index]
            agent_index += 1
            place = rng.choice(places)
            broken = rng.random() < 0.25
            facts.append((f"the {entity} {role}", agent))
            if not broken:
                facts.append((f"{agent} {relation}", place))
                agent_relations.setdefault(agent, {})[relation] = place
            else:
                forbidden.add((agent, relation))
            attr = rng.choice(_PLACE_ATTRS)
            expected = (None if broken
                        else str(place_values[(place, attr)]))
            chain_questions.append(
                (f"what is the {entity} {role} {relation} {attr}?",
                 expected, agent, place))
        elif spot < 7:
            # An epithet decoy: the role is held by "{agent} the
            # {epithet}", and only PLAIN {agent} has this relation
            # stored. The pun answers the plain agent's place for the
            # epithet holder. Reuse an existing (agent, relation) pair
            # rather than writing the key again (a second write would
            # REVISE the genuine chain's place), and never store a pair
            # a broken chain depends on staying absent - the first
            # world build re-linked a broken chain exactly that way,
            # and its "refusal" expectation was scoring a lie.
            role = rng.choice(_ROLES)
            base = agents[rng.randrange(0, agent_index)]
            epithet = rng.choice(_EPITHETS)
            usable = [r for r in _RELATIONS
                      if (base, r) not in forbidden]
            relation = rng.choice(usable)
            held = agent_relations.setdefault(base, {})
            if relation not in held:
                held[relation] = rng.choice(places)
                facts.append((f"{base} {relation}", held[relation]))
            facts.append((f"the {entity} {role}",
                          f"{base} the {epithet}"))
            attr = rng.choice(_PLACE_ATTRS)
            epithet_decoys.append(
                f"what is the {entity} {role} {relation} {attr}?")

    ghost_decoys = []
    for _ in range(3):
        ghost = f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)}"
        if ghost in seen:
            continue
        ghost_decoys.append(
            f"what is the {ghost} {rng.choice(_ROLES)} "
            f"{rng.choice(_RELATIONS)} {rng.choice(_PLACE_ATTRS)}?")

    directs = []
    pool = [f for f in facts if f[0].split()[-1] in _PLACE_ATTRS]
    rng.shuffle(pool)
    for key, value in pool[:4]:
        directs.append((f"what is {key}?", value.split()[0]))

    rng.shuffle(facts)
    return facts, chain_questions, epithet_decoys, ghost_decoys, directs


def _assertive(response: str) -> bool:
    lowered = response.lower()
    if lowered.startswith(("i don't hold", "nothing believed",
                           "i can't plan")):
        return False
    return "(confidence" in lowered or "inferred" in lowered


def run_gate(seeds: int = 12) -> DepthReport:
    """Run both arms over generated depth worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`DepthReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    chains = {"one-hop": [], "two-hop": []}
    recall = {"one-hop": [], "two-hop": []}
    partial_value = 0
    ghosts = 0

    real_hops = inference_module._MAX_HOPS
    for seed in range(seeds):
        rng = random.Random(seed)
        (facts, chain_questions, epithet_decoys,
         ghost_decoys, directs) = _build_world(rng)

        for arm in ("one-hop", "two-hop"):
            inference_module._MAX_HOPS = 1 if arm == "one-hop" else 2
            root = Path(tempfile.mkdtemp(prefix=f"uq_depth_{seed}_"))
            try:
                session = build_session(root, seed=seed)
                for key, value in facts:
                    session.memory.remember_fact(key, value,
                                                 confidence=0.6)

                won = []
                for question, expected, agent, place in chain_questions:
                    response, _trace = run_pipeline(question, session)
                    if expected is None:
                        won.append(0.0)
                        continue
                    good = (expected in response
                            and "inferred" in response
                            and agent in response
                            and place in response)
                    won.append(float(good))
                chains[arm].append(statistics.fmean(won) if won else 0.0)

                for question in epithet_decoys:
                    response, _trace = run_pipeline(question, session)
                    if _assertive(response):
                        partial_value += 1
                for question in ghost_decoys:
                    response, _trace = run_pipeline(question, session)
                    if _assertive(response):
                        ghosts += 1

                hits = []
                for question, expected in directs:
                    response, _trace = run_pipeline(question, session)
                    hits.append(float(expected in response))
                recall[arm].append(statistics.fmean(hits)
                                   if hits else 1.0)
            finally:
                inference_module._MAX_HOPS = real_hops
                shutil.rmtree(root, ignore_errors=True)

    report = DepthReport(
        chains={arm: statistics.fmean(vals)
                for arm, vals in chains.items()},
        recall={arm: statistics.fmean(vals)
                for arm, vals in recall.items()},
        partial_value=partial_value,
        ghosts=ghosts,
    )

    contrasts = [chains["two-hop"][i] - chains["one-hop"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    if report.partial_value > 0:
        report.reason = (
            f"the partial-value control failed: {report.partial_value} "
            "epithet decoy(s) asserted - a dropped qualifier substituted "
            "one entity for another at depth")
        return report
    if report.ghosts > 0:
        report.reason = (f"the ghost control failed: {report.ghosts} "
                         "unknown-entity chain(s) asserted")
        return report
    if min(report.recall.values()) < 1.0:
        report.reason = ("recall regressed: depth work disturbed the "
                         "retrieval below it")
        return report
    if report.delta <= 0:
        report.reason = f"the second hop buys nothing ({report.delta:+.3f})"
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
        f"three-fact chains answer at {report.chains['two-hop']:.3f} "
        f"against the one-hop baseline's {report.chains['one-hop']:.3f} "
        f"({report.delta:+.3f}, {report.margin_ratio:.2f}x seed sd) with "
        "zero assertions on either decoy form and recall at 1.000 in "
        "both arms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
