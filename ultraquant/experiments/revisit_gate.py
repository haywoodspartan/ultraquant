"""The density revisit: a session of capabilities at ten thousand facts.

§11.37's lesson has governed every unit since it was learned: found is
not believed, and clean-room success means nothing until density has
its say. The §11.45-§11.57 arc — depth-4 chains, polarity, polar and
why questions, comparatives with derived operands, superlatives,
aggregates — was gated at small worlds (dozens of facts each), which
is where mechanisms are debuggable. This gate is the capstone the
lesson demands: one §11.37-style world of ~10,000 colliding facts, and
every question form of the session asked against it.

Two things are measured and one is priced. Measured: correctness
floors, pre-registered per form below — and FABRICATION, at absolute
zero across every decoy family the session built (epithet agents at
depth, ghost entities, unheld polar subjects, denial bridges). Priced:
latency per question form, reported and not gated — the superlative
and aggregate full-store scans are an explicit cost by design (§11.56:
"correctness before optimization"), and this run is where that price
is finally seen at scale.

**The criteria, written before the run.**

1. **Fabrication is zero at scale.** Epithet-agent depth questions,
   ghost-entity chains, unheld polar subjects, and denial-bridge
   questions must be asserted ZERO times. One is a fail.
2. **The floors hold.** Unbroken depth-4 chains >= 0.75; stored polar
   questions = 1.000; why questions >= 0.9; stored comparatives =
   1.000; superlative winners = 1.000 (exact enumeration is
   scale-independent by design — any miss is a mechanism bug, not a
   density effect); aggregates exact = 1.000.
3. **The price is named.** Mean per-question latency reported for
   every form, full-scan forms included. Reported, never gated.

SKIPS nothing and contrasts nothing: this is a floors-and-controls
gate in the §11.41 containment tradition — the arms already ran, one
gate at a time, all session long.

**It was run twice: the first run indicted the harness, the second
PASSED on every floor.**

The first run built 2,731 facts while claiming ten thousand — the
three-word name pool held ~900 combinations and ran dry — and scored
polar, why, and compare at 0.000 with second-long latencies while
depth-4 sat at 1.000. The asymmetry was the diagnosis: the builder
ingested keys WITH articles ("the north tower hall material") through
direct `remember_fact`, but the live statement path strips articles
and the recall ladders look up stripped keys, so every recall-backed
form missed facts the spread's token-based retrieval could still see.
A density harness must key its world the way live usage would — pools
were enlarged (20 x 20 x 19) and the keys stripped. Then, at 9,631
facts:

| form | floor | ms/query |
|---|---:|---:|
| depth4 | 1.000 | 726.5 |
| polar | 1.000 | 1.3 |
| why | 1.000 | 1.3 |
| compare | 1.000 | 4.6 |
| superlative | 1.000 | 49.5 |
| aggregate | 1.000 | 18.3 |

**Every floor at 1.000, zero fabrications across every decoy family.**
The price list reads exactly as designed: the recall-backed forms ride
the addressed index at a millisecond or two, the full-scan forms cost
tens of milliseconds and say so, and depth-4's three synchronous
frontier rounds over a dense store cost 0.7 seconds — the expensive
capability, expensively honest. The session's arc survives density
whole.

**Run three extended the battery to the units that landed after run
two** — §11.64's history and §11.66's choices joined the floors at
1.000 each — and re-read the whole table under §11.59's frontier
clock: depth-4 at 339 ms (down from 726.5), history and choice at 1.2
and 1.3 ms on the addressed index, every other floor and price
holding. Eight forms, one dense world, nothing fabricated anywhere:
the closing measurement of the session it capstones.

Run it::

    python -m ultraquant.experiments.revisit_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RevisitReport", "run_gate"]

_PREFIXES = ["north", "south", "east", "west", "old", "new", "high",
             "low", "royal", "great", "inner", "outer", "upper",
             "lower", "grand", "little", "far", "near", "twin",
             "broken"]
_NOUNS = ["tower", "bridge", "spire", "tunnel", "gate", "dome", "mill",
          "keep", "wall", "hall", "arch", "fort", "barn", "pier",
          "vault", "court", "lodge", "stair", "cellar", "roof"]
_ATTRIBUTES = ["height", "width", "length"]
_ROLES = ["architect", "founder", "sculptor", "keeper"]
_AGENTS = ["wren", "aldous", "berta", "cosimo", "delia", "edmund",
           "flora", "gideon", "hesper", "ivo"]
_PLACES = ["york", "delft", "siena", "bruges", "cadiz", "trento"]
_REGIONS = ["norland", "esterre", "vastmark", "sudmoor"]
_ATTR_VALUES = ["temperate", "humid", "arid", "rocky"]
_EPITHETS = ["younger", "elder"]


@dataclass
class RevisitReport:
    """Floors, fabrications, and the price list.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        floors: Form -> correctness at 10k.
        fabricated: Decoy assertions across all forms (must be 0).
        latency_ms: Form -> mean per-question milliseconds (reported).
        facts: The world size actually built.
        reason: Plain-language verdict.
    """

    passes: bool = False
    floors: dict = field(default_factory=dict)
    fabricated: int = 0
    latency_ms: dict = field(default_factory=dict)
    facts: int = 0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"world: {self.facts:,} facts",
                 f"{'form':<14} {'floor':>7} {'ms/query':>9}"]
        for form in sorted(self.floors):
            lines.append(
                f"{form:<14} {self.floors[form]:>7.3f} "
                f"{self.latency_ms.get(form, 0.0):>9.1f}")
        lines += [
            f"fabrications across every decoy family: {self.fabricated}",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _entities(rng: random.Random, wanted: int) -> list[str]:
    out, seen = [], set()
    for _attempt in range(wanted * 60):
        if len(out) >= wanted:
            break
        name = (f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)} "
                f"{rng.choice(_NOUNS)}")
        first, second, third = name.split()
        if second == third or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def run_gate(n_facts: int = 10_000, seed: int = 0) -> RevisitReport:
    """Build one dense world and ask it everything.

    Args:
        n_facts: Target world size.
        seed: World seed.

    Returns:
        The :class:`RevisitReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    rng = random.Random(seed)
    root = Path(tempfile.mkdtemp(prefix="uq_revisit_"))
    report = RevisitReport()
    try:
        session = build_session(root, seed=seed)
        memory = session.memory

        # --- the filler mass: colliding three-word entities.
        names = _entities(rng, max(40, n_facts // 3))
        stored = 0
        heights = {}
        for entity in names:
            for attr in _ATTRIBUTES:
                value = rng.randrange(3, 900)
                memory.remember_fact(f"{entity} {attr}",
                                     f"{value} meters", confidence=0.6)
                if attr == "height":
                    heights[entity] = value
                stored += 1
                if stored >= n_facts - 400:
                    break
            if stored >= n_facts - 400:
                break

        # --- region attributes and place regions (depth scaffolding).
        attr_values = {}
        for region in _REGIONS:
            for attr in ("climate", "terrain"):
                value = rng.choice(_ATTR_VALUES)
                attr_values[(region, attr)] = value
                memory.remember_fact(f"{region} {attr}", value,
                                     confidence=0.6)
                stored += 1
        place_regions = {}
        for place in _PLACES:
            region = rng.choice(_REGIONS)
            place_regions[place] = region
            memory.remember_fact(f"{place} region", region,
                                 confidence=0.6)
            stored += 1

        # --- depth-4 chains over distinct entities.
        chain_entities = names[:6]
        agents = rng.sample(_AGENTS, 6)
        fours = []
        for spot, entity in enumerate(chain_entities):
            role = rng.choice(_ROLES)
            agent = agents[spot]
            place = rng.choice(_PLACES)
            memory.remember_fact(f"{entity} {role}", agent,
                                 confidence=0.6)
            memory.remember_fact(f"{agent} hometown", place,
                                 confidence=0.6)
            stored += 2
            attr = rng.choice(("climate", "terrain"))
            region = place_regions[place]
            fours.append(
                (f"what is the {entity} {role} hometown region {attr}?",
                 attr_values[(region, attr)],
                 [agent, place, region]))

        # --- decoys: epithet agents, ghosts, denial bridges.
        decoys = []
        for spot, entity in enumerate(names[6:8]):
            role = rng.choice(_ROLES)
            base = agents[spot]
            memory.remember_fact(
                f"{entity} {role}",
                f"{base} the {rng.choice(_EPITHETS)}", confidence=0.6)
            stored += 1
            decoys.append(f"what is the {entity} {role} hometown "
                          f"region climate?")
        for _ in range(2):
            ghost = (f"{rng.choice(_PREFIXES)} {rng.choice(_NOUNS)} "
                     f"{rng.choice(_NOUNS)}")
            if ghost in set(names):
                continue
            decoys.append(f"what is the {ghost} "
                          f"{rng.choice(_ROLES)} hometown region "
                          "climate?")
        denial_entity = names[8]
        memory.remember_fact(f"{denial_entity} material", "steel",
                             confidence=0.6, negated=True)
        memory.remember_fact("steel conductivity", "700 units",
                             confidence=0.6)
        stored += 2
        decoys.append(f"what is the {denial_entity} conductivity?")

        # --- polar and why targets.
        polar_entity = names[9]
        memory.remember_fact(f"{polar_entity} material", "iron",
                             confidence=0.6)
        stored += 1
        polar_qs = [
            (f"is the {polar_entity} material iron?", "yes"),
            (f"is the {polar_entity} material steel?", "no"),
            (f"is the {denial_entity} material steel?", "no"),
        ]
        choice_qs = [
            (f"is the {polar_entity} material steel or iron?", "iron"),
            (f"is the {polar_entity} material oak or granite?",
             "neither"),
        ]
        history_entity = names[11]
        memory.remember_fact(f"{history_entity} material", "iron",
                             confidence=0.6)
        memory.remember_fact(f"{history_entity} material", "steel",
                             confidence=0.6)
        stored += 1
        history_q = (f"what was the {history_entity} material?",
                     ["It was iron", "it is steel now"])
        unheld_polar = [f"is the {names[10]} material steel?"]
        why_qs = [
            (f"why is the {polar_entity} material iron?",
             "stated directly"),
            (f"why is the {polar_entity} material steel?",
             "It isn't"),
        ]

        # --- comparatives and the family forms.
        a, b = names[0], names[1]
        bigger, smaller = ((a, b) if heights[a] > heights[b]
                           else (b, a))
        if heights[a] == heights[b]:
            heights[bigger] += 1
            memory.remember_fact(f"{bigger} height",
                                 f"{heights[bigger]} meters",
                                 confidence=0.6)
        comp_qs = [
            (f"is the {bigger} height taller than the {smaller} "
             "height?", "yes"),
            (f"is the {smaller} height taller than the {bigger} "
             "height?", "no"),
        ]
        tallest = max(heights, key=lambda n: heights[n])
        ties = [n for n in heights
                if heights[n] == heights[tallest]]
        super_q = ("which height is the tallest?",
                   f"{tallest} height" if len(ties) == 1 else None)
        count_q = ("how many climate facts do you hold?",
                   f"I hold {len(_REGIONS)} climate fact(s)")

        report.facts = stored

        def timed(question):
            started = time.perf_counter()
            response, _trace = run_pipeline(question, session)
            return response, 1000.0 * (time.perf_counter() - started)

        fabricated = 0
        floors = {}
        latency = {}

        # depth-4
        wins, times = [], []
        for question, expected, middles in fours:
            response, ms = timed(question)
            times.append(ms)
            wins.append(float(expected in response
                              and "inferred" in response
                              and all(m in response for m in middles)))
        floors["depth4"] = statistics.fmean(wins)
        latency["depth4"] = statistics.fmean(times)

        # decoys (chains + denial bridge)
        times = []
        for question in decoys:
            response, ms = timed(question)
            times.append(ms)
            if "inferred" in response:
                fabricated += 1
        latency["decoys"] = statistics.fmean(times)

        # polar
        wins, times = [], []
        for question, verdict in polar_qs:
            response, ms = timed(question)
            times.append(ms)
            wins.append(float(response.strip().lower()
                              .startswith(verdict)))
        floors["polar"] = statistics.fmean(wins)
        latency["polar"] = statistics.fmean(times)
        for question in unheld_polar:
            response, ms = timed(question)
            lowered = response.strip().lower()
            if lowered.startswith(("yes", "no")):
                fabricated += 1

        # why
        wins, times = [], []
        for question, marker in why_qs:
            response, ms = timed(question)
            times.append(ms)
            wins.append(float(marker in response))
        floors["why"] = statistics.fmean(wins)
        latency["why"] = statistics.fmean(times)

        # comparatives
        wins, times = [], []
        for question, verdict in comp_qs:
            response, ms = timed(question)
            times.append(ms)
            wins.append(float(response.strip().lower()
                              .startswith(verdict)))
        floors["compare"] = statistics.fmean(wins)
        latency["compare"] = statistics.fmean(times)

        # superlative (full scan)
        question, winner = super_q
        response, ms = timed(question)
        latency["superlative"] = ms
        if winner is None:
            floors["superlative"] = float(
                response.startswith("Tied"))
        else:
            floors["superlative"] = float(
                response.startswith(f"The {winner}"))

        # aggregate count (full scan)
        question, expected = count_q
        response, ms = timed(question)
        latency["aggregate"] = ms
        floors["aggregate"] = float(expected in response)

        # choices (§11.66)
        wins, times = [], []
        for question, lead in choice_qs:
            response, ms = timed(question)
            times.append(ms)
            wins.append(float(response.strip().lower()
                              .startswith(lead)))
        floors["choice"] = statistics.fmean(wins)
        latency["choice"] = statistics.fmean(times)

        # history (§11.64)
        question, payload = history_q
        response, ms = timed(question)
        latency["history"] = ms
        floors["history"] = float(all(p in response for p in payload))

        report.floors = floors
        report.fabricated = fabricated
        report.latency_ms = latency

        if fabricated > 0:
            report.reason = (f"{fabricated} fabrication(s) at scale - "
                             "the decoy families found a door")
            return report
        checks = [("depth4", 0.75), ("polar", 1.0), ("why", 0.9),
                  ("compare", 1.0), ("superlative", 1.0),
                  ("aggregate", 1.0), ("choice", 1.0),
                  ("history", 1.0)]
        for form, floor in checks:
            if floors[form] < floor:
                report.reason = (f"the {form} floor broke: "
                                 f"{floors[form]:.3f} against "
                                 f"{floor}")
                return report
        report.passes = True
        report.reason = (
            "every floor held and nothing fabricated at "
            f"{report.facts:,} facts; the full-scan forms cost what "
            f"the table says and no more")
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
