"""The depth clock: licensed by measurement, held to parity. The gate.

The house rule is capability before optimization, and §11.58 finally
licensed this unit: depth-4 chains cost 726.5 ms/query at density, the
largest number on the capstone's price list, with 92% of the time in
index-search volume — every activated node re-relaying in every
synchronous round, recomputing identical bridge probes, and shared
bridge values re-querying the same buckets. Two structural
call-eliminations close it, both provably identical in output:
**frontier relaying** (after round one, only nodes that gained or
strengthened an origin last round relay — a node's direct sources
never change after seeding, and equal-strength re-arrivals were
already discarded unread by the strict comparison) and a **per-spread
probe memo** (bridge probes depend only on the value and the fixed
question; within one spread the same value queries the index once).

§11.45's precedent governs the shape: the simulator is the reference,
accelerants are never semantics — parity is absolute and comes first,
the clock is read second. Both arms run against the SAME session (the
flag touches only read-path computation), so parity is compared on
identical store bytes.

**The criteria, written before the run.**

1. **Parity** — every question in the battery (depth-4 chains, depth-3
   and depth-2 chains, modifier tolerance, negated terminals, and
   every decoy family) must produce a BYTE-IDENTICAL response in both
   arms. One divergence is a fail: a fast wrong answer is not an
   optimization, and a fast differently-worded answer is not parity.
2. **The clock pays** — mean per-query wall time over repeated depth-4
   questions must be lower in the frontier arm by more than the
   repetition-to-repetition sd of the paired contrast.

**It was run once, and PASSED — parity absolute, then a 3.2x clock.**

| measure | result |
|---|---:|
| battery divergences | **0** of 10 |
| per-query, every-node | 270.3 ms |
| per-query, frontier | **84.8 ms** |

**+185.5 ms per depth-4 query at 20.27x repetition sd**, with every
battery question — chains at every depth, modifier tolerance, negated
terminals, every decoy family — answering byte-identically in both
arms on identical store bytes. The two eliminations did exactly what
the profile said they would: the frontier stopped ~200 activated
nodes from re-relaying rounds they had nothing new to say in, and the
memo stopped shared bridge values from re-querying the same buckets.
Rule 1 held the whole way: parity first, the clock second, §11.45's
words unchanged one tier up.

Run it::

    python -m ultraquant.experiments.clock_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ClockReport", "run_gate"]

_REPEATS = 8


@dataclass
class ClockReport:
    """Parity, the paired clock, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        divergences: Battery questions whose responses differed
            (must be 0).
        battery: Questions in the parity battery.
        old_ms: Mean per-query ms, every-node-every-round arm.
        new_ms: Mean per-query ms, frontier arm.
        delta_ms: Mean paired saving per query.
        rep_sd: Repetition sd of the paired contrast.
        margin_ratio: ``delta_ms / rep_sd``.
        facts: World size.
        reason: Plain-language verdict.
    """

    passes: bool = False
    divergences: int = 0
    battery: int = 0
    old_ms: float = 0.0
    new_ms: float = 0.0
    delta_ms: float = 0.0
    rep_sd: float = 0.0
    margin_ratio: float = 0.0
    facts: int = 0
    reason: str = ""

    def as_text(self) -> str:
        """The numbers, then the verdict."""
        lines = [
            f"world: {self.facts:,} facts; battery: {self.battery} "
            f"questions, {self.divergences} divergence(s)",
            f"per-query: every-node {self.old_ms:.1f} ms | "
            f"frontier {self.new_ms:.1f} ms",
            f"delta {self.delta_ms:+.1f} ms   rep sd {self.rep_sd:.1f}"
            f"   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def run_gate(n_facts: int = 5_000, seed: int = 0) -> ClockReport:
    """Build one dense world, hold parity, then read the clock.

    Args:
        n_facts: Target world size.
        seed: World seed.

    Returns:
        The :class:`ClockReport`.
    """
    from ultraquant.experiments.revisit_gate import (_AGENTS,
                                                     _ATTR_VALUES,
                                                     _ATTRIBUTES,
                                                     _PLACES, _REGIONS,
                                                     _ROLES, _entities)
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.reason import inference as inference_module

    rng = random.Random(seed)
    root = Path(tempfile.mkdtemp(prefix="uq_clock_"))
    report = ClockReport()
    real_flag = inference_module._FRONTIER_SPREAD
    try:
        session = build_session(root, seed=seed)
        memory = session.memory
        names = _entities(rng, max(40, n_facts // 3))
        stored = 0
        for entity in names:
            for attr in _ATTRIBUTES:
                memory.remember_fact(
                    f"{entity} {attr}",
                    f"{rng.randrange(3, 900)} meters", confidence=0.6)
                stored += 1
                if stored >= n_facts - 200:
                    break
            if stored >= n_facts - 200:
                break
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

        agents = rng.sample(_AGENTS, 4)
        depth4_qs = []
        for spot, entity in enumerate(names[:4]):
            role = rng.choice(_ROLES)
            agent = agents[spot]
            place = rng.choice(_PLACES)
            memory.remember_fact(f"{entity} {role}", agent,
                                 confidence=0.6)
            memory.remember_fact(f"{agent} hometown", place,
                                 confidence=0.6)
            stored += 2
            attr = rng.choice(("climate", "terrain"))
            depth4_qs.append(
                f"what is the {entity} {role} hometown region {attr}?")

        chain_entity = names[4]
        memory.remember_fact(f"{chain_entity} material", "bronzite",
                             confidence=0.6)
        memory.remember_fact("bronzite base", "bronze", confidence=0.6)
        memory.remember_fact("bronze melting point", "913 degrees",
                             confidence=0.6)
        denial_entity = names[5]
        memory.remember_fact(f"{denial_entity} material", "steel",
                             confidence=0.6, negated=True)
        memory.remember_fact("steel conductivity", "700 units",
                             confidence=0.6)
        neg_entity = names[6]
        memory.remember_fact(f"{neg_entity} city", "york",
                             confidence=0.6)
        memory.remember_fact("york climate", "temperate",
                             confidence=0.6, negated=True)
        epithet_entity = names[7]
        memory.remember_fact(f"{epithet_entity} architect",
                             f"{agents[0]} the younger",
                             confidence=0.6)
        stored += 8
        report.facts = stored

        battery = list(depth4_qs) + [
            f"what is the {chain_entity} melting point?",
            f"what is the weathered {chain_entity} melting point?",
            f"what is the {denial_entity} conductivity?",
            f"what is the {neg_entity} city climate?",
            f"what is the {epithet_entity} architect hometown region "
            "climate?",
            "what is the missing spire vault architect hometown "
            "region climate?",
        ]
        report.battery = len(battery)

        # Parity first: byte-identical responses on identical bytes.
        divergences = 0
        for question in battery:
            inference_module._FRONTIER_SPREAD = False
            old_response, _trace = run_pipeline(question, session)
            inference_module._FRONTIER_SPREAD = True
            new_response, _trace = run_pipeline(question, session)
            if old_response != new_response:
                divergences += 1
        report.divergences = divergences
        if divergences > 0:
            report.reason = (f"parity failed on {divergences} "
                             "question(s): a fast different answer is "
                             "not an optimization")
            return report

        # The clock second: paired repetitions over the depth-4 set.
        contrasts = []
        old_times, new_times = [], []
        for _rep in range(_REPEATS):
            for question in depth4_qs:
                inference_module._FRONTIER_SPREAD = False
                started = time.perf_counter()
                run_pipeline(question, session)
                mid = time.perf_counter()
                inference_module._FRONTIER_SPREAD = True
                run_pipeline(question, session)
                done = time.perf_counter()
                old_ms = 1000.0 * (mid - started)
                new_ms = 1000.0 * (done - mid)
                old_times.append(old_ms)
                new_times.append(new_ms)
                contrasts.append(old_ms - new_ms)

        report.old_ms = statistics.fmean(old_times)
        report.new_ms = statistics.fmean(new_times)
        report.delta_ms = statistics.fmean(contrasts)
        # stdev, not pstdev: repetitions are a sample of timings.
        report.rep_sd = (statistics.stdev(contrasts)
                         if len(contrasts) > 1 else 0.0)
        report.margin_ratio = (report.delta_ms / report.rep_sd
                               if report.rep_sd > 0 else 0.0)

        if report.delta_ms <= 0:
            report.reason = (f"the frontier saves nothing "
                             f"({report.delta_ms:+.1f} ms)")
            return report
        if report.rep_sd <= 0.0:
            report.reason = ("every repetition timed identically; "
                             "nothing varied")
            return report
        if report.margin_ratio <= 1.0:
            report.reason = (f"saving {report.delta_ms:+.1f} ms is "
                             f"{report.margin_ratio:.2f}x the "
                             "repetition sd")
            return report
        report.passes = True
        report.reason = (
            f"parity held on all {report.battery} questions and the "
            f"clock fell {report.old_ms:.1f} -> {report.new_ms:.1f} ms "
            f"per depth-4 query ({report.delta_ms:+.1f} ms, "
            f"{report.margin_ratio:.2f}x rep sd)")
        return report
    finally:
        inference_module._FRONTIER_SPREAD = real_flag
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
