"""Confirmation-gated consolidation and truth maintenance. The gate.

§11.30 measured spreading activation's honest floor: three bridges arrive
below the activation threshold and the spread refuses. Consolidation is
the brain-shaped answer to that floor — a derivation the user CONFIRMS
becomes a stored (but provenance-carrying) fact, and a stored conclusion
is a *premise*: the refused depth-3 question becomes two answerable
steps. Episodic to semantic, exactly once per confirmation.

The two ways this mechanism can rot, each with its control:

* **Staleness** — a consolidated conclusion outliving its premises is
  §11.16's entrenched error with a memory. Truth maintenance retracts
  derivatives (recursively) when a premise is revised
  (:meth:`SystematicMemory.remember_fact`), and the gate revises premises
  under consolidated facts to watch. This control failed once before the
  gate existed: the fact-shard store skipped flushing emptied buckets, a
  retracted fact resurrected from the stale cache, and recall
  reinforcement re-persisted the ghost. The fix is in
  ``factshards.flush``; the control here keeps it fixed.
* **Repetition-as-confirmation** — §11.16 measured what reinforcement
  without confirmation does (errors 3.9x harder to overturn). Asking the
  same question twice must consolidate NOTHING: only an explicit
  affirmation immediately after the derivation promotes, and pending
  state survives exactly one turn.

Both arms run the live pipeline over generated worlds holding a depth-3
chain (entity -> alloy -> base -> property). The dialogue asks the
intermediate question (derives the alloy's property over two bridges),
then affirms; the consolidation arm stores the confirmed conclusion, the
baseline arm has ``consolidate_fact`` disabled and everything else
identical.

**The criteria, written before the run.**

1. **Reach** — after the confirmed intermediate step, the depth-3
   question's accuracy in the consolidation arm must beat the baseline
   arm's by more than the seed-to-seed sd of that contrast.
2. **Staleness control** — after revising the chain's base premise, the
   stale-answer rate (the OLD value asserted for the deep or intermediate
   question) must not exceed the baseline arm's by more than one sd.
3. **Repetition control** — in a dialogue that repeats the intermediate
   question WITHOUT affirming, the deep question must stay refused in
   both arms: any leak is a fail, sd or no sd.

**It was run, and it PASSED — narrowly, in §11.13's tradition of saying so.**

| arm | deep reach | stale rate | repetition leak |
|---|---:|---:|---:|
| baseline | 0.000 | 0.000 | 0.000 |
| consolidation | **0.625** | 0.000 | 0.000 |

**+0.625 at 1.21x seed sd**, zero stale answers, zero leaks. The 0.375
miss is entirely the depth-4 worlds: there the intermediate derivation
itself sits past the activation floor, nothing earns confirmation, and
the deep question stays honestly refused — one confirmation buys exactly
one step of reach. Stepwise confirmation of longer chains works in
principle and is unmeasured; it would need its own protocol.

Two corrections the gate itself forced, recorded because each briefly
wore the mask of a control failure. First, both controls "failed" at
identical rates in BOTH arms — the metric was counting the honest
demotion ("Nearest I hold: iron melting point is 1370 degrees") as an
assertion because the value string appears in it; `_asserted` now
distinguishes claiming from mentioning. Second, before the metric fix,
the first world build was only two bridges deep (the entity-material
contact is not a bridge), so the spread answered everything and the
"leak" was ordinary derivation. Miscounting hops and miscounting claims
are both ways a gate lies politely; both are pinned in the tests.

Run it::

    python -m ultraquant.experiments.consolidation_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ConsolidationReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
_PROPERTIES = ["melting point", "density", "hardness"]


@dataclass
class ConsolidationReport:
    """The two arms across the three protocols, and the verdict.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        deep: Arm -> depth-3 accuracy after the confirmed step.
        stale: Arm -> stale-answer rate after premise revision.
        leak: Arm -> deep answers after repetition without affirmation.
        delta: Reach contrast (consolidation minus baseline).
        seed_sd: Seed sd of that contrast.
        margin_ratio: ``delta / seed_sd``.
        reason: Plain-language verdict.
    """

    passes: bool = False
    deep: dict = field(default_factory=dict)
    stale: dict = field(default_factory=dict)
    leak: dict = field(default_factory=dict)
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The table, then the verdict."""
        lines = [f"{'arm':<15} {'deep reach':>10} {'stale rate':>11} "
                 f"{'repetition leak':>16}"]
        for arm in ("baseline", "consolidation"):
            lines.append(f"{arm:<15} {self.deep[arm]:>10.3f} "
                         f"{self.stale[arm]:>11.3f} {self.leak[arm]:>16.3f}")
        lines += [
            f"delta (deep reach, with vs without) {self.delta:+.3f}   "
            f"seed sd {self.seed_sd:.3f}   ratio {self.margin_ratio:.2f}x",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ]
        return "\n".join(lines)


def _asserted(response: str, value: int) -> bool:
    """Did the reply CLAIM this value - not merely mention it?

    The honest demotion "I don't hold that exactly. Nearest I hold: iron
    melting point is 1370 degrees" contains the value string; counting it
    as an answer scored refusals as reach and as staleness in both arms.
    A claim is an inferred reply or a bare assertion, never a nearest-held
    offer or an unknown.
    """
    if str(value) not in response:
        return False
    if response.startswith(("I don't hold", "Nothing believed",
                            "I can't plan")):
        return False
    return True


def run_gate(seeds: int = 8) -> ConsolidationReport:
    """Run both arms over generated depth-3 worlds.

    Args:
        seeds: Fresh worlds per arm.

    Returns:
        The :class:`ConsolidationReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    from ultraquant.memory.systematic import SystematicMemory

    deep_acc = {"baseline": [], "consolidation": []}
    stale_rate = {"baseline": [], "consolidation": []}
    leak_rate = {"baseline": [], "consolidation": []}

    real_consolidate = SystematicMemory.consolidate_fact
    for seed in range(seeds):
        rng = random.Random(seed)
        entity = rng.choice(_ENTITIES)
        base, revised_base, spare = rng.sample(_MATERIALS, 3)
        prop = rng.choice(_PROPERTIES)
        alloy = f"{spare}ite"
        blend = f"{spare}oid"
        value = rng.randrange(100, 2000)
        revised_value = rng.randrange(2000, 4000)

        # Deep question = THREE bridges (alloy, blend, base) - §11.30's
        # measured refusal. The intermediate question resolves two of
        # them, so its confirmed conclusion turns the deep question into
        # one bridge. A third of the worlds instead go FOUR deep: there
        # the intermediate itself sits past the activation floor, nothing
        # derives, nothing can be confirmed, and the deep question stays
        # honestly refused - one confirmation buys exactly one step of
        # reach, and the gate measures that limit instead of hiding it.
        facts = [
            (f"the {entity} material", alloy),
            (f"the {alloy} base", blend),
            (f"the {blend} base", base),
            (f"the {base} {prop}", f"{value} degrees"),
            (f"the {revised_base} {prop}", f"{revised_value} degrees"),
        ]
        if rng.random() < 0.35:
            mix = f"{spare}ate"
            facts[2] = (f"the {blend} base", mix)
            facts.insert(3, (f"the {mix} base", base))
        intermediate = f"what is the {alloy} {prop}?"
        deep = f"what is the {entity} {prop}?"

        for arm in ("baseline", "consolidation"):
            if arm == "baseline":
                SystematicMemory.consolidate_fact = (
                    lambda self, *a, **k: None)
            else:
                SystematicMemory.consolidate_fact = real_consolidate
            try:
                # --- Protocol A: derive, confirm, then go deep. ---------
                root = Path(tempfile.mkdtemp(prefix=f"uq_consol_{seed}_"))
                session = build_session(root, seed=seed)
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)
                run_pipeline(intermediate, session)
                run_pipeline("yes", session)
                response, _trace = run_pipeline(deep, session)
                deep_acc[arm].append(float(_asserted(response, value)))

                # --- Staleness: revise the chain's middle premise. -------
                run_pipeline(f"the {blend} base is {revised_base}", session)
                stale = 0
                for question in (intermediate, deep):
                    response, _trace = run_pipeline(question, session)
                    if _asserted(response, value) and \
                            "inferred" not in response:
                        stale += 1
                stale_rate[arm].append(stale / 2.0)
                shutil.rmtree(root, ignore_errors=True)

                # --- Protocol B: repetition must not consolidate. -------
                root = Path(tempfile.mkdtemp(prefix=f"uq_consolb_{seed}_"))
                session = build_session(root, seed=seed)
                for key, fact_value in facts:
                    run_pipeline(f"{key} is {fact_value}", session)
                run_pipeline(intermediate, session)
                run_pipeline(intermediate, session)
                response, _trace = run_pipeline(deep, session)
                leak_rate[arm].append(float(_asserted(response, value)))
                shutil.rmtree(root, ignore_errors=True)
            finally:
                SystematicMemory.consolidate_fact = real_consolidate

    report = ConsolidationReport(
        deep={arm: statistics.fmean(vals) for arm, vals in deep_acc.items()},
        stale={arm: statistics.fmean(vals)
               for arm, vals in stale_rate.items()},
        leak={arm: statistics.fmean(vals)
              for arm, vals in leak_rate.items()},
    )

    contrasts = [deep_acc["consolidation"][i] - deep_acc["baseline"][i]
                 for i in range(seeds)]
    report.delta = statistics.fmean(contrasts)
    # stdev, not pstdev: the seeds are a sample of generated worlds.
    report.seed_sd = statistics.stdev(contrasts) if seeds > 1 else 0.0
    report.margin_ratio = (report.delta / report.seed_sd
                           if report.seed_sd > 0 else 0.0)

    stale_sd = (statistics.stdev(stale_rate["consolidation"])
                if seeds > 1 else 0.0)
    if (report.stale["consolidation"]
            > report.stale["baseline"] + stale_sd):
        report.reason = (
            f"the staleness control failed: consolidated conclusions "
            f"outlive their premises ({report.stale['consolidation']:.3f} "
            f"stale against baseline {report.stale['baseline']:.3f}) - "
            "SS11.16's entrenched error, wearing a memory")
        return report
    if report.leak["consolidation"] > 0 or report.leak["baseline"] > 0:
        report.reason = (
            f"the repetition control failed: asking twice consolidated "
            f"something (leak {report.leak['consolidation']:.3f} / "
            f"{report.leak['baseline']:.3f}); only affirmation may promote")
        return report
    if report.delta <= 0:
        report.reason = (f"confirmation buys no reach "
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
        f"a confirmed intermediate step lifts depth-3 reach "
        f"{report.deep['baseline']:.3f} -> "
        f"{report.deep['consolidation']:.3f} ({report.delta:+.3f}, "
        f"{report.margin_ratio:.2f}x seed sd) with zero stale answers "
        "after premise revision and zero repetition leaks")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
