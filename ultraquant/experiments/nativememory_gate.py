"""Two stores that remember the same way. The native memory gate.

Facts are the substrate every question form stands on, so parity
here is heavier than "the same value comes back". A store is
observed through what it SAYS about storing: whether a statement was
new, reinforced or a revision; what the replaced belief was, in
spoken form with its polarity ("was not steel", not "was steel");
what confidence a fact carries after being told twice; which
conclusions a revised premise took down with it; and which key
retrieval picks when several overlap.

Polarity is the sharpest of those. "the dome material is not steel"
after "the dome material is steel" is a CHANGE OF MIND, not a
reinforcement, because polarity is part of a fact's identity - and
that rule lives inside the store rather than above it, so a native
tier that put it above would pass every value check and be wrong
about what happened.

**What is deliberately not ported, and why.** The Python store can
put facts in the pageable shard library; the native store keeps them
in a map. That is a storage decision rather than a semantic one -
§11.86 already measured what the paging costs and established that
the price is the bucket cache rather than the lookup - so this gate
compares against the store's plain mode, which is the mode that
defines what a fact IS. A native tier that reproduced the paging
would be reproducing a deployment choice and calling it semantics.

Both tiers are driven through the same generated script: statements
that are new, statements repeated verbatim, statements that revise a
value, statements that flip polarity in both directions, derived
facts consolidated onto premises that are then revised out from
under them, confirmations, recalls of held and unheld keys,
retrievals over colliding keys, and the episode log read back.

**The criteria, written before the run.**

1. **Parity is total** - every record identical, script step for
   script step. One mismatch is a fail.
2. **Outcomes match, not just values** - new / reinforced / revised
   is compared explicitly, as is the spoken form of the replaced
   belief.
3. **Retractions match by SET and by order** - a revised premise
   must take down the same conclusions, named the same way.
4. **Retrieval order matches** - `find_facts` is compared as an
   ordered list, because §11.44's tie-break is a decision about
   which fact answers, not a convenience.

**It took five runs, and every failure was the native tier being
reasonable instead of being faithful.**

That is the finding, and it is worth more than the pass. None of the
five was a wrong VALUE. Each was a defensible implementation choice
that produced a different observable, and each would have survived a
gate that only checked what a fact recalls to:

1. **Consolidation went through the statement path.** Storing a
   derived fact by calling `remember_fact` is the obvious reuse - and
   it logs a revision episode the Python tier never logs, and keeps a
   reinforcement count the Python tier resets. A consolidation is not
   a statement. The record is written whole.
2. **Retraction order.** The Python tier walks a stack of revised
   keys and appends each casualty as it finds it; the port swept
   repeatedly until nothing moved and then sorted. Same facts
   dropped, different order - and the retracted list is spoken aloud
   when a belief changes, so the order is part of what the system
   says.
3. **Episode order.** `recall_episodes` returns most recent FIRST,
   because a history answer reads the newest change first. The port
   returned insertion order, which is the same information in the
   wrong sentence.
4. **The plural fold in retrieval.** The port folded "towers" onto
   "tower" before comparing tokens, which is what the router does -
   but the store's own retrieval compares RAW tokens, and the fold
   belongs above it. A native tier that folded here would find facts
   the Python store does not, which is the same kind of wrong as
   missing them.
5. **`confirm_fact` raised rather than set.** Being told "yes, that
   is correct" is direct testimony: it SETS confidence, and it can
   lower one. The port took a maximum, and skipped the reinforcement
   bump - which is exactly what §11.44's retrieval tie-break reads,
   so the mistake changed which fact answered a question two steps
   later. That is how it surfaced: not as a wrong confidence, but as
   a differently ordered `find_facts`.

| | |
|---|---:|
| script steps compared | 4,800 |
| records that differed | **0** |

Across twelve sessions: 2,502 statements, 461 recalls, 439
retrievals, 431 consolidations, 403 confirmations, 286 key
enumerations, 278 episode reads.

Run it::

    python -m ultraquant.experiments.nativememory_gate
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.memory.systematic import SystematicMemory

__all__ = ["MemoryParityReport", "run_gate", "probe_path"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_memory.exe"


def probe_path() -> Path:
    """Where the built store lives, so tests can skip without it."""
    return _PROBE


@dataclass
class MemoryParityReport:
    """Parity across a scripted session, and where it parted.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        steps: Script steps compared.
        mismatches: Steps whose records differed.
        kinds: Command -> how many steps it contributed.
        first_mismatch: The first disagreement, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    steps: int = 0
    mismatches: int = 0
    kinds: dict = field(default_factory=dict)
    first_mismatch: str = ""
    reason: str = ""


_ENTITIES = ["tower", "bridge", "keep", "mill", "spire", "dome",
             "north tower", "south bridge", "old keep"]
_ATTRIBUTES = ["height", "material", "weight", "length"]
_VALUES = ["300 meters", "iron", "steel", "oak", "450 meters",
           "2 kilometers", "bronze", "120 meters"]


def _script(rng: random.Random, steps: int):
    """The session both stores are driven through."""
    plan: list[tuple[str, tuple]] = []
    keys: list[str] = []
    for _ in range(steps):
        pick = rng.random()
        if pick < 0.42 or not keys:
            key = f"{rng.choice(_ENTITIES)} {rng.choice(_ATTRIBUTES)}"
            if key not in keys:
                keys.append(key)
            plan.append(("remember", (key, rng.choice(_VALUES),
                                      round(rng.uniform(0.3, 0.9), 2),
                                      rng.random() < 0.25)))
        elif pick < 0.52:
            # Repeat something already said, verbatim or with its
            # polarity flipped - the difference between a
            # reinforcement and a change of mind.
            key = rng.choice(keys)
            plan.append(("remember", (key, rng.choice(_VALUES),
                                      round(rng.uniform(0.3, 0.9), 2),
                                      rng.random() < 0.5)))
        elif pick < 0.62:
            plan.append(("recall", (rng.choice(keys + ["nobody holds this"]),)))
        elif pick < 0.72:
            plan.append(("find", (rng.randint(1, 5),
                                  rng.choice([
                                      "what is the tower height?",
                                      "how tall is the north tower?",
                                      "the keep material",
                                      "what are towers?",
                                      "bridge",
                                  ]))))
        elif pick < 0.80:
            plan.append(("consolidate", (
                f"derived {rng.randint(1, 6)}", rng.choice(_VALUES),
                round(rng.uniform(0.3, 0.9), 2),
                rng.sample(keys, min(len(keys), rng.choice((1, 2)))))))
        elif pick < 0.88:
            plan.append(("confirm", (rng.choice(keys),
                                     round(rng.uniform(0.5, 1.0), 2))))
        elif pick < 0.94:
            plan.append(("keys", ()))
        else:
            plan.append(("episodes", ()))
    return plan


def _python_run(plan) -> list[str]:
    """What the tier that DEFINES the semantics says, step by step."""
    memory = SystematicMemory()
    out: list[str] = []
    for command, args in plan:
        if command == "remember":
            key, value, confidence, negated = args
            result = memory.remember_fact(key, value, confidence, negated)
            out.append("{}|{}|{}".format(
                result["outcome"], result.get("was", ""),
                ";".join(result.get("retracted", []))))
        elif command == "consolidate":
            key, value, confidence, premises = args
            memory.consolidate_fact(key, value, confidence,
                                    [(p, "") for p in premises])
            out.append("consolidated")
        elif command == "confirm":
            key, confidence = args
            out.append("ok|{}".format(
                1 if memory.confirm_fact(key, confidence) else 0))
        elif command == "recall":
            record = memory.recall_fact(args[0])
            if record is None:
                out.append("absent")
            else:
                out.append("fact|{}|{:.2f}|{}|{}".format(
                    record["value"], record["confidence"],
                    1 if record.get("negated") else 0,
                    record.get("reinforcements", 0)))
        elif command == "find":
            top_k, text = args
            out.append("found|" + ";".join(memory.find_facts(text, top_k)))
        elif command == "keys":
            out.append("keys|" + ";".join(memory.fact_keys()))
        else:
            spoken = []
            for episode in memory.recall_episodes(limit=10000):
                content = episode.get("content", {})
                spoken.append(":".join([
                    episode.get("kind", ""), content.get("key", ""),
                    content.get("old_value", ""),
                    content.get("new_value", ""),
                    content.get("because", "")]))
            out.append("ep|" + ";".join(spoken))
    return out


def _native_lines(plan) -> list[str]:
    """The same script, spoken to the native store."""
    lines: list[str] = []
    for command, args in plan:
        if command == "remember":
            key, value, confidence, negated = args
            lines.append(f"remember {confidence} {1 if negated else 0} "
                         f"{key}|{value}")
        elif command == "consolidate":
            key, value, confidence, premises = args
            lines.append(f"consolidate {confidence} {key}|{value}|"
                         + ";".join(premises))
        elif command == "confirm":
            key, confidence = args
            lines.append(f"confirm {confidence} {key}")
        elif command == "recall":
            lines.append(f"recall {args[0]}")
        elif command == "find":
            top_k, text = args
            lines.append(f"find {top_k} {text}")
        elif command == "keys":
            lines.append("keys")
        else:
            lines.append("episodes")
    return lines


def run_gate(steps: int = 400, sessions: int = 12,
             seed: int = 0) -> MemoryParityReport:
    """Drive both stores through the same sessions and compare."""
    if not _PROBE.exists():
        return MemoryParityReport(
            passes=False,
            reason="FAIL - the native store is not built; run "
                   "native/uq/build.ps1")
    compared = 0
    mismatches = 0
    kinds: dict = {}
    first = ""
    for session in range(sessions):
        rng = random.Random(31000 + seed * 100 + session)
        plan = _script(rng, steps)
        for command, _args in plan:
            kinds[command] = kinds.get(command, 0) + 1
        want = _python_run(plan)
        got = subprocess.run(
            [str(_PROBE)], input="\n".join(_native_lines(plan)) + "\n",
            capture_output=True, text=True, check=True)
        have = got.stdout.splitlines()
        for index, expected in enumerate(want):
            compared += 1
            actual = have[index] if index < len(have) else "(no answer)"
            if actual == expected:
                continue
            mismatches += 1
            if not first:
                command, args = plan[index]
                first = (f"session {session}, step {index} [{command}] "
                         f"{args!r}\n    python: {expected}\n"
                         f"    native: {actual}")
    passes = mismatches == 0
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {compared} script steps "
        f"compared across {sessions} sessions, {mismatches} records "
        f"differed" + (f"\nfirst mismatch:\n{first}" if first else ""))
    return MemoryParityReport(passes=passes, steps=compared,
                              mismatches=mismatches, kinds=kinds,
                              first_mismatch=first, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"script steps compared: {report.steps}")
    print(f"records that differed: {report.mismatches}")
    print()
    for command, count in sorted(report.kinds.items()):
        print(f"  {command:<14} {count:5d} steps")
    print()
    print(report.reason)
