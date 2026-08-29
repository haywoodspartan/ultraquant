"""The engine in the pipeline. The wiring gate.

§11.113 and §11.114 built a retrieval engine and made it a genuine
superset of what the pipeline retrieves. It was still not used: the
pipeline called the five mechanisms directly and the engine was a
measurement.

This is the replacement, and it touches the most heavily gated code
in the system - the question pipeline, with a hundred units of
measured behaviour behind it. So the bar is not "better". It is
**identical answers, fewer reads**, and the first half is the one
that matters. A refactor that changes an answer is a regression
whatever it saves.

**Where the saving is, precisely.** `Recall` formed candidate keys
from the question - lead-stripped, article-stripped, token n-grams
longest first, about twenty-five of them - and looked up **every
one**, then used `facts[0]` and nothing else. Both consumers of that
list read `facts[0]`; nothing reads the rest, which was checked
rather than assumed. The ladder is ordered most-specific-first, so
the first hit IS `facts[0]` and stopping there cannot change it.
§11.86 measured the same waste from the other side and switched off
the wrong half of it.

**The flag is the arm.** `_RETRIEVAL_ENGINE = False` restores the
exhaustive ladder byte for byte, which is what makes this a
measurement rather than an assertion: both arms run in the same
process over the same worlds and are compared directly.

**The criteria, written before the run.**

1. **Every answer is byte-identical** - across every question in
   every world. One character of difference is a fail, and no saving
   redeems it.
2. **The store is read less** - mean `recall_fact` calls per question
   must fall, by more than the seed-to-seed sd. If it does not, the
   engine is a tidier way to do the same work and should be recorded
   as one.
3. **The stores end identical** - same keys, values, polarity and
   confidences after every conversation. A pipeline can say the right
   thing and remember the wrong one.
4. **The saving is where it was claimed** - the reduction must come
   from questions that HIT the ladder. A question with no hit reads
   the whole ladder either way, so its cost must be unchanged; if the
   saving appears there instead, it is coming from somewhere this
   unit has not explained.
5. **The traces agree on intent** - every turn classified the same
   way in both arms. A right answer reached through a different door
   diverges later, not now.

**PASSED on all five, and criterion 4 caught a defect the other four
would have shipped.**

| question kind | ladder | engine | saved |
|---|---:|---:|---:|
| **hit** | 16.06 | **4.00** | **+12.06** |
| statement | 16.39 | 14.59 | +1.80 |
| chain | 28.95 | 28.95 | +0.00 |
| miss | 21.62 | 21.62 | +0.00 |
| chat | 2.58 | 2.58 | +0.00 |
| **all** | 16.22 | **12.71** | **+3.51** |

**284 turns, 0 answers differing, 0 intents differing, 0 stores
differing.** Store reads fall **21.6%** per turn at 4.3x the seed
noise, and a question that hits the ladder costs **75% less**.

**Run one saved on hits and LOST on everything else.** Misses cost
1.14 reads more and chains 3.18 more, because `Recall` asked the
engine for its whole cascade while consuming only the exact route -
and the pipeline's own fallback runs the lexical and reach routes
again a moment later. **An engine that eagerly does work its caller
will redo is not an economy**, and only criterion 4 could see it:
answers were already byte-identical, the overall saving was already
positive, and the other three criteria were already green.

The fix is `routes=("exact",)` - the caller names what it consumes.
With it, misses and chains cost exactly what they did before, which
is what makes the headline honest: the saving is not an average
hiding a regression.

**What is not claimed.** The pipeline still calls `find_facts`, the
phrase probes and the suggester directly in its fallback chain; what
has been replaced is `Recall`'s ladder. The engine is one caller's
dependency now rather than none, and the rest of the unification is
unmeasured.

Run it::

    python -m ultraquant.experiments.wiring_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["WiringReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "keep", "mill", "spire", "granary",
             "old tower", "north bridge"]
_ATTRS = ["height", "length", "material", "colour", "weight"]
_MATERIALS = ["steel", "oak", "bronze", "granite"]
_ABSENT = ["obelisk", "aqueduct", "lighthouse", "campanile"]


def _conversation(seed: int, turns: int = 24) -> list:
    """Statements and questions, with and without ladder hits."""
    rng = random.Random(seed)
    plan: list = []
    said: list = []
    for _ in range(turns):
        pick = rng.random()
        if pick < 0.34 or len(said) < 3:
            entity = rng.choice(_ENTITIES)
            attr = rng.choice(_ATTRS)
            key = f"{entity} {attr}"
            value = (f"{rng.randrange(25, 900, 25)} meters"
                     if attr in ("height", "length")
                     else rng.choice(_MATERIALS))
            said.append(key)
            plan.append((f"the {key} is {value}", "statement"))
        elif pick < 0.62:
            plan.append((f"what is the {rng.choice(said)}?", "hit"))
        elif pick < 0.78:
            plan.append((f"what is the {rng.choice(_ABSENT)} "
                         f"{rng.choice(_ATTRS)}?", "miss"))
        elif pick < 0.88:
            entity = rng.choice(_ENTITIES)
            material = rng.choice(_MATERIALS)
            said.append(f"{entity} material")
            plan.append((f"the {entity} material is {material}", "statement"))
            plan.append((f"{material} hardness is high", "statement"))
            plan.append((f"what is the {entity} hardness?", "chain"))
        else:
            plan.append((rng.choice(["hello there", "thanks",
                                     "good morning"]), "chat"))
    return plan


def _walk(plan: list, engine_on: bool) -> tuple:
    """One conversation through one arm, counting store reads.

    Returns ``(answers, intents, reads_by_kind, final store)``.
    `recall_fact` is wrapped rather than sampled, so the count is
    every lookup the pipeline actually performed.
    """
    from ultraquant.interpreter import thoughts as T

    root = Path(tempfile.mkdtemp(prefix="uq_wiring_"))
    previous = T._RETRIEVAL_ENGINE
    T._RETRIEVAL_ENGINE = engine_on
    try:
        session = T.build_session(root, seed=0)
        memory = session.memory
        original = memory.recall_fact
        counter = {"n": 0}

        def counted(key):
            counter["n"] += 1
            return original(key)

        memory.recall_fact = counted

        answers, intents = [], []
        reads: dict = {}
        for text, kind in plan:
            before = counter["n"]
            response, trace = T.run_pipeline(text, session)
            answers.append(response)
            intent = ""
            for entry in trace:
                if entry.get("thought") == "Perceive":
                    intent = str(entry.get("intent", ""))
                    break
            intents.append(intent)
            reads.setdefault(kind, []).append(counter["n"] - before)

        memory.recall_fact = original
        store = {}
        for key in memory.fact_keys():
            record = memory.recall_fact(key)
            store[key] = (str(record.get("value", "")),
                          bool(record.get("negated")),
                          round(float(record.get("confidence", 0.0)), 2))
        return answers, intents, reads, store
    finally:
        T._RETRIEVAL_ENGINE = previous
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class WiringReport:
    """Whether the replacement changed anything but the cost.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        worlds: Conversations compared.
        turns: Turns compared.
        answer_differences: Turns whose reply differed at any byte.
        intent_differences: Turns classified differently.
        store_differences: Conversations ending with unequal stores.
        reads_before: Mean recall_fact calls per turn, ladder arm.
        reads_after: The same through the engine.
        by_kind: Question kind -> (before, after) mean reads.
        saving_sd: Seed sd of the per-world saving.
        first_difference: The first answer that differed, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    worlds: int = 0
    turns: int = 0
    answer_differences: int = 0
    intent_differences: int = 0
    store_differences: int = 0
    reads_before: float = 0.0
    reads_after: float = 0.0
    by_kind: dict = field(default_factory=dict)
    saving_sd: float = 0.0
    first_difference: str = ""
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> WiringReport:
    """Both arms, same worlds, same process. Compare everything."""
    answer_differences = intent_differences = store_differences = 0
    compared = 0
    before_all: list = []
    after_all: list = []
    savings: list = []
    kinds: dict = {}
    first = ""

    for seed in range(worlds):
        plan = _conversation(seed, turns)
        old_a, old_i, old_r, old_s = _walk(plan, engine_on=False)
        new_a, new_i, new_r, new_s = _walk(plan, engine_on=True)

        for index, ((text, kind), a, b) in enumerate(
                zip(plan, old_a, new_a)):
            compared += 1
            if a != b:
                answer_differences += 1
                if not first:
                    first = (f"world {seed}, turn {index} [{kind}] "
                             f"{text!r}\n    ladder: {a}\n    engine: {b}")
        intent_differences += sum(1 for a, b in zip(old_i, new_i) if a != b)
        if old_s != new_s:
            store_differences += 1

        old_flat = [n for values in old_r.values() for n in values]
        new_flat = [n for values in new_r.values() for n in values]
        before_all.extend(old_flat)
        after_all.extend(new_flat)
        savings.append(statistics.fmean(old_flat)
                       - statistics.fmean(new_flat))
        for kind in set(old_r) | set(new_r):
            entry = kinds.setdefault(kind, [[], []])
            entry[0].extend(old_r.get(kind, []))
            entry[1].extend(new_r.get(kind, []))

    reads_before = statistics.fmean(before_all) if before_all else 0.0
    reads_after = statistics.fmean(after_all) if after_all else 0.0
    saving_sd = statistics.pstdev(savings) if len(savings) > 1 else 0.0
    by_kind = {kind: (statistics.fmean(a) if a else 0.0,
                      statistics.fmean(b) if b else 0.0)
               for kind, (a, b) in kinds.items()}

    identical = answer_differences == 0
    cheaper = (reads_before - reads_after) > saving_sd
    # Criterion 4: a question with no ladder hit reads the whole
    # ladder either way, so its cost must not move.
    miss_before, miss_after = by_kind.get("miss", (0.0, 0.0))
    honest_saving = abs(miss_before - miss_after) < 0.5
    passes = (identical and cheaper and intent_differences == 0
              and store_differences == 0 and honest_saving)

    report = WiringReport(
        passes=passes, worlds=worlds, turns=compared,
        answer_differences=answer_differences,
        intent_differences=intent_differences,
        store_differences=store_differences, reads_before=reads_before,
        reads_after=reads_after, by_kind=by_kind, saving_sd=saving_sd,
        first_difference=first)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {compared} turns across "
        f"{worlds} worlds; {answer_differences} answers differed "
        + ("(byte-identical)" if identical
           else "- THE REFACTOR CHANGED AN ANSWER")
        + f"; store reads per turn {reads_before:.2f} against "
        f"{reads_after:.2f}, a saving of "
        f"{reads_before - reads_after:+.2f} at seed sd {saving_sd:.2f} "
        + ("(cheaper)" if cheaper else "- NOT BEYOND THE NOISE")
        + f"; misses cost {miss_before:.2f} against {miss_after:.2f} "
        + ("(unchanged, as claimed)" if honest_saving
           else "- THE SAVING IS NOT WHERE IT WAS CLAIMED")
        + f"; {intent_differences} intent differences, "
        f"{store_differences} stores differing"
        + (f"\nfirst:\n{first}" if first else ""))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"turns compared:        {result.turns}")
    print(f"answers differing:     {result.answer_differences}")
    print(f"intents differing:     {result.intent_differences}")
    print(f"stores differing:      {result.store_differences}")
    print()
    print(f"{'question kind':<14}{'ladder':>9}{'engine':>9}{'saved':>9}")
    for kind in sorted(result.by_kind):
        a, b = result.by_kind[kind]
        print(f"{kind:<14}{a:>9.2f}{b:>9.2f}{a - b:>+9.2f}")
    print(f"{'ALL':<14}{result.reads_before:>9.2f}"
          f"{result.reads_after:>9.2f}"
          f"{result.reads_before - result.reads_after:>+9.2f}")
    print()
    print(result.reason)
