"""The question that needed no memory. The recall-skip gate.

§11.85's price line reported that literal arithmetic cost 38.9 ms
and registered the reason as an opportunity. This gate cashes it,
after finishing the diagnosis the capstone started. At 4,000 facts,
with the store flushed as it is in ordinary use::

    Recall   91.072 ms   bucket loads  25   facts found 0
    Recall   55.667 ms   bucket loads  25   facts found 0
    Recall   52.862 ms   bucket loads  25   facts found 0

Twenty-five bucket loads, every one paged from disk, to find nothing
- for the question "what is 12 + 3 * 4?", which names no belief at
all. The candidate ladder builds keys out of whatever words the
question contains, and an expression contains none, so every key it
forms is a guaranteed miss. The pageable library is working exactly
as designed; it is simply being asked a question no store can
answer.

So Recall skips its candidate ladder when the input is nothing but
numbers, operators and parentheses. The predicate is §11.76's
literal reader with NO memory behind it, which is the same claim
said in code: an expression that mentions no words cannot be about
anything stored. Episodes and the context window are left alone -
they cost 0.002 ms and 0.000 ms, and a saving that small is not
worth a behaviour change.

**Parity first, clock second** - §11.45's discipline, which is the
only reason a speed change is allowed to be interesting. A faster
system that answers differently has not been optimised; it has been
broken, and the difference is measured rather than assumed.

Both arms run the live pipeline over identical stores; the baseline
arm turns `_ARITHMETIC_SKIPS_RECALL` off. The corpus covers what the
skip must not touch as well as what it should: expressions,
percentages, powers, roots, rounding requests, belief arithmetic,
chains, polar questions, comparisons, superlatives, aggregates,
history, statements, and chat.

**The criteria, written before the run.**

1. **Parity** - every response, over the whole corpus, byte-identical
   between arms. One difference is a fail, and no clock reading
   redeems it.
2. **The facts do not move** - for every non-expression input, the
   facts Recall finds are identical between arms. Response parity
   could hide a difference that later inputs would expose; this
   checks the stage's own output.
3. **Gain (the clock)** - mean latency on expression questions must
   be LOWER in the treatment arm by more than the seed-to-seed sd of
   that contrast.
4. **No regression elsewhere** - mean latency on belief-backed
   questions must not be higher in the treatment arm by more than
   the seed-to-seed sd.

**It was run twice, and FAILED - on criterion 4, for a real
reason. The mechanism is switched off and kept here.**

Run one passed parity perfectly and posted a 61x clock, and failed
criterion 4: belief questions went from 100.62 ms to 149.81 ms. Run
one's corpus always asked the expressions FIRST, which hands the
ladder arm a prefetch it would not have in ordinary use, so the
corpus was shuffled per seed. Run two, unchanged in the mechanism:

| questions | ladder | skip | |
|---|---:|---:|---|
| expressions | 22.71 ms | **1.13 ms** | 20.1x |
| beliefs | 119.00 ms | 133.11 ms | +14.11 at 11.45 sd |

**Zero response differences and zero recalled-fact differences** -
parity is perfect, and the clock is real. Criterion 4 failed again,
and the decisive measurement is one the criteria never asked for,
which is why it was taken:

    ladder   total corpus    1497.5 ms (sd 21.8)
    skip     total corpus    1530.3 ms (sd 33.9)
    saved per corpus: -32.8 ms (sd 50.4) over 24 questions

**The skip saves nothing.** It makes expression questions 20-60x
faster and the whole session no faster at all, because the twenty-
five bucket loads it removes were warming the bucket cache for every
question that followed. The work was not wasted; it was prefetch
wearing the costume of waste.

That is the finding, and it is worth more than the optimisation
would have been: **the price of a question at this density is the
bucket cache, not the candidate ladder.** A question that pages
twenty-five buckets pays for them and leaves them warm; a question
that pages none pays nothing and leaves the next question to pay.
Any real win has to be at the resident-set level - a larger or
smarter cache, or an eviction policy that knows what a session keeps
asking about - and per-question cleverness above it just moves the
same milliseconds around.

Kept switched off rather than deleted, in §11.60's tradition, so the
result stays reproducible: the treatment arm turns it on.

Run it::

    python -m ultraquant.experiments.recallskip_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RecallSkipReport", "run_gate"]

_ATTRIBUTES = ["height", "length", "span"]


@dataclass
class RecallSkipReport:
    """Parity, the facts, and the clock.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        differences: Responses that differ between arms.
        fact_differences: Non-expression inputs whose recalled facts
            differ between arms.
        expression_ms: Arm -> mean ms on expression questions.
        belief_ms: Arm -> mean ms on belief-backed questions.
        speedup: Baseline expression ms divided by treatment.
        clock_delta: Baseline minus treatment, expression ms.
        clock_sd: Seed sd of that contrast.
        belief_delta: Treatment minus baseline, belief ms.
        belief_sd: Seed sd of that contrast.
        reason: Plain-language verdict.
    """

    passes: bool
    differences: int = 0
    fact_differences: int = 0
    expression_ms: dict = field(default_factory=dict)
    belief_ms: dict = field(default_factory=dict)
    speedup: float = 0.0
    clock_delta: float = 0.0
    clock_sd: float = 0.0
    belief_delta: float = 0.0
    belief_sd: float = 0.0
    reason: str = ""


def _corpus(rng: random.Random, entities: list[str]):
    """(question, kind) over everything the skip must not touch."""
    a, b, c = entities[0], entities[1], entities[2]
    expressions = [
        (f"what is {rng.randint(2, 90)} + {rng.randint(2, 40)} * "
         f"{rng.randint(2, 9)}?", "expression"),
        (f"what is ({rng.randint(2, 40)} + {rng.randint(2, 40)}) * "
         f"{rng.randint(2, 9)}?", "expression"),
        (f"what is {rng.randrange(5, 95, 5)}% of "
         f"{rng.randrange(20, 900, 20)}?", "expression"),
        (f"what is {rng.randint(2, 9)} ^ {rng.randint(2, 5)}?",
         "expression"),
        (f"what is the square root of {rng.randint(2, 200)}?",
         "expression"),
        (f"what is {rng.randint(10, 400)} / {rng.choice((3, 7, 9))} "
         f"to 2 decimal places?", "expression"),
        (f"what is {rng.randint(2, 90)} - {rng.randint(2, 90)}?",
         "expression"),
        (f"{rng.randint(2, 40)} * {rng.randint(2, 12)}", "expression"),
    ]
    beliefs = [
        (f"what is the {a} height?", "belief"),
        (f"what is the {b} length?", "belief"),
        (f"what is the {a} height * 3?", "belief"),
        (f"what is the {a} height + the {b} height?", "belief"),
        (f"is the {a} height greater than 100 meters?", "belief"),
        (f"is the {a} height 300 meters?", "belief"),
        (f"which height is the tallest?", "belief"),
        (f"what is the total height?", "belief"),
        (f"what was the {c} material?", "belief"),
        (f"why is the {a} height {rng.randint(2, 900)} meters?",
         "belief"),
    ]
    others = [
        ("the newest keep material is oak", "other"),
        ("hello there", "other"),
        (f"what is the {c} architect hometown?", "other"),
        ("is 2 + 2 = 4?", "other"),
        ("is 2 + 2 equal to 5?", "other"),
        ("what is 300 meters + 2 kilometers?", "other"),
    ]
    corpus = expressions + beliefs + others
    # Order is not semantics here - these are independent questions -
    # and always asking the expressions FIRST hands the ladder arm a
    # prefetch it would not have in ordinary use: twenty-five bucket
    # loads per expression warm the cache for the belief questions
    # that follow. Shuffled, each arm meets the store in the same
    # state on average.
    rng.shuffle(corpus)
    return corpus


def _build(root: Path, seed: int, entities: list[str]):
    from ultraquant.interpreter.thoughts import (build_session,
                                                 run_pipeline)

    rng = random.Random(seed)
    session = build_session(root, seed=seed)
    memory = session.memory
    for index in range(1200):
        name = f"holder{index}"
        for attr in _ATTRIBUTES:
            memory.remember_fact(f"{name} {attr}",
                                 f"{rng.randrange(25, 900, 25)} meters",
                                 confidence=0.6)
    for entity in entities:
        for attr in _ATTRIBUTES:
            memory.remember_fact(f"{entity} {attr}",
                                 f"{rng.randrange(25, 900, 25)} meters",
                                 confidence=0.6)
    memory.remember_fact(f"{entities[2]} material", "iron",
                         confidence=0.6)
    # Flush, so the store is on disk the way ordinary use leaves it -
    # which is the state the whole cost lives in.
    run_pipeline("hello", session)
    return session


def run_gate(seeds: int = 6) -> RecallSkipReport:
    from ultraquant.interpreter import thoughts
    from ultraquant.interpreter.thoughts import run_pipeline

    arms = {"ladder": False, "skip": True}
    responses: dict = {}
    facts_seen: dict = {}
    expression_seed: dict = {arm: [] for arm in arms}
    belief_seed: dict = {arm: [] for arm in arms}

    for arm, flag in arms.items():
        for seed in range(seeds):
            rng = random.Random(21000 + seed)
            entities = [f"world{seed} tower", f"world{seed} bridge",
                        f"world{seed} keep"]
            corpus = _corpus(rng, entities)
            root = Path(tempfile.mkdtemp(prefix="uq_skipgate_"))
            old = thoughts._ARITHMETIC_SKIPS_RECALL
            thoughts._ARITHMETIC_SKIPS_RECALL = flag
            try:
                session = _build(root, seed, entities)
                expression_ms = []
                belief_ms = []
                for question, kind in corpus:
                    start = time.perf_counter()
                    response, trace = run_pipeline(question, session)
                    spent = (time.perf_counter() - start) * 1000.0
                    responses.setdefault((seed, question),
                                         {})[arm] = response
                    if kind != "expression":
                        found = tuple(
                            sorted(entry.get("facts", []) or []
                                   for entry in trace
                                   if entry.get("thought") == "Recall")
                            [0]) if any(
                                entry.get("thought") == "Recall"
                                for entry in trace) else ()
                        facts_seen.setdefault((seed, question),
                                              {})[arm] = found
                    if kind == "expression":
                        expression_ms.append(spent)
                    elif kind == "belief":
                        belief_ms.append(spent)
                expression_seed[arm].append(
                    statistics.mean(expression_ms))
                belief_seed[arm].append(statistics.mean(belief_ms))
            finally:
                thoughts._ARITHMETIC_SKIPS_RECALL = old
                shutil.rmtree(root, ignore_errors=True)

    differences = sum(1 for by_arm in responses.values()
                      if len(set(by_arm.values())) > 1)
    fact_differences = sum(1 for by_arm in facts_seen.values()
                           if len(set(by_arm.values())) > 1)

    clock = [b - t for b, t in zip(expression_seed["ladder"],
                                   expression_seed["skip"])]
    clock_delta = statistics.mean(clock)
    clock_sd = statistics.stdev(clock)
    belief = [t - b for b, t in zip(belief_seed["ladder"],
                                    belief_seed["skip"])]
    belief_delta = statistics.mean(belief)
    belief_sd = statistics.stdev(belief)

    expression_ms = {arm: statistics.mean(values)
                     for arm, values in expression_seed.items()}
    belief_ms = {arm: statistics.mean(values)
                 for arm, values in belief_seed.items()}
    speedup = (expression_ms["ladder"] / expression_ms["skip"]
               if expression_ms["skip"] else float("inf"))

    passes = (differences == 0 and fact_differences == 0
              and clock_sd > 0.0 and clock_delta > clock_sd
              and belief_delta <= belief_sd)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {differences} response "
        f"differences, {fact_differences} recalled-fact differences; "
        f"expressions {expression_ms['ladder']:.2f} -> "
        f"{expression_ms['skip']:.2f} ms ({speedup:.1f}x, "
        f"{clock_delta:+.2f} ms at {clock_sd:.2f} sd); beliefs "
        f"{belief_ms['ladder']:.2f} -> {belief_ms['skip']:.2f} ms "
        f"({belief_delta:+.2f} ms at {belief_sd:.2f} sd)")
    return RecallSkipReport(
        passes=passes, differences=differences,
        fact_differences=fact_differences,
        expression_ms=expression_ms, belief_ms=belief_ms,
        speedup=speedup, clock_delta=clock_delta, clock_sd=clock_sd,
        belief_delta=belief_delta, belief_sd=belief_sd, reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"response differences: {report.differences}")
    print(f"recalled-fact differences: {report.fact_differences}")
    print()
    print("questions      ladder     skip     speedup")
    print(f"expressions  {report.expression_ms['ladder']:8.2f} "
          f"{report.expression_ms['skip']:8.2f}   "
          f"{report.speedup:6.1f}x")
    print(f"beliefs      {report.belief_ms['ladder']:8.2f} "
          f"{report.belief_ms['skip']:8.2f}")
    print()
    print(f"clock delta {report.clock_delta:+.2f} ms at "
          f"{report.clock_sd:.2f} sd")
    print(f"belief delta {report.belief_delta:+.2f} ms at "
          f"{report.belief_sd:.2f} sd")
    print(report.reason)
