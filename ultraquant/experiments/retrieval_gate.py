"""One call, and the bill. The retrieval engine gate.

`reason/retrieval.py` composes five mechanisms that already existed
separately - index ranking, phrase reach, the semantic suggester,
shard prediction and the byte budget - into one call that answers
*"what did this question actually cost?"*

**A unification's real risk is not that it works badly. It is that it
quietly drops something.** Five callers each had their own way of
finding facts; a sixth that finds fewer is a regression wearing a
tidier interface. Criterion 1 exists for exactly that and is the one
worth failing on.

**The economic claim is criterion 2**, and it is the thesis in one
number: a question the index answers must cost strictly less than one
that needs the semantic route. A dense model has no way to spend less
on an easy question. If this engine also spends the same on both, the
composition has bought nothing but tidiness.

**The criteria, written before the run.**

1. **The routes lose nothing, and the policy loses nothing that
   matters** - two things, which the first version ran together and
   got wrong twice over.

   *1a, the routes*: run exhaustively, the engine's keys must include
   everything the pipeline's own retrieval finds - the
   `_candidate_keys` ladder looked up directly AND `find_facts` at
   the same width. The first version compared against `find_facts`
   alone and passed **while the engine was missing the exact-key
   route entirely**, because the pipeline does not retrieve through
   the index first. A criterion measured against the wrong baseline
   passes for the wrong reason.

   *1b, the policy*: run normally, the engine must still find every
   key that COVERS the question. Stopping early drops keys by design
   - that is the saving - so demanding a superset there would forbid
   the economy the engine exists for. What it may never drop is an
   answer.
2. **Cost scales with the question** - easy questions must examine
   strictly fewer keys, on average, than hard ones. Equal cost means
   the routes are not stopping early and the claim is empty.
3. **The expensive route is not paid when it is not needed** - the
   semantic route must run zero times on questions the earlier routes
   covered. One call is a fail: that is the whole point of ordering
   them.
4. **Coverage is reported, never decided** - `covered` must be False
   whenever no retrieved key covers the question's informative
   tokens, and True only when one does. This layer has no refusal
   machinery and must not appear to.
5. **Absence costs nothing** - with no suggester, every other route
   must return exactly what it returns with one. The optional tier
   stays optional.

**Run one PASSED against the wrong baseline, and that is the most
useful thing here.**

Criterion 1 originally asked whether the composition lost anything
*against `find_facts`*. It did not, and the gate passed - **while the
engine was missing the exact-key route entirely.** The pipeline does
not retrieve through the index first: `Recall` forms candidate keys
from the question's own text and looks them up directly, finding keys
no index query returns. An engine without that route could never have
replaced the code it was built to unify, and a criterion measured
against the wrong baseline certified it anyway.

Strengthened to the pipeline's own retrieval, it failed at once - 12
keys lost across 12 worlds. Adding the exact route fixed the
completeness AND halved the cost of an easy question, because a dict
lookup that covers the question ends the search immediately.

**A second thing had to be separated before it could pass honestly.**
"The engine must find everything the pipeline finds" and "the engine
may stop early" are contradictory as stated - stopping early drops
keys BY DESIGN, and that is the saving. So the criterion is now two:
the **routes** must lose nothing when run exhaustively, and the
**policy** must never drop a key that COVERS the question. Conflating
them is what let the missing route hide.

**Run two, PASSED on all five:**

| | |
|---|---:|
| keys lost by the routes (exhaustive) | **0** |
| covering keys dropped by stopping early | **0** |
| examined, index-answerable questions | **1.00** |
| examined, questions needing more | **2.00** |
| semantic calls wasted on covered questions | **0** of 24 |
| coverage disagreements | 0 |
| routes changed by the suggester's absence | 0 |

**Cost scales with the question**, 1.00 against 2.00 - and on a
four-fact world, so directional rather than a headline. **The
expensive route was never paid when it was not needed**, which is the
entire reason for ordering them and the kind of thing that is easy to
believe and easy to get wrong.

The engine is now a genuine superset of what the pipeline retrieves,
which is the precondition for replacing it. **That replacement has
not been done**: the pipeline still calls the five mechanisms
directly, and swapping it is a change to the most heavily gated code
in the system.

Run it::

    python -m ultraquant.experiments.retrieval_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.reason.retrieval import (DEFAULT_WIDTH, RetrievalEngine,
                                         _fold)

__all__ = ["RetrievalReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "keep", "mill", "spire", "granary"]
_ATTRS = ["height", "length", "material", "colour", "weight"]
_MATERIALS = ["steel", "oak", "bronze", "granite"]
_ABSENT = ["obelisk", "aqueduct", "lighthouse"]


class _Counting:
    """A suggester that answers nothing and counts being asked.

    Criterion 3 is about whether the expensive route is REACHED, not
    about what it would have said, so the cheapest honest stand-in is
    one that records the call and declines.
    """

    def __init__(self) -> None:
        self.calls = 0

    def suggest(self, question, memory):
        self.calls += 1
        return None


def _world(seed: int) -> tuple:
    """Facts, and questions of three known difficulties."""
    rng = random.Random(seed)
    entity, other = rng.sample(_ENTITIES, 2)
    material = rng.choice(_MATERIALS)
    absent = rng.choice(_ABSENT)
    value = rng.randrange(50, 950)
    facts = [
        (f"the {entity} height", f"{value} meters"),
        (f"the {entity} material", material),
        (f"{material} hardness", "high"),
        (f"the {other} length", f"{rng.randrange(2, 40)} kilometers"),
    ]
    easy = [f"what is the {entity} height?", f"what is the {other} length?"]
    hard = [f"what is the {entity} hardness?",
            f"what is the {absent} height?"]
    return facts, easy, hard


@dataclass
class RetrievalReport:
    """What the composition kept, and what it charged.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        worlds: Worlds measured.
        lost: Keys the pipeline finds that the EXHAUSTIVE engine
            does not - completeness of the routes.
        dropped: COVERING keys the stopping policy discarded.
        easy_examined: Mean keys examined on index-answerable questions.
        hard_examined: Mean keys examined on questions needing more.
        semantic_calls_when_covered: Times the expensive route ran on a
            question already covered. Must be zero.
        semantic_calls_total: Times it ran at all.
        coverage_errors: Questions where `covered` disagreed with the
            §11.29 rule applied directly.
        absence_differences: Routes that behaved differently with no
            suggester present.
        reason: Plain-language verdict.
    """

    passes: bool
    worlds: int = 0
    lost: int = 0
    dropped: int = 0
    easy_examined: float = 0.0
    hard_examined: float = 0.0
    semantic_calls_when_covered: int = 0
    semantic_calls_total: int = 0
    coverage_errors: int = 0
    absence_differences: int = 0
    reason: str = ""


def run_gate(worlds: int = 12, width: int = DEFAULT_WIDTH) -> RetrievalReport:
    """Build worlds, retrieve, and check the composition kept everything."""
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    lost = 0
    dropped = 0
    easy_cost: list = []
    hard_cost: list = []
    leaked = 0
    total_calls = 0
    coverage_errors = 0
    absence = 0

    for seed in range(worlds):
        facts, easy, hard = _world(seed)
        root = Path(tempfile.mkdtemp(prefix="uq_retr_"))
        try:
            session = build_session(root, seed=seed)
            for key, value in facts:
                run_pipeline(f"{key} is {value}", session)

            counter = _Counting()
            engine = RetrievalEngine(session.memory, suggester=counter,
                                     width=width)
            bare = RetrievalEngine(session.memory, width=width)

            for question in easy + hard:
                before = counter.calls
                result = engine.retrieve(question)
                called = counter.calls - before
                total_calls += called

                # Criterion 1: nothing the PIPELINE would have found may
                # be missing - the candidate ladder and the index both.
                from ultraquant.interpreter.thoughts import (_candidate_keys,
                                                             _tokens)

                direct = set(session.memory.find_facts(question,
                                                       top_k=width))
                direct |= set(_candidate_keys(question, _tokens(question)))
                held = {key for key in direct
                        if session.memory.recall_fact(key) is not None}
                # 1a: the ROUTES, run exhaustively.
                everything = set(
                    engine.retrieve(question, exhaustive=True).keys)
                lost += len(held - everything)
                # 1b: the POLICY - no COVERING key may be dropped.
                asked_tokens = _fold(question)
                covering = {key for key in held
                            if asked_tokens <= _fold(key)}
                dropped += len(covering - set(result.keys))

                # Criterion 4: coverage, computed independently.
                asked = _fold(question)
                truth = any(asked <= _fold(key) for key in result.keys)
                coverage_errors += int(truth != result.covered)

                # Criterion 3: the expensive route only when needed.
                if result.covered and called:
                    leaked += called

                (easy_cost if question in easy else hard_cost).append(
                    result.examined)

                # Criterion 5: absence changes nothing else.
                without = bare.retrieve(question)
                if {k for k in without.keys} != {
                        k for k in result.keys
                        if not any(item.route == "semantic"
                                   and item.key == k
                                   for item in result.facts)}:
                    absence += 1
        finally:
            shutil.rmtree(root, ignore_errors=True)

    easy_mean = statistics.fmean(easy_cost) if easy_cost else 0.0
    hard_mean = statistics.fmean(hard_cost) if hard_cost else 0.0
    scales = hard_mean > easy_mean
    passes = (lost == 0 and dropped == 0 and scales
              and leaked == 0 and coverage_errors == 0
              and absence == 0)

    report = RetrievalReport(
        passes=passes, worlds=worlds, lost=lost, dropped=dropped,
        easy_examined=easy_mean,
        hard_examined=hard_mean, semantic_calls_when_covered=leaked,
        semantic_calls_total=total_calls, coverage_errors=coverage_errors,
        absence_differences=absence)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {worlds} worlds at width "
        f"{width}; {lost} keys lost by the routes and {dropped} covering "
        f"keys dropped by stopping early "
        + ("(nothing lost)" if lost == 0 and dropped == 0
           else "- THE COMPOSITION FINDS LESS THAN ITS PARTS")
        + f"; examined {easy_mean:.2f} on index-answerable questions "
        f"against {hard_mean:.2f} on harder ones "
        + ("(cost scales)" if scales
           else "- EQUAL COST, so the routes are not stopping early")
        + f"; the semantic route ran {total_calls} times, "
        f"{leaked} of them on already-covered questions "
        + ("(never wasted)" if leaked == 0 else "- PAID WHEN NOT NEEDED")
        + f"; {coverage_errors} coverage disagreements, "
        f"{absence} routes changed by the suggester's absence")
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    print(f"worlds:                    {result.worlds}")
    print(f"keys lost by the routes:   {result.lost}")
    print(f"covering keys dropped:     {result.dropped}")
    print(f"examined, easy questions:  {result.easy_examined:.2f}")
    print(f"examined, hard questions:  {result.hard_examined:.2f}")
    print(f"semantic route calls:      {result.semantic_calls_total}")
    print(f"  of those, wasted:        {result.semantic_calls_when_covered}")
    print(f"coverage disagreements:    {result.coverage_errors}")
    print(f"changed by absence:        {result.absence_differences}")
    print()
    print(result.reason)
