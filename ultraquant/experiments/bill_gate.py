"""The bill must describe the work. Finding 9's independent exam.

**The defect.** Retrieval.examined promises keys inspected whether retained
or not, but reports the size of a set containing only successful keys.
Empty memory can therefore report zero after many lookups and index probes.
The implementer does not write its own exam: these criteria are frozen
before calibration, and this module changes no retrieval implementation.

**The criteria, written before the run**

1. **Every reported counter is honest, on every call in every case.** The
   fixed Retrieval must expose these six nonnegative integer fields:
   `unique_facts_returned` = distinct keys in the returned facts;
   `lookup_attempts` = actual memory.recall_fact invocations, including
   misses, repeated keys and calls made inside the semantic suggester;
   `index_probes` = actual memory.find_facts invocations, including empty
   results, repeated probe texts and the suggester's nested queries;
   `phrase_probes` = the subset of those index invocations made inside
   RetrievalEngine._reach or inference._reachable_facts (including the
   semantic candidate search; a nested invocation is counted only once);
   `semantic_calls` = actual suggester.suggest invocations, even when no
   suggestion is returned. The old name `examined` MUST remain and equal
   `lookup_attempts`: attempted fact reads, not unique keys or a sum of
   unlike operations. All counts are per retrieve call, never cumulative.
   Missing fields, wrong types and unequal counts fail. `routes` must also
   equal the number of returned facts attributed to each executed route.
   The oracle wraps real calls and the phrase path; it never uses the
   engine's accounting to predict a count. The cases are empty memory,
   exact hit, populated-memory miss, chain-shaped question, the same
   question twice on one engine, exact-only hit and miss, full cascade
   with a constant-vector semantic suggester, semantic-only retrieval,
   exhaustive retrieval, and an empty question.
2. **The OLD arm must expose the defect, or the exam is VOID.** On the
   empty-memory case it must perform positive lookup and index work and
   its existing `examined` must disagree with the wrapped lookup count;
   missing newly required fields alone do not establish calibration.
   Path controls must also hold in each arm: the exact hit stops at exact;
   exact-only cases make no index, phrase or semantic calls; the chain
   reaches phrase probing; the full cascade executes all four routes,
   calls the suggester and embedder, and observes its nested memory work;
   semantic-only retrieval returns a semantic fact. Otherwise the exam
   has not exercised its claimed paths and is VOID.
3. **The bill changes, the answer does not.** On every case and each
   repeated call, OLD and fixed arms must have identical question,
   ordered retrieved facts (full records, keys, route and strength),
   ordered keys, routes, `covered` and `stopped_after`. No accounting
   improvement redeems an answer difference. Both arms start from the
   same deterministic records; instrumentation delegates without changing
   arguments, results or exceptions.
4. **The wiring still passes.** With `_HONEST_BILL = True`, call
   ultraquant.experiments.wiring_gate.run_gate() with its default battery
   and require its report.passes to be True. An exception or a false
   verdict fails this criterion.

**The flag is the arm.** retrieval._HONEST_BILL = False must restore OLD
behaviour; True selects the fix. When the flag is absent, run ONLY the OLD
arm, print every discrepancy, and return a calibration report with
passes=False. Criteria 3 and 4 are then explicitly unmeasured, never passes.
Once the flag exists, both arms and the wiring gate run in one process.
All wrappers and flags are restored, including on exceptions. Temporary
SystematicMemory stores and the constant embedder require no service.

Run without creating bytecode files::

    python -B -m ultraquant.experiments.bill_gate
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

__all__ = ["BillReport", "run_gate"]

_COUNTERS = ("unique_facts_returned", "lookup_attempts", "index_probes",
             "phrase_probes", "semantic_calls", "examined")
_FACTS = (("tower height", "300 meters"),
          ("tower material", "steel"), ("steel hardness", "high"))


@dataclass(frozen=True)
class _Case:
    name: str
    question: str
    facts: tuple = _FACTS
    repeats: int = 1
    routes: tuple | None = None
    exhaustive: bool = False
    semantic: bool = False


_CASES = (
    _Case("empty-memory", "what is the tower height?", facts=()),
    _Case("exact-hit", "what is the tower height?"),
    _Case("miss", "what is the obelisk colour?"),
    _Case("chain", "what is the tower hardness?"),
    _Case("repeated", "what is the tower hardness?", repeats=2),
    _Case("exact-only-hit", "what is the tower height?", routes=("exact",)),
    _Case("exact-only-miss", "what is the obelisk colour?", routes=("exact",)),
    _Case("full-cascade", "how tall is the tower?", semantic=True),
    _Case("semantic-only", "how tall is the tower?", semantic=True,
          routes=("semantic",)),
    _Case("exhaustive", "what is the tower height?", exhaustive=True,
          semantic=True),
    _Case("empty-question", "", semantic=True),
)


class _Constant:
    """Every text gets one identical vector; no live embedding service."""

    def __init__(self):
        self.calls = 0

    def available(self):
        return True

    def embed(self, texts, model=None):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


@dataclass
class _Observation:
    case: str
    measured: dict
    reported: dict
    answer: dict
    discrepancies: list = field(default_factory=list)
    controls: list = field(default_factory=list)


@dataclass
class BillReport:
    """A calibration is evidence of the defect, never a passing fix."""

    passes: bool = False
    calibration_only: bool = True
    valid: bool = False
    old: list = field(default_factory=list)
    fixed: list = field(default_factory=list)
    answer_differences: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    wiring_passes: bool | None = None
    wiring_reason: str = "unmeasured"
    reason: str = ""


def _observe(engine, case, label, embedder):
    """Count call boundaries, including work below the semantic boundary."""
    from ultraquant.reason import inference

    counts = dict.fromkeys(_COUNTERS, 0)
    state = {"phrase_depth": 0, "semantic_depth": 0,
             "nested_lookups": 0, "nested_index": 0, "phrase_builds": 0}
    memory = engine.memory
    original_recall, original_find = memory.recall_fact, memory.find_facts
    before_embeddings = embedder.calls

    @wraps(original_recall)
    def recall(*args, **kwargs):
        counts["lookup_attempts"] += 1
        state["nested_lookups"] += bool(state["semantic_depth"])
        return original_recall(*args, **kwargs)

    @wraps(original_find)
    def find(*args, **kwargs):
        counts["index_probes"] += 1
        counts["phrase_probes"] += bool(state["phrase_depth"])
        state["nested_index"] += bool(state["semantic_depth"])
        return original_find(*args, **kwargs)

    def in_phrase_path(original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            state["phrase_depth"] += 1
            try:
                return original(*args, **kwargs)
            finally:
                state["phrase_depth"] -= 1
        return wrapped

    original_phrases = inference._probe_phrases

    @wraps(original_phrases)
    def phrases(*args, **kwargs):
        state["phrase_builds"] += 1
        return original_phrases(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(memory, "recall_fact", recall))
        stack.enter_context(patch.object(memory, "find_facts", find))
        stack.enter_context(patch.object(engine, "_reach",
                                         in_phrase_path(engine._reach)))
        stack.enter_context(patch.object(inference, "_reachable_facts",
                                         in_phrase_path(inference._reachable_facts)))
        stack.enter_context(patch.object(inference, "_probe_phrases", phrases))
        if engine.suggester is not None:
            original_suggest = engine.suggester.suggest

            @wraps(original_suggest)
            def suggest(*args, **kwargs):
                counts["semantic_calls"] += 1
                state["semantic_depth"] += 1
                try:
                    return original_suggest(*args, **kwargs)
                finally:
                    state["semantic_depth"] -= 1

            stack.enter_context(patch.object(engine.suggester, "suggest", suggest))
        result = engine.retrieve(case.question, exhaustive=case.exhaustive,
                                 routes=case.routes)

    # Output cardinality is observed directly, never inferred from examined.
    counts["unique_facts_returned"] = len({item.key for item in result.facts})
    counts["examined"] = counts["lookup_attempts"]
    reported = {name: getattr(result, name, None) for name in _COUNTERS}
    differences = []
    for name in _COUNTERS:
        got = reported[name]
        if type(got) is not int or got < 0 or got != counts[name]:
            shown = "<missing>" if not hasattr(result, name) else repr(got)
            differences.append(f"{name}: reported={shown}, measured={counts[name]}")
    route_counts = {name: sum(item.route == name for item in result.facts)
                    for name in result.routes}
    if (result.routes != route_counts
            or any(item.route not in result.routes for item in result.facts)
            or any(type(n) is not int or n < 0 for n in result.routes.values())):
        differences.append(f"routes: reported={result.routes!r}, "
                           f"observed contributions={route_counts!r}")
    answer = deepcopy({
        "question": result.question,
        "facts": [(item.key, item.record, item.route, item.strength)
                  for item in result.facts],
        "keys": result.keys, "routes": result.routes,
        "covered": result.covered, "stopped_after": result.stopped_after,
    })
    controls = []
    if case.name == "empty-memory" and not (
            counts["lookup_attempts"] > 0 and counts["index_probes"] > 0):
        controls.append("empty memory did not perform lookup and index work")
    if case.name == "exact-hit" and not (
            answer["keys"] == ["tower height"] and answer["covered"]
            and answer["stopped_after"] == "exact"):
        controls.append("exact hit did not stop with the expected fact")
    if case.name.startswith("exact-only") and any(
            counts[name] for name in ("index_probes", "phrase_probes", "semantic_calls")):
        controls.append("exact-only executed an excluded path")
    if case.name == "chain" and not (
            counts["phrase_probes"] > 0 and state["phrase_builds"] > 0):
        controls.append("chain did not exercise phrase probing")
    if case.name == "full-cascade" and not (
            tuple(result.routes) == ("exact", "lexical", "reach", "semantic")
            and counts["semantic_calls"] > 0 and embedder.calls > before_embeddings
            and state["nested_lookups"] > 0 and state["nested_index"] > 0
            and counts["phrase_probes"] > 0):
        controls.append("full cascade did not exercise all required paths")
    if case.name == "semantic-only" and not any(
            item.route == "semantic" for item in result.facts):
        controls.append("semantic-only did not return a semantic fact")
    return _Observation(label, counts, reported, answer, differences, controls)


def _arm(honest: bool | None):
    from ultraquant.memory import systematic
    from ultraquant.reason import retrieval
    from ultraquant.reason.semantic import SemanticSuggester

    flag = (nullcontext() if honest is None else
            patch.object(retrieval, "_HONEST_BILL", honest))
    observations = []
    with flag, TemporaryDirectory(prefix="uq_bill_") as directory:
        for case in _CASES:
            memory = systematic.SystematicMemory(Path(directory) / f"{case.name}.json")
            # Identical full records across arms, including timestamp fields.
            with patch.object(systematic, "_utc_now", return_value="2000-01-01T00:00:00+00:00"):
                for key, value in case.facts:
                    memory.remember_fact(key, value)
            embedder = _Constant()
            suggester = SemanticSuggester(embedder=embedder) if case.semantic else None
            engine = retrieval.RetrievalEngine(memory, suggester=suggester)
            for index in range(case.repeats):
                label = f"{case.name}/{index + 1}" if case.repeats > 1 else case.name
                observations.append(_observe(engine, case, label, embedder))
    return observations


def run_gate() -> BillReport:
    """Calibrate without the flag; otherwise examine both arms and wiring."""
    from ultraquant.reason import retrieval

    has_flag = hasattr(retrieval, "_HONEST_BILL")
    report = BillReport(calibration_only=not has_flag)
    try:
        report.old = _arm(False if has_flag else None)
    except Exception as exc:
        report.errors.append(f"OLD arm raised {type(exc).__name__}: {exc}")
        report.reason = "VOID: OLD arm could not be calibrated"
        return report
    empty = next(item for item in report.old if item.case == "empty-memory")
    report.valid = (type(empty.reported["examined"]) is int
                    and empty.reported["examined"] != empty.measured["lookup_attempts"]
                    and not any(item.controls for item in report.old))
    if not has_flag:
        report.reason = ("CALIBRATION ONLY: OLD defect confirmed; _HONEST_BILL absent; "
                         "fixed counters, answer parity and wiring unmeasured"
                         if report.valid else "VOID: OLD defect or path controls not established")
        return report
    try:
        report.fixed = _arm(True)
    except Exception as exc:
        report.errors.append(f"fixed arm raised {type(exc).__name__}: {exc}")
    report.valid = report.valid and not any(item.controls for item in report.fixed)
    old_answers = {item.case: item.answer for item in report.old}
    new_answers = {item.case: item.answer for item in report.fixed}
    report.answer_differences = [name for name in old_answers.keys() | new_answers.keys()
                                 if old_answers.get(name) != new_answers.get(name)]
    try:
        from ultraquant.experiments.wiring_gate import run_gate as wiring_gate
        with patch.object(retrieval, "_HONEST_BILL", True):
            wiring = wiring_gate()
        report.wiring_passes = wiring.passes is True
        report.wiring_reason = wiring.reason
    except Exception as exc:
        report.wiring_passes = False
        report.wiring_reason = f"raised {type(exc).__name__}: {exc}"
        report.errors.append(f"wiring {report.wiring_reason}")
    report.passes = (report.valid and not report.errors
                     and len(report.fixed) == len(report.old)
                     and not any(item.discrepancies for item in report.fixed)
                     and not report.answer_differences and report.wiring_passes is True)
    verdict = "VOID" if not report.valid else "PASS" if report.passes else "FAIL"
    report.reason = (f"{verdict}: {len(report.old)} calls per arm; "
                     f"{sum(len(item.discrepancies) for item in report.fixed)} fixed discrepancies; "
                     f"{len(report.answer_differences)} answer differences; "
                     f"wiring={report.wiring_passes}")
    return report


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    for arm, observations in (("OLD", report.old), ("FIXED", report.fixed)):
        for item in observations:
            print(f"{arm} {item.case}: " + ", ".join(
                f"{name}={item.measured[name]}" for name in _COUNTERS))
            for discrepancy in item.discrepancies:
                print(f"  DISCREPANCY {discrepancy}")
            for control in item.controls:
                print(f"  VOID {control}")
    for name in sorted(report.answer_differences):
        print(f"ANSWER DIFFERENCE {name}")
    for error in report.errors:
        print(f"ERROR {error}")
    print(f"wiring: {report.wiring_reason}")
    print(f"passes={report.passes}; valid={report.valid}; "
          f"calibration_only={report.calibration_only}")
    raise SystemExit(0 if report.passes or (report.calibration_only and report.valid) else 1)
