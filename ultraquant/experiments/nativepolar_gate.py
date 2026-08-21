"""The yes/no family, two tiers, the same words. The polar gate.

§11.91 ported the spine of the pipeline and named what it did not
port. This takes four items off that list at once, because they
share one door and one discipline and splitting them would gate a
half-open door.

**What this unit ports**: "is X Y?" against a held belief, with
polarity on both the claim and the store; "is X A or B?", where the
disjuncts are offered VALUES and the store elects one; "is A taller
than B?" with units, derived operands, written quantities and
literal expressions on either side; and "is <derivable subject>
Y?", which derives its own subject through the chain machinery
before it can check anything.

**The discipline they share is the thing worth measuring: absence
is never No.** A subject the library holds nothing about gets no
verdict - it falls through to the hedge, because "no" is reserved
for actual contrary belief, held either as a different value or as
a stored denial. That line is easy to state and easy to break in a
port, because every way of breaking it produces a confident,
plausible, wrong sentence. So the corpus asks about subjects that
are not there, and both tiers have to decline in the same words.

**One known gap, provoked rather than avoided.** A comparison side
may be an expression over BELIEFS - "is the tower height * 2 taller
than the bridge length?" - which needs the derived-operand reader
(§11.78), and that is not in this tier. Rather than choose a corpus
that never asks, the corpus asks freely and the gate CLASSIFIES
every difference: a turn differing only because a belief-operand
expression could not be read is counted as a named gap; a turn
differing in any other way is a fail. The number is the size of the
debt the next unit inherits, and it may not grow silently.

**The criteria, written before the run.**

1. **No unexplained difference** - every turn either matches
   byte-for-byte or differs ONLY because a comparison side was an
   expression over beliefs. One difference of any other kind is a
   fail.
2. **Absence is never No, in both tiers** - for every question
   about a subject neither tier holds, neither tier may answer
   with a verdict. A single "No" or "Yes" to an unheld subject is
   a fail on its own, separately from criterion 1, because a
   matching pair of wrong answers would pass criterion 1 happily.
3. **The gap is bounded and reported** - belief-operand differences
   are counted and named.
4. **The store agrees at the end** - a pipeline can say the right
   thing and remember the wrong one, so both tiers are dumped and
   compared after every conversation.
5. **Every form is provoked** - the run reports how many turns of
   each form the corpus produced, and a form with zero turns
   means the gate measured nothing about it and is a fail.

**The run.** 1,624 turns across 24 conversations:

| | |
|---|---:|
| unexplained differences | **0** |
| differing only by the belief-operand gap | 74 (of 89 asked) |
| intent differences | **0** |
| conversations ending with unequal stores | **0** |
| fabricated verdicts | **0** over 109 unheld subjects |

Coverage: 613 statements, 176 polar questions about held facts, 140
comparisons that must refuse aloud, 114 comparisons against a
written quantity, 113 choices, 111 comparisons between two held
numbers, 109 about subjects nobody holds, 92 polar questions whose
subject had to be derived, 67 comparisons over literal expressions.

**A pass on the first run is a claim about the gate as much as the
port, so the gate was checked against itself**: with the branch
disabled and the tier rebuilt, 80 of 180 turns differ. A gate that
cannot fail has measured nothing, and this one can.

Two numbers are quieter than the headline. **Zero fabricated
verdicts over 109 unheld subjects** is criterion 2, and it is
checked separately from parity on purpose - two tiers agreeing on a
confident wrong answer would sail through criterion 1 without
either of them being right. And **74 of 89**, not 89 of 89: fifteen
belief-operand comparisons agreed anyway, because a side the Python
tier also could not read refuses in the same words. The gap is
narrower than the form that provokes it.

Run it::

    python -m ultraquant.experiments.nativepolar_gate
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from ultraquant.interpreter.thoughts import build_session, run_pipeline

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_chat.exe"

__all__ = ["PolarParityReport", "run_gate", "probe_path"]


def probe_path() -> Path:
    """Where the built pipeline lives, so tests can skip without it."""
    return _PROBE


class PolarParityReport:
    """Turn-by-turn parity, the named gap, and the absence check."""

    def __init__(self, passes: bool, **fields) -> None:
        self.passes = passes
        self.turns = fields.get("turns", 0)
        self.real_differences = fields.get("real_differences", 0)
        self.belief_operand = fields.get("belief_operand", 0)
        self.intent_differences = fields.get("intent_differences", 0)
        self.store_differences = fields.get("store_differences", 0)
        self.fabricated_verdicts = fields.get("fabricated_verdicts", 0)
        self.absence_checks = fields.get("absence_checks", 0)
        self.kinds = fields.get("kinds", {})
        self.first_difference = fields.get("first_difference", "")
        self.reason = fields.get("reason", "")


_ENTITIES = ["tower", "bridge", "keep", "mill", "dome", "spire",
             "old tower", "north bridge", "great keep"]
_MATERIALS = ["iron", "steel", "oak", "bronze", "granite"]
_ABSENT = ["obelisk", "aqueduct", "lighthouse", "campanile"]

#: A comparison side written as an expression over BELIEFS. The
#: derived-operand reader is not in the native tier, so these are
#: the named gap - provoked deliberately, counted, never hidden.
_BELIEF_OPERAND = " * "


def _conversation(rng: random.Random, turns: int):
    """A session both tiers are walked through, turn by turn."""
    plan: list[tuple[str, str]] = []
    said: list[tuple[str, str]] = []
    numeric: list[str] = []
    for _ in range(turns):
        pick = rng.random()
        if pick < 0.26 or len(said) < 3:
            entity = rng.choice(_ENTITIES)
            attribute = rng.choice(["material", "height", "weight",
                                    "colour", "length"])
            key = f"{entity} {attribute}"
            if attribute in ("height", "length"):
                value = f"{rng.randrange(25, 900, 25)} meters"
                numeric.append(key)
            elif attribute == "weight":
                value = f"{rng.randrange(5, 400, 5)} tonnes"
                numeric.append(key)
            else:
                value = rng.choice(_MATERIALS)
            if rng.random() < 0.16:
                value = "not " + value
            said.append((key, value))
            plan.append((f"the {key} is {value}", "statement"))
            continue
        if pick < 0.38:
            # "Is X Y?" against something held - the answer is a
            # verdict, and which verdict depends on polarity twice.
            key, value = rng.choice(said)
            bare = value[4:] if value.startswith("not ") else value
            claim = bare if rng.random() < 0.6 else rng.choice(_MATERIALS)
            if rng.random() < 0.25:
                claim = "not " + claim
            plan.append((f"is the {key} {claim}?", "polar held"))
        elif pick < 0.46:
            # Absence is never No. Neither tier holds this subject,
            # and neither may hand out a verdict about it.
            plan.append((f"is the {rng.choice(_ABSENT)} "
                         f"{rng.choice(_MATERIALS)}?", "polar absent"))
        elif pick < 0.54:
            key, value = rng.choice(said)
            other = rng.choice(_MATERIALS)
            plan.append((f"is the {key} {rng.choice(_MATERIALS)} "
                         f"or {other}?", "choice"))
        elif pick < 0.62 and len(numeric) >= 2:
            left, right = rng.sample(numeric, 2)
            word = rng.choice(["taller", "heavier", "longer", "shorter",
                               "greater", "less", "bigger", "lighter"])
            plan.append((f"is the {left} {word} than the {right}?",
                         "compare held"))
        elif pick < 0.70 and numeric:
            key = rng.choice(numeric)
            unit = rng.choice(["meters", "kilometers", "tonnes", "degrees"])
            plan.append((f"is the {key} greater than "
                         f"{rng.randrange(25, 900, 25)} {unit}?",
                         "compare literal"))
        elif pick < 0.76:
            plan.append((rng.choice([
                f"is {rng.randint(2, 20)} * {rng.randint(2, 20)} greater "
                f"than {rng.randint(10, 300)}?",
                f"is {rng.randint(2, 9)} + {rng.randint(2, 9)} = "
                f"{rng.randint(4, 18)}?",
                f"is {rng.randint(2, 40)} equal to {rng.randint(2, 40)}?",
            ]), "compare literal expression"))
        elif pick < 0.82 and numeric:
            # The named gap: a side that names a belief inside an
            # expression. Asked freely; classified, never avoided.
            key = rng.choice(numeric)
            plan.append((f"is the {key}{_BELIEF_OPERAND}2 greater than "
                         f"{rng.randrange(50, 900, 50)} meters?",
                         "compare belief operand"))
        elif pick < 0.90:
            # A chain, then a polar question whose SUBJECT has to be
            # derived before anything can be checked.
            subject = rng.choice(_ENTITIES)
            material = rng.choice(_MATERIALS)
            attribute = rng.choice(["hardness", "conductivity"])
            grade = rng.choice(["high", "low", "medium"])
            said.append((f"{subject} material", material))
            plan.append((f"the {subject} material is {material}",
                         "statement"))
            plan.append((f"{material} {attribute} is {grade}", "statement"))
            plan.append((f"is the {subject} {attribute} {grade}?",
                         "polar derived"))
        else:
            # Comparisons that must REFUSE aloud: an unheld side, a
            # denial, a non-numeric value, incomparable units.
            options = [f"is the {rng.choice(_ABSENT)} taller than the "
                       f"{rng.choice(_ENTITIES)} height?"]
            if numeric:
                options.append(f"is the {rng.choice(numeric)} taller than "
                               f"the {rng.choice(_ABSENT)} height?")
            for key, value in said:
                if not any(c.isdigit() for c in value):
                    options.append(f"is the {key} taller than the "
                                   f"{rng.choice(_ENTITIES)} height?")
                    break
            plan.append((rng.choice(options), "compare refusal"))
    return plan


def _python_walk(plan):
    """The tier that defines the semantics, turn by turn."""
    root = Path(tempfile.mkdtemp(prefix="uq_polarparity_"))
    try:
        session = build_session(root, seed=0)
        spoken: list[tuple[str, str]] = []
        for text, _kind in plan:
            response, trace = run_pipeline(text, session)
            intent = ""
            for entry in trace:
                if entry.get("thought") == "Perceive":
                    intent = str(entry.get("intent", ""))
                    break
            spoken.append((intent, response))
        store = {}
        for key in session.memory.fact_keys():
            record = session.memory.recall_fact(key)
            store[key] = (str(record.get("value", "")),
                          bool(record.get("negated")),
                          round(float(record.get("confidence", 0.0)), 2))
        return spoken, store
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _native_walk(plan):
    """The same conversation, spoken to the native pipeline."""
    lines = [text for text, _kind in plan] + ["::dump"]
    got = subprocess.run([str(_PROBE)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    spoken: list[tuple[str, str]] = []
    store: dict = {}
    for line in got.stdout.splitlines():
        if line.startswith("store|"):
            body = line[len("store|"):]
            if not body:
                continue
            for entry in body.split(" ;; "):
                key, value, negated, confidence = entry.split(" :: ")
                store[key] = (value, negated == "1",
                              round(float(confidence), 2))
            continue
        intent, _, response = line.partition("|")
        spoken.append((intent, response))
    return spoken, store


def _is_verdict(said: str) -> bool:
    """Whether a sentence hands out a yes/no about a claim.

    "Neither" is a verdict too - it says the store holds something
    contrary. What is NOT a verdict is a hedge, a refusal, or a
    nearest-held note; those are the honest answers to a question
    about a subject nobody holds. The check is deliberately narrow:
    it looks at the first word only, because every verdict in this
    family leads with one and no hedge does.
    """
    return said.split(" ", 1)[0].strip(",-") in ("Yes", "No", "Neither")


def run_gate(turns: int = 60, conversations: int = 24,
             seed: int = 0) -> PolarParityReport:
    """Walk both tiers through the same conversations and compare."""
    if not _PROBE.exists():
        return PolarParityReport(
            passes=False,
            reason="FAIL - the native pipeline is not built; run "
                   "native/uq/build.ps1")
    compared = 0
    real = 0
    belief_operand = 0
    intent_differences = 0
    store_differences = 0
    fabricated = 0
    absence_checks = 0
    kinds: Counter = Counter()
    first = ""
    for index in range(conversations):
        rng = random.Random(58000 + seed * 100 + index)
        plan = _conversation(rng, turns)
        for _text, kind in plan:
            kinds[kind] += 1
        want_turns, want_store = _python_walk(plan)
        have_turns, have_store = _native_walk(plan)
        for step, (text, kind) in enumerate(plan):
            compared += 1
            want_intent, want_said = want_turns[step]
            have_intent, have_said = (have_turns[step]
                                      if step < len(have_turns)
                                      else ("", "(no answer)"))
            if want_intent != have_intent:
                intent_differences += 1
            if kind == "polar absent":
                # Criterion 2, checked SEPARATELY from parity: two
                # tiers agreeing on a fabricated verdict would sail
                # through criterion 1 without either of them being
                # right.
                absence_checks += 1
                if _is_verdict(want_said):
                    fabricated += 1
                    if not first:
                        first = (f"python fabricated a verdict about an "
                                 f"unheld subject: {text!r}\n"
                                 f"    python: {want_said}")
                if _is_verdict(have_said):
                    fabricated += 1
                    if not first:
                        first = (f"native fabricated a verdict about an "
                                 f"unheld subject: {text!r}\n"
                                 f"    native: {have_said}")
            if want_said == have_said:
                continue
            if kind == "compare belief operand":
                belief_operand += 1
                continue
            real += 1
            if not first:
                first = (f"conversation {index}, turn {step} [{kind}] "
                         f"{text!r}\n    python: {want_said}\n"
                         f"    native: {have_said}")
        if want_store != have_store:
            store_differences += 1
            if not first:
                only_python = {k: v for k, v in want_store.items()
                               if have_store.get(k) != v}
                only_native = {k: v for k, v in have_store.items()
                               if want_store.get(k) != v}
                first = (f"conversation {index}: stores differ\n"
                         f"    python-only: {only_python}\n"
                         f"    native-only: {only_native}")
    # Criterion 5: a form the corpus never produced is a form this
    # gate measured nothing about, and silence is not a pass.
    wanted_kinds = {"statement", "polar held", "polar absent", "choice",
                    "compare held", "compare literal",
                    "compare literal expression",
                    "compare belief operand", "polar derived",
                    "compare refusal"}
    missing = sorted(wanted_kinds - set(kinds))
    passes = (real == 0 and intent_differences == 0
              and store_differences == 0 and fabricated == 0
              and not missing)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {compared} turns across "
        f"{conversations} conversations; {real} unexplained "
        f"differences, {belief_operand} differing only by the "
        f"belief-operand gap, {intent_differences} intent differences, "
        f"{store_differences} conversations ending with unequal stores, "
        f"{fabricated} fabricated verdicts over {absence_checks} "
        f"questions about unheld subjects"
        + (f"; forms never provoked: {', '.join(missing)}" if missing
           else "")
        + (f"\nfirst:\n{first}" if first else ""))
    return PolarParityReport(passes=passes, turns=compared,
                             real_differences=real,
                             belief_operand=belief_operand,
                             intent_differences=intent_differences,
                             store_differences=store_differences,
                             fabricated_verdicts=fabricated,
                             absence_checks=absence_checks,
                             kinds=dict(kinds), first_difference=first,
                             reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"turns compared: {report.turns}")
    print(f"unexplained differences: {report.real_differences}")
    print(f"belief-operand gap: {report.belief_operand}")
    print(f"intent differences: {report.intent_differences}")
    print(f"stores differing at the end: {report.store_differences}")
    print(f"fabricated verdicts: {report.fabricated_verdicts} over "
          f"{report.absence_checks} unheld subjects")
    print()
    for kind, count in sorted(report.kinds.items()):
        print(f"  {kind:<28} {count:5d} turns")
    print()
    print(report.reason)
