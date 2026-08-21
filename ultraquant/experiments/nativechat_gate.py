"""One conversation, two tiers, the same words. The native chat gate.

This is where the native tier stops being a library and starts being
the system: text in, a sentence out, a store that changed. The
parity obligation is at its strictest here, because what a pipeline
produces IS a sentence - so this gate compares the words, not a
structure that could be worded two ways, and it compares them TURN
BY TURN over a shared conversation, because a store carries state
and the third answer depends on the first two.

**What this unit ports, stated rather than implied**: intent
(statement, question, expression, chat); statements parsed, stored
and spoken back, with polarity, with a revision named aloud, with
§11.63's near-key adjacency note; and the question ladder as far as
the arithmetic and list readers, the exact recall branch under
§11.29's coverage rule, nearest-held, and the honest unknown.

**What it does not port**, and therefore does not claim: polar
questions, why, comparatives, history, choices, testimony,
conjunctions, clause splitting, glyphs, goals, URLs and learning
mode. Those arrive with their own gates. Chains left that list with
§11.92 and superlatives and aggregates with §11.93; this corpus
provokes all three now, so what was once out of scope is under test
rather than merely no longer mentioned.

**One known gap is measured rather than avoided.** The Python tier
appends a curiosity hint to a hedged answer when a failed question
points at a fact worth asking for: "... If I knew the iron bridge, I
could work this out - ':learn' will ask." That hint comes from
`missing_premise`, which is part of the spreading-activation
machinery and belongs to the inference unit. Rather than choose a
corpus that never triggers it - which would be a gate quietly
avoiding what the tier cannot do, and would have measured nothing -
the corpus provokes it freely and the gate CLASSIFIES every
difference. A response differing only by that hint is counted as a
named gap; a response differing in any other way, anywhere, is a
fail - so the number is a measurement of what is missing
rather than a silence about it.

**The criteria, written before the run.**

1. **No unexplained difference** - every turn either matches
   byte-for-byte or differs ONLY by the curiosity-hint suffix. One
   difference of any other kind is a fail.
2. **The gap is bounded and reported** - hint-only differences are
   counted, and the count is the size of the debt the inference unit
   inherits. It may not grow silently.
3. **Intents match** - the native tier must classify each turn the
   same way (statement, question, chat), because a right answer
   reached by the wrong door will diverge on the next input.
4. **The store agrees at the end** - after every conversation, both
   tiers hold the same keys with the same values, polarity and
   confidences. A pipeline can say the right thing and remember the
   wrong one.

**It was run twice: once with a debt, and once after the debt was
paid.**

Run one, before the inference port: 1,200 turns, zero unexplained
differences, zero intent differences, zero stores differing - and
**71 turns differing only by the curiosity hint**, the named gap
this gate was built to measure rather than avoid.

Run two, after §11.92 ported the spread and `missing_premise`, with
the corpus widened to build CHAINS (a subject whose material names a
material whose property is held) so that inference is provoked
rather than merely available:

| | |
|---|---:|
| turns compared | 1,454 |
| unexplained differences | **0** |
| differing only by the curiosity hint | **0** |
| intent differences | **0** |
| conversations ending with unequal stores | **0** |

Coverage: 528 statements, 202 questions about held facts, 169
arithmetic questions, 127 chain questions, 121 that may or may not
be held, 119 revisions, 97 lines of chat, 91 about nothing held.

Two of those zeros are quieter than the headline and matter as much.
**Zero intent differences**: every turn went through the same door in
both tiers, which is what keeps the NEXT turn comparable - a right
answer reached the wrong way diverges later, not now. **Zero stores
differing at the end**: the pipeline that said the right thing also
remembered the right thing.

And the debt closing is worth as much as the pass. The 71 was never
an excuse; it was a number, carried in the open across two units,
and the honest test of the second unit was whether it went to zero
without anything else moving. It did.

Run it::

    python -m ultraquant.experiments.nativechat_gate
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ChatParityReport", "run_gate", "probe_path"]

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "native" / "uq" / "_build" / "uq_chat.exe"

#: The suffix the Python tier adds when a refusal points somewhere.
_HINT = " If I knew the "


def probe_path() -> Path:
    """Where the built pipeline lives, so tests can skip without it."""
    return _PROBE


@dataclass
class ChatParityReport:
    """Turn-by-turn parity, the named gap, and the final stores.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        turns: Turns compared.
        real_differences: Differences not explained by the hint.
        hint_only: Turns differing only by the curiosity hint.
        intent_differences: Turns classified differently.
        store_differences: Conversations ending with unequal stores.
        kinds: Turn kind -> how many turns it contributed.
        first_difference: The first unexplained difference, in full.
        reason: Plain-language verdict.
    """

    passes: bool
    turns: int = 0
    real_differences: int = 0
    hint_only: int = 0
    intent_differences: int = 0
    store_differences: int = 0
    kinds: dict = field(default_factory=dict)
    first_difference: str = ""
    reason: str = ""


_ENTITIES = ["tower", "bridge", "keep", "mill", "dome", "spire",
             "old tower", "north bridge", "great keep"]
_ATTRIBUTES = ["material", "height", "weight", "colour"]
_MATERIALS = ["iron", "steel", "oak", "bronze", "granite"]


def _conversation(rng: random.Random, turns: int):
    """A session both tiers are walked through, turn by turn."""
    plan: list[tuple[str, str]] = []
    said: list[tuple[str, str]] = []
    for _ in range(turns):
        pick = rng.random()
        if pick < 0.10:
            # A chain: a subject whose material names a material whose
            # property is held. Without these the corpus never asks a
            # question inference could answer, and a gate that never
            # provokes a mechanism has not tested it.
            subject = rng.choice(_ENTITIES)
            material = rng.choice(_MATERIALS)
            attribute = rng.choice(["hardness", "conductivity"])
            said.append((f"{subject} material", material))
            plan.append((f"the {subject} material is {material}",
                         "statement"))
            plan.append((f"{material} {attribute} is "
                         f"{rng.randint(100, 900)} units", "statement"))
            plan.append((f"what is the {subject} {attribute}?", "chain"))
            continue
        if pick < 0.34 or not said:
            entity = rng.choice(_ENTITIES)
            attribute = rng.choice(_ATTRIBUTES)
            key = f"{entity} {attribute}"
            if attribute in ("height", "weight"):
                value = f"{rng.randrange(25, 900, 25)} meters"
            else:
                value = rng.choice(_MATERIALS)
            if rng.random() < 0.18:
                value = "not " + value
            said.append((key, value))
            plan.append((f"the {key} is {value}", "statement"))
        elif pick < 0.44:
            key, value = rng.choice(said)
            plan.append((f"the {key} is {rng.choice(_MATERIALS)}",
                         "revision"))
        elif pick < 0.62:
            key, _value = rng.choice(said)
            plan.append((f"what is the {key}?", "question held"))
        elif pick < 0.72:
            plan.append((f"what is the {rng.choice(_ENTITIES)} "
                         f"{rng.choice(_ATTRIBUTES)}?", "question maybe"))
        elif pick < 0.80:
            plan.append((rng.choice([
                "what is the melting point of tungsten?",
                "what is the sky colour?",
                "who is the architect?",
            ]), "question unheld"))
        elif pick < 0.86:
            # The enumerated-family forms: a claim about a SET, which
            # must not be answered by whichever member matched first.
            attribute = rng.choice(["height", "weight"])
            plan.append((rng.choice([
                f"which {attribute} is the tallest?",
                f"which {attribute} is the shortest?",
                "which is the tallest?",
                "which is the biggest?",
                f"what is the total {attribute}?",
                f"what is the average {attribute}?",
                f"how many {attribute} facts do you hold?",
            ]), "family"))
        elif pick < 0.92:
            plan.append((rng.choice([
                f"what is {rng.randint(2, 90)} + {rng.randint(2, 40)} * "
                f"{rng.randint(2, 9)}?",
                f"what is {rng.randrange(5, 95, 5)}% of "
                f"{rng.randrange(20, 900, 20)}?",
                f"what is the square root of {rng.randint(2, 200)}?",
                f"what is {rng.randint(10, 400)} / "
                f"{rng.choice((3, 7, 9))} to 2 decimal places?",
                f"what is the average of {rng.randint(1, 40)}, "
                f"{rng.randint(1, 40)} and {rng.randint(1, 40)}?",
                f"what is {rng.randint(2, 9)} ^ {rng.randint(2, 5)}?",
            ]), "arithmetic"))
        else:
            plan.append((rng.choice([
                "hello there", "thanks", "good morning",
            ]), "chat"))
    return plan


def _python_walk(plan):
    """The tier that defines the semantics, turn by turn."""
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_chatparity_"))
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


def _hint_only(want: str, have: str) -> bool:
    """Whether the sole difference is the curiosity hint suffix."""
    return want.startswith(have) and _HINT in want[len(have):]


def run_gate(turns: int = 60, conversations: int = 20,
             seed: int = 0) -> ChatParityReport:
    """Walk both tiers through the same conversations and compare."""
    if not _PROBE.exists():
        return ChatParityReport(
            passes=False,
            reason="FAIL - the native pipeline is not built; run "
                   "native/uq/build.ps1")
    compared = 0
    real = 0
    hint_only = 0
    intent_differences = 0
    store_differences = 0
    kinds: dict = {}
    first = ""
    for index in range(conversations):
        rng = random.Random(41000 + seed * 100 + index)
        plan = _conversation(rng, turns)
        for _text, kind in plan:
            kinds[kind] = kinds.get(kind, 0) + 1
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
            if want_said == have_said:
                continue
            if _hint_only(want_said, have_said):
                hint_only += 1
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
    passes = (real == 0 and intent_differences == 0
              and store_differences == 0)
    reason = (
        f"{'PASS' if passes else 'FAIL'} - {compared} turns across "
        f"{conversations} conversations; {real} unexplained "
        f"differences, {hint_only} differing only by the curiosity "
        f"hint, {intent_differences} intent differences, "
        f"{store_differences} conversations ending with unequal stores"
        + (f"\nfirst:\n{first}" if first else ""))
    return ChatParityReport(passes=passes, turns=compared,
                            real_differences=real, hint_only=hint_only,
                            intent_differences=intent_differences,
                            store_differences=store_differences,
                            kinds=kinds, first_difference=first,
                            reason=reason)


if __name__ == "__main__":                        # pragma: no cover
    report = run_gate()
    print(f"turns compared: {report.turns}")
    print(f"unexplained differences: {report.real_differences}")
    print(f"hint-only differences: {report.hint_only}")
    print(f"intent differences: {report.intent_differences}")
    print(f"stores differing at the end: {report.store_differences}")
    print()
    for kind, count in sorted(report.kinds.items()):
        print(f"  {kind:<16} {count:5d} turns")
    print()
    print(report.reason)
