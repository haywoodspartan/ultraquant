"""The denial the semantic route forgot. The polarity gate.

The first unit found by the pair: GPT-6 Astra's read-only architecture
review flagged it, and Claude reproduced it independently with controls
before anything was changed. Two models, two probes, one defect.

**The defect.** `Reason._question` has its own copy of the semantic
route, and it built the reply from `Suggestion.value` - the stored
value and nothing else. A held denial reached that way was spoken as
an assertion: "the kettle is not hot", asked "what is the copper
kettle?", answered "kettle is hot". The same record reached lexically
was spoken correctly, the affirmed record reached semantically was
spoken correctly, and the engine's own semantic route re-reads the
full record and keeps its polarity. The defect lived only in the
duplicate. It is the worst failure this system has a name for: not an
abstention, not a wrong neighbour, but the confident opposite of what
it holds.

Five sites in the pipeline speak a raw value rather than
`_shown_value`. Four were checked and are guarded - the superlative
counts denials separately because a denial names no number, and the
comparison refuses a negated operand on both sides before it renders.
This was the fifth.

**The flag is the arm.** `_SEMANTIC_POLARITY = False` restores the old
render byte for byte, so both arms run in one process over the same
inputs.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 5cf77417...), with one amendment also made before any run
(abed6543...):

1. **No inverted assertion.** Ten negated facts - single-word,
   multi-word and numeric values - each reached through the semantic
   route. Fixed arm: 0 replies assert the bare value and every reply
   carries the denial. The OLD arm must invert all ten, or the battery
   is not reaching the path and the gate is void. Calibrated against
   the old arm only: 10 of 10 inverted.
2. **Positives untouched.** The same ten affirmed: replies
   byte-identical across arms.
3. **Changes are exactly the negated semantic readings.** Worlds with
   a suggester attached and denials in them: the set of turns whose
   reply differs across arms equals the set of turns answered by a
   semantic reading of a negated record, in both directions. Void if
   that set is empty. (Amendment A: the §11.115 worlds state no
   denials, so on them alone this would pass whatever the fix did.)
4. **Nothing else moves.** The §11.115 worlds with a default session
   and no suggester: 0 replies, 0 intents, 0 stores differ.
5. **The full suite is green.** Checked outside this module.

The harness embedder returns one vector for every text, so every
cosine is 1.0 and the semantic route fires wherever the anchor and
head rules allow. That is a stress device, not a model of similarity:
it maximises how many turns reach the path the fix touches, which is
what a specificity criterion needs.

**PASSED on all four measured criteria; the fifth is the suite.**

| battery (10 negated facts) | old arm | fixed arm |
|---|---:|---:|
| spoken as an assertion | 10 | **0** |
| spoken as a denial | 0 | **10** |

The same ten affirmed: 0 replies moved. In the denial-bearing
worlds, 240 turns: **29 replies changed, and they were exactly the 29
semantic readings of a negated record** - 0 spurious, 0 missed. In
the §11.115 worlds, 284 turns with no suggester: 0 replies, 0 intents
and 0 stores differed.

The exposure figure carries its caveat: 29 of 240 is what the stress
embedder produces, not a field rate. What it does establish is that
the defect was not a corner - wherever the semantic route answered a
denial, it answered it backwards.

**Found by one model, reproduced by the other.** GPT-6 Astra's review
named the defect and probed it ("not steel" became "steel"); Claude
reproduced it with controls before anything changed and wrote this
gate; GPT-6 Astra wrote the fix. The implementer did not write its own
exam, and its first attempt was refused by a read-only sandbox - which
it reported rather than papered over.
"""

from __future__ import annotations

import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["BATTERY", "PolarityReport", "run_gate"]

#: (subject, value, question). Every question is head-final and carries
#: one leading modifier, so exact coverage fails and the semantic route
#: is the rung that answers.
BATTERY = [
    ("kettle", "hot", "what is the copper kettle?"),
    ("tower material", "steel", "what is the old tower material?"),
    ("bridge colour", "red", "what is the north bridge colour?"),
    ("mill height", "300 meters", "what is the stone mill height?"),
    ("spire weight", "40 tons", "what is the east spire weight?"),
    ("granary roof", "slate grey", "what is the east granary roof?"),
    ("keep gate", "open", "what is the west keep gate?"),
    ("river depth", "12 meters", "what is the lower river depth?"),
    ("lamp colour", "deep blue", "what is the brass lamp colour?"),
    ("engine fuel", "diesel", "what is the ship engine fuel?"),
]

_SEMANTIC = re.compile(r"semantic reading '(.+?)'")


class _Constant:
    """Every text embeds to the same vector: every cosine is 1.0."""

    def embed(self, texts, model=None):
        return [[1.0, 0.0] for _ in texts]

    def available(self):
        return True


def _session(root: Path, semantic: bool):
    from ultraquant.interpreter import thoughts as T
    from ultraquant.reason.semantic import SemanticSuggester

    if semantic:
        return T.build_session(
            root, seed=0, semantic=SemanticSuggester(embedder=_Constant()))
    return T.build_session(root, seed=0)


def _walk(plan: list, polarity_on: bool, semantic: bool) -> tuple:
    """One conversation through one arm.

    Returns ``(replies, intents, readings, store)``, where ``readings``
    holds, per turn, the negated flag of the record a semantic reading
    answered from - or None when no semantic reading answered.
    """
    from ultraquant.interpreter import thoughts as T

    if not hasattr(T, "_SEMANTIC_POLARITY"):
        raise RuntimeError("thoughts._SEMANTIC_POLARITY is missing: "
                           "the two arms would be the same code")
    root = Path(tempfile.mkdtemp(prefix="uq_polarity_"))
    previous = T._SEMANTIC_POLARITY
    T._SEMANTIC_POLARITY = polarity_on
    try:
        session = _session(root, semantic)
        replies, intents, readings = [], [], []
        for text in plan:
            reply, trace = T.run_pipeline(text, session)
            replies.append(reply)
            intent, reading = "", None
            for entry in trace:
                if entry.get("thought") == "Perceive" and not intent:
                    intent = str(entry.get("intent", ""))
                found = _SEMANTIC.search(str(entry.get("summary", "")))
                if found:
                    record = session.memory.recall_fact(found.group(1))
                    reading = bool(record and record.get("negated"))
            intents.append(intent)
            readings.append(reading)
        store = {}
        for key in session.memory.fact_keys():
            record = session.memory.recall_fact(key)
            store[key] = (str(record.get("value", "")),
                          bool(record.get("negated")),
                          round(float(record.get("confidence", 0.0)), 2))
        return replies, intents, readings, store
    finally:
        T._SEMANTIC_POLARITY = previous
        shutil.rmtree(root, ignore_errors=True)


def _battery(negated: bool, polarity_on: bool) -> list:
    """Each case in its own session; returns the replies."""
    replies = []
    for subject, value, question in BATTERY:
        stated = (f"the {subject} is not {value}" if negated
                  else f"the {subject} is {value}")
        got, _i, _r, _s = _walk([stated, question], polarity_on,
                                semantic=True)
        replies.append(got[1])
    return replies


_ENTITIES = ["tower", "bridge", "keep", "mill", "spire", "granary"]
_MODIFIERS = ["east", "west", "old", "north", "stone"]
_VALUES = {
    "height": lambda r: f"{r.randrange(25, 900, 25)} meters",
    "material": lambda r: r.choice(["steel", "oak", "bronze", "granite"]),
    "colour": lambda r: r.choice(["red", "grey", "deep blue", "white"]),
    "weight": lambda r: f"{r.randrange(5, 90, 5)} tons",
    "roof": lambda r: r.choice(["slate", "thatch", "copper", "tile"]),
}


def _denial_world(seed: int, turns: int = 24) -> list:
    """Amendment A: statements negated at 0.35, modified questions."""
    rng = random.Random(seed)
    plan: list = []
    said: list = []
    for _ in range(turns):
        pick = rng.random()
        if pick < 0.36 or len(said) < 3:
            key = f"{rng.choice(_ENTITIES)} {rng.choice(list(_VALUES))}"
            value = _VALUES[key.split()[-1]](rng)
            deny = "not " if rng.random() < 0.35 else ""
            said.append((key, value))
            plan.append(f"the {key} is {deny}{value}")
        elif pick < 0.56:
            key, _value = rng.choice(said)
            plan.append(f"what is the {rng.choice(_MODIFIERS)} {key}?")
        elif pick < 0.72:
            key, _value = rng.choice(said)
            plan.append(f"what is the {key}?")
        elif pick < 0.84:
            key, value = rng.choice(said)
            plan.append(f"is the {key} {value}?")
        elif pick < 0.93:
            plan.append(f"what is the obelisk "
                        f"{rng.choice(list(_VALUES))}?")
        else:
            plan.append(rng.choice(["hello there", "thanks"]))
    return plan


@dataclass
class PolarityReport:
    """Whether the fix changed exactly what it claimed and nothing else.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        old_inverted: Battery cases the old arm inverted (validity: 10).
        new_inverted: Battery cases the fixed arm inverted (must be 0).
        new_denied: Battery cases the fixed arm spoke as a denial.
        positives_differing: Affirmed battery replies that moved.
        denial_turns: Turns compared in the denial-bearing worlds.
        changed: Turns whose reply differed across arms there.
        negated_readings: Turns answered by a semantic reading of a
            negated record there.
        spurious: Changed turns that were not such readings.
        missed: Such readings whose reply did not change.
        standard_turns: Turns compared in the §11.115 worlds.
        standard_replies: Replies differing there (must be 0).
        standard_intents: Intents differing there (must be 0).
        standard_stores: Stores differing there (must be 0).
        example: One before/after pair from the battery.
        reason: Plain-language verdict.
    """

    passes: bool
    old_inverted: int = 0
    new_inverted: int = 0
    new_denied: int = 0
    positives_differing: int = 0
    denial_turns: int = 0
    changed: int = 0
    negated_readings: int = 0
    spurious: int = 0
    missed: int = 0
    standard_turns: int = 0
    standard_replies: int = 0
    standard_intents: int = 0
    standard_stores: int = 0
    example: tuple = field(default_factory=tuple)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> PolarityReport:
    """Every criterion, both arms, one process."""
    from ultraquant.experiments.wiring_gate import _conversation

    # 1 - no inverted assertion, and the battery must reach the path.
    old = _battery(negated=True, polarity_on=False)
    new = _battery(negated=True, polarity_on=True)
    old_inverted = sum(
        f"{s} is {v}" in r and f"not {v}" not in r
        for (s, v, _q), r in zip(BATTERY, old))
    new_inverted = sum(f"{s} is {v}" in r
                       for (s, v, _q), r in zip(BATTERY, new))
    new_denied = sum(f"not {v}" in r for (_s, v, _q), r in zip(BATTERY, new))

    # 2 - positives untouched.
    positives_differing = sum(
        a != b for a, b in zip(_battery(False, False), _battery(False, True)))

    # 3 - changes are exactly the negated semantic readings.
    changed = negated_readings = spurious = missed = denial_turns = 0
    for seed in range(worlds):
        plan = _denial_world(seed, turns)
        r_old, _io, read_old, _so = _walk(plan, False, semantic=True)
        r_new, _in, read_new, _sn = _walk(plan, True, semantic=True)
        for i in range(len(plan)):
            denial_turns += 1
            moved = r_old[i] != r_new[i]
            flagged = read_old[i] is True
            changed += moved
            negated_readings += flagged
            spurious += moved and not flagged
            missed += flagged and not moved

    # 4 - nothing else moves in the §11.115 worlds, default session.
    s_turns = s_replies = s_intents = s_stores = 0
    for seed in range(worlds):
        plan = [text for text, _kind in _conversation(seed, turns)]
        r0, i0, _x0, st0 = _walk(plan, False, semantic=False)
        r1, i1, _x1, st1 = _walk(plan, True, semantic=False)
        s_turns += len(plan)
        s_replies += sum(a != b for a, b in zip(r0, r1))
        s_intents += sum(a != b for a, b in zip(i0, i1))
        s_stores += st0 != st1

    valid = old_inverted == len(BATTERY) and negated_readings > 0
    passes = (valid
              and new_inverted == 0 and new_denied == len(BATTERY)
              and positives_differing == 0
              and spurious == 0 and missed == 0
              and s_replies == 0 and s_intents == 0 and s_stores == 0)
    if not valid:
        reason = (f"VOID: old arm inverted {old_inverted}/{len(BATTERY)}, "
                  f"{negated_readings} negated readings in the worlds")
    elif passes:
        reason = (f"PASS: {len(BATTERY)}/{len(BATTERY)} denials kept "
                  f"(old arm inverted {old_inverted}); "
                  f"{changed} changed replies == {negated_readings} negated "
                  f"semantic readings; {s_turns} standard turns unmoved")
    else:
        reason = (f"FAIL: new inverted {new_inverted}, denied {new_denied}, "
                  f"positives moved {positives_differing}, spurious "
                  f"{spurious}, missed {missed}, standard replies "
                  f"{s_replies}/intents {s_intents}/stores {s_stores}")
    return PolarityReport(
        passes=passes, old_inverted=old_inverted, new_inverted=new_inverted,
        new_denied=new_denied, positives_differing=positives_differing,
        denial_turns=denial_turns, changed=changed,
        negated_readings=negated_readings, spurious=spurious, missed=missed,
        standard_turns=s_turns, standard_replies=s_replies,
        standard_intents=s_intents, standard_stores=s_stores,
        example=(old[0], new[0]), reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    print(f"  battery: old inverted {report.old_inverted}, new inverted "
          f"{report.new_inverted}, new denied {report.new_denied}")
    print(f"  positives moved: {report.positives_differing}")
    print(f"  denial worlds: {report.denial_turns} turns, changed "
          f"{report.changed}, negated readings {report.negated_readings}, "
          f"spurious {report.spurious}, missed {report.missed}")
    print(f"  standard worlds: {report.standard_turns} turns, replies "
          f"{report.standard_replies}, intents {report.standard_intents}, "
          f"stores {report.standard_stores}")
    print(f"  before: {report.example[0]}")
    print(f"  after : {report.example[1]}")
