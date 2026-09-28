"""A "yes" confirms what was asserted. The confirmation gate.

GPT-6 Astra's review, finding 4, reproduced end to end by Claude
through the real chat surface before anything changed. A polar "Yes"
leaves a one-turn slot (§11.68) so the user's next "yes" can confirm
the belief as direct testimony, and a derivation leaves one (§11.31)
so "yes" can consolidate it. Both slots stored a KEY, and "yes" acted
on whatever that key held at the time. Commands bypass Perceive, which
is what clears the slots, so a command between the assertion and the
"yes" could change the belief underneath it:

- "is the tower material iron?" -> "Yes"; `:learn answer steel`;
  "yes" -> "Confirmed: tower material is steel - direct testimony,
  confidence 0.90". The user affirmed iron.
- `:learn answer no` marks the gate colour unconfirmed; "yes" ->
  "Confirmed: gate colour is (unconfirmed) - direct testimony". The
  system confirmed its own placeholder on the user's authority.
- A derivation's premise revised before "yes" -> "Consolidated: tower
  hardness is high" - a belief its own premises no longer support.

**The flag is the arm.** `_CONFIRM_SNAPSHOT = False` restores the old
slots byte for byte.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 c2b01814...) before the fix existed:

1. **No stale confirmation.** Six cases - four stored facts, two
   derivations - with a `:learn` answer between the assertion and the
   "yes". Fixed arm: 0 confirmations or consolidations of anything but
   what was asserted. The OLD arm must confirm the changed belief in
   all six, or the battery is void. Calibrated: 6 of 6.
2. **Immediate confirmations unchanged.** The same cases with nothing
   between: replies and final stores byte-identical across arms.
3. **One-turn freshness preserved.** An ordinary turn between the
   assertion and the "yes" still clears the slot: identical across
   arms.
4. **Nothing else moves**: the §11.115 worlds, 0 replies, intents or
   stores differing.
5. **The full suite is green.** Checked outside this module.

**PASSED on all four measured criteria.**

| | old arm | fixed arm |
|---|---:|---:|
| stale "yes" confirmed or consolidated | 6 of 6 | **0** |
| immediate "yes", reply and store identical | - | **6 of 6** |
| ordinary turn between, identical | - | **6 of 6** |

The §11.115 worlds, 284 turns, moved nothing. A stale "yes" now says
so - "That changed since I said it - gate colour is now (unconfirmed).
Nothing was confirmed." - and a derivation whose premise moved says
which premise, and consolidates nothing.

**The cost, measured and isolated.** Snapshotting a derivation's
premises reads each premise once more when the derivation is made: in
the §11.115 worlds, store reads per turn rose 16.22 -> 16.34 in both of
that gate's arms (its saving unchanged at +3.51), and an isolated copy
without this change reproduced 16.22 exactly. The premise tuples
already carry values; only polarity forced the read, so recording
polarity where inference already holds the record would remove it.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DERIVED", "STORED", "ConfirmReport", "run_gate"]

#: (fact, polar question, key, the :learn answer that changes it)
STORED = [("the tower material is iron", "is the tower material iron?",
           "tower material", "steel"),
          ("the bridge material is oak", "is the bridge material oak?",
           "bridge material", "pine"),
          ("the mill height is 300 meters", "is the mill height 300 meters?",
           "mill height", "450 meters"),
          ("the gate colour is red", "is the gate colour red?",
           "gate colour", "no")]

#: (premises, the question that derives, premise to change, the answer,
#: the derived key)
DERIVED = [(["the tower material is steel", "the steel hardness is high"],
            "what is the tower hardness?", "steel hardness", "low",
            "tower hardness"),
           (["the bridge material is oak", "the oak density is low"],
            "what is the bridge density?", "oak density", "high",
            "bridge density")]


class _Chat:
    """One session behind the real chat surface, under one arm."""

    def __init__(self, snapshot_on: bool) -> None:
        from ultraquant.interpreter import thoughts as T
        from ultraquant.interpreter.chat import ChatCLI

        if not hasattr(T, "_CONFIRM_SNAPSHOT"):
            raise RuntimeError("thoughts._CONFIRM_SNAPSHOT is missing: "
                               "the two arms would be the same code")
        self.T = T
        self.previous = T._CONFIRM_SNAPSHOT
        T._CONFIRM_SNAPSHOT = snapshot_on
        self.root = Path(tempfile.mkdtemp(prefix="uq_confirm_"))
        self.session = T.build_session(self.root, seed=0)
        self.out = io.StringIO()
        self.cli = ChatCLI(self.session, out=self.out)

    def say(self, line: str) -> str:
        mark = self.out.tell()
        self.cli.handle(line, iter([]))
        return self.out.getvalue()[mark:].strip()

    def learn_answer(self, subject: str, answer: str) -> str:
        """Skip, with real commands, to the question about ``subject``."""
        self.say(":learn")
        for _ in range(12):
            learning = getattr(self.cli, "learning", None)
            question = learning.next_question() if learning else None
            if question is None:
                return "no such question"
            if question.subject == subject:
                return self.say(f":learn answer {answer}")
            self.say(":learn skip")
        return "not reached"

    def held(self, key: str):
        record = self.session.memory.recall_fact(key)
        if record is None:
            return None
        return (str(record.get("value")), bool(record.get("negated")),
                round(float(record.get("confidence", 0.0)), 2))

    def close(self) -> None:
        self.T._CONFIRM_SNAPSHOT = self.previous
        shutil.rmtree(self.root, ignore_errors=True)


def _stored(case: tuple, on: bool, between: str | None) -> tuple:
    """``between``: None, "learn" (the :learn answer) or "turn"."""
    fact, polar, key, answer = case
    chat = _Chat(on)
    try:
        chat.say(fact)
        chat.say(polar)
        if between == "learn":
            chat.learn_answer(key, answer)
        elif between == "turn":
            chat.say("hello there")
        yes = chat.say("yes")
        return yes, chat.held(key)
    finally:
        chat.close()


def _derived(case: tuple, on: bool, between: str | None) -> tuple:
    premises, question, premise_key, answer, derived_key = case
    chat = _Chat(on)
    try:
        for premise in premises:
            chat.say(premise)
        chat.say(question)
        if between == "learn":
            chat.learn_answer(premise_key, answer)
        elif between == "turn":
            chat.say("hello there")
        yes = chat.say("yes")
        return yes, chat.held(derived_key)
    finally:
        chat.close()


def _stale_stored(yes: str, held, case: tuple) -> bool:
    """Did "yes" confirm a belief other than the one asserted?"""
    _fact, _polar, key, answer = case
    changed = "(unconfirmed)" if answer == "no" else answer
    return (yes.startswith("Confirmed:") and changed in yes) or bool(
        held and held[0] == changed and held[2] >= 0.9)


def _stale_derived(yes: str, held) -> bool:
    """Did "yes" consolidate a derivation whose premise had moved?"""
    return yes.startswith("Consolidated:") or held is not None


def _standard_worlds(worlds: int, turns: int) -> tuple:
    from ultraquant.experiments.wiring_gate import _conversation
    from ultraquant.interpreter import thoughts as T

    def walk(plan, on):
        previous = T._CONFIRM_SNAPSHOT
        T._CONFIRM_SNAPSHOT = on
        root = Path(tempfile.mkdtemp(prefix="uq_confirm_w_"))
        try:
            session = T.build_session(root, seed=0)
            replies, intents = [], []
            for text in plan:
                reply, trace = T.run_pipeline(text, session)
                replies.append(reply)
                intents.append(next((str(e.get("intent", ""))
                                     for e in trace
                                     if e.get("thought") == "Perceive"), ""))
            store = {}
            for key in session.memory.fact_keys():
                rec = session.memory.recall_fact(key)
                store[key] = (str(rec.get("value", "")),
                              bool(rec.get("negated")),
                              round(float(rec.get("confidence", 0.0)), 2))
            return replies, intents, store
        finally:
            T._CONFIRM_SNAPSHOT = previous
            shutil.rmtree(root, ignore_errors=True)

    n = replies = intents = stores = 0
    for seed in range(worlds):
        plan = [text for text, _kind in _conversation(seed, turns)]
        r0, i0, s0 = walk(plan, False)
        r1, i1, s1 = walk(plan, True)
        n += len(plan)
        replies += sum(a != b for a, b in zip(r0, r1))
        intents += sum(a != b for a, b in zip(i0, i1))
        stores += s0 != s1
    return n, replies, intents, stores


@dataclass
class ConfirmReport:
    """Whether "yes" now confirms only what was asserted.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        stale_old: Cases where the old arm confirmed the changed belief
            (validity: all six).
        stale_new: The same in the fixed arm (must be 0).
        immediate_same: Cases with nothing between whose reply and store
            are identical across arms (need six).
        fresh_same: Cases with an ordinary turn between, identical across
            arms (need six).
        standard: (turns, replies, intents, stores) differing.
        stale_replies: The fixed arm's replies to the stale "yes".
        reason: Plain-language verdict.
    """

    passes: bool
    stale_old: int = 0
    stale_new: int = 0
    immediate_same: int = 0
    fresh_same: int = 0
    standard: tuple = ()
    stale_replies: list = field(default_factory=list)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> ConfirmReport:
    """Every criterion, both arms, one process."""
    stale_old = stale_new = immediate_same = fresh_same = 0
    stale_replies = []
    for case in STORED:
        yes, held = _stored(case, False, "learn")
        stale_old += _stale_stored(yes, held, case)
        yes, held = _stored(case, True, "learn")
        stale_new += _stale_stored(yes, held, case)
        stale_replies.append(yes)
        immediate_same += (_stored(case, False, None)
                           == _stored(case, True, None))
        fresh_same += _stored(case, False, "turn") == _stored(case, True,
                                                               "turn")
    for case in DERIVED:
        yes, held = _derived(case, False, "learn")
        stale_old += _stale_derived(yes, held)
        yes, held = _derived(case, True, "learn")
        stale_new += _stale_derived(yes, held)
        stale_replies.append(yes)
        immediate_same += (_derived(case, False, None)
                           == _derived(case, True, None))
        fresh_same += _derived(case, False, "turn") == _derived(case, True,
                                                                 "turn")
    standard = _standard_worlds(worlds, turns)
    n = len(STORED) + len(DERIVED)
    valid = stale_old == n
    passes = (valid and stale_new == 0 and immediate_same == n
              and fresh_same == n and standard[1:] == (0, 0, 0))
    if not valid:
        reason = f"VOID: old arm confirmed the changed belief in {stale_old}/{n}"
    elif passes:
        reason = (f"PASS: stale confirmations {stale_old} -> 0; immediate "
                  f"{immediate_same}/{n} and fresh-turn {fresh_same}/{n} "
                  f"unchanged; {standard[0]} standard turns unmoved")
    else:
        reason = (f"FAIL: stale {stale_new}, immediate {immediate_same}/{n}, "
                  f"fresh {fresh_same}/{n}, standard {standard}")
    return ConfirmReport(passes=passes, stale_old=stale_old,
                         stale_new=stale_new, immediate_same=immediate_same,
                         fresh_same=fresh_same, standard=standard,
                         stale_replies=stale_replies, reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    print(f"  stale: old {report.stale_old}, new {report.stale_new}")
    print(f"  immediate unchanged {report.immediate_same}, fresh-turn "
          f"unchanged {report.fresh_same}")
    print(f"  standard worlds: {report.standard}")
    for reply in report.stale_replies:
        print(f"    stale yes -> {reply[:100]}")
