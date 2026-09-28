"""A recovered turn meets the standard a stored fact does. The recovery gate.

GPT-6 Astra's review, finding 3, reproduced end to end by Claude
through real eviction before anything changed. The conversation
window (§11.14) pages evicted turns back in, and `_question`'s
recovered-turn rung spoke them as answers - "Earlier in this
conversation: K is V" - whenever the parsed key shared ANY informative
token with the question, from turns of any intent but a question.

- A **code** turn, "code: steel melting point = 9999", asked "what is
  the tungsten melting point?", came back "Earlier in this
  conversation: code: steel melting point is 9999." Wrong source and
  wrong subject in one sentence.
- A chat turn, "steel melting point = 1538", same question: "steel
  melting point is 1538" - the wrong metal, the exact error the §11.29
  coverage rule was built to stop for stored facts. The rung simply
  never had it.

**The flag is the arm.** `_RECOVERY_EVIDENCE = False` restores the old
rung byte for byte.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 a03fb745...), with one amendment also made before the fixed arm
ever ran (33497635...):

1. **No wrong-subject assertion.** Eight decoys: only a turn about
   subject B is recoverable, the question is about subject A sharing
   B's attribute words. Fixed arm: 0 replies assert B's statement as
   the answer. The OLD arm must assert all eight, or the battery is
   void. Calibrated on the old arm: 8 of 8.
2. **No non-statement evidence.** Four code and calc turns: the fixed
   arm quotes none of them in any form. The OLD arm must quote all
   four. Calibrated: 4 of 4, after Amendment A replaced a case the
   aggregate rung (correctly) answered first.
3. **Genuine recoveries survive.** Eight same-subject questions over
   evicted "K = V" turns: the fixed arm's reply is byte-identical to
   the old arm's on every one.
4. **Nothing else moves**: the §11.115 worlds, 0 replies, intents or
   stores differing.
5. **The full suite is green.** Checked outside this module.

Every case runs in its own session with a 64-byte resident window, so
the turn under test is genuinely evicted and genuinely paged back in
through `ContextWindow.recall` - nothing is injected.

**PASSED on all four measured criteria.**

| | old arm | fixed arm |
|---|---:|---:|
| wrong-subject decoys asserted | 8 of 8 | **0** |
| code and calc turns quoted | 4 of 4 | **0** |
| genuine recoveries, byte-identical | - | **8 of 8** |

The §11.115 worlds, 284 turns, moved nothing. A decoy now gets the
honest "I don't hold anything on that yet" - the tungsten question no
longer answers with steel's melting point, and a line of code is no
longer quoted back as something the user said.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CODE", "PAIRS", "RecoveryReport", "run_gate"]

FILLER = ["hello there", "thanks", "good morning", "hello again"]

#: (subject B's key, its value, subject A's key sharing B's attribute)
PAIRS = [("steel melting point", "1538", "tungsten melting point"),
         ("oak density", "750", "pine density"),
         ("north gate code", "4471", "south gate code"),
         ("blue team score", "12", "red team score"),
         ("copper wire gauge", "14", "silver wire gauge"),
         ("east tower height", "300", "west tower height"),
         ("main server port", "8080", "backup server port"),
         ("river depth", "12", "lake depth")]

#: (the code or calc turn, the key it mentions, the value)
CODE = [("code: steel melting point = 9999", "steel melting point", "9999"),
        ("calc: shipping fee = 42", "shipping fee", "42"),
        ("code: max retries = 5", "max retries", "5"),
        ("code: tower height = 300", "tower height", "300")]


def _ask(said: str, question: str, evidence_on: bool) -> str:
    from ultraquant.interpreter import thoughts as T
    from ultraquant.memory.context import ContextWindow

    if not hasattr(T, "_RECOVERY_EVIDENCE"):
        raise RuntimeError("thoughts._RECOVERY_EVIDENCE is missing: "
                           "the two arms would be the same code")
    previous = T._RECOVERY_EVIDENCE
    T._RECOVERY_EVIDENCE = evidence_on
    root = Path(tempfile.mkdtemp(prefix="uq_recovery_"))
    try:
        session = T.build_session(root, seed=0)
        session.context = ContextWindow(root / "context-small",
                                        budget_bytes=64)
        for line in [said] + FILLER:
            T.run_pipeline(line, session)
        return T.run_pipeline(question, session)[0]
    finally:
        T._RECOVERY_EVIDENCE = previous
        shutil.rmtree(root, ignore_errors=True)


def _asserts(reply: str, key: str) -> bool:
    return reply.startswith(f"Earlier in this conversation: {key} is")


def _quotes(reply: str, value: str) -> bool:
    return "Earlier in this conversation" in reply and value in reply


def _standard_worlds(worlds: int, turns: int) -> tuple:
    from ultraquant.experiments.wiring_gate import _conversation
    from ultraquant.interpreter import thoughts as T

    def walk(plan, flag):
        previous = T._RECOVERY_EVIDENCE
        T._RECOVERY_EVIDENCE = flag
        root = Path(tempfile.mkdtemp(prefix="uq_recovery_w_"))
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
            T._RECOVERY_EVIDENCE = previous
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
class RecoveryReport:
    """Whether recovered turns now meet the evidence standard.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        decoys_old: Decoys the old arm asserted (validity: 8).
        decoys_new: Decoys the fixed arm asserted (must be 0).
        code_old: Code turns the old arm quoted (validity: 4).
        code_new: Code turns the fixed arm quoted (must be 0).
        genuine_same: Genuine recoveries byte-identical across arms.
        standard: (turns, replies, intents, stores) differing.
        decoy_replies: The fixed arm's decoy replies, for the record.
        reason: Plain-language verdict.
    """

    passes: bool
    decoys_old: int = 0
    decoys_new: int = 0
    code_old: int = 0
    code_new: int = 0
    genuine_same: int = 0
    standard: tuple = ()
    decoy_replies: list = field(default_factory=list)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> RecoveryReport:
    """Every criterion, both arms, one process."""
    decoys_old = decoys_new = 0
    decoy_replies = []
    for key_b, value, key_a in PAIRS:
        said, question = f"{key_b} = {value}", f"what is the {key_a}?"
        decoys_old += _asserts(_ask(said, question, False), key_b)
        reply = _ask(said, question, True)
        decoys_new += _asserts(reply, key_b)
        decoy_replies.append(reply)

    code_old = code_new = 0
    for turn, key, value in CODE:
        question = f"what is the {key}?"
        code_old += _quotes(_ask(turn, question, False), value)
        code_new += _quotes(_ask(turn, question, True), value)

    genuine_same = 0
    for key, value, _a in PAIRS:
        said, question = f"{key} = {value}", f"what is the {key}?"
        genuine_same += (_ask(said, question, False)
                         == _ask(said, question, True))

    standard = _standard_worlds(worlds, turns)
    valid = decoys_old == len(PAIRS) and code_old == len(CODE)
    passes = (valid and decoys_new == 0 and code_new == 0
              and genuine_same == len(PAIRS)
              and standard[1:] == (0, 0, 0))
    if not valid:
        reason = (f"VOID: old arm asserted {decoys_old}/{len(PAIRS)} decoys "
                  f"and quoted {code_old}/{len(CODE)} code turns")
    elif passes:
        reason = (f"PASS: decoys asserted {decoys_old} -> 0, code turns "
                  f"quoted {code_old} -> 0, genuine recoveries "
                  f"{genuine_same}/{len(PAIRS)} identical, "
                  f"{standard[0]} standard turns unmoved")
    else:
        reason = (f"FAIL: decoys {decoys_new}, code {code_new}, genuine "
                  f"{genuine_same}/{len(PAIRS)}, standard {standard}")
    return RecoveryReport(
        passes=passes, decoys_old=decoys_old, decoys_new=decoys_new,
        code_old=code_old, code_new=code_new, genuine_same=genuine_same,
        standard=standard, decoy_replies=decoy_replies, reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    print(f"  decoys asserted: old {report.decoys_old}, new "
          f"{report.decoys_new}")
    print(f"  code quoted: old {report.code_old}, new {report.code_new}")
    print(f"  genuine identical: {report.genuine_same}/{len(PAIRS)}")
    print(f"  standard worlds: {report.standard}")
    for reply in report.decoy_replies:
        print(f"    decoy -> {reply[:96]}")
