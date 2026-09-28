"""A conjunction of polar questions is answered as polar questions.

GPT-6 Astra's review, finding 5, reproduced end to end by Claude before
anything changed. The compound rung runs before every specialised
handler and rewrote every part of a conjunction as a "what is"
question with its own reduced answer path - which cannot read a polar
part at all. With the tower iron and the bridge steel:

    "is the tower material iron and the bridge material steel?"
    -> "none of it is held. Still missing: is the tower material iron
       (unknown - ':learn' will ask for the iron iron); ..."

Both parts are held and both are true. The reply denies holding
either, and the learn queue gains curiosities named "iron iron" and
"steel steel". Calibration found it on every case in the battery - a
polar conjunction had no working path at all.

**The flag is the arm.** `_COMPOUND_POLAR = False` restores the old
rung byte for byte.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 6b5c6a56...) before the fix existed:

1. **Polar conjunctions answered per part.** Eight "is A x and B y?"
   questions over held facts, mixing true, false, denied and unheld
   parts. Fixed arm: every part takes the stance its single polar
   question takes alone, and no reply claims a held part is not held.
   The OLD arm must fail on every case. Calibrated: 8 of 8.
2. **No junk curiosities.** 0 queued by the battery in the fixed arm;
   the OLD arm must queue at least one per case. Calibrated: 8 of 8.
3. **Supported compounds unchanged.** Eight "what is A and B?"
   questions: replies byte-identical across arms.
4. **Nothing else moves**: the §11.115 worlds, 0 replies, intents or
   stores differing.
5. **The full suite is green.** Checked outside this module, and it
   includes tests/test_compound.py's pinned forms.

An explicit abstention on a conjunction also passes criterion 1,
provided it claims nothing false about what is held.

**PASSED on all four measured criteria** - by the second version.

| | old arm | fixed arm |
|---|---:|---:|
| polar conjunctions answered per part | 0 of 8 | **8 of 8** |
| cases queueing junk curiosities | 8 of 8 | **0** |
| "what is" compounds, byte-identical | - | **8 of 8** |

The §11.115 worlds, 284 turns, moved nothing. "is the tower material
iron and the bridge material steel?" now reads "Yes - tower material
is iron (confidence 0.60); Yes - bridge material is steel (confidence
0.60)."

**Version one passed too, and was not shipped.** Review probed polar
questions this battery does not contain and found the trigger - any
non-wh question holding " and ", checked first - capturing an "and"
that joins two VALUES: "is the tower material steel and iron?" had
answered "No - tower material is steel, not steel and iron" and now
abstained, and "is 7 the sum of 3 and 4?" was misdescribed as a
conjunction. This gate was left as frozen; the finding became its own
tests (tests/test_compoundpolar.py, failing on version one), and
GPT-6 Astra's second version falls through to the old path when a
later part is a bare continuation with no subject of its own. Probed
after: every value form byte-identical to before; "steel and cast
iron" - a two-word value, which the heuristic treats as a clause - now
abstains, where the old path claimed "none of it is held".
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CASES", "FACTS", "WHATS", "CompoundReport", "run_gate"]

FACTS = ["the tower material is iron", "the bridge material is steel",
         "the mill height is 300 meters", "the gate colour is not red"]

#: Polar conjunctions: two parts each.
CASES = [("the tower material iron", "the bridge material steel"),
         ("the tower material steel", "the bridge material steel"),
         ("the tower material iron", "the bridge material iron"),
         ("the tower material steel", "the bridge material iron"),
         ("the tower material iron", "the gate colour red"),
         ("the tower material iron", "the gate colour blue"),
         ("the mill height 300 meters", "the tower material iron"),
         ("the obelisk height 50 meters", "the tower material iron")]

#: Supported "what is" compounds, which must not move.
WHATS = ["what is the tower material and the bridge material?",
         "what is the tower material and the mill height?",
         "what is the bridge material and the gate colour?",
         "what is the mill height and the tower material?",
         "what is the tower material and the obelisk height?",
         "what is the gate colour and the bridge material?",
         "what is the mill height and the bridge material?",
         "what is the bridge material and the tower material?"]


def stance(reply: str) -> str:
    r = reply.strip()
    if r.startswith("Yes"):
        return "supported"
    if r.startswith("No"):
        return "contradicted"
    if r.startswith("I don't know") or r.startswith("I don't hold"):
        return "unknown"
    return "other"


def _run(lines: list, on: bool) -> tuple:
    from ultraquant.interpreter import thoughts as T

    if not hasattr(T, "_COMPOUND_POLAR"):
        raise RuntimeError("thoughts._COMPOUND_POLAR is missing: "
                           "the two arms would be the same code")
    previous = T._COMPOUND_POLAR
    T._COMPOUND_POLAR = on
    root = Path(tempfile.mkdtemp(prefix="uq_compound_"))
    try:
        session = T.build_session(root, seed=0)
        for fact in FACTS:
            T.run_pipeline(fact, session)
        before = len(session.curiosities)
        replies = [T.run_pipeline(line, session)[0] for line in lines]
        return replies, len(session.curiosities) - before
    finally:
        T._COMPOUND_POLAR = previous
        shutil.rmtree(root, ignore_errors=True)


def _case_ok(reply: str, singles: list) -> bool:
    """Every part takes its single question's stance; nothing false."""
    if "none of it is held" in reply:
        return False
    parts = [p.strip() for p in reply.split("; ")]
    if len(parts) == len(singles):
        return all(stance(p) == stance(s) for p, s in zip(parts, singles))
    # An explicit abstention passes if it claims nothing false.
    return reply.startswith("I can't answer that conjunction")


def _standard_worlds(worlds: int, turns: int) -> tuple:
    from ultraquant.experiments.wiring_gate import _conversation
    from ultraquant.interpreter import thoughts as T

    def walk(plan, on):
        previous = T._COMPOUND_POLAR
        T._COMPOUND_POLAR = on
        root = Path(tempfile.mkdtemp(prefix="uq_compound_w_"))
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
            T._COMPOUND_POLAR = previous
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
class CompoundReport:
    """Whether polar conjunctions now get polar answers.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        old_failed: Cases the old arm got wrong (validity: 8).
        new_ok: Cases the fixed arm got right (need 8).
        junk_old: Cases where the old arm queued a curiosity (need 8).
        junk_new: Curiosities the fixed arm queued (must be 0).
        whats_same: "what is" compounds unchanged (need 8).
        standard: (turns, replies, intents, stores) differing.
        replies: The fixed arm's replies, for the record.
        reason: Plain-language verdict.
    """

    passes: bool
    old_failed: int = 0
    new_ok: int = 0
    junk_old: int = 0
    junk_new: int = 0
    whats_same: int = 0
    standard: tuple = ()
    replies: list = field(default_factory=list)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> CompoundReport:
    """Every criterion, both arms, one process."""
    old_failed = new_ok = junk_old = junk_new = 0
    replies = []
    for a, b in CASES:
        question = f"is {a} and {b}?"
        singles, _ = _run([f"is {a}?", f"is {b}?"], False)
        (old,), queued_old = _run([question], False)
        (new,), queued_new = _run([question], True)
        old_failed += not _case_ok(old, singles)
        new_ok += _case_ok(new, singles)
        junk_old += queued_old > 0
        junk_new += queued_new
        replies.append(new)
    whats_old, _ = _run(WHATS, False)
    whats_new, _ = _run(WHATS, True)
    whats_same = sum(a == b for a, b in zip(whats_old, whats_new))
    standard = _standard_worlds(worlds, turns)
    n = len(CASES)
    valid = old_failed == n and junk_old == n
    passes = (valid and new_ok == n and junk_new == 0
              and whats_same == len(WHATS) and standard[1:] == (0, 0, 0))
    if not valid:
        reason = (f"VOID: old arm failed {old_failed}/{n}, queued junk on "
                  f"{junk_old}/{n}")
    elif passes:
        reason = (f"PASS: polar conjunctions {n - old_failed}/{n} -> "
                  f"{new_ok}/{n}; junk curiosities -> 0; 'what is' "
                  f"compounds {whats_same}/{len(WHATS)} unchanged; "
                  f"{standard[0]} standard turns unmoved")
    else:
        reason = (f"FAIL: ok {new_ok}/{n}, junk {junk_new}, whats "
                  f"{whats_same}/{len(WHATS)}, standard {standard}")
    return CompoundReport(passes=passes, old_failed=old_failed,
                          new_ok=new_ok, junk_old=junk_old,
                          junk_new=junk_new, whats_same=whats_same,
                          standard=standard, replies=replies, reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    for (a, b), reply in zip(CASES, report.replies):
        print(f"  is {a} and {b}?\n    -> {reply[:120]}")
