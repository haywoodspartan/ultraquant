"""A curiosity is revalidated before it is asked. The revalidation gate.

GPT-6 Astra's review, finding 10, reproduced end to end by Claude
before anything changed. A refused inference that knows which premise
it lacked leaves a curiosity (§11.33), and `:learn` asks for it - the
system asking to be taught exactly what it needs. But the survey read
the stored gap back verbatim, never checking it against the store:

- told "the steel conductivity is high" directly, `:learn` still asked
  "What is the steel conductivity?" - for something it now held;
- with the tower material revised to iron, the prompt still said "I
  hold that tower material is steel" - misstating its own belief to
  the person it was asking.

**The flag is the arm.** `_CURIOSITY_REVALIDATE = False` restores the
old survey byte for byte.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 5722b1c1...) before the fix existed:

1. **No fulfilled prompts.** Six gaps whose premise is then stated
   directly: the fixed arm asks for none of them. The OLD arm must
   ask for every one, or the battery is void. Calibrated: 6 of 6.
2. **No stale-bridge prompts.** Six gaps whose bridge is then revised:
   the fixed arm states no bridge value that is not currently held.
   The OLD arm must state the stale bridge in every one. Calibrated:
   6 of 6.
3. **Live gaps survive.** An unanswered, unchanged gap yields exactly
   one prompt, with the same text in both arms, and asking the
   question twice does not duplicate it.
4. **The teaching loop still closes.** Answering the prompt through
   `:learn answer` stores the premise and the gap is gone, identically
   in both arms (§11.33).
5. **Nothing else moves**: the §11.115 worlds, 0 replies, intents or
   stores differing.
6. **The full suite is green.** Checked outside this module.

**Amendment A** (sha256 0a5c16f7...), made after code review of the
fix and before its fixed arm ever ran: the first implementation
compared the bridge's value but not its polarity, and counted a
premise held only as a denial as fulfilled. The criteria stand; the
battery widens to two bridges revised to a denial of the same value
(criterion 2) and two premises held only as a denial, which remain
live and must be asked exactly once (criterion 3). Both calibrated on
the old arm first.

**PASSED on all five measured criteria.**

| | old arm | fixed arm |
|---|---:|---:|
| asks for a premise it now holds | 6 of 6 | **0** |
| states a bridge it no longer holds | 8 of 8 | **0** |

The eight stale bridges include the two polarity flips review found:
the first implementation compared values and let "the tower material
is not steel" through as "I hold that tower material is steel". Live
gaps held at one prompt each (6 of 6), a premise held only as a denial
is still asked exactly once (2 of 2), the `:learn` loop closed on 6 of
6 in both arms, and the §11.115 worlds moved nothing.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FLIPS", "GAPS", "CuriosityReport", "run_gate"]

#: (entity, material, property, value to teach, revised material)
GAPS = [("tower", "steel", "conductivity", "high", "iron"),
        ("bridge", "oak", "density", "low", "pine"),
        ("mill", "granite", "hardness", "high", "marble"),
        ("spire", "copper", "colour", "orange", "bronze"),
        ("gate", "bronze", "weight", "heavy", "brass"),
        ("keep", "limestone", "porosity", "high", "basalt")]


#: Amendment A: bridges revised to a denial, and premises held only as one.
FLIPS = GAPS[:2]


def _flag(on: bool):
    from ultraquant.interpreter import learning as L

    if not hasattr(L, "_CURIOSITY_REVALIDATE"):
        raise RuntimeError("learning._CURIOSITY_REVALIDATE is missing: "
                           "the two arms would be the same code")
    previous = L._CURIOSITY_REVALIDATE
    L._CURIOSITY_REVALIDATE = on
    return previous


def _unflag(previous: bool) -> None:
    from ultraquant.interpreter import learning as L

    L._CURIOSITY_REVALIDATE = previous


def _prompts(lines: list, on: bool) -> list:
    """Missing-premise prompts the survey offers after ``lines``."""
    from ultraquant.interpreter import thoughts as T
    from ultraquant.interpreter.learning import LearningSession

    previous = _flag(on)
    root = Path(tempfile.mkdtemp(prefix="uq_revalidate_"))
    try:
        session = T.build_session(root, seed=0)
        for line in lines:
            T.run_pipeline(line, session)
        return [q.prompt for q in LearningSession(session).survey()
                if q.kind == "missing-premise"]
    finally:
        _unflag(previous)
        shutil.rmtree(root, ignore_errors=True)


def _teach_through_learn(entity, material, prop, value, on: bool) -> tuple:
    """Answer the prompt via ChatCLI; return (premise held, prompts left)."""
    from ultraquant.interpreter import thoughts as T
    from ultraquant.interpreter.chat import ChatCLI
    from ultraquant.interpreter.learning import LearningSession

    previous = _flag(on)
    root = Path(tempfile.mkdtemp(prefix="uq_revalidate_l_"))
    try:
        session = T.build_session(root, seed=0)
        out = io.StringIO()
        cli = ChatCLI(session, out=out)
        for line in (f"the {entity} material is {material}",
                     f"what is the {entity} {prop}?", ":learn"):
            cli.handle(line, iter([]))
        for _ in range(12):
            question = cli.learning.next_question()
            if question is None:
                break
            if question.kind == "missing-premise":
                cli.handle(f":learn answer {value}", iter([]))
                break
            cli.handle(":learn skip", iter([]))
        record = session.memory.recall_fact(f"{material} {prop}")
        held = bool(record and str(record.get("value")) == value)
        left = [q for q in LearningSession(session).survey()
                if q.kind == "missing-premise"]
        return held, len(left)
    finally:
        _unflag(previous)
        shutil.rmtree(root, ignore_errors=True)


def _standard_worlds(worlds: int, turns: int) -> tuple:
    from ultraquant.experiments.wiring_gate import _conversation
    from ultraquant.interpreter import thoughts as T

    def walk(plan, on):
        previous = _flag(on)
        root = Path(tempfile.mkdtemp(prefix="uq_revalidate_w_"))
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
            _unflag(previous)
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
class CuriosityReport:
    """Whether the survey now asks only what is still worth asking.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        fulfilled_old: Fulfilled gaps the old arm still asked (need 6).
        fulfilled_new: The same in the fixed arm (must be 0).
        stale_old: Revised bridges the old arm misstated (need 6).
        stale_new: The same in the fixed arm (must be 0).
        live_same: Live gaps giving one identical prompt in both arms,
            also when asked twice (need 6).
        loop_closes: (old, new) gaps whose :learn answer stored the
            premise and cleared the prompt (need 6 and 6).
        standard: (turns, replies, intents, stores) differing.
        example: One stale-bridge prompt, before and after.
        reason: Plain-language verdict.
    """

    passes: bool
    fulfilled_old: int = 0
    fulfilled_new: int = 0
    stale_old: int = 0
    stale_new: int = 0
    live_same: int = 0
    loop_closes: tuple = ()
    standard: tuple = ()
    example: tuple = field(default_factory=tuple)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> CuriosityReport:
    """Every criterion, both arms, one process."""
    fulfilled_old = fulfilled_new = stale_old = stale_new = live_same = 0
    loop_old = loop_new = 0
    example: tuple = ()
    for entity, material, prop, value, revised in GAPS:
        base = [f"the {entity} material is {material}",
                f"what is the {entity} {prop}?"]
        asks = f"What is the {material} {prop}?"
        stale = f"I hold that {entity} material is {material};"

        told = base + [f"the {material} {prop} is {value}"]
        fulfilled_old += any(asks in p for p in _prompts(told, False))
        fulfilled_new += any(asks in p for p in _prompts(told, True))

        moved = base + [f"the {entity} material is {revised}"]
        before, after = _prompts(moved, False), _prompts(moved, True)
        stale_old += any(stale in p for p in before)
        stale_new += any(stale in p for p in after)
        if not example:
            example = (before, after)

        once_old, once_new = _prompts(base, False), _prompts(base, True)
        twice_new = _prompts(base + [f"what is the {entity} {prop}?"], True)
        live_same += (len(once_new) == 1 and once_old == once_new
                      and len(twice_new) == 1)

        held, left = _teach_through_learn(entity, material, prop, value, False)
        loop_old += held and left == 0
        held, left = _teach_through_learn(entity, material, prop, value, True)
        loop_new += held and left == 0

    flips_old = flips_new = denied_live = 0
    for entity, material, prop, _value, _revised in FLIPS:
        base = [f"the {entity} material is {material}",
                f"what is the {entity} {prop}?"]
        stale = f"I hold that {entity} material is {material};"
        flipped = base + [f"the {entity} material is not {material}"]
        flips_old += any(stale in p for p in _prompts(flipped, False))
        flips_new += any(stale in p for p in _prompts(flipped, True))
        asks = f"What is the {material} {prop}?"
        denied = base + [f"the {material} {prop} is not high"]
        denied_live += sum(asks in p for p in _prompts(denied, True)) == 1
    stale_old += flips_old
    stale_new += flips_new

    standard = _standard_worlds(worlds, turns)
    n = len(GAPS)
    valid = fulfilled_old == n and stale_old == n + len(FLIPS)
    passes = (valid and fulfilled_new == 0 and stale_new == 0
              and live_same == n and denied_live == len(FLIPS)
              and loop_old == n and loop_new == n
              and standard[1:] == (0, 0, 0))
    if not valid:
        reason = (f"VOID: old arm asked {fulfilled_old}/{n} fulfilled and "
                  f"misstated {stale_old}/{n + len(FLIPS)} bridges")
    elif passes:
        reason = (f"PASS: fulfilled asks {fulfilled_old} -> 0, stale "
                  f"bridges {stale_old} -> 0 (incl. {len(FLIPS)} polarity "
                  f"flips), live gaps {live_same}/{n} + denied premises "
                  f"{denied_live}/{len(FLIPS)} still asked once, "
                  f"unchanged, loop closes {loop_new}/{n}, "
                  f"{standard[0]} standard turns unmoved")
    else:
        reason = (f"FAIL: fulfilled {fulfilled_new}, stale {stale_new}, "
                  f"live {live_same}/{n}, denied-live {denied_live}/"
                  f"{len(FLIPS)}, loop {loop_old}/{loop_new}, "
                  f"standard {standard}")
    return CuriosityReport(
        passes=passes, fulfilled_old=fulfilled_old,
        fulfilled_new=fulfilled_new, stale_old=stale_old,
        stale_new=stale_new, live_same=live_same,
        loop_closes=(loop_old, loop_new), standard=standard,
        example=example, reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    print(f"  fulfilled asked: old {report.fulfilled_old}, new "
          f"{report.fulfilled_new}")
    print(f"  stale bridges: old {report.stale_old}, new {report.stale_new}")
    print(f"  live gaps unchanged: {report.live_same}")
    print(f"  loop closes (old, new): {report.loop_closes}")
    print(f"  standard worlds: {report.standard}")
    print(f"  stale example before: {report.example[0]}")
    print(f"  stale example after : {report.example[1]}")
