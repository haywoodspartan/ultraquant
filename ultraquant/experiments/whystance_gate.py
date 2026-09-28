"""A why-question agrees with the polar question inside it. The why-stance gate.

Found by GPT-6 Astra's review (its finding 2) and reproduced end to end
by Claude before anything changed: with only "the tower material is not
steel" held, "is the tower material iron?" answered "I don't know" and
"why is the tower material iron?" answered "It isn't". The same belief,
asked two ways, got two stances - and the why-question's was the §11.48
error, a denial of one value read as a denial of another.

**Why it happened.** The polar handler compares a claim with what is
held three ways: supported, contradicted, unknown. The why handler
compared two ways - agree, or "It isn't" - for a stored fact and again,
separately, for a derivation. Two copies of a rule that has to agree
had drifted, which is the §11.94 lesson in a new place.

**Wider than reported.** Laying out the whole matrix for this gate
found a second face of the same defect: holding "steel", "why is the
tower material not iron?" answered "It isn't - tower material is steel"
- a TRUE claim denied - where polar says "Yes - steel, not iron".

**The flag is the arm.** `_WHY_THREE_WAY = False` restores the two-way
test byte for byte.

**The matrix.** Held {affirmed V, denied V} x claim {V, W, not V,
not W}, once for a STORED fact and once for a fact DERIVED on the spot
from a chain - sixteen cells, each in its own fresh session so no
pending state crosses between them. Stance is read from the reply:
why "Because..." is supported, "It isn't" contradicted, "I don't know"
/ "I hold only" unknown; polar "Yes", "No", "I don't know" likewise.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 ea9fd0b6...) before the fix existed:

1. **Why agrees with polar on every cell** of the fixed arm - 16 of 16.
2. **The old arm disagrees exactly where the two-way test is wrong** -
   predicted from the code: affirmed + "not W", denied + "W", denied +
   "not W", stored and derived alike. If the old arm agrees on those
   cells the matrix is not reaching the defect and the gate is void.
   Calibrated against the old code only: exactly those 6 of 16.
3. **Anti-rationalisation holds**: 0 cells in either arm where why says
   "Because..." for a claim polar does not call supported.
4. **Nothing else moves**: the §11.47 why gate reproduces its recorded
   numbers (why 1.000, recall 1.000, 0 rationalised, 0 fabricated), and
   the §11.115 worlds show 0 replies, intents or stores differing.
5. **The full suite is green.** Checked outside this module.

**PASSED on all four measured criteria.**

| arm | why agrees with polar | "Because" for an unsupported claim |
|---|---:|---:|
| two-way (old) | 10 of 16 | 0 |
| three-way (fixed) | **16 of 16** | **0** |

The old arm disagreed on exactly the six predicted cells and no
others. Under the fix the §11.47 why gate reproduced its record -
why 1.000, recall 1.000, 0 rationalised, 0 fabricated - and the
§11.115 worlds, 284 turns, moved nothing.

**The reference was checked too.** The fix moved polar onto the shared
helper as well, and that is not behind the flag, so neither arm runs
the old polar code and the matrix alone could not see a polar
regression. Polar's sixteen replies were therefore byte-compared
against the pre-fix code itself: 0 of 16 differ.

Where the old arm said "It isn't" to a claim it could not judge, the
fix now says "I don't know that it is - I hold only that tower
material is not steel"; where it denied a true claim, "why is the
tower material not iron?" holding steel, it now explains it.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PREDICTED_DEFECT", "WhyStanceReport", "matrix", "run_gate"]

STORED = {"affirmed": ["the tower material is steel"],
          "denied": ["the tower material is not steel"]}
DERIVED = {"affirmed": ["the tower material is steel",
                        "the steel hardness is high"],
           "denied": ["the tower material is steel",
                      "the steel hardness is not high"]}
STORED_CLAIMS = [
    ("V", "why is the tower material steel?", "is the tower material steel?"),
    ("W", "why is the tower material iron?", "is the tower material iron?"),
    ("not V", "why is the tower material not steel?",
     "is the tower material not steel?"),
    ("not W", "why is the tower material not iron?",
     "is the tower material not iron?"),
]
DERIVED_CLAIMS = [
    ("V", "why is the tower hardness high?", "is the tower hardness high?"),
    ("W", "why is the tower hardness low?", "is the tower hardness low?"),
    ("not V", "why is the tower hardness not high?",
     "is the tower hardness not high?"),
    ("not W", "why is the tower hardness not low?",
     "is the tower hardness not low?"),
]

#: Criterion 2's prediction, read off the two-way test before any run.
PREDICTED_DEFECT = frozenset({
    ("stored", "affirmed", "not W"), ("stored", "denied", "W"),
    ("stored", "denied", "not W"), ("derived", "affirmed", "not W"),
    ("derived", "denied", "W"), ("derived", "denied", "not W"),
})


def why_stance(reply: str) -> str:
    r = reply.strip()
    if r.startswith("Because"):
        return "supported"
    if r.startswith("It isn't"):
        return "contradicted"
    if "I don't know" in r or "I hold only" in r or "I don't hold" in r:
        return "unknown"
    return "other"


def polar_stance(reply: str) -> str:
    r = reply.strip()
    if r.startswith("Yes"):
        return "supported"
    if r.startswith("No"):
        return "contradicted"
    if r.startswith("I don't know"):
        return "unknown"
    return "other"


def _ask(facts: list, question: str) -> str:
    from ultraquant.interpreter import thoughts as T

    root = Path(tempfile.mkdtemp(prefix="uq_whystance_"))
    try:
        session = T.build_session(root, seed=0)
        for fact in facts:
            T.run_pipeline(fact, session)
        return T.run_pipeline(question, session)[0]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def matrix(three_way: bool) -> list:
    """Every cell under one arm, as dicts."""
    from ultraquant.interpreter import thoughts as T

    if not hasattr(T, "_WHY_THREE_WAY"):
        raise RuntimeError("thoughts._WHY_THREE_WAY is missing: "
                           "the two arms would be the same code")
    previous = T._WHY_THREE_WAY
    T._WHY_THREE_WAY = three_way
    try:
        cells = []
        for kind, states, claims in (("stored", STORED, STORED_CLAIMS),
                                     ("derived", DERIVED, DERIVED_CLAIMS)):
            for held, facts in states.items():
                for label, why_q, polar_q in claims:
                    why = _ask(facts, why_q)
                    polar = _ask(facts, polar_q)
                    cells.append({
                        "cell": (kind, held, label), "why": why,
                        "why_stance": why_stance(why), "polar": polar,
                        "polar_stance": polar_stance(polar)})
        return cells
    finally:
        T._WHY_THREE_WAY = previous


def _standard_worlds(worlds: int, turns: int) -> tuple:
    """§11.115's worlds through both arms: replies, intents, stores."""
    from ultraquant.experiments.wiring_gate import _conversation
    from ultraquant.interpreter import thoughts as T

    def walk(plan, flag):
        previous = T._WHY_THREE_WAY
        T._WHY_THREE_WAY = flag
        root = Path(tempfile.mkdtemp(prefix="uq_whystance_w_"))
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
            T._WHY_THREE_WAY = previous
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
class WhyStanceReport:
    """Whether why now takes polar's stance, and nothing else moved.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        new_agree: Fixed-arm cells where why and polar agree (need 16).
        old_disagree: The cells the old arm disagreed on.
        rationalised: "Because" for a claim polar does not support, in
            either arm (need 0).
        why_gate: The §11.47 gate's numbers under the fixed arm.
        standard: (turns, replies, intents, stores) differing.
        cells: The fixed arm's cells, for the record.
        reason: Plain-language verdict.
    """

    passes: bool
    new_agree: int = 0
    old_disagree: frozenset = frozenset()
    rationalised: int = 0
    why_gate: dict = field(default_factory=dict)
    standard: tuple = ()
    cells: list = field(default_factory=list)
    reason: str = ""


def run_gate(worlds: int = 10, turns: int = 24) -> WhyStanceReport:
    """Every criterion, both arms, one process."""
    from ultraquant.experiments import why_gate
    from ultraquant.interpreter import thoughts as T

    old = matrix(False)
    new = matrix(True)
    new_agree = sum(c["why_stance"] == c["polar_stance"] for c in new)
    old_disagree = frozenset(c["cell"] for c in old
                             if c["why_stance"] != c["polar_stance"])
    rationalised = sum(c["why_stance"] == "supported"
                       and c["polar_stance"] != "supported"
                       for c in old + new)

    previous = T._WHY_THREE_WAY
    T._WHY_THREE_WAY = True
    try:
        wg = why_gate.run_gate()
    finally:
        T._WHY_THREE_WAY = previous
    why_numbers = {"why": wg.why.get("answers"),
                   "recall": wg.recall.get("answers"),
                   "rationalised": wg.rationalised,
                   "fabricated": wg.fabricated, "passes": wg.passes}
    standard = _standard_worlds(worlds, turns)

    valid = old_disagree == PREDICTED_DEFECT
    why_holds = (wg.passes and why_numbers["why"] == 1.0
                 and why_numbers["recall"] == 1.0
                 and wg.rationalised == 0 and wg.fabricated == 0)
    passes = (valid and new_agree == len(new) and rationalised == 0
              and why_holds and standard[1:] == (0, 0, 0))
    if not valid:
        reason = (f"VOID: old arm disagreed on {sorted(old_disagree)}, "
                  f"predicted {sorted(PREDICTED_DEFECT)}")
    elif passes:
        reason = (f"PASS: why agrees with polar on {new_agree}/{len(new)} "
                  f"cells (old arm: {len(new) - len(old_disagree)}/"
                  f"{len(new)}); 0 rationalised; why gate held; "
                  f"{standard[0]} standard turns unmoved")
    else:
        reason = (f"FAIL: agree {new_agree}/{len(new)}, rationalised "
                  f"{rationalised}, why gate {why_numbers}, standard "
                  f"{standard}")
    return WhyStanceReport(
        passes=passes, new_agree=new_agree, old_disagree=old_disagree,
        rationalised=rationalised, why_gate=why_numbers, standard=standard,
        cells=new, reason=reason)


if __name__ == "__main__":
    report = run_gate()
    print(report.reason)
    print(f"  old arm disagreed on: {sorted(report.old_disagree)}")
    print(f"  why gate under the fix: {report.why_gate}")
    print(f"  standard worlds (turns, replies, intents, stores): "
          f"{report.standard}")
    for cell in report.cells:
        mark = "  " if cell["why_stance"] == cell["polar_stance"] else "XX"
        print(f"  {mark} {' '.join(cell['cell']):22} "
              f"why={cell['why_stance']:12} | {cell['why'][:78]}")
