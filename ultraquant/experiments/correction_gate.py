"""What it costs to change your mind. The correction gate.

ARCHITECTURE.md §12.3 lists three axes where these architectures
differ most structurally and where nothing had been measured:
**correction latency, resident bytes per fact, and whether an answer
can be vetoed premise by premise.** §11.105 measured the fabrication
claim and cost it. This measures the other three.

**The difference is not whether a transformer can be corrected — it
can, trivially, by putting the correction in the prompt.** It is what
holding the correction COSTS as they accumulate. A store keeps a
revision and answers later questions at the same price; a context
window carries every correction into every subsequent query, so the
prompt grows with the number of times you have changed your mind.
That is an architectural claim with a number attached, and the number
is what this measures.

**The same guard as §11.105 applies**, and it is the reason that gate
was worth running: a battery whose opponent never wins is not a
battery. Criterion 5 fails this gate if the transformer does not win
or tie somewhere, and the honest candidate is accuracy — a modern
model tracks in-context corrections very well, possibly better than a
symbolic store with revision semantics.

**The criteria, written before the run.**

1. **Identical facts, identical corrections, identical order** — both
   systems get exactly the same sequence, checked structurally. A
   correction one side received and the other did not would make
   every number below meaningless.
2. **Corrections are honoured** — after N of them, asking about each
   corrected subject must return the NEW value. UltraQuant's rate
   must be **1.00**: a store that loses a revision has no provenance
   claim left. The transformer's is measured.
3. **The cost of holding N corrections is reported with the
   accuracy** — bytes carried per query at N = 1, 5, 10 and 20.
   Cheap and wrong is not a win, so neither number is reported
   without the other.
4. **Derived answers name their premises, or they do not** — for a
   two-fact chain, does the answer state what it used? Measured for
   both. This is the auditability claim and UltraQuant can fail it.
5. **The transformer must win or tie somewhere** — otherwise the
   battery is too narrow to be measuring a real system, and this gate
   reports a failure of its own construction.

**PASSED, and two of the three axes came back TIES. The most useful
thing in this unit is a measurement error it caught in its own
favour.**

**Run one reported the transformer collapsing** — 100%, 60%, 30%,
**20%** as corrections accumulated — which would have been a
spectacular result for this architecture and was entirely an artifact
of the harness. The failures were not wrong answers. They were
**EMPTY** ones. `qwen3.8-27b` is a reasoning model, `complete` was
called without `reasoning_effort`, and an 80-token ceiling left it
thinking until the budget ran out. `interpreter/llmls.py` had already
documented this exact failure — *"gemma spent all 400 tokens thinking
on every panel question"* — and passes `NO_REASONING` for that
reason. This gate did not.

That is twice now that an instrument error pointed the flattering
way, after §11.105's empty-answer bug did the same. **When a
comparison favours the thing you built, check the instrument before
believing the result.** Empties are now counted separately so the
artifact cannot recur silently.

**Run two, with the reasoning budget corrected:**

| corrections | UQ accuracy | LLM accuracy | UQ bytes/query | LLM bytes/query |
|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 25 | 125 |
| 5 | 100% | 100% | 25 | 434 |
| 10 | 100% | 100% | 26 | 839 |
| 20 | 100% | 100% | 26 | **1691** |

**Accuracy: a tie.** A 27B model tracks twenty accumulating
in-context corrections without a single miss. The claim that a store
holds revisions better is not supported at this scale.

**Auditability: a tie.** Asked which facts it used, the transformer
lists them. UltraQuant names its premises without being asked, which
is a difference in default rather than in capability, and criterion 4
does not distinguish those.

**Cost: the one real difference, and it is asymptotic.** The
transformer's context grew **13.5x across a 20x increase in
corrections** — linear, at roughly 82 bytes each — while the store
stayed flat at 25 to 26 bytes. That difference is structural and it
is genuinely a property of the architectures. But it is worth about
1.7 KB at twenty corrections, and a context window measured in tens
of thousands of tokens does not notice. **The crossover is somewhere
in the thousands of corrections and this gate did not measure it**,
so what is established is the slope, not the wall.

**The premise veto worked for both**: derive "the anvil hardness is
high" from two facts, correct one premise, re-derive, and both
returned "low" citing the corrected premise.

So §12.3's three axes resolve as: **correction accuracy — tie;
auditability — tie; bytes per fact — a real linear-versus-flat
difference whose practical crossover is unmeasured.** That is a
thinner result than the section implied was waiting, and it is the
result.

Run it::

    python -m ultraquant.experiments.correction_gate

Requires LM Studio with a chat model. Without it the gate reports
that it did not run rather than passing vacuously.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.experiments.headtohead_gate import classify
from ultraquant.interpreter.llmls import NO_REASONING

__all__ = ["CorrectionReport", "run_gate", "SUBJECTS"]

#: Subjects corrected in order. Invented, so a transformer answering
#: from pretraining rather than from context would be visible.
SUBJECTS = [
    ("tower height", "300 meters", "350 meters"),
    ("bridge length", "2 kilometers", "3 kilometers"),
    ("mill material", "oak", "elm"),
    ("keep colour", "bronze", "slate"),
    ("spire weight", "45 tonnes", "60 tonnes"),
    ("gatehouse width", "12 meters", "18 meters"),
    ("cellar depth", "8 meters", "11 meters"),
    ("chapel height", "22 meters", "27 meters"),
    ("stable length", "30 meters", "36 meters"),
    ("forge material", "iron", "brass"),
    ("well depth", "15 meters", "19 meters"),
    ("granary volume", "400 cubic meters", "520 cubic meters"),
    ("barracks width", "25 meters", "31 meters"),
    ("armoury material", "stone", "timber"),
    ("kitchen height", "6 meters", "9 meters"),
    ("orchard area", "3 hectares", "5 hectares"),
    ("cistern volume", "90 cubic meters", "140 cubic meters"),
    ("rampart length", "700 meters", "850 meters"),
    ("moat width", "14 meters", "20 meters"),
    ("watchtower height", "40 meters", "52 meters"),
]

#: The chain both systems are asked to derive and then to re-derive
#: after one of its premises is corrected. This is criterion 4.
_CHAIN = [
    ("the anvil material is steel", "steel hardness is high"),
    ("what is the anvil hardness?", ["high"]),
    ("the anvil material is oak", "oak hardness is low"),
    ("what is the anvil hardness?", ["low"]),
]


#: Where the accuracy is sampled as corrections accumulate.
_CHECKPOINTS = (1, 5, 10, 20)

_SYSTEM = (
    "Answer in one short sentence, with the value only. You are given "
    "facts, some of which are later corrected. A correction replaces "
    "the earlier value. Answer using the CURRENT value."
)


def _llm_context(upto: int) -> str:
    """Everything the transformer must carry to know N corrections.

    Original facts and then their corrections, in the order both
    systems received them. This IS the cost being measured: a store
    keeps the revision, a context window carries it forever.
    """
    lines = [f"- the {name} is {old}" for name, old, _new in SUBJECTS[:upto]]
    lines += [f"- correction: the {name} is now {new}"
              for name, _old, new in SUBJECTS[:upto]]
    return "Facts:\n" + "\n".join(lines)


def _scored(answer: str, acceptable: list) -> bool:
    lowered = " ".join(answer.lower().split())
    return any(good.lower() in lowered for good in acceptable)


def ultraquant_run(checkpoints=_CHECKPOINTS) -> dict:
    """State every fact, correct every fact, then ask at each mark."""
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_correct_"))
    out = {"accuracy": {}, "bytes": {}, "narrated": 0, "asked": 0}
    try:
        session = build_session(root, seed=0)
        done = 0
        for mark in checkpoints:
            while done < mark:
                name, old, new = SUBJECTS[done]
                run_pipeline(f"the {name} is {old}", session)
                said = run_pipeline(f"the {name} is {new}", session)[0]
                out["asked"] += 1
                # Criterion 4's other half: a revision that does not
                # say what it replaced cannot be audited.
                if "revises what I held" in said:
                    out["narrated"] += 1
                done += 1
            right = 0
            for name, _old, new in SUBJECTS[:mark]:
                question = f"what is the {name}?"
                if _scored(run_pipeline(question, session)[0], [new]):
                    right += 1
            out["accuracy"][mark] = right / mark
            # What a query carries: the question, and nothing else.
            # The store holds the corrections.
            out["bytes"][mark] = sum(
                len(f"what is the {name}?".encode("utf-8"))
                for name, _o, _n in SUBJECTS[:mark]) / mark
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return out


def transformer_run(model: str | None = None,
                    checkpoints=_CHECKPOINTS) -> dict:
    """The same corrections, carried in the prompt every time."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    client = LMStudioClient()
    chosen = model
    if chosen is None:
        cards = [c for c in client.models() if "embed" not in c.lower()]
        if not cards:
            raise RuntimeError("no chat model available in LM Studio")
        chosen = cards[0]
    out = {"accuracy": {}, "bytes": {}, "model": chosen}
    for mark in checkpoints:
        context = _llm_context(mark)
        right = 0
        carried = []
        for name, _old, new in SUBJECTS[:mark]:
            prompt = f"{context}\n\nQuestion: what is the {name}?"
            carried.append(len(prompt.encode("utf-8")))
            # reasoning_effort and a real ceiling, because run one had
            # neither and measured the wrong thing. See the docstring:
            # 80 tokens with no reasoning cap left a thinking model
            # emitting EMPTY answers, which scored as wrong.
            answer = client.complete(prompt, model=chosen, system=_SYSTEM,
                                     max_tokens=256, temperature=0.0,
                                     reasoning_effort=NO_REASONING)
            text = getattr(answer, "text", str(answer)).strip()
            right += int(_scored(text, [new]))
            out["empty"] = out.get("empty", 0) + int(classify(text)
                                                     == "empty")
        out["accuracy"][mark] = right / mark
        out["bytes"][mark] = sum(carried) / len(carried)
    return out


def chain_audit(model: str | None = None) -> dict:
    """Criterion 4: does a derived answer say what it derived from?

    Both systems are given a two-fact chain, asked to derive, then one
    premise is corrected and the question re-asked. Getting the new
    answer is necessary; SAYING which premises produced it is the
    auditability claim, and it is checked separately because a system
    can be right without being checkable.
    """
    from ultraquant.interpreter.lmstudio import LMStudioClient
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    (first_a, first_b), (ask, want), (second_a, second_b), \
        (again, want_after) = _CHAIN

    root = Path(tempfile.mkdtemp(prefix="uq_chain_"))
    try:
        session = build_session(root, seed=0)
        run_pipeline(first_a, session)
        run_pipeline(first_b, session)
        uq_before = run_pipeline(ask, session)[0]
        run_pipeline(second_a, session)
        run_pipeline(second_b, session)
        uq_after = run_pipeline(again, session)[0]
    finally:
        shutil.rmtree(root, ignore_errors=True)

    client = LMStudioClient()
    chosen = model
    if chosen is None:
        chosen = [c for c in client.models() if "embed" not in c.lower()][0]

    def ask_llm(facts: list, question: str) -> str:
        prompt = ("Facts:\n" + "\n".join(f"- {f}" for f in facts)
                  + f"\n\nQuestion: {question}\nAnswer, and state which "
                    "facts you used.")
        answer = client.complete(prompt, model=chosen, system=None,
                                 max_tokens=256, temperature=0.0,
                                 reasoning_effort=NO_REASONING)
        return getattr(answer, "text", str(answer)).strip()

    llm_before = ask_llm([first_a, first_b], ask)
    llm_after = ask_llm([second_a, second_b], again)

    def cites(text: str, premises: list) -> bool:
        lowered = " ".join(text.lower().split())
        return all(any(word in lowered for word in p.lower().split()
                       if len(word) > 3) for p in premises)

    return {
        "uq_before": uq_before, "uq_after": uq_after,
        "llm_before": llm_before, "llm_after": llm_after,
        "uq_right": (_scored(uq_before, want)
                     and _scored(uq_after, want_after)),
        "llm_right": (_scored(llm_before, want)
                      and _scored(llm_after, want_after)),
        "uq_cites": cites(uq_before, [first_a, first_b]),
        "llm_cites": cites(llm_before, [first_a, first_b]),
        "model": chosen,
    }


@dataclass
class CorrectionReport:
    """What holding a correction cost each architecture.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ran: False when LM Studio was unreachable.
        model: The transformer measured.
        uq: UltraQuant's accuracy and bytes per checkpoint.
        llm: The same for the transformer.
        chain: The premise-veto and citation result.
        narration: Share of revisions UltraQuant said aloud.
        reason: Plain-language verdict.
    """

    passes: bool
    ran: bool = False
    model: str = ""
    uq: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    chain: dict = field(default_factory=dict)
    narration: float = 0.0
    reason: str = ""


def run_gate(model: str | None = None) -> CorrectionReport:
    """Run both architectures through the same changes of mind."""
    try:
        llm = transformer_run(model=model)
        chain = chain_audit(model=model)
    except Exception as exc:                      # noqa: BLE001
        return CorrectionReport(
            passes=False, ran=False,
            reason=f"COULD NOT RUN - LM Studio unavailable: "
                   f"{type(exc).__name__}: {str(exc)[:120]}. Reported as "
                   "not run rather than passed.")
    uq = ultraquant_run()

    narration = uq["narrated"] / uq["asked"] if uq["asked"] else 0.0
    last = max(_CHECKPOINTS)
    uq_honours = uq["accuracy"][last] == 1.0
    # Criterion 5: the transformer must win or tie somewhere.
    wins_somewhere = any(
        llm["accuracy"][k] >= uq["accuracy"][k] for k in _CHECKPOINTS)
    passes = uq_honours and wins_somewhere

    growth = llm["bytes"][last] / llm["bytes"][_CHECKPOINTS[0]]
    report = CorrectionReport(
        passes=passes, ran=True, model=llm["model"], uq=uq, llm=llm,
        chain=chain, narration=narration)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - against {llm['model']}; at "
        f"{last} corrections UltraQuant answers {uq['accuracy'][last]:.0%} "
        f"and the transformer {llm['accuracy'][last]:.0%}"
        + ("" if uq_honours else " - UltraQuant LOST a revision, which "
                                "ends its provenance claim")
        + f"; context carried per query grew {growth:.1f}x for the "
        f"transformer ({llm['bytes'][_CHECKPOINTS[0]]:.0f} to "
        f"{llm['bytes'][last]:.0f} bytes) against a flat "
        f"{uq['bytes'][last]:.0f} for the store; revisions narrated "
        f"{narration:.0%}; premises cited "
        f"{'both' if chain['uq_cites'] and chain['llm_cites'] else 'UltraQuant only' if chain['uq_cites'] else 'transformer only' if chain['llm_cites'] else 'neither'}"
        + ("" if wins_somewhere else "; THE TRANSFORMER NEVER WON OR "
                                     "TIED, so this battery is too "
                                     "narrow to be measuring anything"))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"transformer: {result.model}")
    print()
    print(f"{'corrections':>12}{'UQ acc':>9}{'LLM acc':>9}"
          f"{'UQ bytes':>10}{'LLM bytes':>11}")
    for mark in _CHECKPOINTS:
        print(f"{mark:>12}{result.uq['accuracy'][mark]:>9.0%}"
              f"{result.llm['accuracy'][mark]:>9.0%}"
              f"{result.uq['bytes'][mark]:>10.0f}"
              f"{result.llm['bytes'][mark]:>11.0f}")
    print()
    print(f"revisions narrated aloud: {result.narration:.0%}")
    print()
    chain = result.chain
    print("premise veto — derive, correct a premise, re-derive:")
    print(f"  uq  before: {chain['uq_before'][:98]}")
    print(f"  uq  after : {chain['uq_after'][:98]}")
    print(f"  llm before: {chain['llm_before'][:98]}")
    print(f"  llm after : {chain['llm_after'][:98]}")
    print(f"  correct: uq={chain['uq_right']} llm={chain['llm_right']}; "
          f"cites premises: uq={chain['uq_cites']} llm={chain['llm_cites']}")
    print()
    print(result.reason)
