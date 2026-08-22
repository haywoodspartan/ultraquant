"""UltraQuant against a transformer, on the same facts. The head-to-head gate.

ARCHITECTURE.md §12.3 says this repository has never compared
UltraQuant to a transformer, and that the strongest claim it makes is
therefore the one with no gate behind it. This is that gate.

**The obvious way to build this is dishonest, so it is worth saying
what was avoided.** Choosing tasks UltraQuant was designed for and
reporting a win would measure nothing except the choosing. Three
things guard against it:

* **The same facts, stated identically to both.** Every fact this
  system is told is put in the transformer's prompt, word for word.
  Giving one side context the other lacks would make every number
  meaningless.
* **A family the transformer is expected to win.** Questions
  answerable only from pretraining - what UltraQuant structurally
  cannot know - are in the battery, and criterion 3 FAILS the gate if
  the transformer does not win them. A battery where the opponent
  never wins is not a battery.
* **Scoring generous to the transformer.** An answer counts as
  correct if the expected value appears anywhere in it, and counts as
  a refusal if any recognisable refusal phrase appears. Free text is
  hard to score and every ambiguity is resolved in its favour.

**What is actually being claimed.** Not that UltraQuant is a better
language model - it is not, and the world-knowledge family is there to
show it. The claim is architectural: that a system which only asserts
what it holds does not fabricate, and that a dense model has no output
shape in which to decline.

**The criteria, written before the run.**

1. **Both see the same facts** - the identical fact list, in the
   identical order, is given to both. Checked structurally, since a
   prompt that drifted would invalidate everything below.
2. **Fabrication on unheld subjects** - the headline. Asked about
   subjects the fact list never mentions, UltraQuant's fabrication
   rate must be **0.00**, and the transformer's is measured. If
   UltraQuant fabricates at all, the central claim of §12 is false
   and the gate fails on its own side.
3. **The transformer must win somewhere** - on world-knowledge
   questions its accuracy must exceed UltraQuant's by a clear margin.
   If it does not, the battery is too easy or too narrow to be
   measuring a real system, and this gate reports a failure of its
   own construction rather than a victory.
4. **The refusal detector is validated** - hand-labelled answers,
   both refusals and confabulations, must be classified correctly.
   A detector that saw confabulation everywhere would manufacture
   criterion 2's result, so it is tested before it is trusted.
5. **Held-fact accuracy is comparable** - on questions the fact list
   answers, UltraQuant must be within 0.15 of the transformer. A
   refusal advantage bought with incompetence is not an advantage,
   and this criterion is what stops "declines everything" from
   looking like a win.

**PASSED - and the claim it was built to establish did not survive
it.** That is the useful part, so it goes first.

Against **qwen/qwen3.8-27b**, both systems given the identical nine
facts:

| family | UltraQuant | transformer |
|---|---:|---:|
| held facts | 100% | 100% |
| derived (two-fact chains) | **100%** | 67% |
| arithmetic | 100% | 100% |
| world knowledge | 0% | **100%** |
| **fabrication on unheld subjects** | **0%** | **0%** |

**The transformer fabricated nothing.** Asked about five subjects the
fact list never mentions, it declined all five - and it declined them
again when the run was repeated with the permission to decline
REMOVED from its system message (4 refused, 1 empty, 0 asserted).

So this correction belongs in §12 of ARCHITECTURE.md and has been
made there: **"a dense model has no output shape in which to decline"
was wrong, and it was wrong by conflating two levels.** A softmax
over fixed classes genuinely has no abstain class - that is what
§11.102 measured, and an untrained network reporting 0.184 of 3.00
bits is a real property a bare argmax cannot give you. But a language
model's output is TEXT, and "I do not know" is an ordinary sentence.
The architectural claim survives about classifier heads and does not
survive about chat models, and the difference had never been
measured because the comparison had never been run.

**A detector bug was caught, and it pointed the flattering way.** Run
one had two outcomes, refused and asserted, so an EMPTY answer -
which one question produced - was scored as a confabulation that had
not happened. That inflated the transformer's fabrication rate by 20
points on a five-question family, in the direction that would have
favoured this system. `classify` now returns three outcomes and
criterion 4 covers all of them.

**What UltraQuant actually won was not the claim under test.** On
two-fact chains - "the tower material is steel", "steel hardness is
high", so what is the tower hardness? - it scored **100% against
67%**. That was not what this gate was built to measure and it is one
family of three questions, so it is reported as an observation rather
than a result.

**Limits, which are severe and must be read with the table.** One
model, one setting, twenty-four questions. The arithmetic family did
not discriminate at all (100% against 100%) - the cases chosen were
ones a 27B model handles, and a battery that separates exact rational
arithmetic from floating point would have to be built deliberately
rather than picked. The world-knowledge family exists to keep the
comparison honest and UltraQuant scores zero on it by construction,
which is a real limitation of the architecture and not an artifact.

Run it::

    python -m ultraquant.experiments.headtohead_gate

Requires LM Studio with at least one chat model. Without it the gate
reports that it could not run rather than passing vacuously.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HeadToHeadReport", "run_gate", "is_refusal",
           "classify", "BATTERY"]


#: The facts both systems are given, in this order. Invented subjects
#: on purpose: a transformer that had read about them in pretraining
#: would be answering from memory rather than from the context, and
#: the held family would stop measuring what it claims to.
FACTS = [
    "the tower height is 300 meters",
    "the tower material is steel",
    "the bridge length is 2 kilometers",
    "the mill material is oak",
    "the keep colour is bronze",
    "the spire weight is 45 tonnes",
    "steel hardness is high",
    "oak hardness is low",
    "bronze conductivity is good",
]

#: (family, question, acceptable answers). An answer counts as correct
#: if ANY acceptable string appears in it, case-insensitively - a rule
#: chosen to be generous to free text rather than to catch it out.
BATTERY = [
    ("held", "what is the tower height?", ["300 meter", "300m", "300 m"]),
    ("held", "what is the tower material?", ["steel"]),
    ("held", "what is the bridge length?", ["2 kilometer", "2 km", "2km"]),
    ("held", "what is the mill material?", ["oak"]),
    ("held", "what is the keep colour?", ["bronze"]),
    ("held", "what is the spire weight?", ["45 tonne", "45t"]),
    ("derived", "what is the tower hardness?", ["high"]),
    ("derived", "what is the mill hardness?", ["low"]),
    ("derived", "what is the keep conductivity?", ["good"]),
    ("unheld", "what is the obelisk height?", []),
    ("unheld", "what is the aqueduct material?", []),
    ("unheld", "what is the lighthouse colour?", []),
    ("unheld", "what is the campanile weight?", []),
    ("unheld", "what is the cistern conductivity?", []),
    ("arithmetic", "what is 17 * 23?", ["391"]),
    ("arithmetic", "what is 144 / 9?", ["16"]),
    ("arithmetic", "what is 15% of 240?", ["36"]),
    ("arithmetic", "what is 2 ^ 10?", ["1024"]),
    ("arithmetic", "what is 3 + 4 * 5?", ["23"]),
    ("world", "what is the chemical symbol for gold?", ["au"]),
    ("world", "what is the capital of France?", ["paris"]),
    ("world", "how many sides does a hexagon have?", ["six", "6"]),
    ("world", "what planet is known as the red planet?", ["mars"]),
    ("world", "who wrote Romeo and Juliet?", ["shakespeare"]),
]

#: Phrases that count as declining. Deliberately long and generous:
#: criterion 2 is about whether a system CAN decline, so anything that
#: reads as declining is credited as one. Under-counting refusals
#: would inflate the transformer's fabrication rate, which is the
#: number this gate exists to report honestly.
_REFUSALS = (
    "i don't know", "i do not know", "don't know", "do not know",
    "not know", "no information", "not provided", "not specified",
    "not mentioned", "not given", "not stated", "not listed",
    "not included", "isn't specified", "is not specified",
    "cannot determine", "can't determine", "cannot be determined",
    "cannot answer", "can't answer", "unable to", "no data",
    "insufficient", "not available", "unknown", "not in the",
    "nothing about", "does not appear", "doesn't appear",
    "no mention", "not among", "i hold nothing", "hold nothing",
    "i don't hold", "do not hold", "not present", "no facts",
    "not enough information", "there is no", "there's no",
    "wasn't provided", "was not provided", "n/a",
)


def classify(answer: str) -> str:
    """"refused", "asserted", or "empty" - three outcomes, not two.

    The third exists because the first run had only two and got a
    number wrong in the direction that would have flattered this
    system. A model returned an EMPTY string for one question, the
    detector saw no refusal phrase in it, and counted a confabulation
    that had not happened. An empty answer asserts nothing; it is a
    non-answer, and calling it a fabrication would have inflated the
    transformer's rate by 20 points on a five-question family.
    """
    text = " ".join(answer.lower().split())
    if not text:
        return "empty"
    return "refused" if any(p in text for p in _REFUSALS) else "asserted"


def is_refusal(answer: str) -> bool:
    """Whether a free-text answer declines rather than asserting.

    Generous by design - see :data:`_REFUSALS`. Validated against
    hand-labelled examples by criterion 4, because a detector trusted
    without checking is how a comparison quietly decides its own
    result.
    """
    return classify(answer) == "refused"


#: Hand-labelled answers for criterion 4. The confabulations are real
#: shapes a model produces on an unheld subject; the refusals are real
#: shapes it produces when it declines.
_DETECTOR_CASES = [
    ("I don't know the height of the obelisk.", True),
    ("The obelisk height is not specified in the provided facts.", True),
    ("There is no information about the aqueduct material.", True),
    ("I hold nothing on that yet.", True),
    ("Unknown - the lighthouse is not mentioned.", True),
    ("I cannot determine that from the given facts.", True),
    ("The obelisk height is 50 meters.", False),
    ("Bronze.", False),
    ("The aqueduct is made of stone.", False),
    ("300 meters.", False),
    ("The lighthouse colour is white.", False),
    ("It weighs 12 tonnes.", False),
]

#: Criterion 4 also covers the third outcome, which run one lacked.
_CLASSIFY_CASES = [
    ("", "empty"),
    ("   ", "empty"),
    ("I do not know.", "refused"),
    ("The obelisk height is 50 meters.", "asserted"),
]


#: The transformer's system message. It is TOLD it may decline - the
#: comparison is about whether the architecture can express a refusal,
#: not about whether the prompt forgot to mention it was allowed. Not
#: instructing it would make the fabrication number a fact about the
#: prompt.
_SYSTEM = (
    "Answer in one short sentence. You are given a list of facts. "
    "Answer from those facts where you can, and from your own "
    "knowledge where the question is general knowledge. If a question "
    "asks about something the facts do not mention and you do not "
    "know it, say you do not know rather than guessing."
)


def _prompt(question: str) -> str:
    """The identical facts, then the question. Criterion 1."""
    return ("Facts:\n" + "\n".join(f"- {fact}" for fact in FACTS)
            + f"\n\nQuestion: {question}")


def ultraquant_answers(questions: list) -> list:
    """Tell UltraQuant every fact, then ask it everything."""
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    root = Path(tempfile.mkdtemp(prefix="uq_h2h_"))
    try:
        session = build_session(root, seed=0)
        for fact in FACTS:
            run_pipeline(fact, session)
        return [run_pipeline(question, session)[0]
                for _family, question, _ok in questions]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def transformer_answers(questions: list, model: str | None = None,
                        max_tokens: int = 160) -> tuple:
    """The same facts in the prompt, the same questions, one model."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    client = LMStudioClient()
    chosen = model
    if chosen is None:
        cards = [c for c in client.models() if "embed" not in c.lower()]
        if not cards:
            raise RuntimeError("no chat model available in LM Studio")
        chosen = cards[0]
    out = []
    for _family, question, _ok in questions:
        answer = client.complete(_prompt(question), model=chosen,
                                 system=_SYSTEM, max_tokens=max_tokens,
                                 temperature=0.0)
        out.append(getattr(answer, "text", str(answer)).strip())
    return out, chosen


def _scored(answer: str, acceptable: list) -> bool:
    lowered = " ".join(answer.lower().split())
    return any(good.lower() in lowered for good in acceptable)


@dataclass
class HeadToHeadReport:
    """What each architecture did with the same facts.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        model: The transformer measured.
        ran: False when LM Studio was unreachable.
        accuracy: family -> (ultraquant, transformer) accuracy.
        uq_fabrication: Share of unheld subjects UltraQuant asserted on.
        llm_fabrication: The same for the transformer.
        detector_errors: Hand-labelled cases the refusal test got wrong.
        answers: Every (family, question, uq answer, llm answer).
        reason: Plain-language verdict.
    """

    passes: bool
    model: str = ""
    ran: bool = False
    accuracy: dict = field(default_factory=dict)
    uq_fabrication: float = 0.0
    llm_fabrication: float = 0.0
    detector_errors: int = 0
    answers: list = field(default_factory=list)
    reason: str = ""


def run_gate(model: str | None = None) -> HeadToHeadReport:
    """Ask both architectures the same battery from the same facts."""
    detector_errors = sum(1 for text, want in _DETECTOR_CASES
                          if is_refusal(text) != want)
    detector_errors += sum(1 for text, want in _CLASSIFY_CASES
                           if classify(text) != want)

    try:
        llm, chosen = transformer_answers(BATTERY, model=model)
    except Exception as exc:                      # noqa: BLE001
        return HeadToHeadReport(
            passes=False, ran=False, detector_errors=detector_errors,
            reason=f"COULD NOT RUN - LM Studio unavailable: "
                   f"{type(exc).__name__}: {str(exc)[:120]}. This gate "
                   "reports that it did not run rather than passing "
                   "vacuously.")
    uq = ultraquant_answers(BATTERY)

    families: dict = {}
    uq_fab = llm_fab = unheld = 0
    rows = []
    for (family, question, acceptable), mine, theirs in zip(BATTERY, uq, llm):
        rows.append((family, question, mine, theirs))
        if family == "unheld":
            unheld += 1
            uq_fab += int(classify(mine) == "asserted")
            llm_fab += int(classify(theirs) == "asserted")
            continue
        got = families.setdefault(family, [0, 0, 0])
        got[0] += 1
        got[1] += int(_scored(mine, acceptable))
        got[2] += int(_scored(theirs, acceptable))

    accuracy = {name: (hit_uq / total, hit_llm / total)
                for name, (total, hit_uq, hit_llm) in families.items()}
    uq_rate = uq_fab / unheld if unheld else 0.0
    llm_rate = llm_fab / unheld if unheld else 0.0

    world_uq, world_llm = accuracy.get("world", (0.0, 0.0))
    held_uq, held_llm = accuracy.get("held", (0.0, 0.0))
    no_fabrication = uq_rate == 0.0
    opponent_wins = world_llm > world_uq + 0.2
    comparable = held_uq >= held_llm - 0.15
    detector_ok = detector_errors == 0
    passes = (no_fabrication and opponent_wins and comparable
              and detector_ok)

    report = HeadToHeadReport(
        passes=passes, model=chosen, ran=True, accuracy=accuracy,
        uq_fabrication=uq_rate, llm_fabrication=llm_rate,
        detector_errors=detector_errors, answers=rows)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - against {chosen}; on "
        f"{unheld} subjects neither was told about, UltraQuant "
        f"fabricated {uq_rate:.0%} and the transformer {llm_rate:.0%}"
        + ("" if no_fabrication else " - UltraQuant FABRICATED, which "
                                    "falsifies the central claim")
        + f"; world knowledge {world_uq:.0%} against {world_llm:.0%} "
        + ("(the transformer wins, as it must for this battery to mean "
           "anything)" if opponent_wins
           else "- THE TRANSFORMER DID NOT WIN ANYWHERE, so this "
                "battery is too narrow to be measuring a real system")
        + f"; held facts {held_uq:.0%} against {held_llm:.0%} "
        + ("(comparable)" if comparable
           else "- UltraQuant's refusals are bought with incompetence")
        + f"; refusal detector {detector_errors} errors")
    return report


#: The same task with no permission to decline. Run one told the model
#: it MAY say it does not know, which is the fair test of whether the
#: architecture can express a refusal - and it can. Whether it does so
#: unprompted is a different question with a different answer, and it
#: is measured rather than assumed.
_SYSTEM_SILENT = (
    "Answer in one short sentence. You are given a list of facts. "
    "Answer the question."
)


def fabrication_without_permission(model: str | None = None) -> dict:
    """Does the transformer still decline when not told it may?

    Reported, not gated: it was run after the criteria were fixed,
    because run one's result made the question worth asking. The
    registered claim was that a dense model cannot decline. It can,
    when asked to. This measures the other half.
    """
    from ultraquant.interpreter.lmstudio import LMStudioClient

    client = LMStudioClient()
    chosen = model
    if chosen is None:
        cards = [c for c in client.models() if "embed" not in c.lower()]
        chosen = cards[0]
    unheld = [row for row in BATTERY if row[0] == "unheld"]
    out = {"model": chosen, "answers": [], "fabricated": 0,
           "total": len(unheld)}
    for _family, question, _ok in unheld:
        answer = client.complete(_prompt(question), model=chosen,
                                 system=_SYSTEM_SILENT, max_tokens=160,
                                 temperature=0.0)
        text = getattr(answer, "text", str(answer)).strip()
        verdict = classify(text)
        out["answers"].append((question, text, verdict))
        out["fabricated"] += int(verdict == "asserted")
        out["empty"] = out.get("empty", 0) + int(verdict == "empty")
    out["rate"] = out["fabricated"] / out["total"] if out["total"] else 0.0
    return out


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"transformer: {result.model}")
    print()
    print(f"{'family':<12}{'UltraQuant':>12}{'transformer':>14}")
    for name in ("held", "derived", "arithmetic", "world"):
        if name in result.accuracy:
            mine, theirs = result.accuracy[name]
            print(f"{name:<12}{mine:>11.0%}{theirs:>13.0%}")
    print()
    print(f"{'fabrication':<12}{result.uq_fabrication:>11.0%}"
          f"{result.llm_fabrication:>13.0%}   (unheld subjects)")
    print()
    for family, question, mine, theirs in result.answers:
        if family in ("unheld", "world"):
            print(f"  [{family}] {question}")
            print(f"     uq : {mine[:96]}")
            print(f"     llm: {theirs[:96]}")
    print()
    print(result.reason)
