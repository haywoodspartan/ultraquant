"""Requests and unknown subjects, through the catalogue. The gate.

§11.141 (failed, kept) answered through the catalogue on the question
path only, and refused an unknown subject only when every other word was
unknown to the library. Measured on the rebuilt library:
- "Tell me the capital of X." was answered 57.5% of the time. It is
  classified as chat before any answering runs;
- 18 of 50 unknown-subject questions still got another subject's fact:
  "What is the capital of the US state of Kessaway?" said Montgomery,
  because "us" and "state" are words of held subjects' names;
- 0 wrong answers, and 0 junk hints outside the one ungated form.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 aa657285...) before any code:
1. **Requests answered**: "Tell me the capital of X." at least 95%
   (baseline 57.5%), and every other gated form stays at or above 95%.
2. **Unknown subjects**: 0 of 50 answered with another subject's fact
   (baseline 18), and no curiosity.
3. **Chat stays chat**: a chat-taught "freedonia capital" = "Fredville"
   answers "What is the capital of Freedonia?", and "I visited the
   capital of Kenya last year." gets no catalogue answer.
4. **Nothing regresses**: 0 wrong assertions and 0 curiosity hints on
   the 14 gated forms. "X was written by whom?" is reported.
5. **The exam can fail**: P43 (chat never offered to the catalogue)
   breaches 1, P44 (§11.141's refusal rule) breaches 2, and P45
   (readings given for chat messages) and P46 (the first sharing
   candidate decides) breach 3.
6. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

Amendment A (sha256 94ae9c3c..., recorded before any code):
- the "I visited..." check reads the trace for a catalogue step, not
  the reply, because today's chat template names recalled facts;
- criterion 3's Freedonia half failed on every commit since before
  §11.139. The keyword loop returned at the first candidate sharing a
  word ("capital of ghana") and never reached "freedonia capital". Both
  tiers now look for a covering key before demoting.

**PASSED** twice on Claude's machine and in Astra's run: 4 of 4 cases
and 4 of 4 plants. "Tell me the capital of X." is answered 100%
(baseline 57.5%), and no unknown subject is answered with another
subject's fact (baseline 18). Claude's review added the empty-question
guard in both tiers: "What is it?" would otherwise have asserted any
candidate.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.answers_gate import FORMS, UNGATED, _REFUSALS
from ultraquant.experiments.catalogue_gate import REAL_NOT_HELD, _built

__all__ = ["RequestsReport", "run_gate"]


def _normalize(text) -> str:
    from ultraquant.experiments import knowledge_bench as K
    return K.normalize(str(text))


def measure(root: Path) -> dict:
    """Every distilled fact in every form, with hints counted per form."""
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    session = build_session(root, seed=0)
    memory = session.memory
    facts = [(k, memory._fact_record(k)) for k in memory.fact_keys()]
    facts = [(k, r) for k, r in facts if r and r.get("subject")]
    tally, hints = defaultdict(Counter), Counter()
    wrong = []
    for key, record in facts:
        value = _normalize(record["value"])
        for form in FORMS.get(record["attribute"], []):
            question = form.format(s=record["subject"])
            reply, _trace = run_pipeline(question, session)
            answered = (not reply.startswith(_REFUSALS)
                        and value in _normalize(reply))
            tally[form]["answered" if answered else "missed"] += 1
            if not reply.startswith(_REFUSALS) and not answered:
                wrong.append((question, reply[:120]))
            if "If I knew the" in reply:
                hints[form] += 1
    unknown = [i.question for i in list(K.FICTITIOUS) + list(H.FICTITIOUS)
               if i.category == "capital"]
    unknown += [f"What is the capital of {s}?" for s in REAL_NOT_HELD]
    before = len(session.curiosities)
    named_other = []
    for question in unknown:
        reply, _trace = run_pipeline(question, session)
        if "Nearest I hold" in reply or not reply.startswith(_REFUSALS):
            named_other.append((question, reply[:120]))
    return {"forms": {form: dict(c) for form, c in tally.items()},
            "wrong": wrong, "hints": dict(hints),
            "unknown asked": len(unknown), "unknown named another": named_other,
            "unknown curiosities": len(session.curiosities) - before}


_CACHE: dict = {}


def _measured() -> dict:
    if "m" not in _CACHE:
        if "lib" not in _CACHE:
            _CACHE["lib"] = _built()
        _CACHE["m"] = measure(_CACHE["lib"].root)
    return _CACHE["m"]


def _forget(all_of_it: bool) -> None:
    _CACHE.pop("m", None)
    if all_of_it:
        lib = _CACHE.pop("lib", None)
        if lib is not None:
            shutil.rmtree(lib.root, ignore_errors=True)


def _rate(counts) -> float:
    total = counts.get("answered", 0) + counts.get("missed", 0)
    return counts.get("answered", 0) / total if total else 0.0


def _gated():
    return [f for forms in FORMS.values() for f in forms if f not in UNGATED]


def requests_answered() -> bool:
    m = _measured()
    return all(_rate(m["forms"].get(form, {})) >= 0.95 for form in _gated())


def unknown_subjects() -> bool:
    m = _measured()
    return (m["unknown asked"] == 50 and not m["unknown named another"]
            and m["unknown curiosities"] == 0)


def chat_stays_chat() -> bool:
    from ultraquant.interpreter.thoughts import build_session, run_pipeline
    root = Path(tempfile.mkdtemp(prefix="uq_requests_chat_"))
    try:
        session = build_session(root, seed=0)
        session.memory.remember_fact("capital of kenya", "Nairobi", 0.9,
                                     subject="Kenya", attribute="capital")
        session.memory.remember_fact("capital of ghana", "Accra", 0.9,
                                     subject="Ghana", attribute="capital")
        run_pipeline("freedonia capital is Fredville", session)
        freedonia, _ = run_pipeline("What is the capital of Freedonia?", session)
        visited, trace = run_pipeline(
            "I visited the capital of Kenya last year.", session)
        catalogue_used = any("catalogue" in str(entry) for entry in trace)
        # Amendment A: the chat template may name a recalled fact; only a
        # catalogue step in the trace is an answer from the catalogue.
        return "Fredville" in freedonia and not catalogue_used
    finally:
        shutil.rmtree(root, ignore_errors=True)


def nothing_regresses() -> bool:
    m = _measured()
    gated_hints = sum(n for form, n in m["hints"].items() if form not in UNGATED)
    return not m["wrong"] and gated_hints == 0


CASES = {
    "1 requests answered": requests_answered,
    "2 unknown subjects": unknown_subjects,
    "3 chat stays chat": chat_stays_chat,
    "4 nothing regresses": nothing_regresses,
}


def _plants():
    from ultraquant.interpreter import thoughts as T
    from ultraquant.memory import systematic as SM
    from ultraquant.reason.inference import _library_unknown

    from ultraquant.shards.router import _informative, normalize_token

    def first_sharing(candidates, question_tokens, memory):
        """The loop before Amendment A: the first sharing candidate decides."""
        for key in candidates:
            key_tokens = {normalize_token(tok)
                          for tok in T._TOKEN_RE.findall(key.lower())
                          if _informative(tok)}
            if not (key_tokens & question_tokens):
                continue
            if memory.recall_fact(key) is None:
                continue
            return key if question_tokens <= key_tokens else None
        return None

    def old_refusal(self, words, known):
        return all(_library_unknown(word, self) for word in words - known)

    return [
        ("P43 chat never offered to the catalogue", "1 requests answered",
         [mock.patch.object(SM.SystematicMemory, "catalogue_request",
                            lambda self, text: None)]),
        ("P44 the §11.141 refusal rule", "2 unknown subjects",
         [mock.patch.object(SM.SystematicMemory, "_unheld_subject", old_refusal)]),
        ("P45 readings given for chat messages", "3 chat stays chat",
         [mock.patch.object(SM.SystematicMemory, "catalogue_request",
                            lambda self, text: self.catalogue_answer(text))]),
        ("P46 the first sharing candidate decides", "3 chat stays chat",
         [mock.patch.object(T, "_first_covering", first_sharing)]),
    ]


@dataclass
class RequestsReport:
    """Whether requests and unknown subjects go through the catalogue.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The rebuilt library's measurements.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> RequestsReport:
    report = RequestsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        m = _measured()
        report.measured = {
            "answered": {f: round(_rate(c), 3) for f, c in m["forms"].items()},
            "wrong": m["wrong"][:10], "hints": m["hints"],
            "unknown named another": m["unknown named another"][:10]}
    except Exception as exc:
        report.errors["measure"] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        _forget(False)                  # answering is read-time: keep the build
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
        finally:
            _forget(False)
    _forget(True)
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = ("FAIL: "
                         f"{[n for n, ok in report.cases.items() if not ok]}")
    else:
        report.reason = "PASS: 4 cases; 4 of 4 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
