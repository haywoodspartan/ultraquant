"""Ways of asking learned from confirmed answers. The gate.

The catalogue learned its ways of asking only from distillation questions.
Measured at 3f02048, in a catalogued world (the authors of Emma, Dune and
Ulysses, with "wrote" attested):
- "Emma was written by whom?" answered "I don't hold that exactly. Nearest
  I hold: author of emma is Jane Austen (confidence 0.90).";
- a following "yes" answered "I have nothing on that yet.";
- after the same for Dune, "Ulysses was written by whom?" still got only
  the nearest-held reply.
Here the user's "yes" to a nearest-held catalogued fact teaches the way
of asking, and the catalogue's own attestation rule (two subjects)
decides when it counts.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 c4afc88c...) before any code:
1. **Learned from confirmation**: after Emma + "yes" and Dune + "yes",
   "Ulysses was written by whom?" answers exactly with James Joyce.
2. **Only from confirmation**: without "yes", or with "no", nothing is
   learned, and "no" changes no fact's confidence.
3. **Two subjects, as the catalogue requires**: Emma alone leaves
   Ulysses at the nearest-held reply.
4. **Nothing regresses**: §11.142's and §11.145's gates pass here. The
   §11.151 ledger sweep of the change is checked outside this module.
5. **The exam can fail**: P66 (learning on any nearest-held reply)
   breaches 2, and P67 (a threshold of one subject) breaches 3.

**Reproduced first**: at 3f02048, case 1 failed ("yes" answered "I have
nothing on that yet").

**PASSED** on Claude's machine and in Astra's run: 4 of 4 cases, 2 of 2
plants caught. The unit's tree, swept, matches the §11.151 ledger on
every gate. The native gates were re-measured after building native/uq.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["AskingReport", "run_gate"]

BOOKS = (("Emma", "Jane Austen"), ("Dune", "Frank Herbert"), ("Ulysses", "James Joyce"))


@contextlib.contextmanager
def _session():
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_asking_"))
    try:
        session = build_session(root, seed=0)
        memory = session.memory
        for subject, author in BOOKS:
            memory.remember_fact(f"author of {subject.lower()}", author, 0.9,
                                 subject=subject, attribute="author")
            memory.learn_asking("author", subject, f"Who wrote {subject}?")
        yield session
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _say(session, text: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(text, session)[0]


def _exact(reply: str) -> bool:
    return (reply.startswith("author of ulysses is James Joyce")
            and "Nearest I hold" not in reply)


def _asked_by(session) -> dict:
    return dict(session.memory._attribute_vocabulary().get("author", {})
                .get("asked_by", {}))


def learned_from_confirmation() -> bool:
    with _session() as session:
        for subject in ("Emma", "Dune"):
            _say(session, f"{subject} was written by whom?")
            _say(session, "yes")
        return _exact(_say(session, "Ulysses was written by whom?"))


def only_from_confirmation() -> bool:
    outcomes = []
    for reply in (None, "no"):
        with _session() as session:
            before = _asked_by(session)
            confidence = {s: session.memory.recall_fact(f"author of {s.lower()}")["confidence"]
                          for s, _a in BOOKS}
            for subject in ("Emma", "Dune"):
                _say(session, f"{subject} was written by whom?")
                if reply is not None:
                    _say(session, reply)
            after = _say(session, "Ulysses was written by whom?")
            unchanged = {s: session.memory.recall_fact(f"author of {s.lower()}")["confidence"]
                         for s, _a in BOOKS} == confidence
            outcomes.append("Nearest I hold" in after and _asked_by(session) == before
                            and unchanged)
    return all(outcomes)


def two_subjects() -> bool:
    with _session() as session:
        _say(session, "Emma was written by whom?")
        _say(session, "yes")
        _say(session, "Dune was written by whom?")
        return "Nearest I hold" in _say(session, "Ulysses was written by whom?")


_GATES: dict = {}


def nothing_regresses() -> bool:
    from ultraquant.experiments import requests_gate, review9_gate
    _GATES["requests_gate"] = bool(requests_gate.run_gate().passes)
    _GATES["review9_gate"] = bool(review9_gate.run_gate().passes)
    return all(_GATES.values())


CASES = {
    "1 learned from confirmation": learned_from_confirmation,
    "2 only from confirmation": only_from_confirmation,
    "3 two subjects": two_subjects,
    "4 nothing regresses": nothing_regresses,
}


def _learn_at_once(self, ctx, key, record) -> None:
    """P66: the reading is learned the moment it is said, unconfirmed."""
    ctx.session.memory.learn_asking(record["attribute"], record["subject"], ctx.text)


def _one_subject(attribute: str, item: dict) -> set:
    """P67: any asking word counts once one subject has used it."""
    return set(attribute.split()) | set(item.get("asked_by", {}))


def _plants():
    from ultraquant.interpreter import thoughts as T
    from ultraquant.memory import factshards as FS
    from ultraquant.memory import systematic as SM
    return [
        ("P66 learning on any nearest-held reply", "2 only from confirmation",
         [mock.patch.object(T.Reason, "_arm_reading", _learn_at_once)]),
        ("P67 a threshold of one subject", "3 two subjects",
         [mock.patch.object(FS, "attribute_words", _one_subject),
          mock.patch.object(SM, "attribute_words", _one_subject)]),
    ]


@dataclass
class AskingReport:
    """Whether the user's confirmations teach the catalogue how they ask.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        gates: The gates criterion 4 reran.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> AskingReport:
    report = AskingReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.gates = dict(_GATES)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 4 cases; 2 of 2 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "gates": report.gates,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
