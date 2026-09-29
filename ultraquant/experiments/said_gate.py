"""What the chat says about its sources. The gate.

After §11.163 the library knows, for a distilled fact, how many sources
stand behind it and whether another source disputes it - the stash names
the teachers, the source ledger holds unsettled contests. The chat said
none of it: molybdenum's 95.96, which three sources gave three values for,
read "(confidence 0.91)" exactly like gold's, which two sources agree on.
Here every catalogue reply that states a held fact says so: contested (with
the other answers), N sources agree, or one source. A fact no teacher names
- the user's own, or the user's correction - reads exactly as before.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 95f8a6fd...) before any code, with Amendment A (sha256 f0ffd379...,
also before any code: P102 redefined, because corroboration adds teachers
to the same claim and the frozen P102 could not bite). The world: a copy of
the user's library after §11.155's filing with the corrections, then the
three sources' rounds replayed from their recordings, the ledger saved as
sources.json in the library root; the chat asked through build_session.
1. **Contested values say so**: molybdenum's reply names 95.95 and 95.94;
   einsteinium's names 254.
2. **Agreement is said**: gold's, terbium's and osmium's replies say two
   sources agree.
3. **One source is said**: at least one atomic mass only Qwen filed says
   "one source".
4. **The user's own facts are untouched**: iron's and bohrium's replies are
   byte-identical to the measured ones.
5. **No note without a source**: across the 40 §11.155 calibration items
   (facts no teacher names), no reply carries a note.
6. **The exam can fail**: P100 (contests ignored) breaches 1; P101 (every
   fact noted as one source) breaches 4; P102 (the teachers of every claim
   for the key counted, whatever value they named) breaches 2.

**Amendment B, AFTER the first runs** (sha256 ab9a2473...). As frozen, the
exam FAILED 3, 4 and 5: its premises were wrong. The user's stash names three
teachers for iron and the 40 calibration facts (§11.130 and §11.134 distilled
them), so "3 sources agree" is true; and its one-source probe took the first
Qwen-only mass, einsteinium, which is contested. Of 562 structured facts in
this world only the user's two corrections name no teacher. Amended:
3. every Qwen-only atomic mass is asked; at least one uncontested one says
   "one source", and every contested one says "contested";
4. the user's corrections (bohrium, curium) carry no note, bohrium's reply
   byte-identical to the measured one;
5. iron and the 40 calibration facts each say "N sources agree", N the
   distinct teachers the stash names for the fact.
7. **Nothing regresses**: a sweep matches the §11.151 ledger. Checked
   outside this module.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["SaidReport", "run_gate"]

MEASURED = {
    "What is the atomic number of bohrium?": "atomic number of bohrium is 107 (confidence 0.90).",
}
CORRECTIONS = ["What is the atomic number of bohrium?", "What is the atomic number of curium?"]

_RUN: dict = {}


def _notes() -> dict:
    from ultraquant.distill import provenance
    return json.loads((Path(provenance.__file__).with_name("data") / "provenance.json")
                      .read_text(encoding="utf-8"))


@contextlib.contextmanager
def _world():
    """The three sources' world, replayed, with a chat session on it."""
    from ultraquant.distill import frontier, sources
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _filed_11155
    from ultraquant.interpreter.thoughts import build_session
    root = Path(tempfile.mkdtemp(prefix="uq_said_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    try:
        lib = Library(root)
        _filed_11155(lib)
        ledger = sources.SourceLedger(root / "sources.json")
        plan = [(TG.FIRST, GG._Teacher(), _decided()[3]["wilson_lower"]),
                (TG.SECOND, SG._Second(), None), (TG.THIRD, TG._Third(), None)]
        for name, teacher, bound in plan:
            bound = bound or TG._bound(teacher, name)
            for n in range(1, 9):
                out = frontier.study_round(
                    lib.memory, lib.stash, teacher, ledger, name, confidence=bound,
                    run_id=f"said-{name}-{n}", records_path=root.parent / f"{n}.jsonl",
                    approver=lib.approver())
                if out["used_up"]:
                    break
        lib.memory.save()
        yield lib, build_session(root, seed=0)
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
        shutil.rmtree(root.parent, ignore_errors=True)


def _ask(session, question) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    return run_pipeline(question, session)[0]


def _replies() -> dict:
    if "replies" in _RUN:
        return _RUN["replies"]
    from ultraquant.distill import corroborate
    from ultraquant.experiments.ownquestions_gate import _decided
    with _world() as (lib, session):
        teachers_of: dict = {}
        for entry in lib.stash.entries(status="promoted"):
            fields = entry.get("fields") or {}
            held = lib.memory.recall_fact(fields.get("key") or "") or {}
            if held.get("value") is None or not corroborate.values_agree(
                    held["value"], fields.get("value")):
                continue
            teachers_of.setdefault(fields["key"], set()).update(
                (entry.get("provenance") or {}).get("teachers") or [])
        qwen_only = []
        for key, teachers in teachers_of.items():
            record = lib.memory.recall_fact(key) or {}
            if teachers == {"qwen/qwen3.8-27b"} and record.get("attribute") == "atomic mass":
                qwen_only.append(f"What is the atomic mass of {record['subject']}?")
        pairs = _decided()[1]
        by_slot = {}
        for key in teachers_of:
            record = lib.memory.recall_fact(key) or {}
            by_slot[(record.get("subject"), record.get("attribute"))] = key
        expected = {"What is the atomic number of iron?": len(teachers_of.get("atomic number of iron", ()))}
        for target, _v in pairs:
            key = by_slot.get((target.subject, target.attribute))
            expected[target.question] = len(teachers_of.get(key, ())) if key else 0
        questions = ["What is the atomic mass of molybdenum?", "What is the atomic mass of einsteinium?",
                     "What is the atomic mass of gold?", "What is the atomic number of terbium?",
                     "What is the atomic mass of osmium?", "What is the atomic number of iron?",
                     *CORRECTIONS, *qwen_only]
        _RUN["replies"] = {
            "asked": {q: _ask(session, q) for q in dict.fromkeys(questions)},
            "qwen only": sorted(qwen_only),
            "calibration": {t.question: _ask(session, t.question) for t, _v in pairs},
            "expected": expected,
        }
    return _RUN["replies"]


def contested_said() -> bool:
    r = _replies()["asked"]
    mo, es = r["What is the atomic mass of molybdenum?"], r["What is the atomic mass of einsteinium?"]
    return "95.95" in mo and "95.94" in mo and "254" in es


def agreement_said() -> bool:
    r = _replies()["asked"]
    phrase = _notes()["agree"].format(n=2)
    return all(phrase in r[q] for q in ("What is the atomic mass of gold?",
                                        "What is the atomic number of terbium?",
                                        "What is the atomic mass of osmium?"))


def _markers() -> list:
    notes = _notes()
    return [notes["one"], notes["agree"].split("{n}")[-1].strip(),
            notes["contested"].split("{answers}")[0].strip()]


def one_source_said() -> bool:
    """Amendment B: an uncontested Qwen-only mass says one source; contested ones say so."""
    replies = _replies()
    asked, notes = replies["asked"], _notes()
    contested_marker = notes["contested"].split("{answers}")[0].strip()
    said_one = [q for q in replies["qwen only"] if notes["one"] in asked[q]]
    contested = [q for q in replies["qwen only"] if contested_marker in asked[q]]
    _RUN["one source"] = {"qwen only": len(replies["qwen only"]), "one source": len(said_one),
                          "contested": len(contested)}
    return bool(said_one) and len(said_one) + len(contested) == len(replies["qwen only"])


def own_untouched() -> bool:
    """Amendment B: the user's corrections carry no note; bohrium's is byte-identical."""
    r = _replies()["asked"]
    return (all(not any(m in r[q] for m in _markers()) for q in CORRECTIONS)
            and all(r[q] == reply for q, reply in MEASURED.items()))


def no_note_without_source() -> bool:
    """Amendment B: iron and the calibration facts say how many distinct teachers."""
    replies = _replies()
    notes = _notes()
    everything = {**replies["asked"], **replies["calibration"]}
    wrong = []
    for question, n in replies["expected"].items():
        reply = everything[question]
        want = notes["agree"].format(n=n) if n >= 2 else notes["one"] if n == 1 else None
        if (want is None and any(m in reply for m in _markers())) or (want and want not in reply):
            wrong.append((question, n, reply[-60:]))
    _RUN["noted"] = wrong[:5]
    # Amendment C (after the run, sha256 c1008d38...): iron is one of the 40
    # calibration items, so the distinct questions number 40, not 41.
    return len(replies["expected"]) == 40 and not wrong


CASES = {
    "1 contested values say so": contested_said,
    "2 agreement is said": agreement_said,
    "3 one source is said": one_source_said,
    "4 the user's own facts are untouched": own_untouched,
    "5 the library's distilled facts say how many": no_note_without_source,
}


def _plants():
    from ultraquant.distill import provenance
    real = provenance.provenance

    def no_contests(stash, ledger, key, value):
        return dict(real(stash, ledger, key, value), contests=[])

    def always_one(stash, ledger, key, value):
        found = real(stash, ledger, key, value)
        return found if found.get("teachers") else dict(found, teachers=["a source"])

    def any_value(stash, ledger, key, value):
        """P102: every claim's teachers for the key, whatever value they named."""
        found = real(stash, ledger, key, value)
        teachers = set(found.get("teachers") or [])
        for entry in stash.entries(status="promoted"):
            if (entry.get("fields") or {}).get("key") == key:
                teachers |= set((entry.get("provenance") or {}).get("teachers") or [])
        return dict(found, teachers=sorted(teachers))

    return [
        ("P100 contests ignored", "1 contested values say so",
         [mock.patch.object(provenance, "provenance", no_contests)]),
        ("P101 every fact noted as one source", "4 the user's own facts are untouched",
         [mock.patch.object(provenance, "provenance", always_one)]),
        ("P102 every claim's teachers counted, whatever value", "2 agreement is said",
         [mock.patch.object(provenance, "provenance", any_value)]),
    ]


@dataclass
class SaidReport:
    """Whether the chat says what stands behind a distilled fact.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        replies: The replies asked about.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    replies: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> SaidReport:
    report = SaidReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    replies = _RUN.get("replies") or {}
    report.replies = {"asked": replies.get("asked"), "wrong notes": _RUN.get("noted"),
                      "one source": _RUN.get("one source")}
    for name, target, patches in _plants():
        _RUN.clear()
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
            _RUN.clear()
    valid = len(report.planted) == 3 and all(report.planted.values())
    met = len(report.cases) == 5 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "replies": report.replies,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
