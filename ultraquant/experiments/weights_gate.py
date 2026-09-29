"""A teacher is known by its weights. The gate.

The committed records show that §11.130 and §11.134's three teachers -
"command-r-08-2024", "qwen3.8-27b" and "cydonia-22b" - are the very model
files this session asked as "c4ai-command-r-08-2024", "qwen/qwen3.8-27b"
and "cydonia-v1.3-magnum-v4-22b" (c4ai-command-r-08-2024-Q4_K_S.gguf,
18,849,512,896 bytes; Qwen3.8-27B-Q4_K_M.gguf, 16,810,714,336;
Cydonia-v1.3-Magnum-v4-22B-Q6_K.gguf, 18,252,706,816). Provenance kept only
names, so a claim filed under an old name could be "corroborated" by the
same weights under a new one, and the chat could count one model twice.
Here a teacher's identity is its model file, read from evidence into an
index, and every comparison uses it.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 d9918e5d...) before any code.
1. **The index is evidence**: every teacher name in the committed records
   with gguf fields resolves to the identity its records carry; the old and
   new name of each model share one identity; the three differ.
2. **No self-corroboration**: a claim filed by "qwen3.8-27b" is not checked
   by "qwen/qwen3.8-27b", and is checked by "cydonia-v1.3-magnum-v4-22b".
3. **Sources are counted by weights**: a fact whose claims name
   "command-r-08-2024" and "c4ai-command-r-08-2024" says "one source".
4. **Nothing else changes**: §11.164's measured replies, byte-identical.
5. **New claims carry identities**: every claim the three-source world files
   has "teacher_ids" matching its "teachers" through the index.
6. **The exam can fail**: P106 (names compared) breaches 2; P107 (distinct
   names counted) breaches 3; P108 (the index from names alone) breaches 1.
7. **Nothing regresses**: a sweep matches the §11.151 ledger. Checked
   outside this module.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["WeightsReport", "run_gate"]

RECORDS = Path(__file__).with_name("records")
PAIRS = [("command-r-08-2024", "c4ai-command-r-08-2024"),
         ("qwen3.8-27b", "qwen/qwen3.8-27b"),
         ("cydonia-22b", "cydonia-v1.3-magnum-v4-22b")]
#: §11.164's replies, measured in its passing run (Amendments B and C).
MEASURED_11164 = {
    "What is the atomic mass of molybdenum?": "atomic mass of molybdenum is 95.96 (confidence 0.91; contested: other sources say 95.95, 95.94).",
    "What is the atomic mass of einsteinium?": "atomic mass of einsteinium is 252 (confidence 0.91; contested: other sources say 254).",
    "What is the atomic mass of gold?": "atomic mass of gold is 196.966569 (confidence 0.91; 2 sources agree).",
    "What is the atomic number of terbium?": "atomic number of terbium is 65 (confidence 0.91; 2 sources agree).",
    "What is the atomic mass of osmium?": "atomic mass of osmium is 190.23 (confidence 0.91; 2 sources agree).",
    "What is the atomic number of iron?": "atomic number of iron is 26 (confidence 0.96; 3 sources agree).",
    "What is the atomic number of bohrium?": "atomic number of bohrium is 107 (confidence 0.90).",
    "What is the atomic number of curium?": "atomic number of curium is 96 (confidence 0.90).",
    "What is the atomic mass of hassium?": "atomic mass of hassium is 269 (confidence 0.91; one source).",
    "What is the atomic mass of lawrencium?": "atomic mass of lawrencium is 262 (confidence 0.91; one source).",
    "What is the atomic mass of praseodymium?": "atomic mass of praseodymium is 140.90766 (confidence 0.91; one source).",
    "What is the atomic mass of terbium?": "atomic mass of terbium is 158.925 (confidence 0.91; one source).",
}

_RUN: dict = {}


def _evidence() -> dict:
    """teacher name -> {identity} as the committed records carry them."""
    seen: dict = {}
    for path in sorted(RECORDS.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"gguf_name"' not in line:
                    continue
                row = json.loads(line)
                if row.get("teacher") and row.get("gguf_name") and row.get("gguf_size"):
                    seen.setdefault(row["teacher"], set()).add(f"{row['gguf_name']}:{row['gguf_size']}")
    return seen


def index_is_evidence() -> bool:
    from ultraquant.distill import sources
    evidence = _evidence()
    wrong = [(name, sources.identity(name), sorted(ids)) for name, ids in evidence.items()
             if len(ids) != 1 or sources.identity(name) not in ids]
    pairs = [sources.identity(a) == sources.identity(b) for a, b in PAIRS]
    distinct = len({sources.identity(a) for a, _b in PAIRS}) == len(PAIRS)
    _RUN["index"] = {"names": len(evidence), "wrong": wrong[:5], "pairs": pairs, "distinct": distinct}
    return bool(evidence) and not wrong and all(pairs) and distinct


@contextlib.contextmanager
def _scratch():
    from ultraquant.experiments.shape_gate import _live_copy
    with _live_copy() as lib:
        yield lib


def _file(lib, key, subject, attribute, value, teachers):
    """A promoted claim naming these teachers, approved through the auto-approver."""
    entry = lib.stash.add_claim(
        f"https://distill.invalid/weights/{len(lib.stash.entries())}", f"What is the {attribute} of {subject}?",
        f"The {attribute} of {subject} is {value}.", measured_confidence=0.9,
        provenance={"run_id": "weights", "question_id": f"{attribute}:{subject}",
                    "teachers": list(teachers), "lineages": list(teachers)},
        fields={"key": key, "value": value, "subject": subject, "attribute": attribute})
    lib.approver().approve_all()
    return entry


def no_self_corroboration() -> bool:
    from ultraquant.distill import corroborate
    with _scratch() as lib:
        _file(lib, "atomic mass of carbon", "carbon", "atomic mass", "12.011", ["qwen3.8-27b"])
        same = {c["key"] for c in corroborate.claims_to_check(lib.memory, lib.stash, "qwen/qwen3.8-27b")}
        other = {c["key"] for c in corroborate.claims_to_check(lib.memory, lib.stash,
                                                               "cydonia-v1.3-magnum-v4-22b")}
    _RUN["self"] = {"same weights check it": "atomic mass of carbon" in same,
                    "other weights check it": "atomic mass of carbon" in other}
    return "atomic mass of carbon" not in same and "atomic mass of carbon" in other


def counted_by_weights() -> bool:
    from ultraquant.distill import provenance
    with _scratch() as lib:
        _file(lib, "atomic mass of neon", "neon", "atomic mass", "20.180", ["command-r-08-2024"])
        _file(lib, "atomic mass of neon", "neon", "atomic mass", "20.180", ["c4ai-command-r-08-2024"])
        found = provenance.provenance(lib.stash, None, "atomic mass of neon", "20.180")
        note = provenance.note(found)
    words = json.loads((Path(provenance.__file__).with_name("data") / "provenance.json").read_text(encoding="utf-8"))
    _RUN["neon"] = {"teachers": found["teachers"], "note": note}
    return note.endswith(words["one"]) and words["agree"].split("{n}")[-1].strip() not in note


def _said():
    if "said" not in _RUN:
        from ultraquant.experiments import said_gate as SG
        SG._RUN.clear()
        replies = SG._replies()
        claims = []
        with SG._world() as (lib, _session):
            for entry in lib.stash.entries(status="promoted"):
                provenance = entry.get("provenance") or {}
                if str(provenance.get("run_id", "")).startswith("said-"):
                    claims.append((list(provenance.get("teachers") or []),
                                   list(provenance.get("teacher_ids") or [])))
        _RUN["said"] = {"asked": replies["asked"], "claims": claims}
    return _RUN["said"]


def nothing_else_changes() -> bool:
    asked = _said()["asked"]
    differ = [q for q, reply in MEASURED_11164.items() if asked.get(q) != reply]
    _RUN["differ"] = [(q, asked.get(q)) for q in differ][:5]
    return not differ


def claims_carry_ids() -> bool:
    from ultraquant.distill import sources
    claims = _said()["claims"]
    bad = [(t, i) for t, i in claims if len(t) != len(i)
           or any(sources.identity(name) != ident for name, ident in zip(t, i))]
    _RUN["ids"] = {"claims": len(claims), "bad": bad[:5]}
    return bool(claims) and not bad


CASES = {
    "1 the index is evidence": index_is_evidence,
    "2 no self-corroboration": no_self_corroboration,
    "3 sources are counted by weights": counted_by_weights,
    "4 nothing else changes": nothing_else_changes,
    "5 new claims carry identities": claims_carry_ids,
}


def _plants():
    from ultraquant.distill import provenance, sources
    real = provenance.provenance

    def by_names(stash, ledger, key, value):
        """P107: every distinct name counted."""
        found = real(stash, ledger, key, value)
        names = set()
        from ultraquant.distill import corroborate
        for entry in stash.entries(status="promoted"):
            fields = entry.get("fields") or {}
            if fields.get("key") == key and corroborate.values_agree(fields.get("value"), value):
                names |= set((entry.get("provenance") or {}).get("teachers") or [])
        return dict(found, teachers=sorted(names))

    return [
        ("P106 names compared, not identities", "2 no self-corroboration",
         [mock.patch.object(sources, "identity", lambda name: name)]),
        ("P107 distinct names counted", "3 sources are counted by weights",
         [mock.patch.object(provenance, "provenance", by_names)]),
        ("P108 the index from names alone", "1 the index is evidence",
         [mock.patch.object(sources, "_teacher_index", lambda: {})]),
    ]


@dataclass
class WeightsReport:
    """Whether teachers are known by their model files.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: The index check, self-corroboration, neon's note, reply differences.
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


def run_gate() -> WeightsReport:
    report = WeightsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("index", "self", "neon", "differ", "ids")}
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
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
