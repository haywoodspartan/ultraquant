"""Rivals only for the same fact. The gate.

``stash.claim_relation`` treated two claims as one subject when their
subject phrases shared ANY content word. "capital of france" and
"capital of spain" share "capital". Claude reproduced, on 3b01174:
- "The capital of Spain is Madrid" (1 source) REJECTED as "lost to
  entry 1", the better-sourced capital of France. §11.136's recorded
  losses made the old looseness permanent;
- "The capital of Spain is Paris" CORROBORATED by "The capital of France
  is Paris" from another site, and "The bridge height is 324 metres" by
  the tower's.
The user's library is not affected today: on a copy, auto-approval
approves and rejects nothing. Future web claims would be.

A stopgap within today's string keys: the indexing unit replaces it
with subject identity from the index.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 a830be24...) before any fix existed:

1. **Different subjects are never rivals**: Spain (1 source) and France
   (2 sources) are both held, neither is rejected or marked "sources
   disagree", and ``claim_relation`` gives no verdict for France/Spain
   Paris or tower/bridge 324.
2. **The same fact still competes**: Kenya's Mombasa (1) loses to
   Nairobi (3) in both id orders; 324 contradicts 330 metres; Paris
   contradicts Lyon; "the height of the tower" and "the tower height" at
   324 metres agree.
3. **No false corroboration of identity claims**: Spain-is-Paris and
   bridge-324 are not corroborated; the measured descriptive paraphrase
   pair still corroborates, and so does France-is-Paris from two sites.
4. **The user's library is untouched**: on a copy, analyze and
   approve_all still approve and reject nothing (run when present).
5. **The exam can fail**: P28 (the old shared-word rule) breaches 1, and
   P29 (the paraphrase guard removed) breaches 3.
6. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**PASSED** twice on Claude's machine, and in Astra's run: 3 of 3 cases,
2 of 2 plants, and a copy of the user's library untouched. Every earlier
gate stayed green.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.review6_gate import _arbitration, _seed_sources
from ultraquant.experiments.undo_gate import World

__all__ = ["SameFactReport", "run_gate"]

LIVE_HOME = Path(__file__).resolve().parents[2] / "uq_home"

#: The measured true-paraphrase pair (tests/test_codeweb.py).
_PARAPHRASE = (
    "The main arithmetic operations are addition, subtraction, "
    "multiplication, and division.",
    "The four basic arithmetic operations are addition, subtraction, "
    "multiplication, and division.")


def different_subjects_not_rivals() -> bool:
    """1: Spain's capital is not France's rival."""
    from ultraquant.interpreter.stash import claim_relation
    world = World()
    try:
        _seed_sources(world, [
            ("The capital of France is Paris.", ["a.example", "b.example"]),
            ("The capital of Spain is Madrid.", ["c.example"])])
        world.approver().approve_all()
        entries = world.stash.entries()
        held = (world.value("capital of france") == "Paris"
                and world.value("capital of spain") == "Madrid")
        untouched = all(e["status"] != "rejected"
                        and "sources disagree" not in (e["notes"] or "")
                        for e in entries)
    finally:
        world.cleanup()
    no_verdicts = (
        claim_relation("The capital of France is Paris.",
                       "The capital of Spain is Paris.") is None
        and claim_relation("The tower height is 324 metres.",
                           "The bridge height is 324 metres.") is None)
    return held and untouched and no_verdicts


def same_fact_competes() -> bool:
    """2: one fact's rivals still compete, whatever the word order."""
    from ultraquant.interpreter.stash import claim_relation
    relations = (
        claim_relation("The tower height is 324 metres.",
                       "The tower height is 330 metres.") == "contradicts"
        and claim_relation("The capital of France is Paris.",
                           "The capital of France is Lyon.") == "contradicts"
        and claim_relation("The height of the tower is 324 metres.",
                           "The tower height is 324 metres.") == "agrees")
    return relations and _arbitration(True) and _arbitration(False)


def _statuses(claims) -> dict:
    """Analyze a fresh stash holding (url, claim) pairs."""
    from ultraquant.interpreter.stash import ContemporaryStash
    root = Path(tempfile.mkdtemp(prefix="uq_samefact_"))
    try:
        stash = ContemporaryStash(root / "stash.json")
        for url, claim in claims:
            stash.add_claim(url, "t", claim)
        stash.analyze()
        return {e["claim"]: e["status"] for e in stash.entries()}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def no_false_corroboration() -> bool:
    """3: identity claims corroborate only their own subject."""
    spain = _statuses([("https://a.example/1", "The capital of France is Paris."),
                       ("https://b.example/1", "The capital of Spain is Paris.")])
    bridge = _statuses([("https://c.example/1", "The tower height is 324 metres."),
                        ("https://d.example/1", "The bridge height is 324 metres.")])
    paraphrase = _statuses([("https://e.example/1", _PARAPHRASE[0]),
                            ("https://f.example/1", _PARAPHRASE[1])])
    mirror = _statuses([("https://g.example/1", "The capital of France is Paris."),
                        ("https://h.example/1", "The capital of France is Paris.")])
    return (spain["The capital of Spain is Paris."] != "corroborated"
            and bridge["The bridge height is 324 metres."] != "corroborated"
            and all(s == "corroborated" for s in paraphrase.values())
            and all(s == "corroborated" for s in mirror.values()))


def library_untouched() -> bool | None:
    """4: a copy of the user's library, analyzed and auto-approved."""
    if not (LIVE_HOME / "stash.json").exists():
        return None
    from ultraquant.interpreter.autoapprove import AutoApprover
    from ultraquant.interpreter.stash import ContemporaryStash
    from ultraquant.memory.factshards import FactShards
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.shards.vault import ShardVault
    root = Path(tempfile.mkdtemp(prefix="uq_samefact_live_")) / "uq_home"
    try:
        shutil.copytree(LIVE_HOME, root)
        memory = SystematicMemory(path=root / "memory.json")
        memory.shards = FactShards(ShardVault(root / "vault"))
        stash = ContemporaryStash(root / "stash.json")
        before = Counter(e["status"] for e in stash.entries())
        approved = AutoApprover(stash, memory,
                                root / "approvals.jsonl").approve_all()
        entries = stash.entries()
        after = Counter(e["status"] for e in entries)
        lost = [e for e in entries if "lost to entry" in (e["notes"] or "")]
        return not approved and not lost and after == before
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


CASES = {
    "1 different subjects are never rivals": different_subjects_not_rivals,
    "2 the same fact still competes": same_fact_competes,
    "3 no false corroboration of identity claims": no_false_corroboration,
}


def _plants():
    from ultraquant.interpreter import stash as S

    def shared_word(subject_a, subject_b):
        return bool(S._content_tokens(subject_a) & S._content_tokens(subject_b))

    return [
        ("P28 the old shared-word subject rule",
         "1 different subjects are never rivals",
         [mock.patch.object(S, "_same_subject", shared_word)]),
        ("P29 the paraphrase guard removed",
         "3 no false corroboration of identity claims",
         [mock.patch.object(S.ContemporaryStash, "_identity_paraphrase_ok",
                            lambda self, claim_a, claim_b: True)]),
    ]


@dataclass
class SameFactReport:
    """Whether claims compete, and corroborate, only within one fact.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        live: Criterion 4 on a copy of the user's library, or None when
            the library is not present.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    live: bool | None = None
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> SameFactReport:
    report = SameFactReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.live = library_untouched()
    except Exception as exc:
        report.live = False
        report.errors["4 the user's library"] = repr(exc)
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
    met = all(report.cases.values()) and report.live is not False
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        failed = [n for n, ok in report.cases.items() if not ok]
        if report.live is False:
            failed.append("4 the user's library")
        report.reason = f"FAIL: {failed}"
    else:
        live = ("the user's library untouched" if report.live
                else "the user's library not present")
        report.reason = f"PASS: 3 cases; 2 of 2 plants caught; {live}"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "live": report.live,
                      "planted": report.planted, "errors": report.errors},
                     indent=1))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
