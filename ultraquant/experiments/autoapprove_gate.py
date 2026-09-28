"""Quarantined knowledge approved automatically, and disputable exactly.

The user: "You may also auto approve all quarantined knowledge. We can
always dispute a claim later." So promotion out of the stash stops
waiting for a human, and the price is paid on the other side: a dispute
must undo an approval EXACTLY. Every fact record the approval changed
goes back to what it was, including the derived facts truth maintenance
retracted when an approved claim revised their premise. GPT-6 Astra
wrote ``interpreter/autoapprove.py`` and ``distill/file.py``; Claude
wrote this exam.

**What "all" means here.** Before this was written, Claude read the
user's own quarantine: 47 unanalysed entries, many of them not
knowledge at all.
- "What is pixels: pixel dot end turn token": leaked chat-template text.
- "You have mentioned 'code' 6 times...": system chatter.
- Question-form panel claims, which the stash's splitter files under
  the key "what", each overwriting the last.
So "all" is read as all well-formed factual claims. The malformed ones
are rejected with the reason recorded, which a forced promotion can
still reverse.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 a157c912...) before any of the code existed:

1. **Eligible approved, the rest not**: staged, corroborated and
   memory-disputed factual claims; the better-sourced side of a
   two-source conflict, and neither side of a tie; never opinion,
   hedged or unclassified; malformed claims rejected with a reason.
2. **Every approval is on the record**: entry, key, value, sources,
   confidence, time, and the prior state of every fact it touched.
3. **A dispute undoes exactly**, for a new fact, a reinforcement, and a
   revision that retracted a chain of derived facts:
   - every touched record deep-equals its prior state;
   - a fact derived after the approval, resting on the disputed value,
     is retracted;
   - the stash entry is rejected with the reason, and an episode
     records the dispute.
4. **Disputes survive a restart.**
5. **Distilled facts land right**: every known subject §11.130's
   records promote (179: Amendment A corrected an "all 180" that
   overcounted), and none of the 80 invented. Each is approved at 0.959,
   under its display form, and each is right.
6. **The exam can fail**: P1 a dispute that deletes, P2 hedged claims
   approved, P3 malformed claims approved, P4 no snapshot of retracted
   derivatives, P5 distilled facts at the default confidence.
7. **The full suite is green.** Checked outside this module.

Two amendments were made to this exam before any verdict run, while
Astra was implementing:
- A: the recorded run promotes 179 known subjects, not the 180 the
  pre-registration said. Crime and Punishment's teachers split between
  "Dostoyevsky" and "Dostoevsky", and the filter reads a transliteration
  as dissent.
- B: Astra found that scenario B read a retraction after the disputes
  that restore it.

**PASSED** in two consecutive runs on Claude's machine: every criterion,
and 5 of 5 plants caught. Astra's two runs agreed.

Claude's review then fixed one defect outside the exam. The session
read the user's settings file for itself, so every gate that builds a
session would have started auto-approving the moment the user switched
the setting on. Now the chat, GUI and TUI pass the setting in, and
nothing else ever sees it.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["AutoApproveReport", "run_gate"]

RECORDS = Path(__file__).with_name("records") / "distill_11130.jsonl"
MEASURED = 0.959


class World:
    """A temporary stash, memory and journal."""

    def __init__(self) -> None:
        from ultraquant.interpreter.stash import ContemporaryStash
        from ultraquant.memory.systematic import SystematicMemory
        self.dir = Path(tempfile.mkdtemp(prefix="uq_autoapprove_"))
        self.stash_path = self.dir / "stash.json"
        self.journal = self.dir / "approvals.jsonl"
        self.memory = SystematicMemory(path=None)
        self._Stash = ContemporaryStash
        self.stash = ContemporaryStash(self.stash_path)

    def seed(self, entries: list) -> None:
        """Write entries straight into the stash file, then reload it."""
        data = {"entries": [], "next_id": len(entries) + 1}
        for index, (claim, sources) in enumerate(entries, start=1):
            data["entries"].append({
                "id": index, "claim": claim, "classification": "unclassified",
                "status": "staged", "sources": list(sources),
                "netloc": sources[0], "url": f"https://{sources[0]}/p{index}",
                "title": f"page {index}", "fetched": "2026-09-28T00:00:00+00:00",
                "notes": ""})
        self.stash_path.write_text(json.dumps(data), encoding="utf-8")
        self.stash = self._Stash(self.stash_path)

    def approver(self):
        from ultraquant.interpreter.autoapprove import AutoApprover
        return AutoApprover(self.stash, self.memory, self.journal)

    def record(self, key):
        found = self.memory._fact_record(key)
        return copy.deepcopy(found) if found is not None else None

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _world(body):
    world = World()
    try:
        return body(world)
    finally:
        world.cleanup()


# -- scenario A: who is approved (criteria 1 and 2) ---------------------------

KINDS = [
    ("staged", "The capital of Kenya is Nairobi.", ["a.example"]),
    ("corroborated", "The boiling point of water is 100 degrees Celsius.",
     ["b.example", "c.example"]),
    # The stash flags a clash with memory only for numbers and identities
    # (claim_relation); "Everest" against a stored "K2" is not one, so a
    # numeric clash stands for the memory-disputed kind.
    ("memory-disputed", "The height of Mount Everest is 8849 metres.",
     ["d.example"]),
    ("conflict winner", "The largest desert is the Sahara.",
     ["e.example", "f.example"]),
    ("conflict loser", "The largest desert is Antarctica.", ["g.example"]),
    ("tie one", "The fastest land animal is the cheetah.", ["h.example"]),
    ("tie two", "The fastest land animal is the pronghorn.", ["i.example"]),
    ("opinion", "The best programming language is Python.", ["j.example"]),
    ("hedged", "The treasure is reportedly buried in Oak Island.",
     ["k.example"]),
    ("unclassified", "Click here to subscribe to our weekly newsletter today",
     ["l.example"]),
    ("malformed question", "What is code: set instructions computer",
     ["lm-studio.invalid"]),
    ("malformed residue", "The pixel is a dot end turn token",
     ["lm-studio.invalid"]),
    ("malformed chatter",
     "You have mentioned 'code' 6 times and I hold nothing about it.",
     ["lm-studio.invalid"]),
]
APPROVED = {"staged", "corroborated", "memory-disputed", "conflict winner"}
MALFORMED = {"malformed question", "malformed residue", "malformed chatter"}
DEFAULTS = {"staged": 0.55, "corroborated": 0.8, "memory-disputed": 0.3,
            "conflict winner": 0.3}


def scenario_who() -> dict:
    def body(world):
        world.memory.remember_fact("height of mount everest", "8000 metres",
                                   confidence=0.6)
        world.seed([(claim, sources) for _kind, claim, sources in KINDS])
        approvals = world.approver().approve_all()
        by_entry = {a.entry_id: a for a in approvals}
        kinds = {index: kind for index, (kind, _c, _s) in
                 enumerate(KINDS, start=1)}
        approved = {kinds[i] for i in by_entry}
        rejected = {kinds[e["id"]] for e in world.stash.entries()
                    if e["status"] == "rejected"}
        reasons_ok = all("malformed" in (world.stash.get(i)["notes"] or "")
                         for i, kind in kinds.items() if kind in MALFORMED)
        rows = [json.loads(line) for line in
                world.journal.read_text(encoding="utf-8").splitlines()
                if line.strip()] if world.journal.exists() else []
        needed = {"entry_id", "key", "value", "sources", "confidence",
                  "time", "before"}
        journaled = (len(rows) >= len(approvals) > 0
                     and all(needed <= set(r) for r in rows))
        confidences = all(
            math.isclose(by_entry[i].confidence, DEFAULTS[kinds[i]],
                         abs_tol=1e-9) for i in by_entry if kinds[i] in DEFAULTS)
        return {"exactly the eligible approved": approved == APPROVED,
                "malformed rejected with a reason":
                    MALFORMED <= rejected and reasons_ok,
                "opinion, hedged, unclassified untouched":
                    not ({"opinion", "hedged", "unclassified"}
                         & (approved | rejected)),
                "tie left for review": not ({"tie one", "tie two"} & approved),
                "every approval journaled": journaled,
                "confidence as the stash sets it": confidences}
    return _world(body)


# -- scenario B: disputes undo exactly (criterion 3) --------------------------

def scenario_dispute() -> dict:
    def body(world):
        memory = world.memory
        memory.remember_fact("boiling point of water", "100 degrees Celsius",
                             confidence=0.5)
        memory.remember_fact("tallest mountain", "K2", confidence=0.6)
        memory.consolidate_fact("tallest mountain country", "Pakistan", 0.7,
                                [("tallest mountain", "K2")])
        memory.consolidate_fact("tallest mountain continent", "Asia", 0.7,
                                [("tallest mountain country", "Pakistan")])
        world.seed([("The capital of Kenya is Nairobi.", ["a.example"]),
                    ("The boiling point of water is 100 degrees Celsius.",
                     ["b.example", "c.example"]),
                    ("The tallest mountain is Mount Everest.", ["d.example"])])
        keys = ["capital of kenya", "boiling point of water",
                "tallest mountain", "tallest mountain country",
                "tallest mountain continent"]
        before = {k: world.record(k) for k in keys}
        approver = world.approver()
        approvals = approver.approve_all()
        outcomes = {a.key: a.outcome for a in approvals}
        revised = world.record("tallest mountain")
        # read NOW: the disputes below restore it (Astra found the exam
        # read it after them, contradicting its own restoration check)
        country_after_approval = world.record("tallest mountain country")
        # a conclusion drawn from the approved value, AFTER the approval
        memory.consolidate_fact("tallest mountain height", "8849 m", 0.7,
                                [("tallest mountain", "Mount Everest")])
        for key in ("capital of kenya", "boiling point of water",
                    "tallest mountain"):
            approver.dispute(key, f"user disputes {key}")
        after = {k: world.record(k) for k in keys}
        entries = world.stash.entries()
        episodes = [e for e in memory.recall_episodes(limit=200)
                    if "dispute" in str(e.get("kind", ""))]
        return {"the three outcomes covered":
                    outcomes == {"capital of kenya": "new",
                                 "boiling point of water": "reinforced",
                                 "tallest mountain": "revised"},
                "the revision really took effect":
                    bool(revised) and revised["value"] == "Mount Everest"
                    and country_after_approval is None,
                "every touched record restored exactly": after == before,
                "a later derivation is retracted":
                    world.record("tallest mountain height") is None,
                "the entries rejected with the reason": all(
                    e["status"] == "rejected" and "user disputes" in
                    (e["notes"] or "") for e in entries),
                "an episode records each dispute": len(episodes) >= 3}
    return _world(body)


# -- scenario C: after a restart (criterion 4) ---------------------------------

def scenario_restart() -> dict:
    def body(world):
        world.seed([("The capital of Kenya is Nairobi.", ["a.example"])])
        first = world.approver()
        first.approve_all()
        from ultraquant.interpreter.autoapprove import AutoApprover
        from ultraquant.interpreter.stash import ContemporaryStash
        second = AutoApprover(ContemporaryStash(world.stash_path),
                              world.memory, world.journal)
        second.dispute("capital of kenya", "disputed after a restart")
        return {"a fresh approver disputes an old approval":
                    world.record("capital of kenya") is None}
    return _world(body)


# -- scenario D: distilled facts (criterion 5) ---------------------------------

def _expected_display(records, qid, promoted):
    """The most common extracted form among samples that agree - computed
    here independently of the filer."""
    from ultraquant.distill import elicit as E
    forms = Counter(E.extract(r.raw) for r in records
                    if r.question_id == qid
                    and not E.is_abstention(r.raw)
                    and E.agree(E.normalize(E.extract(r.raw)), promoted))
    if not forms:
        return None
    best = max(forms.values())
    return sorted(f for f, n in forms.items() if n == best)[0]


def scenario_distilled() -> dict:
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.experiments import knowledge_bench as K

    def body(world):
        records = E.load_records(RECORDS)
        # Amendment B (§11.136, after this gate had passed): filing now
        # refuses a batch holding any invented probe, so the probes are
        # offered on their own and must be refused with nothing filed. That
        # keeps "none invented" and makes it stricter.
        try:
            F.file_distilled(world.stash, records, list(K.FICTITIOUS),
                             MEASURED, "11130")
            probes_refused = False
        except ValueError:
            probes_refused = world.stash.entries() == []
        items = list(K.KNOWN)
        ids = F.file_distilled(world.stash, records, items, MEASURED,
                               "11130")
        approvals = world.approver().approve_all()
        decided = E.decide(records, items)
        by_key = {a.key: a for a in approvals}
        invented = {E.question_id(i) for i in K.FICTITIOUS}
        filed_invented = [e for e in world.stash.entries()
                          if any(q in e["url"] for q in invented)]
        # Amendment A: the recorded run promoted 179 known subjects, not 180
        # (one calibration item was not promoted), so "all" is exactly the
        # ones decide() promotes from the record.
        expected = [i for i in K.KNOWN if decided.get(E.question_id(i))]
        right, display_ok = 0, 0
        for item in expected:
            qid = E.question_id(item)
            match = [a for a in approvals
                     if a.entry_id in ids and qid in world.stash.get(
                         a.entry_id)["url"]]
            if len(match) != 1:
                continue
            value = str(match[0].value)
            right += K.is_correct(item, value)
            display_ok += value == _expected_display(records, qid,
                                                     decided[qid])
        n = len(expected)
        return {"every promoted subject filed, none invented":
                    len(ids) == n and not filed_invented and probes_refused,
                "all approved at the measured confidence":
                    len(approvals) == n and all(
                        math.isclose(a.confidence, MEASURED, abs_tol=1e-9)
                        for a in approvals),
                "every value right": right == n,
                "every value in its display form": display_ok == n,
                "stored as facts": len(by_key) == n}
    return _world(body)


SCENARIOS = (("1-2 who is approved", scenario_who),
             ("3 disputes undo exactly", scenario_dispute),
             ("4 after a restart", scenario_restart),
             ("5 distilled facts", scenario_distilled))


def _plants():
    from ultraquant.interpreter import autoapprove as A

    def deleting(self, before):
        for key in before:
            self.memory._drop_fact(key)

    original_eligible = A.AutoApprover._eligible

    def with_hedged(self, entry):
        return (entry.get("classification") == "hedged"
                or original_eligible(self, entry))

    def key_only(self, key):
        return {key: (copy.deepcopy(self.memory._fact_record(key))
                      if self.memory._fact_record(key) is not None else None)}

    return [
        ("P1 a dispute that deletes", "3 disputes undo exactly",
         "every touched record restored exactly",
         mock.patch.object(A.AutoApprover, "_restore", deleting)),
        ("P2 hedged claims approved", "1-2 who is approved",
         "exactly the eligible approved",
         mock.patch.object(A.AutoApprover, "_eligible", with_hedged)),
        ("P3 malformed claims approved", "1-2 who is approved",
         "malformed rejected with a reason",
         mock.patch.object(A.AutoApprover, "_malformed",
                           lambda self, entry: None)),
        ("P4 no snapshot of retracted derivatives", "3 disputes undo exactly",
         "every touched record restored exactly",
         mock.patch.object(A.AutoApprover, "_snapshot", key_only)),
        ("P5 distilled facts at the default confidence", "5 distilled facts",
         "all approved at the measured confidence",
         mock.patch.object(A.AutoApprover, "_confidence",
                           lambda self, entry: 0.55)),
    ]


@dataclass
class AutoApproveReport:
    """Whether approval is automatic and a dispute undoes it exactly.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        criteria: Scenario -> outcome name -> bool.
        planted: Plant -> whether it breached its target outcome.
        errors: Scenario or plant -> the exception that stopped it.
        reason: Plain-language verdict.
    """

    passes: bool
    criteria: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> AutoApproveReport:
    report = AutoApproveReport(passes=False)
    runners = dict(SCENARIOS)
    for name, scenario in SCENARIOS:
        try:
            report.criteria[name] = scenario()
        except Exception as exc:        # a crash is a failure, not a pass
            report.criteria[name] = {"crashed": False}
            report.errors[name] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, scenario, target, patch in plants:
        try:
            with patch:
                outcome = runners[scenario]().get(target)
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing: not caught
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 5 and all(report.planted.values())
    met = all(all(v.values()) for v in report.criteria.values())
    report.passes = valid and met
    if not valid:
        missed = [n for n, caught in report.planted.items() if not caught]
        report.reason = f"VOID: plants not caught: {missed}"
    elif not met:
        failed = {s: [k for k, v in o.items() if not v]
                  for s, o in report.criteria.items() if not all(o.values())}
        report.reason = f"FAIL: {failed}"
    else:
        report.reason = "PASS: every criterion; 5 of 5 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"criteria": report.criteria, "planted": report.planted,
                      "errors": report.errors}, indent=1))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
