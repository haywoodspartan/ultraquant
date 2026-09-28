"""Facts catalogued by what they are about. The gate (indexing, stage 1).

The user ruled that "the problem is we are hardcoding instead of
indexing". SPEC.md's guiding idea is a library of patterns catalogued
like a brain. Measured before this unit:
- distillation knew each fact's subject and attribute ("capital:Kenya"),
  then filed only the sentence and the key "capital of kenya";
- buckets were addressed by a key's first two words, so the user's four
  large buckets were four ATTRIBUTES (120 capitals, 100 symbols, 89
  authors, 75 atomic numbers);
- readers guessed the structure back with grammar. The curiosity hint
  fired "If I knew the iqaluit veltrania" for 28 of 30 invented capitals
  and 15 of 20 real places the library does not hold.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 d2c1b9cc...), with Amendment A (sha256 39be3f2b..., recorded
before any code) replacing an unmeasured bound:

1. **Structure from the writer**: new filings carry the subject and
   attribute of their item, and migration gives existing distilled facts
   the structure of their provenance. That is 384 of 384 on a copy of
   the user's library, none for chat facts, and values unchanged.
2. **Catalogued by subject**: every structured fact is in its subject's
   bucket, one subject's facts share a bucket, and no bucket holds more
   than 10% of the structured facts. The attribute buckets held 20-31%.
3. **Nothing is lost**: every fact readable by key; every approval keeps
   its dispute mode; disputes still undo.
4. **Curiosity only within a subject**: 0 of 30 and 0 of 20 hints on the
   rebuilt facts, while a structured bridge within one subject and the
   chat-taught compound bridge still ask for "steel conductivity".
5. **A failed flush changes nothing**: a failure while a structured
   flush writes its index leaves the previous state readable.
6. **The exam can fail**: P30 (addressing ignores structure) breaches 2,
   P31 (curiosity ignores the catalogue) breaches 4, and P32 (migration
   without provenance) breaches 1.
7. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

Amendment B (sha256 155208e7..., recorded before any code) adds the
derivation index. One build of this exam's library took 485 s, because
every approval's snapshot paged every fact to find its derivatives, and
every page rewrote and fsync'd the catalog:
8. **The index is exact**: on 200 random derivation graphs,
   ``derivatives_of`` by index equals the full scan, sharded and not.
9. **The library builds in reasonable time**: at most 60 s (baseline 485).
10. **The exam can fail**: P33 (the index returns nothing) breaches 8.

**PASSED** twice on Claude's machine and twice in Astra's runs: 8 of 8
cases, 4 of 4 plants, and a copy of the user's library migrated
losslessly (384 distilled facts structured from provenance, 405
readable, every approval still exact, disputes still undo). The §11.34
ladder and §11.31 consolidation gates fail, and did before this unit:
bisected to §11.52, which moved the limit they assumed.
"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["CatalogueReport", "run_gate"]

LIVE_HOME = Path(__file__).resolve().parents[2] / "uq_home"

#: The filer's templates, as the exam's expectation: attribute by category.
ATTRIBUTE = {"symbol": "chemical symbol", "capital": "capital",
             "author": "author", "number": "atomic number"}

REAL_NOT_HELD = (
    "the Indian state of Gujarat", "the Indian state of Kerala",
    "the Indian state of Odisha", "the Indian state of Nagaland",
    "the Indian state of Rajasthan", "the Chinese province of Shanxi",
    "the Chinese province of Shaanxi", "the Chinese province of Hebei",
    "the Chinese province of Yunnan", "the Chinese province of Gansu",
    "the Brazilian state of Paraná", "the Brazilian state of Bahia",
    "the Brazilian state of Santa Catarina", "the Brazilian state of Pará",
    "the Brazilian state of Amazonas", "Belize", "Palau", "Myanmar",
    "Tanzania", "Kazakhstan")


class Library:
    """A scratch library laid out as a session lays out ``uq_home``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.open()

    def open(self) -> None:
        from ultraquant.interpreter.stash import ContemporaryStash
        from ultraquant.memory.factshards import FactShards
        from ultraquant.memory.systematic import SystematicMemory
        from ultraquant.shards.vault import ShardVault
        self.vault = ShardVault(self.root / "vault")
        self.shards = FactShards(self.vault)
        self.memory = SystematicMemory(path=self.root / "memory.json")
        self.memory.shards = self.shards
        self.stash = ContemporaryStash(self.root / "stash.json")

    def approver(self):
        from ultraquant.interpreter.autoapprove import AutoApprover
        return AutoApprover(self.stash, self.memory,
                            self.root / "approvals.jsonl")

    def keys(self) -> list:
        return list(self.memory.fact_keys())

    def record(self, key):
        found = self.memory._fact_record(key)
        return copy.deepcopy(found) if found is not None else None

    def distilled(self) -> dict:
        """key -> (subject, attribute) expected from each promoted entry."""
        from ultraquant.interpreter.stash import _claim_provenance
        out = {}
        for entry in self.stash.entries():
            provenance = _claim_provenance(entry)
            if entry["status"] != "promoted" or provenance is None:
                continue
            category, _sep, subject = provenance[1].partition(":")
            split = self.stash._split_entry(entry)
            if split is None or category not in ATTRIBUTE:
                continue
            out[split[0]] = (subject, ATTRIBUTE[category])
        return out

    def bucket_payloads(self) -> dict:
        """bucket id -> {key: record}, read from the vault itself."""
        out = {}
        for entry in self.vault.catalog():
            if entry.get("kind") == "fact-bucket":
                out[entry["shard_id"]] = self.vault.get(
                    entry["shard_id"]).get("facts", {})
        return out


def _scratch(prefix) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _built() -> Library:
    """Both recorded runs filed and auto-approved into a scratch library."""
    from ultraquant.distill import elicit as E
    from ultraquant.distill import file as F
    from ultraquant.experiments import distill_facts_gate as D
    from ultraquant.experiments import distill_hard_gate as DH
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    lib = Library(_scratch("uq_catalogue_"))
    F.file_distilled(lib.stash, E.load_records(D.RECORDS), list(K.KNOWN),
                     0.959, "11130")
    F.file_distilled(lib.stash, E.load_records(DH.RECORDS), list(H.KNOWN),
                     0.964, "11134")
    lib.approver().approve_all()
    lib.memory.save()
    lib.open()                          # read everything back from disk
    return lib


_CACHE: dict = {}


def _shared() -> Library:
    """The built library, built once per context and timed (criterion 9)."""
    if "lib" not in _CACHE:
        import time
        started = time.perf_counter()
        _CACHE["lib"] = _built()
        _CACHE["seconds"] = time.perf_counter() - started
    return _CACHE["lib"]


def _forget_shared() -> None:
    lib = _CACHE.pop("lib", None)
    _CACHE.pop("seconds", None)
    if lib is not None:
        shutil.rmtree(lib.root, ignore_errors=True)


@contextlib.contextmanager
def _library():
    """The shared build, read-only."""
    yield _shared()


@contextlib.contextmanager
def _library_copy():
    """A private copy of the shared build, for cases that write."""
    lib = _shared()
    root = _scratch("uq_catalogue_copy_")
    shutil.rmtree(root)
    shutil.copytree(lib.root, root)
    copy_lib = Library(root)
    try:
        yield copy_lib
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _structured_as_expected(lib, expected) -> bool:
    for key, (subject, attribute) in expected.items():
        record = lib.record(key)
        if (record is None or record.get("subject") != subject
                or record.get("attribute") != attribute):
            return False
    return bool(expected)


# -- 1: structure from the writer ---------------------------------------------------

def new_filings_structured() -> bool:
    with _library() as lib:
        expected = lib.distilled()
        return len(expected) >= 380 and _structured_as_expected(lib, expected)


def _strip_structure(lib) -> Library:
    """A pre-index copy of ``lib``: no structure in records or claims.

    Half of the promoted entries also lose ``fields`` altogether, as the
    user's first 179 did.
    """
    pre = Library(_scratch("uq_catalogue_pre_"))
    for key in lib.keys():
        record = lib.record(key)
        record.pop("subject", None)
        record.pop("attribute", None)
        pre.memory.restore_fact(key, record)
    pre.memory.save()
    data = {"entries": [], "next_id": 1}
    for entry in lib.stash.entries():
        entry = copy.deepcopy(entry)
        fields = entry.get("fields")
        if isinstance(fields, dict):
            fields.pop("subject", None)
            fields.pop("attribute", None)
            if entry["id"] % 2:
                entry.pop("fields")
        data["entries"].append(entry)
        data["next_id"] = max(data["next_id"], entry["id"] + 1)
    (pre.root / "stash.json").write_text(json.dumps(data), encoding="utf-8")
    pre.open()
    return pre


def _migrated_ok(pre, expected, before) -> bool:
    from ultraquant.memory import migrate
    migrate.structure_from_provenance(pre.memory, pre.stash)
    pre.open()                          # restart
    if not _structured_as_expected(pre, expected):
        return False
    for key, record in before.items():
        now = pre.record(key)
        if now is None or now["value"] != record["value"] or \
                now["confidence"] != record["confidence"]:
            return False
        if key not in expected and ("subject" in now or "attribute" in now):
            return False
    return True


def migration_from_provenance() -> bool:
    with _library() as lib:
        expected = lib.distilled()
        pre = _strip_structure(lib)
        try:
            before = {key: pre.record(key) for key in pre.keys()}
            return _migrated_ok(pre, expected, before)
        finally:
            shutil.rmtree(pre.root, ignore_errors=True)


def _live_copy():
    root = _scratch("uq_catalogue_live_") / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    return Library(root)


def live_migration() -> dict | None:
    """1-3 on a copy of the user's library, when it is present."""
    if not (LIVE_HOME / "stash.json").exists():
        return None
    lib = _live_copy()
    try:
        expected = lib.distilled()
        before = {key: lib.record(key) for key in lib.keys()}
        modes_before = _modes(lib)
        structured = _migrated_ok(lib, expected, before)
        readable = all(lib.record(key) is not None for key in before)
        modes_after = _modes(lib)
        placement = _placed_by_subject(lib)
        approvals = lib.approver().approvals()
        v1 = next(a for a in approvals if a.after is None)
        v2 = next(a for a in approvals if a.after is not None)
        approver = lib.approver()
        approver.dispute(v1.key, "catalogue gate, on a copy")
        approver.dispute(v2.key, "catalogue gate, on a copy")
        undone = (lib.record(v1.key) is None and lib.record(v2.key) is None)
        return {"distilled": len(expected), "facts": len(before),
                "structured as provenance": structured,
                "all readable": readable,
                "modes kept": modes_before == modes_after
                and set(modes_after.values()) == {"exact"},
                "placed by subject": placement, "disputes undo": undone}
    finally:
        shutil.rmtree(lib.root.parent, ignore_errors=True)


def _modes(lib) -> dict:
    approver = lib.approver()
    return {a.approval_id: approver._dispute_mode(a)
            for a in approver.approvals() if not a.disputed}


# -- 2: catalogued by subject ---------------------------------------------------------

def _placed_by_subject(lib) -> bool:
    payloads = lib.bucket_payloads()
    where = defaultdict(set)            # subject -> buckets
    structured_in = defaultdict(int)    # bucket -> structured facts
    total = 0
    for bucket, facts in payloads.items():
        for key, record in facts.items():
            subject = record.get("subject")
            if subject is None:
                continue
            total += 1
            structured_in[bucket] += 1
            where[subject].add(bucket)
            if bucket != lib.shards.subject_bucket(subject):
                return False
    if not total:
        return False
    one_bucket_each = all(len(buckets) == 1 for buckets in where.values())
    largest = max(structured_in.values()) / total
    return one_bucket_each and largest <= 0.10


def catalogued_by_subject() -> bool:
    with _library() as lib:
        return _placed_by_subject(lib)


# -- 3: nothing is lost (filed library; the live copy is reported) -------------------

def nothing_lost() -> bool:
    with _library_copy() as lib:
        keys = lib.keys()
        readable = all(lib.record(key) is not None for key in keys)
        approvals = lib.approver().approvals()
        modes = _modes(lib)
        first = approvals[0]
        lib.approver().dispute(first.key, "catalogue gate")
        return (readable and len(approvals) == len(keys)
                and set(modes.values()) == {"exact"}
                and lib.record(first.key) is None)


# -- 4: curiosity only within a subject --------------------------------------------

def curiosity_within_subject() -> bool:
    from ultraquant.experiments import knowledge_bench as K
    from ultraquant.experiments import knowledge_bench_hard as H
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.reason.inference import missing_premise
    with _library() as lib:
        invented = [i.question for i in list(K.FICTITIOUS) + list(H.FICTITIOUS)
                    if i.category == "capital"]
        real = [f"What is the capital of {s}?" for s in REAL_NOT_HELD]
        fired = [q for q in invented + real
                 if missing_premise(q, lib.memory) is not None]
        if len(invented) != 30 or len(real) != 20 or fired:
            return False
    structured = SystematicMemory()
    structured.remember_fact("material of the tower", "steel", 0.9,
                             subject="tower", attribute="material")
    compound = SystematicMemory()
    compound.remember_fact("tower material", "steel", 0.9)
    gap_a = missing_premise("What is the conductivity of the tower?", structured)
    gap_b = missing_premise("What is the tower conductivity?", compound)
    return (gap_a is not None and gap_a["premise_key"] == "steel conductivity"
            and gap_b is not None and gap_b["premise_key"] == "steel conductivity")


# -- 5: a failed flush changes nothing --------------------------------------------

def failed_flush_changes_nothing() -> bool:
    from ultraquant.shards import vault as V
    lib = Library(_scratch("uq_catalogue_flush_"))
    try:
        m = lib.memory
        m.remember_fact("capital of kenya", "Nairobi", 0.9,
                        subject="Kenya", attribute="capital")
        m.remember_fact("capital of ghana", "Accra", 0.9,
                        subject="Ghana", attribute="capital")
        m.save()
        m.remember_fact("capital of kenya", "Mombasa", 0.9,
                        subject="Kenya", attribute="capital")
        m.remember_fact("capital of peru", "Lima", 0.9,
                        subject="Peru", attribute="capital")
        real = V.ShardVault.add_shard

        def index_fails(self, shard_id, category, payload, kind="expert-net",
                        associations=None):
            if kind == "fact-index":
                raise OSError("injected: the index write fails")
            return real(self, shard_id, category, payload, kind=kind,
                        associations=associations)

        with mock.patch.object(V.ShardVault, "add_shard", index_fails):
            try:
                m.save()
            except OSError:
                pass
        lib.open()                      # restart
        return (lib.record("capital of kenya") is not None
                and lib.record("capital of kenya")["value"] == "Nairobi"
                and lib.record("capital of ghana")["value"] == "Accra"
                and lib.record("capital of peru") is None)
    finally:
        shutil.rmtree(lib.root, ignore_errors=True)


# -- 8-9: the derivation index (Amendment B) ----------------------------------------

def _scan(records: dict, key) -> set:
    """The old derivatives_of, over a snapshot: the closure by record."""
    found, stack = set(), [key]
    while stack:
        changed = stack.pop()
        for name, record in records.items():
            if name in found or not record or "derived_from" not in record:
                continue
            if any(p_key == changed for p_key, _v in record["derived_from"]):
                found.add(name)
                stack.append(name)
    return found


def _graph_memory(root: Path, sharded: bool):
    from ultraquant.memory.systematic import SystematicMemory
    memory = SystematicMemory(path=root / "memory.json")
    if sharded:
        from ultraquant.memory.factshards import FactShards
        from ultraquant.shards.vault import ShardVault
        memory.shards = FactShards(ShardVault(root / "vault"))
    return memory


def index_exact() -> bool:
    """8: 200 random derivation graphs; the index against the full scan."""
    import random
    for trial in range(200):
        rnd = random.Random(1000 + trial)
        sharded = trial % 2 == 1
        root = _scratch("uq_catalogue_graph_")
        try:
            memory = _graph_memory(root, sharded)
            names = [f"node {i} of graph {trial}" for i in range(10)]
            for name in names[:3]:
                memory.remember_fact(name, f"v{rnd.randrange(3)}", 0.6)
            restarted = False
            for step in range(16):
                held = list(memory.fact_keys())
                op = rnd.random()
                if op < 0.45 and held:
                    target = rnd.choice(names)
                    premises = [p for p in rnd.sample(
                        held, k=min(len(held), rnd.randint(1, 3))) if p != target]
                    if premises:
                        memory.consolidate_fact(
                            target, f"d{step}", 0.7,
                            [(p, memory._fact_record(p)["value"])
                             for p in premises])
                elif op < 0.65 and held:
                    memory.remember_fact(rnd.choice(held), f"r{step}", 0.8)
                elif op < 0.8 and held:
                    memory.restore_fact(rnd.choice(held), None)
                elif op < 0.9 and held:
                    record = copy.deepcopy(memory._fact_record(rnd.choice(held)))
                    memory.restore_fact(rnd.choice(names), record)
                elif not restarted:
                    memory.save()
                    memory = _graph_memory(root, sharded)
                    restarted = True
                records = {k: memory._fact_record(k)
                           for k in memory.fact_keys()}
                for name in names:
                    if set(memory.derivatives_of(name)) != _scan(records, name):
                        return False
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return True


def builds_in_time() -> bool:
    """9: filing both runs and auto-approving them, at most 60 s."""
    _shared()
    return _CACHE["seconds"] <= 60.0


CASES = {
    "1a new filings carry their structure": new_filings_structured,
    "1b migration from provenance": migration_from_provenance,
    "2 catalogued by subject": catalogued_by_subject,
    "3 nothing is lost": nothing_lost,
    "4 curiosity only within a subject": curiosity_within_subject,
    "5 a failed flush changes nothing": failed_flush_changes_nothing,
    "8 the derivation index is exact": index_exact,
    "9 the library builds in reasonable time": builds_in_time,
}


def _plants():
    from ultraquant.memory import factshards as FS
    from ultraquant.memory import migrate as M
    from ultraquant.reason import inference as I
    return [
        ("P30 addressing ignores structure", "2 catalogued by subject",
         [mock.patch.object(FS.FactShards, "address",
                            lambda self, key, record: self.bucket_of(key))]),
        ("P31 curiosity ignores the catalogue",
         "4 curiosity only within a subject",
         [mock.patch.object(I, "_bridge_allowed",
                            lambda via_record, question_subjects: True)]),
        ("P32 migration without provenance", "1b migration from provenance",
         [mock.patch.object(M, "_structure_of", lambda entry: None)]),
        ("P33 the index returns nothing", "8 the derivation index is exact",
         [mock.patch.object(_systematic().SystematicMemory,
                            "_derived_candidates", lambda self, key: set())]),
    ]


def _systematic():
    from ultraquant.memory import systematic
    return systematic


@dataclass
class CatalogueReport:
    """Whether facts are catalogued by what they are about.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        live: Criteria 1-3 on a copy of the user's library, or None when
            it is not present.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    live: dict | None = None
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def _live_ok(live) -> bool:
    if live is None:
        return True
    return (live["distilled"] == 384 and live["facts"] == 405
            and all(live[k] for k in ("structured as provenance",
                                      "all readable", "modes kept",
                                      "placed by subject", "disputes undo")))


def run_gate() -> CatalogueReport:
    report = CatalogueReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.live = live_migration()
    except Exception as exc:
        report.live = {"error": repr(exc)}
        report.errors["live copy"] = repr(exc)
    try:
        plants = _plants()
    except Exception as exc:
        report.errors["plants"] = repr(exc)
        plants = []
    for name, target, patches in plants:
        # Addressing is decided at write time: P30's library is built under
        # the plant, never borrowed from the unplanted build.
        rebuild = name.startswith("P30")
        if rebuild:
            _forget_shared()
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
            if rebuild:
                _forget_shared()
    _forget_shared()
    live_ok = "error" not in (report.live or {}) and _live_ok(report.live)
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = all(report.cases.values()) and live_ok
    report.passes = valid and met
    if not valid:
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        failed = [n for n, ok in report.cases.items() if not ok]
        if not live_ok:
            failed.append("the user's library (copy)")
        report.reason = f"FAIL: {failed}"
    else:
        live = ("the user's library migrated losslessly on a copy"
                if report.live else "the user's library not present")
        report.reason = f"PASS: 8 cases; 4 of 4 plants caught; {live}"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "live": report.live,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
