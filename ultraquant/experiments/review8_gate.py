"""Addresses that stay put, flushes that finalize late, undo that keeps
structure. The gate for GPT-6 Astra's eighth review.

Claude reproduced each finding on 7599e4d (§11.139):
- **a migration could keep a stale duplicate.** §11.139 made
  ``bucket_of`` normalize its prefix, so "the tower material" moved
  address, and the one-time scan forgot the old one. With the current
  "steel" in the old bucket and an obsolete "wood" in the full-key
  bucket, only "wood" survived. The user's library has no key beginning
  with an article; its 405 values were verified unchanged;
- **a nested flush committed its index early.** Inside an outer batch
  that then rolled back, it still cleared its staged buckets and
  advanced its index baselines. After a later unrelated flush, "k" was
  listed but unreadable;
- **undo stripped catalogue structure.** Disputing an approval
  journalled before structure existed restored a record with no subject
  or attribute.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 6bcea2d4...), with Amendment A (sha256 55811844..., recorded
before any fix) correcting criterion 2 to write-behind semantics:

1. **A migration keeps the current record** for keys beginning "the",
   "a" and "an", and no unstructured key's address differs from its
   pre-§11.139 address (1,000 random keys).
2. **A nested flush commits with its outer batch**: after a rollback and
   a later flush, every listed key is readable and "k" holds what memory
   held (Ghana); after a commit, the nested writes are readable.
3. **Undo keeps structure**, for reinforced and revised approvals and for
   a dispute replayed after a crash.
4. **The user's library** (a copy): 405 facts readable, 384 approvals
   exact, and a question about Kenya names Kenya.
5. **The exam can fail**: P37 (the normalizing ``bucket_of``) breaches 1,
   P38 (finalize immediately) breaches 2, and P39 (restore records as
   journalled) breaches 3.
6. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**PASSED** twice on Claude's machine and in Astra's run: 3 of 3 cases, 3
of 3 plants, and a copy of the user's library intact. §11.139's
catalogue gate and every earlier gate listed stayed green.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library

__all__ = ["Review8Report", "run_gate"]


def _scratch(prefix) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _hash_bucket(text: str, buckets: int = 256) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return f"fact:{int.from_bytes(digest, 'big') % buckets:03d}"


def old_prefix_bucket(key: str) -> str:
    """``bucket_of`` exactly as it was before §11.139."""
    tokens = re.findall(r"[a-z0-9]+", key.lower())
    return _hash_bucket(" ".join(tokens[:2]) or key.lower())


def _record(value, stamp):
    return {"value": value, "confidence": 0.9, "reinforcements": 0,
            "first_seen": stamp, "last_seen": stamp}


# -- 1: addresses stay put ---------------------------------------------------------

def migration_keeps_current() -> bool:
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards.vault import ShardVault
    for article in ("the", "a", "an"):
        key = f"{article} tower material"
        current, stale = old_prefix_bucket(key), _hash_bucket(key.lower())
        if current == stale:
            continue
        root = _scratch("uq_r8_mig_")
        try:
            vault = ShardVault(root)
            vault.add_shard(current, current,
                            {"facts": {key: _record("steel", "t2")}},
                            kind="fact-bucket")
            vault.add_shard(stale, stale,
                            {"facts": {key: _record("wood", "t1")}},
                            kind="fact-bucket")
            shards = FactShards(ShardVault(root))
            shards.get(key)
            shards.flush()
            again = FactShards(ShardVault(root)).get(key)
            if again is None or again["value"] != "steel":
                return False
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return True


def addresses_unchanged() -> bool:
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards.vault import ShardVault
    rnd = random.Random(8)
    words = ["the", "a", "an", "tower", "Kenya", "capital", "of", "height",
             "The", "An", "x-ray", "café", "zorbenium", "1945", "Ärzte"]
    root = _scratch("uq_r8_addr_")
    try:
        shards = FactShards(ShardVault(root))
        for _ in range(1000):
            key = " ".join(rnd.choice(words)
                           for _ in range(rnd.randint(1, 5))).lower()
            if shards.bucket_of(key) != old_prefix_bucket(key):
                return False
        return True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def addresses_stay_put() -> bool:
    """1: both halves of criterion 1."""
    return migration_keeps_current() and addresses_unchanged()


# -- 2: a nested flush finalizes with its outer batch -----------------------------

def nested_rollback_consistent() -> bool:
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards.vault import ShardVault
    root = _scratch("uq_r8_nest_")
    try:
        shards = FactShards(ShardVault(root))
        shards.put("k", {"value": 1, "subject": "Kenya", "attribute": "a"})
        shards.flush()
        try:
            with shards.vault.batch():
                shards.put("k", {"value": 2, "subject": "Ghana",
                                 "attribute": "a"})
                shards.flush()
                raise OSError("injected: the outer transaction fails")
        except OSError:
            pass
        page = shards._index_id("keys", "k")
        extra = next(f"extra{i}" for i in range(10 ** 5)
                     if shards._index_id("keys", f"extra{i}") == page)
        shards.put(extra, {"value": 3, "subject": "Peru", "attribute": "a"})
        shards.flush()
        fresh = FactShards(ShardVault(root))
        listed = fresh.keys()
        readable = all(fresh.get(key) is not None for key in listed)
        k = fresh.get("k")
        return (readable and "k" in listed and k is not None
                and k["value"] == 2 and k["subject"] == "Ghana")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def nested_commit_readable() -> bool:
    from ultraquant.memory.factshards import FactShards
    from ultraquant.shards.vault import ShardVault
    root = _scratch("uq_r8_nestok_")
    try:
        shards = FactShards(ShardVault(root))
        with shards.vault.batch():
            for i, subject in enumerate(("Kenya", "Ghana", "Peru")):
                shards.put(f"key {i}", {"value": i, "subject": subject,
                                        "attribute": "a"})
            shards.flush()
        fresh = FactShards(ShardVault(root))
        return all(fresh.get(f"key {i}") is not None
                   and fresh.get(f"key {i}")["value"] == i for i in range(3))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def nested_flush() -> bool:
    """2: both halves of criterion 2."""
    return nested_rollback_consistent() and nested_commit_readable()


# -- 3: undo keeps structure ---------------------------------------------------------

def _seed_distilled(lib, value="Nairobi") -> None:
    """A distilled claim filed before structure existed (no subject)."""
    lib.stash.add_claim(
        "https://distill.invalid/11130/capital:Kenya",
        "What is the capital of Kenya?", f"The capital of Kenya is {value}.",
        measured_confidence=0.959,
        provenance={"run_id": "11130", "question_id": "capital:Kenya"},
        fields={"key": "capital of kenya", "value": value})


def _structured(lib, value) -> bool:
    record = lib.record("capital of kenya")
    return (record is not None and record["value"] == value
            and record.get("subject") == "Kenya"
            and record.get("attribute") == "capital"
            and "kenya" in lib.memory.subjects_in(
                "What is the capital of Kenya?"))


def _undo_scenario(held_value, crash=False) -> bool:
    from ultraquant.interpreter.autoapprove import AutoApprover
    from ultraquant.memory import migrate
    lib = Library(_scratch("uq_r8_undo_"))
    try:
        lib.memory.remember_fact("capital of kenya", held_value, 0.6)
        lib.memory.save()
        _seed_distilled(lib)
        approver = lib.approver()
        approver.approve_all()
        migrate.structure_from_provenance(lib.memory, lib.stash)
        approver = lib.approver()
        if crash:
            def failed(self, *args, **kwargs):
                raise OSError("injected: power cut before the commit")
            with mock.patch.object(AutoApprover, "_persist", failed):
                try:
                    approver.dispute("capital of kenya", "undo it")
                except OSError:
                    pass
            lib.open()
            lib.approver()              # recovery completes the dispute
        else:
            approver.dispute("capital of kenya", "undo it")
            lib.open()
        return _structured(lib, held_value)
    finally:
        shutil.rmtree(lib.root, ignore_errors=True)


def undo_keeps_structure() -> bool:
    """3: reinforced, revised, and replayed after a crash."""
    return (_undo_scenario("Nairobi") and _undo_scenario("Mombasa")
            and _undo_scenario("Nairobi", crash=True))


# -- 4: the user's library ------------------------------------------------------------

def live_library() -> bool | None:
    if not (LIVE_HOME / "stash.json").exists():
        return None
    root = _scratch("uq_r8_live_") / "uq_home"
    try:
        shutil.copytree(LIVE_HOME, root)
        lib = Library(root)
        keys = lib.keys()
        readable = all(lib.record(key) is not None for key in keys)
        approver = lib.approver()
        modes = [approver._dispute_mode(a) for a in approver.approvals()
                 if not a.disputed]
        return (len(keys) == 405 and readable and len(modes) == 384
                and set(modes) == {"exact"}
                and "kenya" in lib.memory.subjects_in(
                    "What is the capital of Kenya?"))
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


CASES = {
    "1 addresses stay put": addresses_stay_put,
    "2 a nested flush commits with its outer batch": nested_flush,
    "3 undo keeps structure": undo_keeps_structure,
}


def _plants():
    from ultraquant.interpreter import autoapprove as A
    from ultraquant.memory import factshards as FS
    from ultraquant.shards import vault as V

    def normalizing(self, key):
        return self.subject_bucket(" ".join(self.tokens(key)[:2]) or key.lower())

    return [
        ("P37 the normalizing bucket_of", "1 addresses stay put",
         [mock.patch.object(FS.FactShards, "bucket_of", normalizing)]),
        ("P38 finalize immediately",
         "2 a nested flush commits with its outer batch",
         [mock.patch.object(V.ShardVault, "after_batch",
                            lambda self, callback: callback(True))]),
        ("P39 restore records as journalled", "3 undo keeps structure",
         [mock.patch.object(A.AutoApprover, "_with_catalogue",
                            lambda self, key, record: record)]),
    ]


@dataclass
class Review8Report:
    """The eighth review's fixes, measured.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        live: Criterion 4 on a copy of the user's library, or None.
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


def run_gate() -> Review8Report:
    report = Review8Report(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    try:
        report.live = live_library()
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
    valid = len(report.planted) == 3 and all(report.planted.values())
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
        live = ("the user's library intact" if report.live
                else "the user's library not present")
        report.reason = f"PASS: 3 cases; 3 of 3 plants caught; {live}"
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
