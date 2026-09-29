"""Reads that don't write. The gate.

``ShardVault.get`` ended with ``touch``, which counted the access and
rewrote the whole catalog (tmp + fsync + replace) unless a batch deferred
it. Measured on the user's library (a copy, 197 KiB catalog):
- enumerating the fact keys: 191 catalog writes (1.22 s);
- recalling all 405 facts: 468 writes, about 94 MB fsync'd (1.93 s);
- one catalogue answer: 29 writes, about 5.7 MB (175 ms).
The counts are read only by displays and by ``selflearn``'s packing
threshold, so they are hints. Here they ride along with the next real
write, or with an explicit ``flush_stats()``.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 8a32478a...) before any code:
1. **Reads write nothing**: key enumeration, a recall of every fact and
   ten catalogue answers cause 0 catalog writes.
2. **Counts still land**: ``flush_stats()`` persists exactly the reads
   made, and so does the next real write without it.
3. **A crash loses only hints**: reads and an abandoned vault leave every
   payload byte-identical and every catalog field unchanged except
   ``access_count`` and ``last_access``.
4. **Faster**: timings before and after are reported, with no threshold.
5. **Nothing regresses**: the storage gates (§11.136, §11.137, §11.140)
   and the catalogue gate pass.
6. **The exam can fail**: P58 (``touch`` saves the catalog again)
   breaches 1, and P59 (accesses never recorded) breaches 2.
7. **Every earlier gate stays PASS, and the suite is green.** Checked
   outside this module.

**Reproduced first**: on the committed code, case 1 failed (reads wrote
the catalog), and case 2 could not run (no ``flush_stats``).

**PASSED** on Claude's machine and in Astra's run: 4 of 4 cases, 2 of 2
plants caught. Reads took 3.36 s before and 0.38 s after in Astra's run,
and 3.70 s before and 0.49 s after in Claude's (run beside the suite).
The storage and catalogue gates pass.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

from ultraquant.experiments.catalogue_gate import LIVE_HOME, Library

__all__ = ["ReadsReport", "run_gate"]

QUESTIONS = (
    "What is the atomic number of iron?", "What is the chemical symbol of gold?",
    "What is the capital of Kenya?", "Who wrote Moby-Dick?",
    "What is the atomic number of carbon?", "What is the chemical symbol of sodium?",
    "What is the capital of Japan?", "Who wrote Nineteen Eighty-Four?",
    "What is the capital of Australia?", "What is the atomic number of gold?",
)


# -- the committed touch (the reference for P58), verbatim -------------------------

def _touch_committed(self, shard_id: str) -> None:
    from ultraquant.shards.vault import _utc_now
    entry = self._catalog[shard_id]
    entry["access_count"] = int(entry["access_count"]) + 1
    entry["last_access"] = _utc_now()
    self._save_catalog()


# -- instruments ---------------------------------------------------------------------

@contextlib.contextmanager
def _live_copy():
    root = Path(tempfile.mkdtemp(prefix="uq_reads_")) / "uq_home"
    shutil.copytree(LIVE_HOME, root)
    try:
        yield root
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


@contextlib.contextmanager
def _counting_writes():
    from ultraquant.shards.vault import ShardVault
    real = ShardVault._write_catalog
    count = {"writes": 0}

    def counted(self):
        count["writes"] += 1
        return real(self)

    with mock.patch.object(ShardVault, "_write_catalog", counted):
        yield count


@contextlib.contextmanager
def _counting_gets():
    """Accesses per shard, counted at ``get`` whatever ``touch`` does."""
    from ultraquant.shards.vault import ShardVault
    real = ShardVault.get
    gets = Counter()

    def counted(self, shard_id):
        payload = real(self, shard_id)
        gets[shard_id] += 1
        return payload

    with mock.patch.object(ShardVault, "get", counted):
        yield gets


def _reads(lib) -> None:
    keys = lib.memory.fact_keys()
    for key in keys:
        lib.memory.recall_fact(key)
    for question in QUESTIONS:
        lib.memory.catalogue_answer(question)


def _counts(root: Path) -> dict:
    from ultraquant.shards.vault import ShardVault
    return {e["shard_id"]: int(e["access_count"])
            for e in ShardVault(root / "vault").catalog()}


def _snapshot(root: Path) -> dict:
    """Every shard's bytes and catalog fields, less the access hints."""
    from ultraquant.shards.vault import ShardVault
    vault = ShardVault(root / "vault")
    return {e["shard_id"]: (
                hashlib.sha256(vault.load_bytes(e["shard_id"])).hexdigest(),
                json.dumps({k: v for k, v in e.items()
                            if k not in ("access_count", "last_access")},
                           sort_keys=True, default=str))
            for e in vault.catalog()}


# -- the cases -------------------------------------------------------------------------

def reads_write_nothing() -> bool:
    with _live_copy() as root:
        with _counting_writes() as count:
            _reads(Library(root))
        return count["writes"] == 0


def counts_still_land() -> bool:
    with _live_copy() as root:
        start = _counts(root)
        with _counting_gets() as gets:
            lib = Library(root)
            _reads(lib)
        lib.vault.flush_stats()
        flushed = (sum(gets.values()) > 0
                   and _counts(root) == {sid: n + gets[sid] for sid, n in start.items()})
        middle = _counts(root)
        with _counting_gets() as gets:
            lib = Library(root)
            _reads(lib)
            lib.memory.remember_fact("reads gate probe", "landed", 0.5)
            lib.memory.save()           # a real write, no flush_stats()
        after = _counts(root)
        landed = all(after.get(sid) == n + gets[sid] for sid, n in middle.items())
        return flushed and landed


def crash_loses_only_hints() -> bool:
    with _live_copy() as root:
        before = _snapshot(root)
        lib = Library(root)
        _reads(lib)
        del lib                         # abandoned: no flush, no close
        return _snapshot(root) == before


_TIMES: dict = {}


def _timed(label: str) -> None:
    with _live_copy() as root:
        lib = Library(root)
        started = time.perf_counter()
        _reads(lib)
        _TIMES[label] = round(time.perf_counter() - started, 3)


_GATES = ("review6_gate", "review7_gate", "review8_gate", "catalogue_gate")
_REGRESSIONS: dict = {}


def nothing_regresses() -> bool:
    for name in _GATES:
        module = importlib.import_module(f"ultraquant.experiments.{name}")
        _REGRESSIONS[name] = bool(module.run_gate().passes)
    return all(_REGRESSIONS[name] for name in _GATES)


CASES = {
    "1 reads write nothing": reads_write_nothing,
    "2 counts still land": counts_still_land,
    "3 a crash loses only hints": crash_loses_only_hints,
    "5 nothing regresses": nothing_regresses,
}


def _plants():
    from ultraquant.shards.vault import ShardVault
    return [
        ("P58 touch saves the catalog again", "1 reads write nothing",
         [mock.patch.object(ShardVault, "touch", _touch_committed)]),
        ("P59 accesses never recorded", "2 counts still land",
         [mock.patch.object(ShardVault, "touch", lambda self, shard_id: None)]),
    ]


@dataclass
class ReadsReport:
    """Whether reading the library stops writing it.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        seconds: Criterion 1's reads, timed "after" and "before" (P58).
        regressions: Gate -> whether it passed (criterion 5).
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    regressions: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> ReadsReport:
    report = ReadsReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.regressions = dict(_REGRESSIONS)
    try:
        _timed("after")
        with mock.patch.object(__import__("ultraquant.shards.vault", fromlist=["x"]).ShardVault,
                               "touch", _touch_committed):
            _timed("before")
        report.seconds = dict(_TIMES)
    except Exception as exc:
        report.errors["timing"] = repr(exc)
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
        report.reason = (f"PASS: 4 cases; 2 of 2 plants caught; reads "
                         f"{report.seconds.get('before')} s -> {report.seconds.get('after')} s")
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "seconds": report.seconds,
                      "regressions": report.regressions, "planted": report.planted,
                      "errors": report.errors}, indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
