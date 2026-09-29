"""Merge what a staged session learned into the live library, keeping the live library's own changes.

Usage:
    python tools/merge_session.py STAGING_ROOT LIVE_ROOT --backup-dir DIR [--base DIR] [--check]

A session (ultraquant.distill.session) run on a staging copy backs the copy
up first; that backup is the BASE both libraries started from (by default
the one ``uq_home-*`` directory beside the staging root, in ``backups``).

- **Nothing changed in the live library** (byte-identical to the base): the
  staging library replaces it whole. Exact.
- **The live library changed meanwhile** (the GUI saved): the session is
  replayed into it, exactly as the session wrote it:
  1. refuse unless the live ledger is a prefix of the staging ledger
     (nothing else asked sources since the copy was taken);
  2. the session's claims (provenance run_id "session-...") in staging
     order, approved one run_id at a time - the session approved once per
     round, so the batches, and which claim wins, are the same;
  3. the teachers staging added to claims both libraries hold;
  4. the kinds and property verdicts the session learned (its difference
     from the base);
  5. the staging ledger.

Either way the result is checked three ways against the base - every
structured fact's value and the attribute vocabulary (kinds, None kinds
included; property verdicts; adoptions; the words each attribute was asked
by) - and the exit code is nonzero unless it is clean:
- "lost": a value the session set that the merge did not reproduce;
- "overwritten": a value the live library set that the merge changed;
- "drifted": a value neither side changed that the merge changed;
- "touched": a stash entry not from the session whose status differs from
  the three-way expectation (the session's effect on entries both sides
  share; the live library's own entries left exactly as they were);
- "conflicts" (reported, not failing): a key both sides changed differently.

While the session's approvals replay, the stash entries the session never
saw (added to the live library since the copy) are hidden from the
approver: it promoted, rejected and re-analysed on a copy without them.

The merge always runs on a scratch copy of the live library first. --check
stops there; otherwise the live library is backed up and written only when
that check is clean.
"""

import argparse
import contextlib
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _open(root: Path):
    from ultraquant.interpreter.autoapprove import AutoApprover
    from ultraquant.interpreter.stash import ContemporaryStash
    from ultraquant.memory.factshards import FactShards
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.shards.vault import ShardVault
    memory = SystematicMemory(path=root / "memory.json")
    if (root / "vault").exists():
        memory.shards = FactShards(ShardVault(root / "vault"))
    stash = ContemporaryStash(root / "stash.json")
    return memory, stash, AutoApprover(stash, memory, root / "approvals.jsonl")


def _hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _provenance_id(entry):
    provenance = entry.get("provenance") or {}
    return provenance.get("run_id"), provenance.get("question_id")


def _from_session(entry) -> bool:
    return str(_provenance_id(entry)[0] or "").startswith("session-")


def _vocabulary(memory) -> dict:
    """What a session learns about attributes: kinds (None too), property verdicts, adoptions."""
    out = {}
    for attribute, item in memory._attribute_vocabulary().items():
        if "kind" in item:
            out[("kind", attribute)] = repr(item["kind"])
        if "extends" in item:
            out[("extends", attribute)] = str(item["extends"])
        for prop, verdict in (item.get("properties") or {}).items():
            out[("property", attribute, prop)] = str(verdict)
        for word, subjects in (item.get("asked_by") or {}).items():
            for subject in subjects:
                out[("asked", attribute, word, subject)] = "1"
    return out


def _state(root: Path) -> dict:
    """Everything the three-way check compares: structured facts and the vocabulary."""
    memory, _, _ = _open(root)
    out = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        if record.get("subject") and record.get("attribute"):
            out[("fact", key)] = str(record.get("value"))
    out.update(_vocabulary(memory))
    return out


def _label(item) -> str:
    return f"{item[0]}: " + " / ".join(item[1:])


def _statuses(root: Path) -> dict:
    _, stash, _ = _open(root)
    return {e["id"]: e["status"] for e in stash.entries() if not _from_session(e)}


@contextlib.contextmanager
def _as_the_session_saw_it(stash, hidden):
    """Hide the entries the session never saw while its approvals replay.

    The stash numbers entries from a counter (``next_id``), never reusing an
    id, so replayed claims cannot take a hidden entry's id.
    """
    held = {i: stash._entries.pop(i) for i in sorted(hidden) if i in stash._entries}
    try:
        yield
    finally:
        stash._entries.update(held)
        stash.save()


def default_base(staging: Path) -> Path:
    found = sorted((staging.parent / "backups").glob(f"{staging.name}-*"))
    if len(found) != 1:
        raise ValueError(f"expected one base backup beside {staging}, found {len(found)}: pass --base")
    return found[0]


def replay(staging: Path, live: Path, base: Path) -> dict:
    """Replay a staged session's writes into ``live`` (which must already be backed up)."""
    staging_ledger = json.loads((staging / "sources.json").read_text(encoding="utf-8")) \
        if (staging / "sources.json").exists() else {}
    live_ledger = json.loads((live / "sources.json").read_text(encoding="utf-8")) \
        if (live / "sources.json").exists() else {}
    for source, rows in live_ledger.items():
        if staging_ledger.get(source, [])[:len(rows)] != rows:
            raise ValueError(f"the live ledger for {source!r} is not a prefix of staging's: "
                             "something else asked sources since the copy was taken")
    s_memory, s_stash, _ = _open(staging)
    b_memory, b_stash, _ = _open(base)
    memory, stash, approver = _open(live)
    known = {_provenance_id(e) for e in stash.entries()}
    hidden = {e["id"] for e in stash.entries()} - {e["id"] for e in b_stash.entries()}

    # The session's claims, in staging order, approved one run_id (one round) at a time.
    replayed, batches, current = 0, 0, None
    with _as_the_session_saw_it(stash, hidden):
        for entry in sorted(s_stash.entries(), key=lambda e: e["id"]):
            if not _from_session(entry):
                continue
            run_id = _provenance_id(entry)[0]
            if current is not None and run_id != current:
                approver.approve_all()
                batches += 1
            current = run_id
            if _provenance_id(entry) in known:
                continue
            stash.add_claim(entry["url"], entry["title"], entry["claim"],
                            measured_confidence=entry.get("measured_confidence"),
                            provenance=entry.get("provenance"), fields=entry.get("fields"))
            replayed += 1
        if current is not None:
            approver.approve_all()
            batches += 1

    # The teachers staging added to claims both hold.
    from ultraquant.distill import sources
    added = 0
    by_id = {_provenance_id(e): e for e in stash.entries() if _provenance_id(e) != (None, None)}
    for entry in s_stash.entries():
        mine = by_id.get(_provenance_id(entry))
        if mine is None:
            continue
        theirs = entry.get("provenance") or {}
        names = theirs.get("teachers") or []
        ids = theirs.get("teacher_ids") or [sources.identity(n) for n in names]
        for name, teacher_id in zip(names, ids):
            ours = stash.get(mine["id"]).get("provenance") or {}
            if name in (ours.get("teachers") or []):
                continue
            prior = ours.get("teacher_ids")
            if prior is None:
                prior = [sources.identity(n) for n in ours.get("teachers") or []]
            stash.add_teacher(mine["id"], name, teacher_id, prior_teacher_ids=prior)
            added += 1

    # The kinds (a None kind too: asked, and no shared kind) and property
    # verdicts the session learned: exactly its difference from the base.
    learned = 0
    before = b_memory._attribute_vocabulary()
    for attribute, item in s_memory._attribute_vocabulary().items():
        was = before.get(attribute, {})
        name = item.get("name", attribute)
        if "kind" in item and ("kind" not in was or was["kind"] != item["kind"]):
            memory.learn_kind(name, item["kind"])
            learned += 1
        for prop, verdict in (item.get("properties") or {}).items():
            if (was.get("properties") or {}).get(prop) != verdict:
                memory.learn_property(name, prop, verdict)
                learned += 1
    memory.save()

    # The ledger.
    (live / "sources.json").write_text(json.dumps(staging_ledger, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    return {"replayed": replayed, "batches": batches, "teachers added": added,
            "vocabulary learned": learned, "hidden": len(hidden)}


def _replay(staging: Path, live: Path, base: Path) -> dict:
    """``replay``, looked up as it is called: ``merge``'s keyword of that name hides it."""
    return replay(staging, live, base)


def _replace(staging: Path, live: Path) -> None:
    """Replace ``live`` with a copy of ``staging`` (the live library is already backed up)."""
    incoming = live.with_name(live.name + ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(staging, incoming)
    outgoing = live.with_name(live.name + ".outgoing")
    shutil.rmtree(outgoing, ignore_errors=True)
    live.rename(outgoing)          # fails, and changes nothing, if a file in it is open
    incoming.rename(live)
    shutil.rmtree(outgoing, ignore_errors=True)


def merge(staging: Path, live: Path, base: Path, *, replay=False) -> dict:
    """Merge ``staging`` into ``live`` (already backed up); check the result against ``base``.

    ``replay=True`` replays even into a live library unchanged since the base
    (§11.171): the GUI merges into the library it holds open, which a
    replacement would move.
    """
    b_values, s_values, l0_values = _state(base), _state(staging), _state(live)
    b_statuses, s_statuses, l0_statuses = _statuses(base), _statuses(staging), _statuses(live)
    unchanged = not replay and _hashes(live) == _hashes(base)
    if unchanged:
        _replace(staging, live)
        report = {"mode": "the live library was unchanged: replaced by the staging library"}
    else:
        report = {"mode": "the live library changed meanwhile: the session replayed into it",
                  **_replay(staging, live, base)}
    l1_values, l1_statuses = _state(live), _statuses(live)
    keys = set(b_values) | set(s_values) | set(l0_values) | set(l1_values)
    session = {k for k in keys if s_values.get(k) != b_values.get(k)}
    gui = {k for k in keys if l0_values.get(k) != b_values.get(k)}

    def labels(found):
        return sorted(_label(k) for k in found)
    report.update({
        "facts": sum(1 for k in l1_values if k[0] == "fact"),
        "session changed": len(session), "live changed": len(gui),
        "lost": labels(k for k in session - gui if l1_values.get(k) != s_values.get(k)),
        "overwritten": labels(k for k in gui - session if l1_values.get(k) != l0_values.get(k)),
        "drifted": labels(k for k in keys - session - gui if l1_values.get(k) != b_values.get(k)),
        "touched": sorted(i for i in l0_statuses if l1_statuses.get(i) != (
            s_statuses.get(i, b_statuses.get(i)) if b_statuses.get(i) == l0_statuses[i]
            else l0_statuses[i])),
        "conflicts": labels(k for k in session & gui if l0_values.get(k) != s_values.get(k)),
    })
    report["clean"] = not any(report[name] for name in ("lost", "overwritten", "drifted", "touched"))
    return report


def _check(staging: Path, live: Path, base: Path, *, replay=False) -> dict:
    """Merge into a scratch copy of ``live``: the check, which never writes ``live``."""
    scratch = Path(tempfile.mkdtemp(prefix="uq_merge_check_")) / live.name
    shutil.copytree(live, scratch)
    try:
        report = merge(staging, scratch, base, replay=replay)
    finally:
        shutil.rmtree(scratch.parent, ignore_errors=True)
    report["check"] = "on a scratch copy; the live library was not written"
    return report


def checked_merge(staging: Path, live: Path, base: Path, backup_dir: Path, *, replay=False) -> dict:
    """Merge ``staging`` into ``live`` only after the same merge checks clean on a scratch copy.

    The command line's merge, as a function: the GUI takes a finished session
    in itself through it, always by the replay (§11.171). Clean: ``live`` is
    backed up into ``backup_dir`` and merged into, and that report comes back
    with its "backup". Not clean: the check's report comes back and ``live`` is
    not written. Errors (a ledger that moved on) propagate.
    """
    staging, live, base = Path(staging), Path(live), Path(base)
    report = _check(staging, live, base, replay=replay)
    if not report["clean"]:
        return report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(backup_dir) / f"{live.name}-{stamp}-pre-merge"
    shutil.copytree(live, backup)
    report = merge(staging, live, base, replay=replay)
    report["backup"] = str(backup)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("staging", type=Path)
    parser.add_argument("live", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    base = args.base or default_base(args.staging)
    # Always merge into a scratch copy first; the live library is written only after a clean check.
    if args.check:
        report = _check(args.staging, args.live, base)
    else:
        report = checked_merge(args.staging, args.live, base, args.backup_dir)
    report["base"] = str(base)
    print(json.dumps(report, indent=2))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
