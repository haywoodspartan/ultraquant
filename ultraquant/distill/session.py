"""Run an ordered study session, backing up the library and restoring models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import elicit, frontier, session, sources
from ultraquant.interpreter.autoapprove import AutoApprover
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.memory.factshards import FactShards, normalize_subject
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


PLAN_PATH = Path(__file__).with_name("data") / "session.json"
TOTALS = ("asked", "filed", "agreed", "contested", "revised")


@dataclass(frozen=True)
class SourcePlan:
    name: str
    gguf: str
    context_length: int


def load_plan(path=None) -> list[SourcePlan]:
    data = json.loads(Path(path or PLAN_PATH).read_text(encoding="utf-8"))
    return [SourcePlan(row["name"], row["gguf"], data["context_length"])
            for row in data["sequence"]]


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def backup(root, backup_dir) -> Path:
    """Copy the entire library, refusing a destination inside the library."""
    root, backup_dir = Path(root).resolve(), Path(backup_dir).resolve()
    if backup_dir == root or root in backup_dir.parents:
        raise ValueError("backup_dir must be outside the library")
    destination = backup_dir / f"{root.name}-{_stamp()}"
    shutil.copytree(root, destination)
    return destination


class LMStudioSwapper:
    def __init__(self, cli, run=None):
        self.cli = str(cli)
        self.run = subprocess.run if run is None else run

    def _command(self, *args):
        # lms prints UTF-8 (its progress bars too); the Windows code page cannot
        # decode every byte of it (§11.166 live).
        done = self.run([self.cli, *args], capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
        done.check_returncode()
        return done

    def snapshot(self):
        return json.loads(self._command("ps", "--json").stdout)

    def load(self, name, context_length):
        current = self.snapshot()
        for model in current:
            if model.get("type") == "llm" and model["identifier"] != name:
                self._command("unload", model["identifier"])
        if not any(model["identifier"] == name for model in current):
            self._command("load", name, "--context-length", str(context_length), "-y")

    def restore(self, snapshot):
        wanted = {model["identifier"]: model for model in snapshot}
        current = {model["identifier"]: model for model in self.snapshot()}
        settings = ("contextLength", "parallel", "ttlMs")
        for name, model in list(current.items()):
            if name not in wanted or any(
                    model.get(key) != wanted[name].get(key) for key in settings):
                self._command("unload", name)
                del current[name]
        for name, model in wanted.items():
            if name in current:
                continue
            args = ["load", name, "--context-length", str(model["contextLength"])]
            if model.get("parallel") is not None:
                args.extend(["--parallel", str(model["parallel"])])
            if model.get("ttlMs") is not None:
                args.extend(["--ttl", format(Decimal(str(model["ttlMs"])) / 1000, "f")])
            self._command(*args, "-y")


def _open_library(root):
    memory = SystematicMemory(path=root / "memory.json")
    if (root / "vault").exists():
        memory.shards = FactShards(ShardVault(root / "vault"))
    return memory, ContemporaryStash(root / "stash.json")


def _co_distilled(stash, pairs, name):
    """Count pairs whose promoted claims name this source's weights."""
    identity = sources.identity(name)
    held = set()
    for entry in stash.entries(status="promoted"):
        provenance = entry.get("provenance") or {}
        identities = provenance.get("teacher_ids")
        if identities is None:
            identities = [sources.identity(t) for t in provenance.get("teachers", [])]
        fields = entry.get("fields") or {}
        if identity in identities and fields.get("subject") and fields.get("attribute"):
            held.add((normalize_subject(fields["subject"]),
                      normalize_subject(fields["attribute"])))
    return sum((normalize_subject(t.subject), normalize_subject(t.attribute)) in held
               for t, _ in pairs)


def run_session(root, plan, *, swapper, teacher_factory, backup_dir,
                report_path, max_rounds=8, pairs=None) -> dict:
    backup_path = session.backup(root, backup_dir)
    root, report_path = Path(root), Path(report_path)
    report = {"status": "complete", "failed": None, "error": None,
              "backup": str(backup_path), "sources": []}
    snapshot = memory = error = active = None
    stamp = _stamp()

    def failed(exc):
        nonlocal error
        if error is None:
            error = exc
            report.update(status="failed", failed=active, error=repr(exc))
        else:
            error.add_note(f"Session cleanup also failed: {exc!r}")
            report["error"] += f"; cleanup: {exc!r}"

    try:
        snapshot = swapper.snapshot()
        memory, stash = _open_library(root)
        approver = AutoApprover(stash, memory, root / "approvals.jsonl")
        ledger = sources.SourceLedger(root / "sources.json")
        pairs = list(pairs or sources.calibration_items(memory, stash, k=40, seed=155))
        # Keep scratch records outside the library and its byte-identical backup.
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"session-{stamp}-", dir=backup_dir) as scratch:
            for index, source in enumerate(plan):
                active = source.name
                swapper.load(source.name, source.context_length)
                teacher = teacher_factory(source)
                records = elicit.elicit(teacher, source.name, [t for t, _ in pairs],
                                        Path(scratch) / f"{index}-calibration.jsonl")
                calibration = sources.calibrate(records, pairs)
                entry = {"name": source.name,
                         "calibration": {"right": calibration["right"],
                                         "decided": calibration["promoted"],
                                         "bound": calibration["wilson_lower"]},
                         "co_distilled": _co_distilled(stash, pairs, source.name),
                         "rounds": [], "totals": dict.fromkeys(TOTALS, 0)}
                report["sources"].append(entry)
                for n in range(1, max_rounds + 1):
                    result = frontier.study_round(
                        memory, stash, teacher, ledger, source.name,
                        confidence=calibration["wilson_lower"],
                        run_id=f"session-{stamp}-{source.name}-{n}",
                        records_path=Path(scratch) / f"{index}-{n}.jsonl",
                        approver=approver)
                    entry["rounds"].append(result)
                    for key in TOTALS:
                        entry["totals"][key] += result.get(key, 0)
                    if result["used_up"]:
                        break
                active = None
    except BaseException as exc:
        failed(exc)
    finally:
        try:
            if snapshot is not None:
                swapper.restore(snapshot)
        except BaseException as exc:
            failed(exc)
        try:
            if memory is not None:
                memory.save()
        except BaseException as exc:
            failed(exc)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if error is not None:
        raise error
    return report


class _ReadOnlyVault(ShardVault):
    """Only catalog/get reads: no directory creation or access-hint writes."""

    def __init__(self, root):
        self.root = root
        self.loose_dir = root / "loose"
        self.catalog_path = root / "catalog.json"
        self.storage = None
        self._catalog = {}
        if self.catalog_path.exists():
            self._load_catalog()

    def touch(self, *args, **kwargs):
        pass


def _preview_library(root):
    memory = SystematicMemory(path=root / "memory.json")
    if (root / "vault").exists():
        vault = _ReadOnlyVault(root / "vault")
        # Calibration needs held records only, not mutable shard indexes.
        memory._facts = {key: record for entry in vault.catalog()
                         if entry.get("kind") == "fact-bucket"
                         for key, record in vault.get(entry["shard_id"]).get("facts", {}).items()}
    return memory, ContemporaryStash(root / "stash.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = session.load_plan()
    config = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    swapper = session.LMStudioSwapper(config["cli"])
    if args.dry_run:
        snapshot = swapper.snapshot()
        memory, stash = _preview_library(args.root)
        pairs = sources.calibration_items(memory, stash, k=40, seed=155)
        print(json.dumps({"plan": [asdict(source) for source in plan],
                          "snapshot": snapshot, "pairs": len(pairs),
                          "sources": [{"name": source.name,
                                       "co_distilled": _co_distilled(stash, pairs, source.name)}
                                      for source in plan]}, indent=2))
    else:
        backup_dir = args.backup_dir or args.root.resolve().parent / "backups"
        report_path = args.report or backup_dir / f"session-{_stamp()}.json"
        report = session.run_session(
            args.root, plan, swapper=swapper,
            teacher_factory=lambda source: sources.LMStudioTeacher(source.name, source.gguf),
            backup_dir=backup_dir, report_path=report_path)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
