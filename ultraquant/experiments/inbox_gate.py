"""Study sessions come home through the GUI. The gate.

"GUI needs to work while your working also." Since §11.168-§11.169 a study
session runs beside the GUI without touching the user's models, but its
knowledge reached the library only through a merge that needed the GUI
closed: an open GUI holds the library in memory and saves after every turn,
so a merge written beside it would be overwritten. Here a finished session
is announced in ``<home>/inbox/``, and the GUI takes it in itself - when
idle, after a clean check, through the in-place replay - then rebuilds its
session from disk.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 aca303ff...) before the implementation existed, with Amendment A (sha256 0592a2f2..., before any code: the rebuild is also checked as a new session object), and Amendment B (sha256 79809ae4..., POST-RUN: run 1 was VOID - P125 disabled start-up too, P126's 4 s window was shorter than a take-in), and Amendment C (sha256 b7d03409..., POST-RUN: run 2 was VOID - P125's stale session crashed reading a page the merge replaced; such reads are guarded and count as not held). The world: a live copy
of the pinned library after §11.155's filing; a staged copy on which
§11.166's recorded three-source session ran, announced with
``session.write_inbox``; a GUI, built withdrawn and pumped, over the live
copy, with settings that keep LM Studio out of it.
1. **Taken in while running**: every structured fact the staged library
   holds is held by the live library, and the GUI's session was rebuilt: a
   new session object, holding a fact only the staged session added.
2. **The GUI's own work kept**: a fact stored in a GUI chat turn before the
   entry was taken in holds its value afterwards.
3. **Recorded**: the entry is in ``inbox/applied/`` with a clean report; its
   backup holds the library as before the merge (the GUI's fact, not the
   session's).
4. **Held, not applied**: (a) a ledger conflict and (b) a hand-added staged
   fact no claim produced each go to ``inbox/held/``, and the live library's
   structured facts are unchanged.
5. **It waits while busy**: while a GUI job runs, a pending entry stays
   pending; after "done", it is taken in.
6. **The exam can fail**: P125 (no rebuild) breaches 1; P126 (taking in
   beside a running job) breaches 5; P127 (merge without the check)
   breaches 4.
7. **Nothing regresses**: checked outside this module.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["InboxReport", "run_gate"]

REMEMBER = "remember that the favourite colour of the tester is blue"
_RUN: dict = {}


# -- the world ---------------------------------------------------------------------------

def _open(home: Path):
    from ultraquant.memory.factshards import FactShards
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.shards.vault import ShardVault
    memory = SystematicMemory(path=home / "memory.json")
    if (home / "vault").exists():
        memory.shards = FactShards(ShardVault(home / "vault"))
    return memory


def _facts(home: Path, structured: bool = True) -> dict:
    memory = _open(home)
    out = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        if not structured or (record.get("subject") and record.get("attribute")):
            out[key] = str(record.get("value"))
    return out


@contextlib.contextmanager
def _world():
    """A live copy and a staged copy carrying §11.166's recorded session."""
    from ultraquant.distill import session
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import session_gate as SES
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    scratch = Path(tempfile.mkdtemp(prefix="uq_inbox_"))
    settings = scratch / "settings.json"
    settings.write_text(json.dumps({"lmstudio": {"semantic_suggest": False},
                                    "stash_auto_approve": False}), encoding="utf-8")
    old = os.environ.get("ULTRAQUANT_CONFIG")
    os.environ["ULTRAQUANT_CONFIG"] = str(settings)
    try:
        with _live_copy() as lib:
            live = lib.root
            staging_dir = scratch / "staging"
            stage = staging_dir / "uq_home"
            shutil.copytree(live, stage)
            report = session.run_session(
                stage, session.load_plan(), swapper=SES._Fake(), teacher_factory=SES._factory(),
                backup_dir=staging_dir / "backups", report_path=staging_dir / "session-report.json",
                max_rounds=8, pairs=_decided()[1])
            base = next(p for p in sorted((staging_dir / "backups").iterdir())
                        if p.is_dir() and p.name.startswith("uq_home-"))
            yield live, stage, base, report, scratch
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
        if old is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = old
        shutil.rmtree(scratch, ignore_errors=True)


class _GUI:
    """The desktop app, built withdrawn and pumped by hand, as tests/test_gui.py does."""

    def __init__(self, home: Path):
        import tkinter as tk
        self.gui = importlib.import_module("ultraquant.gui")
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = self.gui.UltraQuantGUI(self.root, home)
        if not self.idle(timeout=180):
            raise RuntimeError("the GUI session did not start")

    def idle(self, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if not self.app.busy:
                return True
            time.sleep(0.02)
        return False

    def say(self, text: str) -> None:
        self.app._submit(text)
        if not self.idle():
            raise RuntimeError(f"the GUI did not finish {text!r}")

    def pump_until(self, predicate, timeout: float = 600.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.root.update()
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def close(self):
        try:
            self.root.destroy()
        except Exception:        # already gone
            pass


def _pending(home: Path) -> list:
    return sorted(p for p in (home / "inbox").glob("*.json")) if (home / "inbox").exists() else []


def _moved(home: Path, where: str) -> list:
    folder = home / "inbox" / where
    return sorted(folder.glob("*.json")) if folder.exists() else []


def _announce(stage: Path, base: Path, report: dict, home: Path) -> Path:
    session = importlib.import_module("ultraquant.distill.session")
    return Path(session.write_inbox(home, stage, base, report))


# -- 1, 2, 3: taken in, the GUI's own work kept, recorded ------------------------------

def _take_in_run() -> dict:
    if "take in" in _RUN:
        return _RUN["take in"]
    with _world() as (live, stage, base, report, _scratch):
        gui = _GUI(live)
        try:
            before = _facts(live, structured=False)
            gui.say(REMEMBER)
            after_turn = _facts(live, structured=False)
            gui_keys = sorted(k for k in after_turn if before.get(k) != after_turn.get(k))
            staged = _facts(stage)
            session_only = sorted(k for k in staged if k not in _facts(base))
            gui.gui.INBOX_SECONDS = 0
            old_session = gui.app.session
            entry = _announce(stage, base, report, live)
            done = gui.pump_until(lambda: not _pending(live) and not gui.app.busy)
            gui.idle()
            # A stale session could still read new facts through the vault on
            # a cache miss, so the rebuild is checked as the object it is.
            rebuilt = gui.app.session is not old_session and gui.app.session is not None
            # Amendment C (post-run): a stale session can fail to read a page
            # the merge replaced; a failed read is a fact it does not hold.
            held_now = {}
            for k in session_only[:5]:
                try:
                    held_now[k] = str((gui.app.session.memory.recall_fact(k) or {}).get("value"))
                except Exception as exc:        # the stale session's own failure
                    held_now[k] = f"unreadable: {type(exc).__name__}"
            applied = _moved(live, "applied")
            outcome = json.loads(applied[0].read_text(encoding="utf-8")).get("outcome") if applied else {}
            backup = Path(((outcome or {}).get("report") or {}).get("backup") or "")
            _RUN["take in"] = {
                "done": done, "rebuilt": rebuilt, "entry": entry.name, "gui keys": gui_keys,
                "session only": session_only[:5],
                "live facts": _facts(live), "live all": _facts(live, structured=False),
                "staged": staged, "gui session holds": held_now,
                "applied": [p.name for p in applied], "outcome": outcome,
                "backup facts": _facts(backup, structured=False) if backup.is_dir() else None,
                "after turn": after_turn}
        finally:
            gui.close()
    return _RUN["take in"]


def taken_in() -> bool:
    run = _take_in_run()
    missing = sorted(k for k, v in run["staged"].items() if run["live facts"].get(k) != v)
    stale = {k: v for k, v in run["gui session holds"].items() if v != run["staged"].get(k)}
    _RUN["taken in"] = {"done": run["done"], "rebuilt": run["rebuilt"], "missing": missing[:5],
                        "stale in the GUI": stale, "session only": run["session only"]}
    return (run["done"] and run["rebuilt"] and not missing and bool(run["session only"])
            and not stale)


def own_work_kept() -> bool:
    run = _take_in_run()
    kept = {k: (run["after turn"].get(k), run["live all"].get(k)) for k in run["gui keys"]}
    _RUN["own"] = {"gui keys": run["gui keys"], "kept": kept}
    return bool(run["gui keys"]) and all(a == b for a, b in kept.values())


def recorded() -> bool:
    run = _take_in_run()
    outcome = run["outcome"] or {}
    report = outcome.get("report") or {}
    backup = run["backup facts"]
    good_backup = (backup is not None
                   and all(backup.get(k) == run["after turn"].get(k) for k in run["gui keys"])
                   and not any(k in backup for k in run["session only"]))
    _RUN["recorded"] = {"applied": run["applied"], "clean": report.get("clean"),
                        "backup": report.get("backup"), "good backup": good_backup}
    return run["applied"] == [run["entry"]] and report.get("clean") is True and good_backup


# -- 4: held, not applied ----------------------------------------------------------------

def held() -> bool:
    results = {}
    with _world() as (live, stage, base, report, scratch):
        gui = _GUI(live)
        try:
            gui.gui.INBOX_SECONDS = 0
            # (b) a staged fact no session claim produced.
            stage_b = scratch / "staging-b" / "uq_home"
            shutil.copytree(stage, stage_b)
            memory = _open(stage_b)
            memory.remember_fact("pet of the tester", "a cat", 0.9, subject="the tester", attribute="pet")
            memory.save()
            facts_before = _facts(live)
            entry_b = _announce(stage_b, base, report, live)
            done_b = gui.pump_until(lambda: not _pending(live) and not gui.app.busy)
            gui.idle()
            results["b"] = {"done": done_b, "held": entry_b.name in [p.name for p in _moved(live, "held")],
                            "facts unchanged": _facts(live) == facts_before}
            # (a) a ledger the live library moved on from.
            ledger_path = live / "sources.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
            ledger.setdefault("someone-else", []).append({"question_id": "x:1", "promoted": False})
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
            facts_before = _facts(live)
            time.sleep(1.1)          # a distinct inbox name from entry (b)
            entry_a = _announce(stage, base, report, live)
            done_a = gui.pump_until(lambda: not _pending(live) and not gui.app.busy)
            gui.idle()
            results["a"] = {"done": done_a, "held": entry_a.name in [p.name for p in _moved(live, "held")],
                            "facts unchanged": _facts(live) == facts_before}
        finally:
            gui.close()
    _RUN["held"] = results
    return all(r["done"] and r["held"] and r["facts unchanged"] for r in results.values()) and len(results) == 2


# -- 5: it waits while busy --------------------------------------------------------------

def waits_while_busy() -> bool:
    with _world() as (live, stage, base, report, _scratch):
        gui = _GUI(live)
        try:
            gui.gui.INBOX_SECONDS = 0
            release = threading.Event()
            gui.app._run_async("A long job", lambda: release.wait(120))
            entry = _announce(stage, base, report, live)
            # Amendment B (post-run): a take-in merges for ~20 s; the job is held
            # up to 90 s, and any take-in landing before it ends counts.
            end = time.time() + 90.0
            taken_while_busy = False
            while time.time() < end:
                gui.root.update()
                if not _pending(live) or _moved(live, "applied"):
                    taken_while_busy = True
                    break
                time.sleep(0.05)
            release.set()
            done = gui.pump_until(lambda: not _pending(live) and not gui.app.busy)
            gui.idle()
            applied = [p.name for p in _moved(live, "applied")]
        finally:
            gui.close()
    _RUN["busy"] = {"taken while busy": taken_while_busy, "done": done, "applied": applied}
    return not taken_while_busy and done and applied == [entry.name]


CASES = {"1 taken in": taken_in, "2 own work kept": own_work_kept, "3 recorded": recorded,
         "4 held": held, "5 waits while busy": waits_while_busy}


# -- plants --------------------------------------------------------------------------------

def _plants():
    gui = importlib.import_module("ultraquant.gui")
    merge = importlib.import_module("ultraquant.distill.merge")
    real_merge = merge.merge

    def eager(self):
        # A take-in beside whatever runs: no busy check, a thread of its own.
        pending = sorted((Path(self.home) / "inbox").glob("*.json"))
        if pending and not getattr(self, "_eager_started", False):
            self._eager_started = True
            threading.Thread(target=self._take_in, args=(pending[0],), daemon=True).start()

    def unchecked(staging, live, base, backup_dir, *, replay=False):
        return real_merge(staging, live, base, replay=replay)

    return [
        # Amendment B (post-run): start-up builds through _rebuild_session too,
        # so only a rebuild once a session exists is skipped.
        ("P125 no rebuild after the merge", "1 taken in",
         [mock.patch.object(gui.UltraQuantGUI, "_rebuild_session",
                            lambda self, _real=gui.UltraQuantGUI._rebuild_session:
                            _real(self) if self.session is None else None)]),
        ("P126 taking in beside a running job", "5 waits while busy",
         [mock.patch.object(gui.UltraQuantGUI, "_check_inbox", eager)]),
        ("P127 merge without the check", "4 held",
         [mock.patch.object(merge, "checked_merge", unchecked)]),
    ]


@dataclass
class InboxReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> InboxReport:
    report = InboxReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("taken in", "own", "recorded", "held", "busy")}
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
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
