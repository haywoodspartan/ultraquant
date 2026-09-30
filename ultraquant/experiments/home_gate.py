"""The session folder stays put while the GUI works. The gate.

§11.172's implementer reported that "Open session here" (``_reopen_session``)
and "Change session folder..." (``_choose_home``) set ``self.home`` before
``_run_async`` refuses a busy window: during a take-in (§11.171), which reads
``self.home`` again after its merge, the window could rebuild from the other
folder and file the entry in the other folder's inbox. Here both check busy
first.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 1df952ab...) before the implementation existed. The world: §11.171's,
plus an empty second folder to switch to.
1. **No switch while busy**: with a GUI job running, ``_reopen_session``
   (library field naming the other vault) and ``_choose_home`` (dialog
   answering the other folder) leave ``self.home`` and the session as they
   were, and the dialog is not opened.
2. **The switch still works when idle**: after "done", ``_reopen_session``
   moves to the other folder and starts a session there.
3. **A take-in files where it merged**: switches attempted during a take-in
   leave the entry in the original home's ``inbox/applied/``, the home the
   original, and the session holding the staged-only facts.
4. **The exam can fail**: P130 (``_reopen_session`` unguarded) breaches 1
   and 3; P131 (``_choose_home`` unguarded) breaches 1.
5. **Nothing regresses**: checked outside this module.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["HomeReport", "run_gate"]

_RUN: dict = {}


def _inbox():
    return importlib.import_module("ultraquant.experiments.inbox_gate")


class _Dialog:
    """The folder dialog, answering the other folder and counting its openings."""

    def __init__(self, answer: Path):
        self.answer, self.opened = str(answer), 0

    def __call__(self, *args, **kwargs):
        self.opened += 1
        return self.answer


# -- 1 and 2: no switch while busy; the switch works when idle ---------------------------

def no_switch_while_busy() -> bool:
    IG = _inbox()
    gui_module = importlib.import_module("ultraquant.gui")
    with IG._world() as (live, _stage, _base, _report, scratch):
        other = scratch / "other_home"
        other.mkdir()
        gui = IG._GUI(live)
        try:
            dialog = _Dialog(other)
            release = threading.Event()
            gui.app._run_async("A long job", lambda: release.wait(60))
            session_before = gui.app.session
            gui.app.library_root.set(str(other / "vault"))
            gui.app._reopen_session()
            after_reopen = Path(gui.app.home)
            with mock.patch.object(gui_module.filedialog, "askdirectory", dialog):
                gui.app._choose_home()
            after_choose = Path(gui.app.home)
            release.set()
            gui.idle()
            same_session = gui.app.session is session_before
            # 2: idle, the same switch goes through.
            gui.app.library_root.set(str(other / "vault"))
            gui.app._reopen_session()
            gui.idle()
            switched = Path(gui.app.home) == other and gui.app.session is not None \
                and Path(gui.app.session.root) == other
        finally:
            gui.close()
    _RUN["switch"] = {"home after reopen": str(after_reopen), "home after choose": str(after_choose),
                      "dialog opened": dialog.opened, "same session": same_session,
                      "switched when idle": switched, "live": str(live)}
    return (after_reopen == live and after_choose == live and dialog.opened == 0
            and same_session and switched)


def switch_when_idle() -> bool:
    if "switch" not in _RUN:
        no_switch_while_busy()
    return bool(_RUN["switch"]["switched when idle"])


# -- 3: a take-in files where it merged ---------------------------------------------------

def take_in_stays_home() -> bool:
    IG = _inbox()
    gui_module = importlib.import_module("ultraquant.gui")
    with IG._world() as (live, stage, base, report, scratch):
        other = scratch / "other_home"
        other.mkdir()
        gui = IG._GUI(live)
        try:
            staged = IG._facts(stage)
            session_only = sorted(k for k in staged if k not in IG._facts(base))[:5]
            gui.gui.INBOX_SECONDS = 0
            entry = IG._announce(stage, base, report, live)
            started = gui.pump_until(lambda: gui.app.busy, timeout=60)
            dialog = _Dialog(other)
            gui.app.library_root.set(str(other / "vault"))
            gui.app._reopen_session()
            with mock.patch.object(gui_module.filedialog, "askdirectory", dialog):
                gui.app._choose_home()
            done = gui.pump_until(lambda: not IG._pending(live) and not gui.app.busy)
            gui.idle()
            applied = [p.name for p in IG._moved(live, "applied")]
            stray = [p.name for p in (other / "inbox").rglob("*.json")] if (other / "inbox").exists() else []
            holds = {}
            for k in session_only:
                try:
                    holds[k] = str((gui.app.session.memory.recall_fact(k) or {}).get("value")) == staged[k]
                except Exception:        # a session of another home, or a stale one
                    holds[k] = False
            home_now = Path(gui.app.home)
        finally:
            gui.close()
    _RUN["take in"] = {"started": started, "done": done, "applied": applied, "entry": entry.name,
                       "stray": stray, "home now": str(home_now), "live": str(live),
                       "session holds": holds}
    return (started and done and applied == [entry.name] and not stray and home_now == live
            and bool(holds) and all(holds.values()))


CASES = {"1 no switch while busy": no_switch_while_busy, "2 switch when idle": switch_when_idle,
         "3 take-in stays home": take_in_stays_home}


# -- plants --------------------------------------------------------------------------------

def _plants():
    gui = importlib.import_module("ultraquant.gui")

    def reopen_unguarded(self):
        # _reopen_session before §11.173: the home moves before busy is looked at.
        library = Path(self.library_root.get().strip() or (self.home / "vault"))
        home = library.parent if library.name == "vault" else library
        self.home = home
        self._append(self.storage_log, f"\nReopening session at {home}\n")
        self._run_async("Starting session", self._start_session)

    def choose_unguarded(self):
        # _choose_home before §11.173.
        path = gui.filedialog.askdirectory(title="Choose session folder")
        if not path:
            return
        self.home = Path(path)
        self._append(self.transcript, f"\n[session] switching to {self.home}\n")
        self._run_async("Starting session", self._start_session)

    return [
        ("P130 _reopen_session unguarded", "1 no switch while busy",
         [mock.patch.object(gui.UltraQuantGUI, "_reopen_session", reopen_unguarded)]),
        ("P131 _choose_home unguarded", "1 no switch while busy",
         [mock.patch.object(gui.UltraQuantGUI, "_choose_home", choose_unguarded)]),
    ]


@dataclass
class HomeReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> HomeReport:
    report = HomeReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("switch", "take in")}
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
    valid = len(report.planted) == 2 and all(report.planted.values())
    met = len(report.cases) == 3 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 3 cases; 2 of 2 plants caught"
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
