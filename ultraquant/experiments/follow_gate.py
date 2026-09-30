"""Everything that holds the session follows the rebuild. The gate.

§11.171's implementer reported, before its exam ran, two ways the GUI could
still write a stale library over a merged one: a Learn-tab learner built
before a take-in keeps the old session (its next answer saves pre-merge
memory), and UI-thread actions such as promote and reject write without
looking at ``busy``, beside a take-in running on the worker. Here the
rebuild carries the learner along, and those actions wait.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 67975481...) before the implementation existed. The world: §11.171's
(a live copy of the pinned library after §11.155's filing, a staged copy
carrying the recorded three-source session announced with
``session.write_inbox``, a GUI built withdrawn and pumped, settings keeping
LM Studio out).
1. **The learner follows**: a learner built by the Learn tab's survey before
   the take-in holds the GUI's new session afterwards; after answering one
   of its questions the live library still holds every staged fact.
2. **Writers wait while busy**: while a GUI job runs, ``_promote`` of a
   staged stash entry leaves stash, memory and facts unchanged and
   ``_reject`` leaves the stash unchanged; after "done" the same promote
   promotes.
3. **§11.171 still holds**: checked in the sweep.
4. **The exam can fail**: P128 (learner not rebound) breaches 1; P129
   (``_promote`` without the guard) breaches 2.
5. **Nothing regresses**: checked outside this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["FollowReport", "run_gate"]

CLAIM = "The capital of Testland is Testville."
_RUN: dict = {}


def _inbox():
    return importlib.import_module("ultraquant.experiments.inbox_gate")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _stash_status(home: Path, claim: str) -> list:
    from ultraquant.interpreter.stash import ContemporaryStash
    return [e["status"] for e in ContemporaryStash(home / "stash.json").entries() if e["claim"] == claim]


# -- 1: the learner follows --------------------------------------------------------------

def learner_follows() -> bool:
    IG = _inbox()
    with IG._world() as (live, stage, base, report, _scratch):
        gui = IG._GUI(live)
        try:
            gui.app._survey()
            gui.idle()
            learner = getattr(gui.app, "learner", None)
            questions = len(learner.pending) if learner is not None else 0
            gui.gui.INBOX_SECONDS = 0
            IG._announce(stage, base, report, live)
            done = gui.pump_until(lambda: not IG._pending(live) and not gui.app.busy)
            gui.idle()
            follows = learner is not None and learner.session is gui.app.session
            answered = False
            if learner is not None and learner.next_question() is not None:
                gui.app._answer_question(preset="I do not know")
                gui.idle()
                answered = True
            staged, now = IG._facts(stage), IG._facts(live)
            missing = sorted(k for k, v in staged.items() if now.get(k) != v)
        finally:
            gui.close()
    _RUN["learner"] = {"questions": questions, "done": done, "follows": follows,
                       "answered": answered, "missing": missing[:5]}
    return done and follows and not missing


# -- 2: writers wait while busy ------------------------------------------------------------

def writers_wait() -> bool:
    IG = _inbox()
    from ultraquant.interpreter.stash import ContemporaryStash
    with IG._world() as (live, _stage, _base, _report, _scratch):
        entry_id = ContemporaryStash(live / "stash.json").add_claim(
            "https://example.org/gui-lookup", "A GUI lookup", CLAIM, measured_confidence=0.9)
        gui = IG._GUI(live)
        try:
            gui.app._refresh_stash()
            gui.root.update()
            rows = [item for item in gui.app.stash_view.get_children()
                    if str(gui.app.stash_view.item(item, "values")[0]) == str(entry_id)]
            if not rows:
                raise RuntimeError("the staged claim is not in the stash view")
            release = threading.Event()
            gui.app._run_async("A long job", lambda: release.wait(60))
            gui.app.stash_view.selection_set(rows[0])
            before = (_hash(live / "stash.json"), _hash(live / "memory.json"), IG._facts(live, structured=False))
            gui.app._promote()
            after_promote = (_hash(live / "stash.json"), _hash(live / "memory.json"),
                             IG._facts(live, structured=False))
            gui.app._reject()
            after_reject = _hash(live / "stash.json")
            release.set()
            gui.idle()
            gui.app._refresh_stash()
            gui.root.update()
            rows = [item for item in gui.app.stash_view.get_children()
                    if str(gui.app.stash_view.item(item, "values")[0]) == str(entry_id)]
            if rows:
                gui.app.stash_view.selection_set(rows[0])
            gui.app._promote()
            gui.idle()
            status = _stash_status(live, CLAIM)
        finally:
            gui.close()
    waited = after_promote == before and after_reject == before[0]
    _RUN["writers"] = {"waited on promote": after_promote == before,
                       "waited on reject": after_reject == before[0], "status after": status}
    return waited and status == ["promoted"]


CASES = {"1 learner follows": learner_follows, "2 writers wait": writers_wait}


# -- plants --------------------------------------------------------------------------------

def _plants():
    gui = importlib.import_module("ultraquant.gui")

    def rebuild_alone(self):
        # §11.171's rebuild: a new session and CLI, and nothing carried along.
        from ultraquant.interpreter.chat import ChatCLI
        from ultraquant.interpreter.thoughts import build_session
        semantic = bool(self.settings.get("lmstudio.semantic_suggest", True))
        self.session = build_session(
            self.home, budget_bytes=1024 * 1024, seed=0, semantic=semantic,
            auto_approve=bool(self.settings.get("stash_auto_approve", False)))
        self.cli = ChatCLI(self.session, out=gui._QueueStream(self.events, "out"))

    def promote_unguarded(self, force=False):
        # The promote before §11.172: no look at busy.
        from ultraquant.interpreter.stash import StashError
        entry_id = self._selected_stash_id()
        if entry_id is None or self.session is None:
            return
        try:
            self.session.stash.promote(entry_id, self.session.memory, force=force)
        except StashError:
            return
        self.session.memory.save()

    return [
        ("P128 the learner not rebound", "1 learner follows",
         [mock.patch.object(gui.UltraQuantGUI, "_rebuild_session", rebuild_alone)]),
        ("P129 _promote without the busy guard", "2 writers wait",
         [mock.patch.object(gui.UltraQuantGUI, "_promote", promote_unguarded)]),
    ]


@dataclass
class FollowReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> FollowReport:
    report = FollowReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("learner", "writers")}
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
    met = len(report.cases) == 2 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 2 cases; 2 of 2 plants caught"
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
