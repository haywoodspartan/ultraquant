"""The session unloads only what it loaded. The gate.

§11.168's implementer reported, before its exam ran, that a CPU session still
ended with ``restore(snapshot)``, which unloads every model the snapshot did
not hold - so a model the user loads mid-session (the GUI's LLM panel, or by
hand) would be unloaded when the session ends. §11.168's criteria never
simulated a user acting meanwhile. Here the CPU session's last cleanup
unloads only what the session itself loaded and is still loaded; GPU mode is
§11.166's, unchanged.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 e6fceb5f...) before the implementation existed. The world: §11.168's;
the real LMStudioSwapper over a fake runner that keeps LM Studio's loaded
list, records every command, and also plays the user: when the session's
first CPU load is issued it adds "user-model" (llm, 8192, parallel 2, no TTL)
with no command from the session.
1. **What the user loads meanwhile survives**: at the end the fake holds
   "user-model" with its settings, and no session command names it - in the
   normal run and with the second source's teacher failing.
2. **The session's own models leave anyway**: at the end the fake holds
   exactly the snapshot plus "user-model" - normally, after the teacher
   fault, and when the fake's first unload of a source fails once.
3. **§11.168 still holds**: the three-source world's values; no load or
   unload names a snapshot model; Command-R never loaded; every load carries
   ``--gpu off``; devices shared, cpu, cpu.
4. **GPU mode unchanged**: session_gate passes. Checked outside this module.
5. **The exam can fail**: P119 (the last cleanup restores the snapshot)
   breaches 1; P120 (the last cleanup does nothing) breaches 2.
6. **Nothing regresses**: a sweep matches the ledger. Checked outside.
"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["OwnReport", "run_gate"]

USER_MODEL = {"identifier": "user-model", "modelKey": "user-model", "type": "llm",
              "contextLength": 8192, "parallel": 2, "ttlMs": None}
SETTINGS = ("contextLength", "parallel", "ttlMs")
_RUN: dict = {}


class _UserRunner:
    """A fake ``lms`` that also plays the user, and can fail one unload."""

    def __init__(self, start, fail_first_unload: bool = False):
        from ultraquant.experiments.alongside_gate import _Runner
        self.inner = _Runner(start)
        self.fail_unload = fail_first_unload
        self.user_loaded = False

    @property
    def log(self):
        return self.inner.log

    @property
    def loaded(self):
        return self.inner.loaded

    def __call__(self, args, **kwargs):
        words = [str(a) for a in args]
        if words[1] == "unload" and self.fail_unload:
            self.fail_unload = False
            self.inner.log.append((words[1:], copy.deepcopy(self.inner.loaded)))
            return subprocess.CompletedProcess(args, 1, "", "unload failed (injected)")
        done = self.inner(args, **kwargs)
        if words[1] == "load" and "--gpu" in words and not self.user_loaded:
            # The user, meanwhile: a model loaded by hand, not by the session.
            self.user_loaded = True
            self.inner.loaded.append(copy.deepcopy(USER_MODEL))
        return done


def _session(key: str, fault_at: str | None = None, fail_unload: bool = False) -> dict:
    if key in _RUN:
        return _RUN[key]
    from ultraquant.distill import session
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import session_gate as SES
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    scratch = Path(tempfile.mkdtemp(prefix="uq_own_"))
    runner = _UserRunner(SES.START, fail_first_unload=fail_unload)
    try:
        with _live_copy() as lib:
            error = None
            try:
                report = session.run_session(
                    lib.root, session.load_plan(),
                    swapper=session.LMStudioSwapper("lms", run=runner),
                    teacher_factory=SES._factory(fault_at), backup_dir=scratch / "backups",
                    report_path=scratch / "report.json", max_rounds=8,
                    pairs=_decided()[1], device="cpu")
            except Exception as exc:        # the fault runs are expected to raise
                error = repr(exc)
                report = (json.loads((scratch / "report.json").read_text(encoding="utf-8"))
                          if (scratch / "report.json").exists() else {})
            _RUN[key] = {"report": report, "error": error, "values": SES._values(lib.root),
                         "log": runner.log, "end": copy.deepcopy(runner.loaded),
                         "user loaded": runner.user_loaded}
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN[key]


def _keyed(models) -> list:
    return sorted((m["identifier"], m.get("type"), *(m.get(k) for k in SETTINGS)) for m in models)


def _normal():
    return _session("run")


def _teacher_fault():
    from ultraquant.experiments import third_gate as TG
    return _session("teacher fault", fault_at=TG.SECOND)


def _unload_fault():
    return _session("unload fault", fail_unload=True)


# -- 1: what the user loads meanwhile survives -------------------------------------------

def user_survives() -> bool:
    bad = []
    for key, run in (("run", _normal()), ("teacher fault", _teacher_fault())):
        mine = {m["identifier"]: m for m in run["end"]}.get("user-model")
        if not run["user loaded"]:
            bad.append((key, "the user never loaded"))
        elif mine is None or any(mine.get(k) != USER_MODEL[k] for k in SETTINGS):
            bad.append((key, "user-model gone or changed", mine))
        named = [args for args, _ in run["log"] if "user-model" in args]
        if named:
            bad.append((key, "commands named user-model", named[:2]))
    _RUN["user"] = bad[:5]
    return not bad


# -- 2: the session's own models leave anyway ---------------------------------------------

def own_leave() -> bool:
    from ultraquant.experiments import session_gate as SES
    want = _keyed(list(SES.START) + [USER_MODEL])
    bad = []
    for key, run in (("run", _normal()), ("teacher fault", _teacher_fault()),
                     ("unload fault", _unload_fault())):
        if _keyed(run["end"]) != want:
            bad.append((key, [m["identifier"] for m in run["end"]]))
    failed_unloads = [args for args, _ in _unload_fault()["log"] if args[0] == "unload"]
    _RUN["own"] = {"bad": bad[:5], "unloads in the unload-fault run": failed_unloads[:4],
                   "unload-fault error": _unload_fault()["error"]}
    return not bad and len(failed_unloads) >= 2


# -- 3: §11.168 still holds --------------------------------------------------------------

def alongside_holds() -> bool:
    from ultraquant.experiments import session_gate as SES
    run = _normal()
    got, want = run["values"], SES._three_source_values()
    differ = sorted(k for k in set(got) | set(want) if str(got.get(k)) != str(want.get(k)))
    names = {m["identifier"] for m in SES.START}
    touched = [args for key, r in (("run", run), ("fault", _teacher_fault()))
               for args, _ in r["log"] if args[0] in ("load", "unload") and args[1] in names]
    loads = [args for args, _ in run["log"] if args[0] == "load"]
    on_gpu = [a[1] for a in loads if "--gpu" not in a or a[a.index("--gpu") + 1] != "off"]
    devices = [e.get("device") for e in run["report"].get("sources", [])]
    _RUN["alongside"] = {"differ": differ[:5], "touched": touched[:3], "not --gpu off": on_gpu,
                         "devices": devices}
    return (bool(got) and not differ and not touched and not on_gpu
            and not any(a[1] == "c4ai-command-r-08-2024" for a in loads)
            and devices == ["shared", "cpu", "cpu"])


CASES = {"1 user survives": user_survives, "2 own leave": own_leave,
         "3 alongside holds": alongside_holds}


def _plants():
    from ultraquant.distill import session
    from ultraquant.experiments import session_gate as SES

    return [
        ("P119 the last cleanup restores the snapshot", "1 user survives",
         [mock.patch.object(session, "release",
                            lambda swapper, loaded: swapper.restore(copy.deepcopy(SES.START)))]),
        ("P120 the last cleanup does nothing", "2 own leave",
         [mock.patch.object(session, "release", lambda swapper, loaded: None)]),
    ]


@dataclass
class OwnReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> OwnReport:
    report = OwnReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("user", "own", "alongside")}
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
