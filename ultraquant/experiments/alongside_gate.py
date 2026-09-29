"""The session leaves your models alone. The gate.

"GUI needs to work while your working also." §11.166's session swaps the
GPU: live, it unloaded the user's Command-R for about ten minutes while Qwen
and Cydonia answered. Here a source already loaded is shared as it is, and
any other source is loaded CPU-only beside whatever is loaded, then unloaded
when it is used up; nothing of the user's is ever unloaded. Measured first:
Qwen on the CPU calibrated 40 of 40 at 1.32 s a request while Command-R, on
the GPU, kept answering in 0.08-0.14 s.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 0df366c2...) before the implementation existed. The world: §11.166's
(a copy of the pinned library after §11.155's filing, the recorded teachers,
§11.155's pairs, the plan in session.json); the swapper is the real
LMStudioSwapper over a fake command runner that keeps LM Studio's loaded list
and records every command; it starts holding §11.166's snapshot; device cpu.
1. **The same knowledge**: every held value equals the three-source world's.
2. **The user's models are untouched**: no load or unload names a snapshot
   model, and at every load the fake holds every snapshot model with its
   context length, parallel and TTL - in the normal run and with a fault in
   the second source.
3. **Cleaned up**: at every load the fake holds exactly the snapshot; at the
   end it holds exactly the snapshot, in both runs.
4. **Shared when loaded**: Command-R is never loaded; every load carries
   ``--gpu off``; the report's devices are shared, cpu, cpu.
5. **Live**: through the implementation's own load_alongside and unload,
   Qwen is placed cpu, decides at least 39 of the 40 pairs with every
   decision right, Command-R answers "Paris" to every probe meanwhile (the
   slowest under 2 s), ``lms ps`` after equals before, and Command-R itself
   is placed shared with nothing but ``ps`` issued. Recorded; --rescore
   scores the record and never reaches LM Studio.
6. **The exam can fail**: P112 (load_alongside does §11.166's GPU swap)
   breaches 2; P113 (unload does nothing) breaches 3; P114 (load_alongside
   loads a source already loaded) breaches 4.
7. **Nothing regresses**: a sweep matches the ledger. Checked outside this
   module.
"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["AlongsideReport", "run_gate"]

HERE = Path(__file__).with_name("records")
LIVE_RECORD = HERE / "alongside_11168_live.json"
LMS = Path(r"C:\Users\Stephen Hawking\.lmstudio\bin\lms.exe")
SETTINGS = ("contextLength", "parallel", "ttlMs")
PROBE = "What is the capital of France?"

_MODE = {"live": False}
_RUN: dict = {}


def _keyed(models) -> list:
    return sorted((m["identifier"], m.get("type"), *(m.get(k) for k in SETTINGS)) for m in models)


class _Runner:
    """A fake ``lms``: keeps the loaded list; records each command and what was loaded before it."""

    def __init__(self, start):
        self.loaded = copy.deepcopy(list(start))
        self.log: list[tuple[list[str], list[dict]]] = []

    def __call__(self, args, **kwargs):
        args = [str(a) for a in args][1:]
        self.log.append((args, copy.deepcopy(self.loaded)))
        verb = args[0]
        if verb == "ps":
            return subprocess.CompletedProcess(args, 0, json.dumps(self.loaded), "")
        if verb == "unload":
            self.loaded = [m for m in self.loaded if m["identifier"] != args[1]]
        elif verb == "load":
            def flag(name, default=None):
                return args[args.index(name) + 1] if name in args else default
            parallel, ttl = flag("--parallel"), flag("--ttl")
            self.loaded.append({"identifier": args[1], "modelKey": args[1], "type": "llm",
                                "contextLength": int(flag("--context-length")),
                                # LM Studio's default, as measured live (§11.168).
                                "parallel": int(parallel) if parallel is not None else 4,
                                "ttlMs": float(ttl) * 1000 if ttl is not None else None,
                                "gpu": flag("--gpu", "auto")})
        else:
            raise AssertionError(f"unexpected lms command {args}")
        return subprocess.CompletedProcess(args, 0, "", "")


def _session(key: str, fault_at: str | None = None) -> dict:
    """Run the session with device cpu on a filed copy; everything the criteria read."""
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
    scratch = Path(tempfile.mkdtemp(prefix="uq_alongside_"))
    runner = _Runner(SES.START)
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
            except Exception as exc:        # a fault run is expected to raise
                error = repr(exc)
                report = (json.loads((scratch / "report.json").read_text(encoding="utf-8"))
                          if (scratch / "report.json").exists() else {})
            _RUN[key] = {"report": report, "error": error, "values": SES._values(lib.root),
                         "log": runner.log, "end": copy.deepcopy(runner.loaded)}
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN[key]


def _runs():
    from ultraquant.experiments import third_gate as TG
    return (("run", _session("run")), ("fault", _session("fault", fault_at=TG.SECOND)))


# -- 1: the same knowledge -----------------------------------------------------------------

def same_knowledge() -> bool:
    from ultraquant.experiments import session_gate as SES
    got, want = _session("run")["values"], SES._three_source_values()
    differ = sorted(k for k in set(got) | set(want) if str(got.get(k)) != str(want.get(k)))
    _RUN["differ"] = differ[:5]
    return bool(got) and not differ


# -- 2: the user's models are untouched ----------------------------------------------------

def untouched() -> bool:
    from ultraquant.experiments import session_gate as SES
    names = {m["identifier"] for m in SES.START}
    bad = []
    for key, run in _runs():
        for args, before in run["log"]:
            if args[0] in ("load", "unload") and args[1] in names:
                bad.append((key, args[0], args[1]))
            if args[0] == "load":
                held = {m["identifier"]: m for m in before}
                for model in SES.START:
                    mine = held.get(model["identifier"])
                    if mine is None or any(mine.get(k) != model.get(k) for k in SETTINGS):
                        bad.append((key, "at load", args[1], "lacked", model["identifier"]))
    _RUN["touched"] = bad[:5]
    return not bad


# -- 3: cleaned up ------------------------------------------------------------------------

def cleaned_up() -> bool:
    from ultraquant.experiments import session_gate as SES
    from ultraquant.experiments import third_gate as TG
    start = _keyed(SES.START)
    bad = []
    for key, run in _runs():
        for args, before in run["log"]:
            if args[0] == "load" and _keyed(before) != start:
                bad.append((key, "at load", args[1], [m["identifier"] for m in before]))
        if _keyed(run["end"]) != start:
            bad.append((key, "end", [m["identifier"] for m in run["end"]]))
    fault = dict(_runs())["fault"]
    failed = fault["report"].get("failed")
    _RUN["cleanup"] = {"bad": bad[:5], "fault failed": failed, "fault error": fault["error"]}
    return not bad and failed == TG.SECOND and fault["error"] is not None


# -- 4: shared when loaded ----------------------------------------------------------------

def shared() -> bool:
    run = _session("run")
    loads = [args for args, _ in run["log"] if args[0] == "load"]
    devices = [entry.get("device") for entry in run["report"].get("sources", [])]
    reloaded = [a[1] for a in loads if a[1] == "c4ai-command-r-08-2024"]
    on_gpu = [a[1] for a in loads if "--gpu" not in a or a[a.index("--gpu") + 1] != "off"]
    _RUN["shared"] = {"devices": devices, "reloaded": reloaded, "not --gpu off": on_gpu,
                      "loads": [a[:2] for a in loads]}
    return not reloaded and not on_gpu and devices == ["shared", "cpu", "cpu"] and len(loads) == 2


# -- 5: live ------------------------------------------------------------------------------

def _probe(model: str) -> dict:
    from ultraquant.distill import elicit
    payload = {"model": model, "messages": [{"role": "system", "content": elicit.SYSTEM},
                                            {"role": "user", "content": PROBE}],
               "temperature": 0, "max_tokens": 24, "seed": 1}
    request = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.load(response)["choices"][0]["message"]["content"]
    except Exception as exc:        # recorded, and scored as a failed probe
        reply = f"ERROR {exc!r}"
    return {"seconds": round(time.monotonic() - start, 3), "reply": reply[:80]}


def _live_run() -> dict:
    from ultraquant.distill import elicit, session, sources
    from ultraquant.experiments.ownquestions_gate import _decided
    plan = {s.name: s for s in session.load_plan()}
    qwen = plan["qwen/qwen3.8-27b"]
    pairs = _decided()[1]
    swapper = session.LMStudioSwapper(LMS)
    record = {"before": _keyed(swapper.snapshot())}
    probes, stop = [], threading.Event()

    def probing():
        while not stop.is_set():
            probes.append(_probe("c4ai-command-r-08-2024"))
            stop.wait(20)
    scratch = Path(tempfile.mkdtemp(prefix="uq_alongside_live_"))
    try:
        record["qwen placed"] = swapper.load_alongside(qwen.name, qwen.context_length)
        thread = threading.Thread(target=probing, daemon=True)
        thread.start()
        try:
            teacher = sources.LMStudioTeacher(qwen.name, qwen.gguf)
            start = time.monotonic()
            records = elicit.elicit(teacher, qwen.name, [t for t, _ in pairs],
                                    scratch / "calibration.jsonl")
            record["calibration seconds"] = round(time.monotonic() - start, 1)
            calibration = sources.calibrate(records, pairs)
            record["calibration"] = {k: calibration[k] for k in ("right", "promoted", "wilson_lower")}
        finally:
            stop.set()
            thread.join(timeout=150)
            swapper.unload(qwen.name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record["probes"] = probes
    record["after"] = _keyed(swapper.snapshot())
    calls: list[list[str]] = []

    def recording(args, **kwargs):
        calls.append([str(a) for a in args][1:])
        return subprocess.run(args, **kwargs)
    record["command-r placed"] = session.LMStudioSwapper(LMS, run=recording).load_alongside(
        "c4ai-command-r-08-2024", 4096)
    record["command-r calls"] = calls
    HERE.mkdir(parents=True, exist_ok=True)
    LIVE_RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def live() -> bool:
    record = (_live_run() if _MODE["live"]
              else json.loads(LIVE_RECORD.read_text(encoding="utf-8")))
    calibration = record.get("calibration") or {}
    probes = record.get("probes") or []
    _RUN["live"] = {k: record.get(k) for k in ("qwen placed", "calibration", "calibration seconds",
                                                "command-r placed", "command-r calls")}
    _RUN["live"]["probes"] = {"count": len(probes),
                              "slowest": max((p["seconds"] for p in probes), default=None),
                              "all Paris": all("paris" in p["reply"].lower() for p in probes)}
    _RUN["live"]["ps unchanged"] = record.get("before") == record.get("after")
    return (record.get("qwen placed") == "cpu"
            and calibration.get("promoted", 0) >= 39
            and calibration.get("right") == calibration.get("promoted")
            and bool(probes) and all("paris" in p["reply"].lower() for p in probes)
            and max(p["seconds"] for p in probes) < 2
            and record.get("before") == record.get("after") and bool(record.get("before"))
            and record.get("command-r placed") == "shared"
            and bool(record.get("command-r calls"))
            and all(call[0] == "ps" for call in record["command-r calls"]))


CASES = {"1 same knowledge": same_knowledge, "2 untouched": untouched, "3 cleaned up": cleaned_up,
         "4 shared": shared, "5 live": live}


# -- plants --------------------------------------------------------------------------------

def _plants():
    from ultraquant.distill import session

    def gpu_swap(self, name, context_length):
        self.load(name, context_length)
        return "cpu"

    def always_load(self, name, context_length):
        self._command("load", name, "--gpu", "off", "--context-length", str(context_length), "-y")
        return "cpu"

    return [
        ("P112 load_alongside does the GPU swap", "2 untouched",
         [mock.patch.object(session.LMStudioSwapper, "load_alongside", gpu_swap)]),
        ("P113 unload does nothing", "3 cleaned up",
         [mock.patch.object(session.LMStudioSwapper, "unload", lambda self, name: None)]),
        ("P114 an already loaded source loaded again", "4 shared",
         [mock.patch.object(session.LMStudioSwapper, "load_alongside", always_load)]),
    ]


@dataclass
class AlongsideReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(live: bool = True) -> AlongsideReport:
    _MODE["live"] = live
    report = AlongsideReport(passes=False)
    try:
        for name, case in CASES.items():
            try:
                report.cases[name] = bool(case())
            except Exception as exc:        # a crash is a failure
                report.cases[name] = False
                report.errors[name] = repr(exc)
        report.measured = {k: _RUN.get(k) for k in ("differ", "touched", "cleanup", "shared", "live")}
        report.measured["devices"] = [e.get("device") for e in
                                      ((_RUN.get("run") or {}).get("report") or {}).get("sources", [])]
        report.measured["seconds"] = [e.get("seconds") for e in
                                      ((_RUN.get("run") or {}).get("report") or {}).get("sources", [])]
        _MODE["live"] = False
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
    finally:
        _MODE["live"] = False
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
    live_mode = "--rescore" not in sys.argv[1:]
    report = run_gate(live=live_mode)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
