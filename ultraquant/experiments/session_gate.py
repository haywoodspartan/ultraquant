"""One session, three sources, everything restored. The gate.

§11.158-§11.165 ran only as exam code, each LM Studio swap typed by hand.
Here one command runs the plan on a library: back it up, snapshot what LM
Studio holds, run each source in order until it is used up, and always put
LM Studio back as it was - also when a round fails - with a report of what
each source did. The report says plainly that every calibration pair was
co-distilled by these same models (§11.165): their calibration is partly
circular.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 284379b5...) before any code, with Amendment A (sha256 a48c48c1...,
also before any code: recorded calibration pairs may be supplied; P111
drops the last source; the swapper's command runner is a seam). The world:
a copy of the user's library after §11.155's filing with the corrections;
the recorded teachers answer; the swapper is a fake keeping a loaded set.
1. **The same knowledge**: every held value equals the three-source world's.
2. **Backup before any write**: the backup is byte-identical to the library
   before the session.
3. **Everything restored**: the fake ends equal to its snapshot; with a
   fault in the second source's first round it is still restored, the
   backup exists, and the report names the second source as failed.
4. **The report is true**: per-source totals equal the rounds' sums; the
   co-distilled count is 40 for each source.
5. **The real swapper round-trips**: snapshot then restore issues no load or
   unload and leaves what ``lms ps --json`` reports loaded unchanged
   (identifier, type, context length, parallel, TTL - not lastUsedTime or
   status, which move on their own; fixed before any run). Live run: the
   real CLI; --rescore: its recorded output, so sweeps never reach LM Studio.
6. **The exam can fail**: P109 (no restore when a round raises) breaches 3;
   P110 (the backup copied after the first round) breaches 2; P111 (the
   plan's last source dropped) breaches 1.
7. **Nothing regresses**: a sweep matches the §11.151 ledger. Checked
   outside this module.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["SessionReport", "run_gate"]

HERE = Path(__file__).with_name("records")
PS_RECORD = HERE / "session_11166_ps.json"
LMS = Path(r"C:\Users\Stephen Hawking\.lmstudio\bin\lms.exe")
START = [
    {"identifier": "c4ai-command-r-08-2024", "modelKey": "c4ai-command-r-08-2024", "type": "llm",
     "contextLength": 128000, "parallel": 4, "ttlMs": None},
    {"identifier": "text-embedding-nomic-embed-text-v1.5", "modelKey": "text-embedding-nomic-embed-text-v1.5",
     "type": "embedding", "contextLength": 2048, "parallel": None, "ttlMs": 3600000},
]

_MODE = {"live": False}
_RUN: dict = {}


class _Fake:
    """A swapper keeping a loaded set, as LM Studio would."""

    def __init__(self):
        self.loaded = {m["identifier"]: dict(m) for m in START}
        self.calls = []

    def snapshot(self):
        self.calls.append(("snapshot",))
        return copy.deepcopy(list(self.loaded.values()))

    def load(self, name, context_length):
        self.calls.append(("load", name, context_length))
        self.loaded = {k: v for k, v in self.loaded.items() if v.get("type") != "llm"}
        self.loaded[name] = {"identifier": name, "modelKey": name, "type": "llm",
                             "contextLength": context_length, "parallel": 4, "ttlMs": None}

    def restore(self, snapshot):
        self.calls.append(("restore",))
        self.loaded = {m["identifier"]: dict(m) for m in snapshot}


def _factory(fault_at: str | None = None):
    """The recorded teachers, by source name; one may be made to fail."""
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import ownquestions_gate as G
    from ultraquant.experiments import roundtrip_gate as R
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import third_gate as TG

    class Failing:
        def __init__(self, spec):
            self.spec = spec

        def ask(self, *args, **kwargs):
            raise RuntimeError("injected fault: the source stopped answering")

    def make(source):
        name = getattr(source, "name", source)
        if name == TG.FIRST:
            teacher = GG._Teacher()
            for question, raws in R._load(G.CALIBRATION).items():
                teacher.recorded.setdefault(question, raws)
        elif name == TG.SECOND:
            teacher = SG._Second()
        elif name == TG.THIRD:
            teacher = TG._Third()
        else:
            raise KeyError(name)
        return Failing(teacher.spec) if name == fault_at else teacher
    return make


def _hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _values(root: Path) -> dict:
    from ultraquant.experiments.catalogue_gate import Library
    lib = Library(root)
    return {k: (lib.record(k) or {}).get("value") for k in lib.keys()}


def _session(key: str, fault_at: str | None = None) -> dict:
    """Run the session on a filed copy; everything the criteria read."""
    if key in _RUN:
        return _RUN[key]
    from ultraquant.distill import session
    from ultraquant.experiments import growth_gate as GG
    from ultraquant.experiments import second_gate as SG
    from ultraquant.experiments import third_gate as TG
    from ultraquant.experiments.ownquestions_gate import _decided
    from ultraquant.experiments.shape_gate import _live_copy
    modes = (GG._MODE["live"], SG._MODE["live"], TG._MODE["live"])
    GG._MODE["live"] = SG._MODE["live"] = TG._MODE["live"] = False
    scratch = Path(tempfile.mkdtemp(prefix="uq_session_"))
    fake = _Fake()
    snapshot = copy.deepcopy(list(fake.loaded.values()))
    try:
        with _live_copy() as lib:
            root = lib.root
            before = _hashes(root)
            plan = session.load_plan()
            error = None
            try:
                report = session.run_session(
                    root, plan, swapper=fake, teacher_factory=_factory(fault_at),
                    backup_dir=scratch / "backups", report_path=scratch / "report.json",
                    max_rounds=8, pairs=_decided()[1])
            except Exception as exc:        # a fault run is expected to raise
                error = repr(exc)
                report = json.loads((scratch / "report.json").read_text(encoding="utf-8")) \
                    if (scratch / "report.json").exists() else {}
            backups = [p for p in (scratch / "backups").glob("*") if p.is_dir()] \
                if (scratch / "backups").exists() else []
            _RUN[key] = {
                "plan": [getattr(s, "name", s) for s in plan], "report": report, "error": error,
                "values": _values(root), "before": before,
                "backup": _hashes(backups[0]) if len(backups) == 1 else None,
                "restored": list(fake.loaded.values()) == snapshot or
                            sorted(fake.loaded) == sorted(m["identifier"] for m in snapshot),
                "calls": fake.calls,
            }
    finally:
        GG._MODE["live"], SG._MODE["live"], TG._MODE["live"] = modes
        shutil.rmtree(scratch, ignore_errors=True)
    return _RUN[key]


def _three_source_values() -> dict:
    if "three" not in _RUN:
        from ultraquant.experiments import said_gate as SD
        with SD._world() as (lib, _session_):
            _RUN["three"] = {k: (lib.record(k) or {}).get("value") for k in lib.keys()}
    return _RUN["three"]


def same_knowledge() -> bool:
    got, want = _session("run")["values"], _three_source_values()
    differ = sorted(k for k in set(got) | set(want) if str(got.get(k)) != str(want.get(k)))
    _RUN["differ"] = differ[:5]
    return bool(got) and not differ


def backup_first() -> bool:
    s = _session("run")
    return s["backup"] is not None and s["backup"] == s["before"]


def always_restored() -> bool:
    from ultraquant.experiments import third_gate as TG
    ok = _session("run")
    bad = _session("fault", fault_at=TG.SECOND)
    report = bad["report"] or {}
    failed = str(report.get("failed") or report.get("status") or "")
    _RUN["fault"] = {"error": bad["error"], "failed": failed, "restored": bad["restored"],
                     "backup": bad["backup"] is not None}
    return (ok["restored"] and bad["restored"] and bad["error"] is not None
            and bad["backup"] is not None and TG.SECOND in failed)


def report_true() -> bool:
    report = _session("run")["report"] or {}
    sources_ = report.get("sources") or []
    wrong = []
    for entry in sources_:
        totals, rounds = entry.get("totals") or {}, entry.get("rounds") or []
        for name in ("asked", "filed", "agreed", "contested", "revised"):
            if totals.get(name) != sum(r.get(name, 0) for r in rounds):
                wrong.append((entry.get("name"), name))
        if entry.get("co_distilled") != 40:
            wrong.append((entry.get("name"), "co_distilled", entry.get("co_distilled")))
    _RUN["report wrong"] = wrong[:5]
    return len(sources_) == 3 and not wrong


def real_swapper_round_trips() -> bool:
    from ultraquant.distill import session
    issued = []

    def guard(args, **kwargs):
        args = [str(a) for a in args]
        if len(args) > 1 and args[1] == "ps":
            if _MODE["live"]:
                done = subprocess.run(args, **kwargs)
                PS_RECORD.write_text(done.stdout if isinstance(done.stdout, str) else
                                     done.stdout.decode("utf-8"), encoding="utf-8")
                return done
            return subprocess.CompletedProcess(args, 0, PS_RECORD.read_text(encoding="utf-8"), "")
        issued.append(args[1:3])
        return subprocess.CompletedProcess(args, 0, "", "")

    def loaded():
        """What is loaded - not lastUsedTime or status, which move on their own."""
        rows = json.loads(guard([str(LMS), "ps", "--json"], capture_output=True, text=True).stdout)
        return sorted((m.get("identifier"), m.get("type"), m.get("contextLength"),
                       m.get("parallel"), m.get("ttlMs")) for m in rows)

    swapper = session.LMStudioSwapper(LMS, run=guard)
    before = loaded()
    swapper.restore(swapper.snapshot())
    after = loaded()
    _RUN["swapper"] = {"issued": issued, "unchanged": before == after}
    return not issued and before == after


CASES = {
    "1 the same knowledge": same_knowledge,
    "2 backup before any write": backup_first,
    "3 everything restored": always_restored,
    "4 the report is true": report_true,
    "5 the real swapper round-trips": real_swapper_round_trips,
}


def _plants():
    import shutil as _sh
    from ultraquant.distill import frontier, session

    real_round, real_plan = frontier.study_round, session.load_plan
    late = {}

    def backup(root, backup_dir):
        late["args"] = (Path(root), Path(backup_dir) / "late")
        return late["args"][1]

    def round_then_backup(*args, **kwargs):
        out = real_round(*args, **kwargs)
        if "args" in late and not late["args"][1].exists():
            _sh.copytree(*late["args"])
        return out

    return [
        ("P109 no restore when a round raises", "3 everything restored",
         [mock.patch.object(_Fake, "restore", lambda self, snapshot: self.calls.append(("restore skipped",)))]),
        ("P110 the backup copied after the first round", "2 backup before any write",
         [mock.patch.object(session, "backup", backup),
          mock.patch.object(frontier, "study_round", round_then_backup)]),
        ("P111 the plan's last source dropped", "1 the same knowledge",
         [mock.patch.object(session, "load_plan", lambda *a, **k: real_plan(*a, **k)[:-1])]),
    ]


@dataclass
class SessionReport:
    """Whether one session runs the plan and always restores what it found.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        cases: Case -> whether it held.
        measured: Differences, fault run, report checks, swapper calls.
        planted: Plant -> whether it broke its case.
        errors: Case or plant -> the exception it raised, if any.
        reason: Plain-language verdict.
    """

    passes: bool
    cases: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(live: bool = True) -> SessionReport:
    _MODE["live"] = live
    report = SessionReport(passes=False)
    try:
        for name, case in CASES.items():
            try:
                report.cases[name] = bool(case())
            except Exception as exc:        # a crash is a failure
                report.cases[name] = False
                report.errors[name] = repr(exc)
        report.measured = {k: _RUN.get(k) for k in ("differ", "fault", "report wrong", "swapper")}
        report.measured["report"] = (_RUN.get("run") or {}).get("report")
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
        report.reason = ("VOID: plants not caught: "
                         f"{[n for n, c in report.planted.items() if not c]}")
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 3 of 3 plants caught"
    return report


def main() -> int:
    import sys
    report = run_gate(live="--rescore" not in sys.argv)
    print(report.reason)
    print(json.dumps({"cases": report.cases, "measured": report.measured,
                      "planted": report.planted, "errors": report.errors},
                     indent=1, default=str))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
