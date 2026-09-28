"""Paid GPUs attach on demand. The on-demand gate, amended after review.

The user: "The Usage needs to be on-Demand" - and, before that, "money
is an issue". lupine.sh bills each attached GPU per second, and nothing
attached is nothing billed. So the only way this project may touch a
paid GPU is a runner that attaches one for a job's own seconds, lets go
on every path, refuses before leasing any job that could carry the
month past its cap, and writes down every GPU-second it spends.

**No money is spent here.** The exam drives the runner through
``lupine_stub.py``, a stand-in CLI that prints what lupine v0.3.1
prints, spawns real child processes, and keeps its own lease state,
clock and call log. That state is what the exam believes: a runner
saying it released a lease, or that a job took so many seconds, is not
evidence.

**Version 1 (§11.127) passed, and was not ready.** GPT-6 Astra's
adversarial review of it found 13 defects. Three were reproduced end to
end before anything was fixed: a grandchild holding the output pipe kept
a 0.5 s job busy 6.2 s, and it would have kept a job busy forever with a
grandchild that never exits. Two runner instances were both admitted
against one $0.05 cap ($0.096). A price row the parser could not read
vanished instead of refusing. Three of the thirteen were holes in this
exam: it believed internally consistent zeros, its stub never started a
real process, and it exited 0 on a failing verdict. Version 1's text and
result stay in the journal. This is the amended exam.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 29a2a606...) before runner v2 existed:

1. **Release on every path**:
   - success, a failing workload, an overrun, garbled CLI output, and a
     transient or a permanent ``end`` failure;
   - an overrun with a real child, and with a whole process tree (no
     descendant survives);
   - a slow ``end`` (the release time is charged), a hung ``end``
     (bounded), and Ctrl-C mid-job.
   Every bounded path finishes within max_seconds + grace().
2. **Refused before attach**, with zero ``run`` calls:
   - a job that fits only at the cheapest rate, lupine's figure over a
     lower ledger, a zero cap, an unreadable usage report, and a pinned
     rate;
   - two runner instances that one cap can pay for once;
   - a partially unreadable price table, and "$0.00" hiding 11 billed
     seconds;
   - a price rise between admissions;
   - an outstanding lease that still will not release. Once it can,
     it is reconciled first;
   - Windows-interop commands, and an open reservation with no receipt.
3. **The ledger matches the stub's own clock**: lease, GPU, workload
   seconds and attached seconds, read back from the ledger file.
4. **The local GPU stays hidden**: LUPINE_DISABLE_LOCAL=1, WSLENV, and
   no ``/mnt/`` entry on lupine's PATH.
5. **One lease at a time**: across two instances, and across two OS
   processes.
6. **The exam can fail**: nine planted defects, P1-P9, each targeting
   one case. A plant that crashes the exam counts as not caught.
7. **Flags reach lupine**: the supervised WSL argv is exact.
8. **Timeouts are named by the deadline**, not by an exit code of 124.
9. **The exam's own exit status** is nonzero unless PASS. Checked by
   tests/test_ondemand.py.
10. **The full suite is green.** Checked outside this module.

Not gated here: a real lease. That is criterion R, run only with the
user's OK and a cost estimate.

**PASSED on every measured criterion, with 9 of 9 planted defects
caught.** Astra's two runs took 62.1 s each, and Claude's own rerun
took 67.5 s. Nothing was spent.

| criterion | cases | result |
|---|---|---|
| 1 release on every path | 11, incl. a process tree, a hung ``end``, Ctrl-C | 11 of 11 |
| 2 refused before attach | 13, incl. two instances, a partial table, "$0.00" | 13 of 13 |
| 3 the ledger matches the stub's clock | lease, GPU, seconds, attached, month | 5 of 5 |
| 4 the local GPU stays hidden | flag over "0", WSLENV, no ``/mnt/`` on PATH | 5 of 5 |
| 5 one lease at a time | two instances; two OS processes | both apart |
| 7 flags reach lupine | flags, supervised WSL argv, path translation | 10 of 10 |
| 8 timeouts named by the deadline | exit 124 vs a real deadline | both right |

**Review F1, reproduced again against v2:** the grandchild that holds
stdout for 6 s now costs a 0.5 s job 1.15 s, against 6.20 s in v1.

**Astra's one concern about this exam**, recorded rather than
changed: ``grace()`` bounds the time from launch to release, not the
wait for the lock before admission. That wait bills nothing, because no
GPU is attached, and every bounded case here runs uncontended.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["OnDemandReport", "run_gate"]

STUB = Path(__file__).with_name("lupine_stub.py")
REPO = Path(__file__).resolve().parents[2]
HIGH, LOW = 1.50, 1.10            # the stub's price list, as lupine's
TIMING = {"kill_grace": 1.0, "release_attempts": 3, "release_timeout": 1.5,
          "drain_timeout": 1.0, "lock_wait": 120.0}
_SUPERVISOR_OVERRIDE = None        # set by plants that alter the supervisor


# -- the stub and its evidence ------------------------------------------------

def _alive(pid: int) -> bool:
    """Whether a process is still running, without disturbing it."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE,
                                                ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259            # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with contextlib.suppress(OSError):
            os.kill(pid, 9)


class Stub:
    """One isolated stand-in lupine: its state, log, ledger and lock."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_lupine_stub_"))
        self.ledger_path = self.dir / "ledger.jsonl"
        self.lock_path = self.dir / "lease.lock"

    def set(self, **faults) -> None:
        state = self.state()
        state.update(faults)
        (self.dir / "state.json").write_text(json.dumps(state),
                                             encoding="utf-8")

    def state(self) -> dict:
        for _ in range(50):
            try:
                return json.loads((self.dir / "state.json").read_text(
                    encoding="utf-8"))
            except FileNotFoundError:
                return {}
            except (OSError, ValueError):
                time.sleep(0.02)
        return {}

    def lease(self):
        return self.state().get("cached_lease")

    def log(self) -> list:
        try:
            lines = (self.dir / "log.jsonl").read_text(
                encoding="utf-8").splitlines()
        except OSError:
            return []
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except ValueError:          # two stubs appending at once
                continue
        return events

    def runs(self) -> list:
        return [e for e in self.log() if e.get("cmd") == "run"
                and e.get("phase") == "start"]

    def survivors(self) -> list:
        """Logged stub processes and their descendants still running."""
        pids = {e["pid"] for e in self.log() if isinstance(e.get("pid"), int)}
        pids.discard(os.getpid())
        return sorted(pid for pid in pids if _alive(pid))

    def rows(self) -> list:
        try:
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            with contextlib.suppress(ValueError):
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
        return out

    def receipt_rows(self) -> list:
        return [r for r in self.rows()
                if r.get("kind", "receipt") == "receipt"]

    def cleanup(self) -> None:
        for pid in self.survivors():
            _reap(pid)
        shutil.rmtree(self.dir, ignore_errors=True)


@contextlib.contextmanager
def _environ(**values):
    saved = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runner(stub: Stub, cap: float = 100.0):
    from ultraquant.cloud import ondemand as od
    extra = {}
    if _SUPERVISOR_OVERRIDE is not None:
        extra["supervisor"] = str(_SUPERVISOR_OVERRIDE)
    launcher = od.Launcher(prefix=(sys.executable, str(STUB)), **extra)
    return od.OnDemand(launcher, od.Ledger(stub.ledger_path), cap,
                       lock_path=stub.lock_path, **TIMING)


def _job(command, max_seconds=30.0, label="exam", **extra):
    from ultraquant.cloud import ondemand as od
    return od.Job(command=tuple(command), max_seconds=max_seconds,
                  label=label, **extra)


def _seed(stub: Stub, cost: float, started: float | None = None) -> None:
    from ultraquant.cloud import ondemand as od
    od.Ledger(stub.ledger_path).append(od.Receipt(
        label="seed", started=time.time() if started is None else started,
        seconds=0.0, sku=None, rate=HIGH, cost=cost, worst_case=cost,
        exit_code=0, timed_out=False, lease=None, gpu=None, released=True,
        release_output="", output_tail=""))


def _scenario(body, **faults):
    """Run ``body(stub)`` against a fresh stub; reap anything left over."""
    stub = Stub()
    try:
        if faults:
            stub.set(**faults)
        with _environ(LUPINE_STUB_DIR=str(stub.dir)):
            return body(stub)
    finally:
        stub.cleanup()


def _timed(stub, job, cap=100.0):
    """(receipt or exception, wall seconds, runner) for one job."""
    runner = _runner(stub, cap)
    started = time.monotonic()
    try:
        outcome = runner.run(job)
    except Exception as exc:            # the case decides what is right
        outcome = exc
    return outcome, time.monotonic() - started, runner


def _run_cases(cases: dict, only=None) -> tuple:
    outcomes, errors = {}, {}
    for name, case in cases.items():
        if only is not None and name not in only:
            continue
        try:
            outcomes[name] = bool(case())
        except Exception as exc:        # a crash is a failure, not a pass
            outcomes[name] = False
            errors[name] = repr(exc)
    detail = {"outcomes": outcomes}
    if errors:
        detail["errors"] = errors
    return all(outcomes.values()) and bool(outcomes), detail


# -- criterion 1: release on every path ---------------------------------------

def _release_case(command, max_seconds=30.0, expect_exit=None,
                  expect_timeout=None, bounded=False, contained=False,
                  **faults):
    def body(stub):
        receipt, wall, runner = _timed(stub, _job(command, max_seconds))
        if isinstance(receipt, Exception):
            return False
        ok = stub.lease() is None and receipt.released
        if expect_exit is not None:
            ok = ok and receipt.exit_code == expect_exit
        if expect_timeout is not None:
            ok = ok and receipt.timed_out is expect_timeout
        if bounded:
            ok = ok and wall <= max_seconds + runner.grace()
        if contained:
            time.sleep(0.5)             # let killed processes finish dying
            ok = ok and not stub.survivors()
        return ok
    return lambda: _scenario(body, **faults)


def _garbled():
    def body(stub):
        receipt, _, _ = _timed(stub, _job(["echo", "hi"]))
        return (not isinstance(receipt, Exception) and stub.lease() is None
                and receipt.released and receipt.lease is None)
    return _scenario(body, garbage=True)


def _flaky_end():
    def body(stub):
        receipt, _, _ = _timed(stub, _job(["echo", "hi"]))
        ends = [e for e in stub.log() if e.get("cmd") == "end"
                and e.get("phase") == "start"]
        return (not isinstance(receipt, Exception) and stub.lease() is None
                and receipt.released and len(ends) >= 2)
    return _scenario(body, end_failures=1)


def _dead_end():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        outcome, _, _ = _timed(stub, _job(["echo", "hi"]))
        rows = stub.receipt_rows()
        return (isinstance(outcome, od.LeaseNotReleased) and bool(rows)
                and rows[-1].get("released") is False)
    return _scenario(body, end_failures=99)


def _slow_end():
    def body(stub):
        receipt, wall, runner = _timed(stub, _job(["echo", "hi"], 30.0))
        ended = [e for e in stub.log() if e.get("outcome") == "ended"]
        if isinstance(receipt, Exception) or not ended:
            return False
        attached = ended[-1]["attach_end"] - ended[-1]["attach_start"]
        return (stub.lease() is None and receipt.released
                and receipt.attached_seconds >= attached - 0.05
                and wall <= 30.0 + runner.grace())
    return _scenario(body, end_delay=0.6)


def _hung_end():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        outcome, wall, runner = _timed(stub, _job(["echo", "hi"], 5.0))
        time.sleep(0.5)                 # let killed processes finish dying
        return (isinstance(outcome, od.LeaseNotReleased)
                and wall <= 5.0 + runner.grace()
                and not stub.survivors())
    return _scenario(body, end_hang=True)


def _ctrl_c():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        runner = _runner(stub)

        def interrupted(self, process, timeout):
            time.sleep(1.5)             # the stub starts and caches a lease
            raise KeyboardInterrupt

        raised = False
        with mock.patch.object(od.OnDemand, "_wait", interrupted):
            try:
                runner.run(_job(["sleep", "30"]))
            except KeyboardInterrupt:
                raised = True
        time.sleep(0.5)
        rows = stub.receipt_rows()
        return (raised and len(stub.runs()) == 1 and stub.lease() is None
                and bool(rows) and rows[-1].get("released") is True
                and rows[-1].get("interrupted") is True
                and rows[-1].get("timed_out") is False
                and not stub.survivors())
    return _scenario(body)


RELEASE_CASES = {
    "a success": _release_case(["echo", "hi"], expect_exit=0),
    "b workload fails": _release_case(["exit", "3"], expect_exit=3),
    "c overrun": _release_case(["sleep", "30"], 1.0, expect_timeout=True,
                               bounded=True),
    "d garbled CLI": _garbled,
    "e transient end failure": _flaky_end,
    "f end never succeeds": _dead_end,
    "g real child contained": _release_case(["sleep", "30"], 1.0,
                                            bounded=True, contained=True),
    "h process tree contained": _release_case(["sleep-tree", "30"], 1.0,
                                              bounded=True, contained=True),
    "i slow end charged": _slow_end,
    "j hung end bounded": _hung_end,
    "k Ctrl-C mid-job": _ctrl_c,
}


def criterion_release(only=None) -> tuple:
    return _run_cases(RELEASE_CASES, only)


# -- criterion 2: refused before attach ---------------------------------------

def _refused(cap, command, max_seconds, seed=None, **faults):
    from ultraquant.cloud import ondemand as od

    def body(stub):
        if seed is not None:
            _seed(stub, seed)
        runner = _runner(stub, cap)
        seconds = max_seconds(runner) if callable(max_seconds) else max_seconds
        try:
            runner.run(_job(command, seconds))
        except od.OverBudget:
            return len(stub.runs()) == 0
        return False
    return lambda: _scenario(body, **faults)


def _grace_seconds(total):
    """max_seconds making max_seconds + grace() equal ``total``."""
    return lambda runner: total - runner.grace()


def _pinned_rate():
    def body(stub):
        runner = _runner(stub)
        grace = runner.grace()
        return (math.isclose(runner.estimate(_job(["x"], 100.0, sku="a100")),
                             LOW * (100.0 + grace) / 3600, rel_tol=1e-12)
                and math.isclose(runner.estimate(_job(["x"], 100.0)),
                                 HIGH * (100.0 + grace) / 3600,
                                 rel_tol=1e-12))
    return _scenario(body)


def _two_instances():
    """One cap pays for one job; two instances race for it."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        # The seed makes the ledger the governing figure: lupine's own
        # "$0.00" carries $0.005 of rounding headroom, which would
        # otherwise hide the first job's cost from the second admission.
        _seed(stub, 0.10)
        first, second = _runner(stub), _runner(stub)
        job = _job(["sleep", "1"], 60.0)
        cap = 0.10 + first.estimate(job) + 0.0001
        first.monthly_cap = second.monthly_cap = cap
        barrier = threading.Barrier(2)
        original = od.OnDemand.month_spent

        def racing(self, *args, **kwargs):
            spent = original(self, *args, **kwargs)
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(3)         # a correct lock never lets both in
            return spent

        results = []

        def work(runner):
            try:
                results.append(runner.run(job))
            except Exception as exc:    # the exam records, never hides
                results.append(exc)

        with mock.patch.object(od.OnDemand, "month_spent", racing):
            threads = [threading.Thread(target=work, args=(r,))
                       for r in (first, second)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(120)
        refused = [r for r in results if isinstance(r, od.OverBudget)]
        return len(stub.runs()) == 1 and len(refused) == 1
    return _scenario(body)


def _partial_table():
    table = ("SKU              GPU                                            VRAM     $/HR\n"
             "a100             NVIDIA A100-SXM4-80GB                          80GB    $1.10\n"
             "rtx-pro-6000     NVIDIA RTX PRO 6000 Blackwell Server Edition   96GB    $1.50/hr\n"
             "\n"
             "Run on a type:  lupine run --gpu <SKU> <command>\n")
    return _refused(0.04, ["echo", "x"], 10.0, gpus_text=table)()


def _rounded_usage():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        runner = _runner(stub)
        job = _job(["echo", "x"], 1.0)
        runner.monthly_cap = runner.estimate(job) + 0.004
        try:
            runner.run(job)
        except od.OverBudget:
            return len(stub.runs()) == 0
        return False
    return _scenario(body, usage_seconds=11)


def _price_rise():
    from ultraquant.cloud import ondemand as od
    from ultraquant.experiments import lupine_stub as S

    def body(stub):
        runner = _runner(stub)
        first = runner.run(_job(["echo", "x"], 10.0))
        if not first.released:
            return False
        stub.set(gpus_text=S.GPUS_TABLE.replace("$1.50", "$3.00"))
        runner.monthly_cap = (runner.month_spent()
                              + HIGH * (100.0 + runner.grace()) / 3600
                              + 0.0005)
        try:
            runner.run(_job(["echo", "y"], 100.0))
        except od.OverBudget:
            return len(stub.runs()) == 1
        return False
    return _scenario(body)


def _outstanding():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        runner = _runner(stub)
        try:
            runner.run(_job(["echo", "x"]))
            return False                # the fault should have stopped it
        except od.LeaseNotReleased:
            pass
        try:                            # still failing: refuse, no launch
            runner.run(_job(["echo", "y"]))
            return False
        except od.LeaseNotReleased:
            if len(stub.runs()) != 1:
                return False
        stub.set(end_failures=0)        # the network is back
        receipt = runner.run(_job(["echo", "z"]))
        reconciled = [r for r in stub.rows()
                      if r.get("kind") == "reconciled"]
        return (receipt.released and len(stub.runs()) == 2
                and len(reconciled) == 1 and stub.lease() is None)
    return _scenario(body, end_failures=99)


def _outstanding_refused():
    """The half of the outstanding case that P8 must break."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        runner = _runner(stub)
        with contextlib.suppress(od.LeaseNotReleased):
            runner.run(_job(["echo", "x"]))
        try:
            runner.run(_job(["echo", "y"]))
        except od.LeaseNotReleased:
            return len(stub.runs()) == 1
        return False
    return _scenario(body, end_failures=99)


def _interop():
    from ultraquant.cloud import ondemand as od
    rejected = 0
    for command in (("/mnt/c/Python313/python.exe", "x.py"),
                    ("/mnt/c/tools/run", "x"), ("C:\\Tools\\CUDA.EXE",)):
        try:
            od.Job(command=command, max_seconds=1.0, label="interop")
        except ValueError:
            rejected += 1
    allowed = od.Job(command=("/mnt/c/tools/run",), max_seconds=1.0,
                     label="interop", allow_interop=True)
    ordinary = od.Job(command=("python3", "teacher.py"), max_seconds=1.0,
                      label="linux")
    return rejected == 3 and allowed.allow_interop and bool(ordinary.command)


def _open_reservation():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        ledger = od.Ledger(stub.ledger_path)
        now = time.time()
        ledger.reserve("r-crashed", now, 0.30, "a runner that died")
        ledger.reserve("r-closed", now, 0.40, "a job that finished")
        ledger.append(od.Receipt(
            label="closed", started=now, seconds=1.0, sku=None, rate=HIGH,
            cost=0.01, worst_case=0.40, exit_code=0, timed_out=False,
            lease=None, gpu=None, released=True, release_output="",
            output_tail="", attached_seconds=24.0, reservation="r-closed"))
        return math.isclose(ledger.spent_in_month(), 0.31, abs_tol=1e-12)
    return _scenario(body)


REFUSAL_CASES = {
    # 1.50 x 140 / 3600 = 0.0583 > 0.05 left; at 1.10 it is 0.0428
    "a unpinned at the highest rate": _refused(
        1.00, ["echo", "x"], _grace_seconds(140.0), seed=0.95),
    # lupine says 0.97 (+0.005 rounding); ledger 0.10; job 0.0479
    "b lupine's figure governs": _refused(
        1.00, ["echo", "x"], _grace_seconds(115.0), seed=0.10,
        usage_cost=0.97),
    "c zero cap": _refused(0.0, ["echo", "x"], 1.0),
    "d unreadable usage": _refused(100.0, ["echo", "x"], 1.0,
                                   usage_broken=True),
    "e pinned SKU at its own rate": _pinned_rate,
    "f two instances, one admitted": _two_instances,
    "g partial price table": _partial_table,
    "h rounded usage bounded": _rounded_usage,
    "i price rise governs": _price_rise,
    "j outstanding lease reconciled": _outstanding,
    "j2 outstanding lease refused": _outstanding_refused,
    "k interop rejected": _interop,
    "l open reservation counts": _open_reservation,
}


def criterion_refusal(only=None) -> tuple:
    return _run_cases(REFUSAL_CASES, only)


# -- criterion 3: the ledger matches the stub's clock -------------------------

def criterion_ledger(only=None) -> tuple:
    from ultraquant.cloud import ondemand as od
    from ultraquant.experiments import lupine_stub as S

    def body(stub):
        _seed(stub, 5.0, started=time.time() - 45 * 86400)  # another month
        runner = _runner(stub)
        runner.run(_job(["echo", "one"]))
        runner.run(_job(["sleep", "1"]))
        rows = stub.receipt_rows()[1:]          # the two jobs, from the file
        log = stub.log()
        starts = [e for e in log if e.get("cmd") == "run"
                  and e.get("phase") == "start"]
        stops = [e for e in log if e.get("cmd") == "run"
                 and e.get("phase") == "stop"]
        ends = [e for e in log if e.get("outcome") == "ended"]
        if not (len(rows) == len(starts) == len(stops) == len(ends) == 2):
            return {"two jobs on the ledger and in the log": False}
        checks = {"lease and GPU are the stub's": True,
                  "seconds cover the stub's workload": True,
                  "attached seconds cover the stub's attachment": True,
                  "cost = rate x attached / 3600": True}
        for row, start, stop, end in zip(rows, starts, stops, ends):
            checks["lease and GPU are the stub's"] &= (
                row.get("lease") == start["lease"]
                and row.get("gpu") == S.GPU_NAME)
            checks["seconds cover the stub's workload"] &= (
                float(row.get("seconds", 0)) >= stop["t"] - start["t"] - 0.05)
            checks["attached seconds cover the stub's attachment"] &= (
                float(row.get("attached_seconds", 0))
                >= end["attach_end"] - end["attach_start"] - 0.05)
            checks["cost = rate x attached / 3600"] &= math.isclose(
                float(row.get("cost", -1)),
                float(row.get("rate", 0))
                * float(row.get("attached_seconds", 0)) / 3600,
                rel_tol=1e-9, abs_tol=1e-15)
        month = od.Ledger(stub.ledger_path).spent_in_month()
        checks["other months excluded"] = math.isclose(
            month, sum(float(r["cost"]) for r in rows), abs_tol=1e-12)
        return checks

    outcomes = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 4: the local GPU stays hidden ----------------------------------

def criterion_hidden(only=None) -> tuple:
    def body(stub):
        fake = "/mnt/c/Windows/fake-cuda"
        path = fake + os.pathsep + os.environ.get("PATH", "")
        with _environ(LUPINE_DISABLE_LOCAL="0", WSLENV="FOO/p", PATH=path):
            runner = _runner(stub)
            runner.run(_job(["echo", "x"]))
            env = runner._environment()
        seen = stub.runs()
        entries = (env.get("WSLENV") or "").split(":")
        return {"every run hides the local GPU":
                    bool(seen) and all(e.get("disable_local") == "1"
                                       for e in seen),
                "lupine's PATH has no /mnt/ entry":
                    bool(seen) and all(
                        not any(part.startswith("/mnt/") for part in
                                e.get("path", "").split(os.pathsep))
                        for e in seen),
                "environment sets the flag":
                    env.get("LUPINE_DISABLE_LOCAL") == "1",
                "WSLENV forwards it": "LUPINE_DISABLE_LOCAL/u" in entries,
                "WSLENV keeps what was there": "FOO/p" in entries}
    outcomes = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 5: one lease at a time -----------------------------------------

def _intervals(stub) -> list:
    spans = {}
    for event in stub.log():
        if event.get("cmd") == "run":
            spans.setdefault(event["pid"], {})[event["phase"]] = event["t"]
    return sorted((s["start"], s.get("stop", math.inf))
                  for s in spans.values() if "start" in s)


def _apart(stub) -> bool:
    intervals = _intervals(stub)
    ends = [e["t"] for e in stub.log() if e.get("cmd") == "end"]
    separate = all(later[0] >= earlier[1]
                   for earlier, later in zip(intervals, intervals[1:]))
    clean = not any(a < t < b for t in ends for a, b in intervals)
    return len(intervals) == 2 and separate and clean


def _child_job(stub_dir: str) -> None:
    """Entry point for the second OS process in criterion 5."""
    stub = Stub.__new__(Stub)
    stub.dir = Path(stub_dir)
    stub.ledger_path = stub.dir / "ledger.jsonl"
    stub.lock_path = stub.dir / "lease.lock"
    os.environ["LUPINE_STUB_DIR"] = stub_dir
    _runner(stub).run(_job(["sleep", "1"], label="second process"))


def criterion_one_lease(only=None) -> tuple:
    def instances(stub):
        first, second = _runner(stub), _runner(stub)
        errors = []

        def work(runner, label):
            try:
                runner.run(_job(["sleep", "1"], label=label))
            except Exception as exc:    # the exam records, never hides
                errors.append(repr(exc))
        threads = [threading.Thread(target=work, args=(r, f"job{i}"))
                   for i, r in enumerate((first, second))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        return not errors and _apart(stub)

    def processes(stub):
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from ultraquant.experiments.ondemand_gate import _child_job; "
             "_child_job(sys.argv[2])", str(REPO), str(stub.dir)],
            cwd=str(REPO))
        time.sleep(0.3)
        _runner(stub).run(_job(["sleep", "1"], label="first process"))
        child.wait(120)
        return child.returncode == 0 and _apart(stub)

    outcomes = {"two instances never overlap": _scenario(instances),
                "two processes never overlap": _scenario(processes)}
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 7: flags reach lupine ------------------------------------------

def criterion_flags(only=None) -> tuple:
    from ultraquant.cloud import ondemand as od

    def after(flags, flag):
        return flags[flags.index(flag) + 1] if flag in flags else None

    def body(stub):
        runner = _runner(stub)
        runner.run(_job(["echo", "x"], sku="a100", remote=True,
                        region="us-east-1", pip=("numpy",)))
        runner.run(_job(["echo", "y"]))
        pinned, unpinned = stub.runs()
        return {"--gpu when pinned": after(pinned["flags"], "--gpu") == "a100",
                "no --gpu when unpinned": "--gpu" not in unpinned["flags"],
                "--remote": "--remote" in pinned["flags"],
                "--region": after(pinned["flags"], "--region") == "us-east-1",
                "--pip": after(pinned["flags"], "--pip") == "numpy",
                "workload after --": pinned["workload"] == ["echo", "x"]}
    outcomes = _scenario(body)
    lupine = "/home/u/.local/bin/lupine"
    wsl = od.Launcher(prefix=(lupine,), wsl_distro="Ubuntu",
                      supervisor=r"H:\AI Model AGI\ultraquant\cloud\supervisor.py")
    outcomes["WSL supervised argv exact"] = (
        wsl.supervised(["run", "--", "nvidia-smi", "-L"], workdir=r"H:\w",
                       deadline=30.0, kill_grace=10.0, release_attempts=3,
                       release_timeout=5.0)
        == ["wsl.exe", "-d", "Ubuntu", "--cd", r"H:\w", "--", "python3",
            "/mnt/h/AI Model AGI/ultraquant/cloud/supervisor.py",
            "--deadline", "30", "--kill-grace", "10",
            "--release-attempts", "3", "--release-timeout", "5",
            "--lupine", lupine, "--", "run", "--", "nvidia-smi", "-L"])
    outcomes["WSL direct argv exact"] = (
        wsl.argv(["end"]) == ["wsl.exe", "-d", "Ubuntu", "--", lupine, "end"])
    plain = od.Launcher(prefix=("lupine",), python=("py",),
                        supervisor="sup.py")
    outcomes["plain argv exact"] = (
        plain.argv(["end"], workdir="x") == ["lupine", "end"]
        and plain.supervised(["run", "--", "x"], workdir="w", deadline=1.5,
                             kill_grace=1.0, release_attempts=2,
                             release_timeout=0.5)
        == ["py", "sup.py", "--deadline", "1.5", "--kill-grace", "1",
            "--release-attempts", "2", "--release-timeout", "0.5",
            "--lupine", "lupine", "--", "run", "--", "x"])
    outcomes["WSL paths translate"] = (
        od.to_wsl_path(r"H:\AI Model AGI\x.py") == "/mnt/h/AI Model AGI/x.py"
        and od.to_wsl_path("C:/a/b") == "/mnt/c/a/b")
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 8: timeouts are named by the deadline --------------------------

def criterion_timeouts(only=None) -> tuple:
    def body(stub):
        runner = _runner(stub)
        quick = runner.run(_job(["exit", "124"], 30.0))
        late = runner.run(_job(["sleep", "30"], 1.0))
        return {"exit 124 is not a timeout":
                    quick.exit_code == 124 and quick.timed_out is False,
                "a deadline is a timeout": late.timed_out is True}
    outcomes = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 6: the planted defects -----------------------------------------

def _wrapper(lines: str) -> Path:
    """A supervisor that is the real one with one function replaced."""
    where = Path(tempfile.mkdtemp(prefix="uq_planted_supervisor_"))
    script = where / "planted_supervisor.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from ultraquant.cloud import supervisor as S\n"
        f"{lines}\n"
        "raise SystemExit(S.main(sys.argv[1:]))\n", encoding="utf-8")
    return script


@contextlib.contextmanager
def _planted(supervisor_lines=None, *patches):
    global _SUPERVISOR_OVERRIDE
    script = _wrapper(supervisor_lines) if supervisor_lines else None
    saved = _SUPERVISOR_OVERRIDE
    try:
        _SUPERVISOR_OVERRIDE = script
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            yield
    finally:
        _SUPERVISOR_OVERRIDE = saved
        if script is not None:
            shutil.rmtree(script.parent, ignore_errors=True)


def _plants():
    """(name, criterion, the case it must break, context) for P1-P9."""
    from ultraquant.cloud import ondemand as od

    def cheapest(self, sku):
        if sku is None:
            return min(self.by_sku.values())
        return self.by_sku[sku]

    def ledger_only(self, *args, **kwargs):
        return self.ledger.spent_in_month()

    def unhidden(self):
        return {k: v for k, v in os.environ.items()
                if k != "LUPINE_DISABLE_LOCAL"}

    def per_instance(self):
        return self.__dict__.setdefault("_planted_lock", threading.Lock())

    original_append = od.Ledger.append

    def zeroed(self, receipt):
        if isinstance(receipt, od.Receipt):
            receipt = od.Receipt(**{**receipt.__dict__, "seconds": 0.0,
                                    "attached_seconds": 0.0, "cost": 0.0})
        return original_append(self, receipt)

    def lenient(text):
        prices = {}
        for line in text.splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[-1].startswith("$"):
                with contextlib.suppress(ValueError):
                    prices[tokens[0]] = float(tokens[-1][1:])
        if not prices:
            raise ValueError("no prices")
        return od.Rates(prices)

    return [
        ("P1 release claimed, never done", criterion_release, "a success",
         lambda: _planted("S.release = lambda *a, **k: (True, 'planted')",
                          mock.patch.object(od.OnDemand, "_release",
                                            lambda self: (True, "planted")))),
        ("P2 unpinned at the cheapest rate", criterion_refusal,
         "a unpinned at the highest rate",
         lambda: _planted(None, mock.patch.object(od.Rates, "for_sku",
                                                  cheapest))),
        ("P3 local GPU left visible", criterion_hidden,
         "every run hides the local GPU",
         lambda: _planted(
             "S.child_environment = lambda base=None: "
             "dict(os.environ if base is None else base)",
             mock.patch.object(od.OnDemand, "_environment", unhidden))),
        ("P4 the per-instance lock of 11.127", criterion_one_lease,
         "two instances never overlap",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "_exclusive",
                                                  per_instance))),
        ("P5 lupine's usage ignored", criterion_refusal,
         "b lupine's figure governs",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "month_spent",
                                                  ledger_only))),
        ("P6 zeroed accounting", criterion_ledger,
         "seconds cover the stub's workload",
         lambda: _planted(None, mock.patch.object(od.Ledger, "append",
                                                  zeroed))),
        ("P7 parent-only kill", criterion_release,
         "h process tree contained",
         lambda: _planted("S.kill_tree = lambda process, grace: "
                          "process.kill()")),
        ("P8 no reconciliation", criterion_refusal,
         "j2 outstanding lease refused",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "_reconcile",
                                                  lambda self: None))),
        ("P9 lenient price parse", criterion_refusal, "g partial price table",
         lambda: _planted(None, mock.patch.object(od, "parse_gpus",
                                                  lenient))),
    ]


@dataclass
class OnDemandReport:
    """Whether the runner spends only what a job needs, and says so.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        criteria: Criterion name -> (ok, detail with per-case outcomes).
        planted: Planted defect -> whether it broke the case it targets.
        plant_errors: Plants that crashed the exam instead of failing a
            case - counted as not caught, since a crash shows nothing
            about the check.
        seconds: Wall time of the whole exam.
        reason: Plain-language verdict.
    """

    passes: bool
    criteria: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    plant_errors: dict = field(default_factory=dict)
    seconds: float = 0.0
    reason: str = ""


CRITERIA = (("1 release on every path", criterion_release),
            ("2 refused before attach", criterion_refusal),
            ("3 the ledger matches the stub's clock", criterion_ledger),
            ("4 the local GPU stays hidden", criterion_hidden),
            ("5 one lease at a time", criterion_one_lease),
            ("7 flags reach lupine", criterion_flags),
            ("8 timeouts are named by the deadline", criterion_timeouts))


def run_gate() -> OnDemandReport:
    """Every criterion, then every planted defect, against the stub."""
    started = time.monotonic()
    report = OnDemandReport(passes=False)
    for name, check in CRITERIA:
        try:
            report.criteria[name] = check()
        except Exception as exc:        # a crash is a failure, not a pass
            report.criteria[name] = (False, {"error": repr(exc)})
    try:
        plants = _plants()
    except Exception as exc:            # no runner, no plants: VOID below
        report.plant_errors["construction"] = repr(exc)
        plants = []
    for name, check, target, context in plants:
        try:
            with context():
                ok, detail = check(only={target})
            outcome = detail.get("outcomes", {}).get(target)
            crashed = detail.get("errors", {}).get(target)
            report.planted[name] = outcome is False and crashed is None
            if outcome is None:
                report.plant_errors[name] = f"target {target!r} not measured"
            elif crashed is not None:
                report.plant_errors[name] = crashed
        except Exception as exc:        # a crash proves nothing: not caught
            report.planted[name] = False
            report.plant_errors[name] = repr(exc)
    report.seconds = time.monotonic() - started
    valid = bool(report.planted) and all(report.planted.values())
    failed = [n for n, (ok, _) in report.criteria.items() if not ok]
    report.passes = valid and not failed
    if not valid:
        missed = [n for n, caught in report.planted.items() if not caught]
        report.reason = f"VOID: planted defects not caught: {missed}"
    elif failed:
        report.reason = f"FAIL: {failed}"
    else:
        report.reason = (f"PASS: all {len(report.criteria)} measured "
                         f"criteria hold; {len(report.planted)} of "
                         f"{len(report.planted)} planted defects caught; "
                         f"{report.seconds:.1f} s")
    return report


def main() -> int:
    result = run_gate()
    print(result.reason)
    for key, (ok, detail) in result.criteria.items():
        print(f"  {'ok ' if ok else 'BAD'} {key}: {detail}")
    for key, caught in result.planted.items():
        note = result.plant_errors.get(key)
        print(f"  {'caught' if caught else 'MISSED'} {key}"
              + (f"  ({note})" if note else ""))
    return 0 if result.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
