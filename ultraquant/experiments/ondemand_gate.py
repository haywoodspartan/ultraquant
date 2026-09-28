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

**§11.128: PASSED on every measured criterion, with 9 of 9 planted
defects caught.** Astra's two runs took 62.1 s each, and Claude's own
rerun took 67.5 s. Nothing was spent.

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

**Round three (§11.129).** Astra's second review found nine more
defects:
- a crashed runner's reservation was never reconciled;
- lupine's usage figure could hide an open reservation;
- a failed release stopped being charged;
- a workload exiting normally could leave a background child running;
- the Windows descendant snapshot could select a process that reused a
  PID, and it missed orphans;
- this exam accepted a zero price;
- no scenario ran the real WSL path;
- a shell could wrap a Windows program past the interop guard.

Claude measured the remedy for the last one in WSL before anything was
written. Removing WSL_INTEROP does not block interop. A private mount
namespace with /run/WSL covered, and every capability dropped, blocks
it, and the job cannot undo it.

Added, and frozen (sha256 e1a12e89..., with Amendments A ab80408f...
and B, recorded before runner v3 existed):
- 1 (l)-(n): a background child after exit 0, and after exit 3; an
  orphaned grandchild. No survivor after any of them.
- 2 (m): a reservation inside its window refuses admission. Past its
  window, it is reconciled first and settled at its worst case.
  2 (n): lupine's usage plus an open reservation, measured at
  ``month_spent()``.
- 3: the rate is the stub's own price, the cost covers the stub's own
  charge, and a failed release is charged until it is reconciled.
- 10: the production WSL path at no cost: the stub as lupine under WSL
  python3, with Linux survivors checked inside WSL. It covers success, a
  process tree, a background child, an orphan, the supervisor's
  wsl.exe killed mid-job, and a shell-wrapped Windows program, which
  must print nothing. A positive control proves the shell itself ran.
- Amendment E, made AFTER Claude's first full v3 run and stated as
  such. Every measured criterion held there, including all of criterion
  10. P4 alone went uncaught: v3's reservation window refused the second
  runner, a correct second layer, and the exam counted that as a crash.
  Like P1, P3 and P11, P4 now removes both layers.
- Amendment D, made AFTER Astra's first v3 run and stated as such.
  Astra found two defects in this exam. The stub's banner repeats the
  command, so the interop marker was present whether or not interop
  ran. The markers now exist only if a program really ran:
  ``INTEROP-%OS%`` and ``LINUX-$((6*7))``. P11 was masked by the
  runner's outer Job Object, so like P1 and P3 it now removes both
  layers.
- Amendment C, found by Claude while checking this exam's own harness:
  ``wsl.exe -- CMD`` hands the command line to a shell, which expanded
  ``$HOME`` and ran backticks in a measured argument. ``--exec``
  passes it literally. The pinned argv now uses ``--exec``, and a WSL
  job must see ``a;b $HOME`` unexpanded.
- Plants P10-P14. Each is aimed where nothing else could mask it
  (Amendment B): on Windows, a Job Object with kill-on-close would
  mask a plant that only skips a kill, so P7 and P10 are judged under
  WSL.

**§11.129: PASSED on every measured criterion, with 14 of 14 planted
defects caught**, in two consecutive runs of 120.0 s and 119.9 s on
Claude's machine, WSL included. Nothing was spent. The record of how
it got there:
- Astra's own runs were VOID at 11 of 14, in a sandbox that could not
  reach WSL (E_ACCESSDENIED even for a plain stub query).
- Amendment D corrected two exam defects that Astra found.
- Claude's first full run held every criterion, but was VOID on P4
  alone. Amendment E corrected that.

| criterion | new in round three | result |
|---|---|---|
| 1 release on every path | background child after exit 0 and 3; an orphan | 14 of 14 cases |
| 2 refused before attach | reservation windows; usage plus a hold | 16 of 16 cases |
| 3 the ledger | the stub's own price and charge; charging until reconciled | 8 of 8 |
| 10 the production WSL path | literal arguments, a process tree, a background child, an orphan, wsl.exe killed mid-job, a sealed shell | 7 of 7 |

**Round four (§11.131).** Astra's third review found five more:
- reconciliation could end a lease through the wrong launcher;
- a late-recovered crash was charged only its original worst case;
- a settled crash went back inside max() with lupine's usage;
- on POSIX, the group was signalled after its leader was reaped;
- in this exam, the WSL "killed mid-job" case read the lease four
  seconds after run() raised, when it must be read at the instant of
  the raise.

Added, and frozen (sha256 68954ec8...) before runner v4 existed:
- 2 (o): a foreign launcher's reservation is refused, with no ``end``.
- 2 (p): a two-hour-old crash accrues two hours.
- 2 (q): a settled crash stays additive, through run().
- 10: the lease is read at the instant of the raise.
- Plants P15-P18.
- Case 2 (m2) now expects a crash to accrue from its start, never less
  than its worst case.

**§11.131: PASSED on every measured criterion, with 18 of 18 planted
defects caught**, in two consecutive runs of 130.2 s and 130.9 s on
Claude's machine, WSL included. Astra's own runs caught all 14 plants
that need no WSL, and passed every criterion that needs no WSL. P14 and
P15 need WSL, and they are caught here. Astra noted that P7 and P10
count as caught in its sandbox only because their WSL cases fail there
anyway. Here they are measured.

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
WSL_DISTRO = "Ubuntu"
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

    def __init__(self, wsl: bool = False) -> None:
        self.wsl = wsl
        # resolve() turns the 8.3 temp path into one WSL can reach
        self.dir = Path(tempfile.mkdtemp(prefix="uq_lupine_stub_")).resolve()
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
        """Logged stub processes and their descendants still running.

        Under WSL the logged PIDs are Linux PIDs, so they are checked
        inside WSL: a Windows handle check on them would be meaningless.
        """
        pids = {e["pid"] for e in self.log() if isinstance(e.get("pid"), int)}
        if not self.wsl:
            pids.discard(os.getpid())
            return sorted(pid for pid in pids if _alive(pid))
        if not pids:
            return []
        script = ("for p in " + " ".join(str(p) for p in sorted(pids))
                  + "; do kill -0 $p 2>/dev/null && echo $p; done; true")
        out = subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "--exec", "sh",
                              "-c", script], capture_output=True, timeout=60)
        text = out.stdout.decode("utf-8", "replace").replace("\x00", "")
        return sorted(int(t) for t in text.split() if t.isdigit())

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
        with contextlib.suppress(Exception):
            left = self.survivors()
            if self.wsl and left:
                subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "--exec", "kill",
                                "-9", *map(str, left)], capture_output=True,
                               timeout=60)
            elif not self.wsl:
                for pid in left:
                    _reap(pid)
        shutil.rmtree(self.dir, ignore_errors=True)


def _wsl_path(path) -> str:
    """The exam's own translation, independent of the runner's."""
    text = str(path)
    return f"/mnt/{text[0].lower()}/" + text[3:].replace("\\", "/")


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
    if getattr(stub, "wsl", False):
        launcher = od.Launcher(prefix=("python3", _wsl_path(STUB)),
                               wsl_distro=WSL_DISTRO, **extra)
    else:
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


def _scenario(body, wsl=False, **faults):
    """Run ``body(stub)`` against a fresh stub; reap anything left over.

    Under WSL the stub directory crosses into Linux through WSLENV with
    the /p flag, which translates the Windows path.
    """
    stub = Stub(wsl=wsl)
    env = {"LUPINE_STUB_DIR": str(stub.dir)}
    if wsl:
        kept = [e for e in os.environ.get("WSLENV", "").split(":")
                if e and not e.startswith("LUPINE_STUB_DIR")]
        env["WSLENV"] = ":".join([*kept, "LUPINE_STUB_DIR/p"])
    try:
        if faults:
            stub.set(**faults)
        with _environ(**env):
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
                  wsl=False, **faults):
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
            time.sleep(1.5 if wsl else 0.5)   # let killed processes finish
            ok = ok and not stub.survivors()
        return ok
    return lambda: _scenario(body, wsl=wsl, **faults)


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
    "l background child after exit 0": _release_case(
        ["background", "30", "0"], 30.0, expect_exit=0, contained=True),
    "m background child after exit 3": _release_case(
        ["background", "30", "3"], 30.0, expect_exit=3, contained=True),
    "n orphaned grandchild": _release_case(["orphan", "30"], 2.0,
                                           bounded=True, contained=True),
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


def _identity(runner) -> str:
    """A launcher's identity as v4 records it on a reservation."""
    return json.dumps([list(runner.launcher.prefix), runner.launcher.wsl_distro])


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
        ledger.reserve("r-crashed", now, 0.30, "a runner that died",
                       now + 3600)
        ledger.reserve("r-closed", now, 0.40, "a job that finished",
                       now + 3600)
        ledger.append(od.Receipt(
            label="closed", started=now, seconds=1.0, sku=None, rate=HIGH,
            cost=0.01, worst_case=0.40, exit_code=0, timed_out=False,
            lease=None, gpu=None, released=True, release_output="",
            output_tail="", attached_seconds=24.0, reservation="r-closed"))
        return math.isclose(ledger.spent_in_month(), 0.31, abs_tol=1e-12)
    return _scenario(body)


def _reservation_in_window():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        now = time.time()
        od.Ledger(stub.ledger_path).reserve("r-live", now, 0.02,
                                             "maybe still running", now + 600)
        try:
            _runner(stub).run(_job(["echo", "x"]))
        except od.LeaseBusy:
            return len(stub.runs()) == 0
        return False
    return _scenario(body)


def _reservation_past_window():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        now = time.time()
        runner = _runner(stub)
        od.Ledger(stub.ledger_path).reserve("r-dead", now - 120, 0.02,
                                             "a runner that died", now - 60,
                                             rate=HIGH, launcher=_identity(runner))
        receipt = runner.run(_job(["echo", "x"]))
        log = stub.log()
        first_end = next((i for i, e in enumerate(log)
                          if e.get("cmd") == "end"), None)
        first_run = next((i for i, e in enumerate(log)
                          if e.get("cmd") == "run"), None)
        settled = [r for r in stub.rows() if r.get("kind") == "settlement"
                   and r.get("reservation") == "r-dead"]
        # v4 (§11.131): accrued from its start, never below its worst case
        return (receipt.released and first_end is not None
                and first_run is not None and first_end < first_run
                and len(settled) == 1
                and float(settled[0]["cost"]) >= max(0.02, HIGH * 120 / 3600)
                - 1e-12)
    return _scenario(body)


def _usage_plus_open():
    from ultraquant.cloud import ondemand as od

    def body(stub):
        _seed(stub, 0.10)
        now = time.time()
        od.Ledger(stub.ledger_path).reserve("r-open", now, 0.30,
                                             "still inside its window",
                                             now + 3600)
        spent = _runner(stub).month_spent()
        return math.isclose(spent, max(0.10, 0.50 + 0.005) + 0.30,
                            abs_tol=1e-9)
    return _scenario(body, usage_cost=0.50)


def _foreign_launcher():
    """R3-1: a reservation made through another launcher is not ours to end."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        now = time.time()
        od.Ledger(stub.ledger_path).reserve(
            "r-foreign", now - 120, 0.02, "made under another distro",
            now - 60, rate=HIGH,
            launcher=json.dumps([["/elsewhere/lupine"], "OtherDistro"]))
        try:
            _runner(stub).run(_job(["echo", "x"]))
        except od.LeaseNotReleased:
            ends = [e for e in stub.log() if e.get("cmd") == "end"]
            return not ends and len(stub.runs()) == 0
        return False
    return _scenario(body)


def _late_recovery():
    """R3-2: a crash recovered two hours later is charged for two hours."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        now = time.time()
        runner = _runner(stub)
        od.Ledger(stub.ledger_path).reserve(
            "r-late", now - 7200, 0.02, "crashed two hours ago", now - 7000,
            rate=HIGH, launcher=_identity(runner))
        runner.run(_job(["echo", "x"]))
        settled = [r for r in stub.rows() if r.get("kind") == "settlement"
                   and r.get("reservation") == "r-late"]
        return (len(settled) == 1
                and float(settled[0]["cost"]) >= HIGH * 7200 / 3600 - 1e-9)
    return _scenario(body)


def _settled_stays_additive():
    """R3-3: a settled crash is not hidden inside max() with lupine's usage."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        _seed(stub, 0.10)
        now = time.time()
        runner = _runner(stub, 0.81)
        od.Ledger(stub.ledger_path).reserve(
            "r-crash", now - 720, 0.02, "crashed twelve minutes ago",
            now - 600, rate=HIGH, launcher=_identity(runner))
        try:
            runner.run(_job(["echo", "x"], 10.0))
        except od.OverBudget:
            return len(stub.runs()) == 0
        return False
    return _scenario(body, usage_cost=0.50)


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
    "m reservation inside its window": _reservation_in_window,
    "m2 reservation past its window": _reservation_past_window,
    "n usage plus open reservation": _usage_plus_open,
    "o foreign launcher refused": _foreign_launcher,
    "p late recovery accrues": _late_recovery,
    "q settled crash stays additive": _settled_stays_additive,
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
                  "cost = rate x attached / 3600": True,
                  "rate is the stub's price": True,
                  "cost covers the stub's own charge": True}
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
            checks["rate is the stub's price"] &= (
                float(row.get("rate", 0)) == HIGH)
            stub_charge = (end["attach_end"] - end["attach_start"]) * HIGH / 3600
            checks["cost covers the stub's own charge"] &= (
                float(row.get("cost", 0)) >= stub_charge - 1e-9)
        month = od.Ledger(stub.ledger_path).spent_in_month()
        checks["other months excluded"] = math.isclose(
            month, sum(float(r["cost"]) for r in rows), abs_tol=1e-12)
        return checks

    def charged_until_reconciled(stub):
        from ultraquant.cloud import ondemand as od
        runner = _runner(stub)
        try:
            runner.run(_job(["echo", "x"]))
            return False                # the fault should have stopped it
        except od.LeaseNotReleased:
            pass
        failed = stub.receipt_rows()[-1]
        time.sleep(2.0)
        stub.set(end_failures=0)
        runner.run(_job(["echo", "y"]))
        settled = [r for r in stub.rows() if r.get("kind") == "settlement"
                   and r.get("reservation") == failed.get("reservation")]
        return (len(settled) == 1
                and float(settled[0]["cost"]) >= HIGH * 2.0 / 3600 - 1e-12)

    outcomes = _scenario(body)
    outcomes["a failed release is charged until reconciled"] = _scenario(
        charged_until_reconciled, end_failures=99)
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
        try:
            _runner(stub).run(_job(["sleep", "1"], label="first process"))
        except Exception:               # recorded as a failure, not a crash
            child.wait(120)
            return False
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
        == ["wsl.exe", "-d", "Ubuntu", "--cd", r"H:\w", "--exec", "python3",
            "/mnt/h/AI Model AGI/ultraquant/cloud/supervisor.py",
            "--deadline", "30", "--kill-grace", "10",
            "--release-attempts", "3", "--release-timeout", "5",
            "--lupine", lupine, "--", "run", "--", "nvidia-smi", "-L"])
    outcomes["WSL direct argv exact"] = (
        wsl.argv(["end"]) == ["wsl.exe", "-d", "Ubuntu", "--exec", lupine,
                              "end"])
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


# -- criterion 10: the production WSL path, at no cost ------------------------

def _wsl_killed():
    """The runner is interrupted; its Job Object takes wsl.exe with it."""
    from ultraquant.cloud import ondemand as od

    def body(stub):
        runner = _runner(stub)

        def interrupted(self, process, timeout):
            time.sleep(4.0)             # WSL starts, the stub takes a lease
            raise KeyboardInterrupt

        raised = False
        lease_at_return = "unread"
        with mock.patch.object(od.OnDemand, "_wait", interrupted):
            try:
                runner.run(_job(["sleep", "60"]))
            except KeyboardInterrupt:
                raised = True
                # R3-5: read at the instant run() raises. A lease the Linux
                # side ends a moment later was still billing at the return.
                lease_at_return = stub.lease()
        time.sleep(4.0)                 # survivors may take a moment to die
        rows = stub.receipt_rows()
        return (raised and len(stub.runs()) == 1 and lease_at_return is None
                and bool(rows) and rows[-1].get("released") is True
                and rows[-1].get("interrupted") is True
                and not stub.survivors())
    return _scenario(body, wsl=True)


def _wsl_interop():
    """A shell-wrapped Windows program runs nothing; Linux still runs.

    The markers exist only if a program really ran (Amendment D): the
    stub echoes its command in the banner, so a marker copied from the
    command line would prove nothing. cmd expands %OS% to Windows_NT; sh
    expands $((6*7)) to 42.
    """
    def body(stub):
        runner = _runner(stub)
        blocked = runner.run(_job(["shell", "/mnt/c/Windows/System32/cmd.exe",
                                   "/c", "echo", "INTEROP-%OS%"]))
        control = runner.run(_job(["shell", "sh", "-c",
                                   "echo LINUX-$((6*7))"]))
        return (blocked.released and control.released
                and "INTEROP-Windows_NT" not in blocked.output_tail
                and "LINUX-42" in control.output_tail)
    return _scenario(body, wsl=True)


def _wsl_literal():
    """Amendment C: nothing between Windows and lupine re-parses arguments."""
    def body(stub):
        receipt = _runner(stub).run(_job(["echo", "a;b $HOME `id -u`"]))
        lines = [line.strip() for line in receipt.output_tail.splitlines()]
        # the echo's own output line, not the banner that repeats the command
        return receipt.released and "a;b $HOME `id -u`" in lines
    return _scenario(body, wsl=True)


WSL_CASES = {
    "arguments arrive literally (WSL)": _wsl_literal,
    "success (WSL)": _release_case(["echo", "hi"], expect_exit=0, wsl=True),
    "process tree overrun (WSL)": _release_case(
        ["sleep-tree", "60"], 2.0, bounded=True, contained=True, wsl=True),
    "background child after exit (WSL)": _release_case(
        ["background", "60", "0"], 30.0, expect_exit=0, contained=True,
        wsl=True),
    "orphaned grandchild (WSL)": _release_case(
        ["orphan", "60"], 2.0, bounded=True, contained=True, wsl=True),
    "supervisor killed mid-job (WSL)": _wsl_killed,
    "4 interop blocked (WSL)": _wsl_interop,
}


def criterion_wsl(only=None) -> tuple:
    return _run_cases(WSL_CASES, only)


# -- criterion 6: the planted defects -----------------------------------------

def _wrapper(lines: str) -> Path:
    """A supervisor that is the real one with one function replaced.

    It loads supervisor.py by path, from Windows or from WSL, so the same
    wrapper serves both sides of criterion 10.
    """
    where = Path(tempfile.mkdtemp(prefix="uq_planted_supervisor_")).resolve()
    script = where / "planted_supervisor.py"
    real = REPO / "ultraquant" / "cloud" / "supervisor.py"
    script.write_text(
        "import importlib.util, os, sys\n"
        f"path = {str(real)!r} if os.name == 'nt' else {_wsl_path(real)!r}\n"
        "spec = importlib.util.spec_from_file_location('uq_planted', path)\n"
        "S = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(S)\n"
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

    def max_only(self, *args, **kwargs):
        rates = self.rates()
        code, output = self._query(["usage"])
        usage = od.parse_usage(output)
        return max(self.ledger.spent_in_month(), usage["cost"] + 0.005,
                   usage["gpu_seconds"] * max(rates.by_sku.values()) / 3600)

    def settled_inside_max(self, *args, **kwargs):
        rates = self.rates()
        code, output = self._query(["usage"])
        usage = od.parse_usage(output)
        now = time.time()
        return (max(self.ledger.settled_in_month(now), usage["cost"] + 0.005,
                    usage["gpu_seconds"] * max(rates.by_sku.values()) / 3600)
                + self.ledger._reserved_in_month(now))

    def halved(self, receipt):
        if isinstance(receipt, od.Receipt):
            receipt = od.Receipt(**{**receipt.__dict__,
                                    "rate": receipt.rate / 2,
                                    "cost": receipt.cost / 2})
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
         # Amendment E: the reservation window is a second, correct layer
         # that also keeps concurrent jobs apart, so the plant removes both.
         lambda: _planted(None, mock.patch.object(od.OnDemand, "_exclusive",
                                                  per_instance),
                          mock.patch.object(od.OnDemand, "_reconcile",
                                            lambda self: None))),
        ("P5 lupine's usage ignored", criterion_refusal,
         "b lupine's figure governs",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "month_spent",
                                                  ledger_only))),
        ("P6 zeroed accounting", criterion_ledger,
         "seconds cover the stub's workload",
         lambda: _planted(None, mock.patch.object(od.Ledger, "append",
                                                  zeroed))),
        ("P7 parent-only kill", criterion_wsl,
         "process tree overrun (WSL)",
         lambda: _planted("S.kill_tree = lambda process, grace: "
                          "process.kill()")),
        ("P8 no reconciliation", criterion_refusal,
         "j2 outstanding lease refused",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "_reconcile",
                                                  lambda self: None))),
        ("P9 lenient price parse", criterion_refusal, "g partial price table",
         lambda: _planted(None, mock.patch.object(od, "parse_gpus",
                                                  lenient))),
        ("P10 no kill after an ordinary exit", criterion_wsl,
         "background child after exit (WSL)",
         lambda: _planted("_kill = S.kill_tree\n"
                          "S.kill_tree = lambda p, g: (None if p.poll() "
                          "is not None else _kill(p, g))")),
        ("P11 the snapshot kill of 11.128, no Job Object", criterion_release,
         "n orphaned grandchild",
         lambda: _planted(_SNAPSHOT_PLANT,
                          mock.patch.object(od.supervisor, "launch",
                                            _plain_launch),
                          mock.patch.object(od.supervisor, "kill_tree",
                                            _snapshot_kill))),
        ("P12 max() hides an open reservation", criterion_refusal,
         "n usage plus open reservation",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "month_spent",
                                                  max_only))),
        ("P13 halved rate and cost", criterion_ledger,
         "rate is the stub's price",
         lambda: _planted(None, mock.patch.object(od.Ledger, "append",
                                                  halved))),
        ("P14 no sealed namespace", criterion_wsl, "4 interop blocked (WSL)",
         lambda: _planted("S.sealed = lambda argv: list(argv)")),
        ("P15 release left to the supervisor's late cleanup", criterion_wsl,
         "supervisor killed mid-job (WSL)",
         lambda: _planted(None, mock.patch.object(
             od.OnDemand, "_release", lambda self: (True, "planted")))),
        ("P16 settlements inside max()", criterion_refusal,
         "q settled crash stays additive",
         lambda: _planted(None, mock.patch.object(od.OnDemand, "month_spent",
                                                  settled_inside_max))),
        ("P17 a crash settled at its fixed worst case", criterion_refusal,
         "p late recovery accrues",
         lambda: _planted(None, mock.patch.object(
             od.OnDemand, "_settlement_cost",
             lambda self, reservation, receipt, now: (
                 float(reservation["worst_case"]) if receipt is None
                 else float(receipt["rate"]) * max(0.0, now - (
                     float(receipt["started"])
                     + float(receipt.get("attached_seconds", 0.0))))
                 / 3600)))),
        ("P18 no launcher check", criterion_refusal,
         "o foreign launcher refused",
         lambda: _planted(None, mock.patch.object(
             od.OnDemand, "_same_launcher", lambda self, reservation: True))),
    ]


#: P11: the §11.128 way to kill, with no Job Object to fall back on - a
#: plain process group, then taskkill following parent links. An orphan
#: whose parent has exited has no link to follow.
_SNAPSHOT_PLANT = """
import subprocess as _sp
def _launch(argv, env, log, *, cwd=None):
    return _sp.Popen(argv, env=env, cwd=cwd, stdin=_sp.DEVNULL, stdout=log,
                     stderr=_sp.STDOUT,
                     creationflags=getattr(_sp, 'CREATE_NEW_PROCESS_GROUP', 0))
def _kill(process, grace):
    _sp.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
S.launch = _launch
S.kill_tree = _kill
"""


def _plain_launch(argv, env, log, *, cwd=None):
    """P11, runner side: the supervisor started with no Job Object."""
    return subprocess.Popen(
        argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _snapshot_kill(process, grace):
    """P11, runner side: taskkill following parent links, as in 11.128."""
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
            ("8 timeouts are named by the deadline", criterion_timeouts),
            ("10 the production WSL path", criterion_wsl))


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
