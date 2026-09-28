"""Paid GPUs attach on demand. The on-demand gate.

The user: "The Usage needs to be on-Demand" - and, before that, "money
is an issue". lupine.sh bills each attached GPU per second, and nothing
attached is nothing billed. So the only way this project may touch a
paid GPU is a runner that attaches one for a job's own seconds, lets go
on every path - failure, overrun, a garbled CLI, a flaky network -
refuses before leasing any job that could carry the month past its cap,
and writes down every GPU-second it spends. GPT-6 Astra wrote
``ultraquant/cloud/ondemand.py``; Claude wrote this exam.

**No money is spent here.** The exam drives the runner through
``lupine_stub.py``, a stand-in CLI that prints what lupine v0.3.1
prints and keeps its own lease state and call log. That state is what
the exam believes: a runner saying it released a lease is not evidence.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 8075e53b...) before the runner existed:

1. **Release on every path**: success, a failing workload, an overrun
   (killed at max_seconds, within GRACE), unparseable CLI output, and a
   transient ``end`` failure - after each the stub holds no lease and the
   receipt says released. When ``end`` never succeeds: LeaseNotReleased,
   and the ledger still records the unreleased receipt.
2. **Refused before attach**: a job that fits at the cheapest rate but
   not the highest; lupine's reported spend governing a lower ledger;
   a zero cap; an unreadable usage report - each refused with zero
   ``run`` calls. A pinned SKU is estimated at its own rate.
3. **The ledger**: one receipt per launched job, cost = rate x seconds /
   3600, other months excluded.
4. **The local GPU stays hidden**: every ``run`` carries
   LUPINE_DISABLE_LOCAL=1, and WSLENV forwards it into WSL.
5. **One lease at a time**: concurrent jobs never overlap, and no
   ``end`` lands inside another job's run.
6. **The exam can fail**: five planted defects (P1-P5), each caught by
   its criterion, or the gate is VOID.
7. **Flags reach lupine**: --gpu only when pinned, --remote, --region,
   --pip, the workload after ``--``; the WSL argv exact.
8. **The full suite is green.** Checked outside this module.

Not gated here: a real lease. That is criterion R, run only with the
user's OK and a cost estimate.

**PASSED on every measured criterion, with 5 of 5 planted defects
caught** - in 11.7 s, spending nothing. Astra's run reported the same
verdict; Claude reran the exam rather than taking its word.

| criterion | cases | result |
|---|---|---|
| 1 release on every path | success, exit 3, overrun, garbled CLI, flaky ``end``, dead ``end`` | 6 of 6 |
| 2 refused before attach | highest rate, lupine's figure, zero cap, unreadable usage, pinned rate | 5 of 5, zero ``run`` calls |
| 3 the ledger | receipt count, cost formula, other months | 3 of 3 |
| 4 the local GPU stays hidden | flag set over a preset "0", WSLENV forwards and keeps | 4 of 4 |
| 5 one lease at a time | two threads | never overlapping |
| 7 flags reach lupine | --gpu, --remote, --region, --pip, ``--``, WSL argv | 7 of 7 |

**The review found what the exam could not.** A job interrupted with
Ctrl-C was released and recorded correctly - but its receipt said
"timed out", because the cleanup path labelled every kill a timeout.
A ledger that misdescribes the work is the §11.122 defect again, in a
new place. Fixed: timeouts are labelled only where they happen, and a
receipt now names the error that stopped its job. Probes beyond the
exam, pinned in ``tests/test_ondemand.py``: an interrupt mid-job, a
launch that fails, an unknown SKU (never launched), and the real
CLI's own text - ``gpus`` in 236 ms, ``usage`` in 267 ms and ``end``
in 95 ms through WSL, well inside the runner's timeouts.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["OnDemandReport", "run_gate"]

STUB = Path(__file__).with_name("lupine_stub.py")
HIGH, LOW = 1.50, 1.10            # the stub's price list, as lupine's


class Stub:
    """One isolated stand-in lupine: its state, its log, its ledger."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_lupine_stub_"))
        self.ledger_path = self.dir / "ledger.jsonl"

    def set(self, **faults) -> None:
        state = self.state()
        state.update(faults)
        (self.dir / "state.json").write_text(json.dumps(state),
                                             encoding="utf-8")

    def state(self) -> dict:
        try:
            return json.loads((self.dir / "state.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
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

    def cleanup(self) -> None:
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
    launcher = od.Launcher(prefix=(sys.executable, str(STUB)))
    return od.OnDemand(launcher, od.Ledger(stub.ledger_path), cap)


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
    """Run ``body(stub, runner)`` against a fresh stub; return its result."""
    stub = Stub()
    try:
        if faults:
            stub.set(**faults)
        with _environ(LUPINE_STUB_DIR=str(stub.dir)):
            return body(stub)
    finally:
        stub.cleanup()


# -- criterion 1 -------------------------------------------------------------

def criterion_release() -> tuple:
    from ultraquant.cloud import ondemand as od
    from ultraquant.experiments import lupine_stub as S
    outcomes = {}

    def plain(command, max_seconds=30.0, **faults):
        def body(stub):
            started = time.monotonic()
            receipt = _runner(stub).run(_job(command, max_seconds))
            wall = time.monotonic() - started
            return receipt, stub.lease(), wall, stub.log()
        return _scenario(body, **faults)

    r, lease, _, log = plain(["echo", "hi"])
    outcomes["a success"] = (lease is None and r.released
                             and r.exit_code == 0)
    parsed = (r.lease is not None and r.gpu == S.GPU_NAME)
    r, lease, _, _ = plain(["exit", "3"])
    outcomes["b workload fails"] = (lease is None and r.released
                                    and r.exit_code == 3)
    r, lease, wall, _ = plain(["sleep", "5"], max_seconds=1.0)
    outcomes["c overrun"] = (lease is None and r.released and r.timed_out
                             and wall <= 1.0 + od.OnDemand.GRACE)
    r, lease, _, _ = plain(["echo", "hi"], garbage=True)
    outcomes["d garbled CLI"] = (lease is None and r.released
                                 and r.lease is None)
    r, lease, _, log = plain(["echo", "hi"], end_failures=1)
    ends = [e for e in log if e.get("cmd") == "end"]
    outcomes["e transient end failure"] = (lease is None and r.released
                                           and len(ends) >= 2)

    def never(stub):
        try:
            _runner(stub).run(_job(["echo", "hi"]))
        except od.LeaseNotReleased:
            receipts = od.Ledger(stub.ledger_path).receipts()
            return bool(receipts) and receipts[-1]["released"] is False
        return False
    outcomes["f end never succeeds"] = _scenario(never, end_failures=99)
    ok = all(outcomes.values())
    return ok, {"outcomes": outcomes, "parsed_lease_and_gpu": parsed}


# -- criterion 2 -------------------------------------------------------------

def criterion_refusal() -> tuple:
    from ultraquant.cloud import ondemand as od
    outcomes = {}

    def refused(cap, job, seed=None, **faults):
        def body(stub):
            if seed is not None:
                _seed(stub, seed)
            try:
                _runner(stub, cap).run(job)
            except od.OverBudget:
                return len(stub.runs()) == 0
            return False
        return _scenario(body, **faults)

    # 1.50 x (130 + 15) / 3600 = 0.0604 > 0.05 left; at 1.10 it is 0.0443
    outcomes["a unpinned at the highest rate"] = refused(
        1.00, _job(["echo", "x"], 130.0), seed=0.95)
    # lupine says 0.97; the ledger says 0.10; 1.50 x 115 / 3600 = 0.0479
    outcomes["b lupine's figure governs"] = refused(
        1.00, _job(["echo", "x"], 100.0), seed=0.10, usage_cost=0.97)
    outcomes["c zero cap"] = refused(0.0, _job(["echo", "x"], 1.0))
    outcomes["d unreadable usage"] = refused(
        100.0, _job(["echo", "x"], 1.0), usage_broken=True)

    def pinned(stub):
        runner = _runner(stub)
        grace = od.OnDemand.GRACE
        return (math.isclose(runner.estimate(_job(["x"], 100.0, sku="a100")),
                             LOW * (100.0 + grace) / 3600, rel_tol=1e-12)
                and math.isclose(runner.estimate(_job(["x"], 100.0)),
                                 HIGH * (100.0 + grace) / 3600,
                                 rel_tol=1e-12))
    outcomes["e pinned SKU at its own rate"] = _scenario(pinned)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 3 -------------------------------------------------------------

def criterion_ledger() -> tuple:
    from ultraquant.cloud import ondemand as od

    def body(stub):
        _seed(stub, 5.0, started=time.time() - 45 * 86400)  # another month
        runner = _runner(stub)
        first = runner.run(_job(["echo", "one"]))
        second = runner.run(_job(["echo", "two"]))
        ledger = od.Ledger(stub.ledger_path)
        receipts = ledger.receipts()
        costs = all(math.isclose(r.cost, r.rate * r.seconds / 3600,
                                 rel_tol=1e-9, abs_tol=1e-15)
                    for r in (first, second))
        month = ledger.spent_in_month()
        return {"one receipt per launched job": len(receipts) == 3,
                "cost = rate x seconds / 3600": costs,
                "other months excluded": math.isclose(
                    month, first.cost + second.cost, abs_tol=1e-12)},             [first.rate, second.rate]
    outcomes, rates = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes,
                                    "unpinned_rates": rates}


# -- criterion 4 -------------------------------------------------------------

def criterion_hidden() -> tuple:
    def body(stub):
        with _environ(LUPINE_DISABLE_LOCAL="0", WSLENV="FOO/p"):
            runner = _runner(stub)
            runner.run(_job(["echo", "x"]))
            env = runner._environment()
        seen = [e.get("disable_local") for e in stub.runs()]
        entries = (env.get("WSLENV") or "").split(":")
        return {"every run hides the local GPU":
                    bool(seen) and all(v == "1" for v in seen),
                "environment sets the flag":
                    env.get("LUPINE_DISABLE_LOCAL") == "1",
                "WSLENV forwards it": "LUPINE_DISABLE_LOCAL/u" in entries,
                "WSLENV keeps what was there": "FOO/p" in entries}
    outcomes = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 5 -------------------------------------------------------------

def criterion_one_lease() -> tuple:
    def body(stub):
        runner = _runner(stub)
        runner.rates()                  # both threads price from one table
        errors = []

        def work(label):
            try:
                runner.run(_job(["sleep", "1.0"], label=label))
            except Exception as exc:    # the exam records, never hides
                errors.append(repr(exc))
        threads = [threading.Thread(target=work, args=(f"job{i}",))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        log = stub.log()
        spans = {}
        for event in log:
            if event.get("cmd") == "run":
                spans.setdefault(event["pid"], {})[event["phase"]] = event["t"]
        intervals = sorted((s["start"], s.get("stop", math.inf))
                           for s in spans.values())
        apart = (len(intervals) == 2
                 and intervals[1][0] >= intervals[0][1])
        ends = [e["t"] for e in log if e.get("cmd") == "end"]
        clean = not any(a < t < b for t in ends for a, b in intervals)
        return {"two jobs ran": len(intervals) == 2 and not errors,
                "never overlapping": apart,
                "no end inside a run": clean}
    outcomes = _scenario(body)
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 7 -------------------------------------------------------------

def criterion_flags() -> tuple:
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
    wsl = od.Launcher(prefix=("/home/u/.local/bin/lupine",),
                      wsl_distro="Ubuntu")
    outcomes["WSL argv exact"] = (
        wsl.argv(["run", "--", "nvidia-smi", "-L"], workdir=r"H:\w",
                 limit=30.0)
        == ["wsl.exe", "-d", "Ubuntu", "--cd", r"H:\w", "--", "timeout",
            "--kill-after=10", "30", "/home/u/.local/bin/lupine", "run",
            "--", "nvidia-smi", "-L"]
        and wsl.argv(["end"])
        == ["wsl.exe", "-d", "Ubuntu", "--", "/home/u/.local/bin/lupine",
            "end"]
        and od.Launcher(prefix=("lupine",)).argv(["end"], workdir="x",
                                                 limit=5.0)
        == ["lupine", "end"])
    return all(outcomes.values()), {"outcomes": outcomes}


# -- criterion 6: the planted defects ----------------------------------------

def _plants():
    """(name, criterion it must fail, patch context) for P1-P5."""
    from ultraquant.cloud import ondemand as od

    def cheapest(self, sku):
        if sku is None:
            return min(self.by_sku.values())
        return self.by_sku[sku]

    def ledger_only(self):
        return self.ledger.spent_in_month()

    def unhidden(self):
        return {k: v for k, v in os.environ.items()
                if k != "LUPINE_DISABLE_LOCAL"}

    original_init = od.OnDemand.__init__

    def lockless(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._lock = contextlib.nullcontext()

    return [
        ("P1 release claimed, never done", criterion_release,
         mock.patch.object(od.OnDemand, "_release",
                           lambda self: (True, "planted"))),
        ("P2 unpinned at the cheapest rate", criterion_refusal,
         mock.patch.object(od.Rates, "for_sku", cheapest)),
        ("P3 local GPU left visible", criterion_hidden,
         mock.patch.object(od.OnDemand, "_environment", unhidden)),
        ("P4 no lock", criterion_one_lease,
         mock.patch.object(od.OnDemand, "__init__", lockless)),
        ("P5 lupine's usage ignored", criterion_refusal,
         mock.patch.object(od.OnDemand, "month_spent", ledger_only)),
    ]


def _planted_detail(name, detail) -> bool:
    """Whether the plant broke the very outcome it targets."""
    outcomes = detail.get("outcomes", {})
    target = {"P2 unpinned at the cheapest rate":
                  "a unpinned at the highest rate",
              "P5 lupine's usage ignored": "b lupine's figure governs"}
    if name in target:
        return outcomes.get(target[name]) is False
    return not all(outcomes.values())


@dataclass
class OnDemandReport:
    """Whether the runner spends only what a job needs, and says so.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        criteria: Criterion name -> (ok, detail with per-case outcomes).
        planted: Planted defect -> whether its criterion caught it.
        plant_errors: Plants that crashed the exam instead of failing a
            criterion - counted as not caught, since a crash shows
            nothing about the check.
        seconds: Wall time of the whole exam.
        reason: Plain-language verdict.
    """

    passes: bool
    criteria: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    plant_errors: dict = field(default_factory=dict)
    seconds: float = 0.0
    reason: str = ""


def run_gate() -> OnDemandReport:
    """Every criterion, then every planted defect, against the stub."""
    started = time.monotonic()
    report = OnDemandReport(passes=False)
    for name, check in (("1 release on every path", criterion_release),
                        ("2 refused before attach", criterion_refusal),
                        ("3 the ledger", criterion_ledger),
                        ("4 the local GPU stays hidden", criterion_hidden),
                        ("5 one lease at a time", criterion_one_lease),
                        ("7 flags reach lupine", criterion_flags)):
        try:
            report.criteria[name] = check()
        except Exception as exc:        # a crash is a failure, not a pass
            report.criteria[name] = (False, {"error": repr(exc)})
    try:
        plants = _plants()
    except Exception as exc:            # no runner, no plants: VOID below
        report.plant_errors["construction"] = repr(exc)
        plants = []
    for name, check, patch in plants:
        try:
            with patch:
                ok, detail = check()
            report.planted[name] = (not ok) and _planted_detail(name, detail)
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


if __name__ == "__main__":
    result = run_gate()
    print(result.reason)
    for key, (ok, detail) in result.criteria.items():
        print(f"  {'ok ' if ok else 'BAD'} {key}: {detail}")
    for key, caught in result.planted.items():
        print(f"  {'caught' if caught else 'MISSED'} {key}")
