"""Paid GPUs attach on demand: the user pays out of pocket.

The user's words are "The Usage needs to be on-Demand", and "money is an
issue". Lupine bills each attached GPU per second. A finished workload
still has a cached lease; only a confirmed end stops the attachment clock.

The review of 11.127 found inherited output pipes that defeated deadlines,
incomplete tree cleanup, competing admissions, missing crash reservations
and reconciliation, release time omitted from bills, stale or partial
prices, rounded usage treated as exact, and exit codes mistaken for timers.
Windows interop could also bypass the local-GPU setting. Three findings
concerned the old exam: simulated children, self-consistent zero accounting,
and a successful exit status on failure.

Version 2 splits ownership. The supervisor runs beside lupine, writes its
output to a file, owns the deadline and process tree, and retries release.
This runner locks admissions across processes, reconciles a pending lease,
reads fresh strict prices and bounded usage, fsyncs a reservation, and keeps
its own bounded backstop and release. Receipts charge attached seconds,
including release, and conservatively count jobs spanning UTC months. The
frozen v2 exam supplies real descendants, an independent attachment clock
and nine planted defects. Interop commands require an explicit opt-in.

Still unverified: --remote kill semantics, lupine's multi-GPU behavior, the
unpinned rate ($1.10 on the page versus $1.50 in the CLI), and descendants
that detach from their parent or session. The environment flag is
configuration, not isolation. --remote (a runner pod with no local GPU) is
the isolation.
"""

from __future__ import annotations

import contextlib
import errno
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import supervisor


class OverBudget(RuntimeError):
    """A job cannot safely be admitted within the monthly budget."""


class LeaseNotReleased(RuntimeError):
    """The receipt or outstanding ledger row needs reconciliation."""

    def __init__(self, receipt):
        self.receipt = receipt
        row = receipt if isinstance(receipt, dict) else asdict(receipt)
        super().__init__(f"Lease release unconfirmed for {row.get('label')!r}: "
                         f"{row.get('release_output', '')}")


class LeaseBusy(RuntimeError):
    """Another admission holds the shared lease lock."""


@dataclass(frozen=True)
class Rates:
    by_sku: dict

    def for_sku(self, sku):
        if sku is None:
            return max(self.by_sku.values())
        try:
            return self.by_sku[sku]
        except KeyError as exc:
            raise ValueError(f"Unknown GPU SKU: {sku!r}") from exc


def parse_gpus(text) -> Rates:
    """Reject the entire table if even one price row is unreadable."""
    prices = {}
    for line in text.replace("\x00", "").splitlines():
        tokens = line.split()
        if (not tokens or tokens[0] == "SKU"
                or line.lstrip().startswith("Run on a type:")):
            continue
        try:
            if len(tokens) < 2 or not tokens[-1].startswith("$"):
                raise ValueError("missing price")
            value = tokens[-1][1:]
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                                value):
                raise ValueError("malformed price")
            price = float(value)
            if not math.isfinite(price) or price < 0:
                raise ValueError("invalid price")
            if tokens[0] in prices:
                raise ValueError("duplicate SKU")
        except ValueError as exc:
            raise ValueError(f"Unreadable GPU price line: {line!r}") from exc
        prices[tokens[0]] = price
    if not prices:
        raise ValueError("No GPU price rows found")
    return Rates(prices)


def parse_usage(text) -> dict | None:
    """Require both a whole-second duration and a finite dollar amount."""
    seconds = cost = None
    for line in text.replace("\x00", "").splitlines():
        duration = re.fullmatch(r"\s*GPU time:\s*(.*?)\s*", line)
        if duration:
            match = re.fullmatch(
                r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?", duration[1])
            if match and any(part is not None for part in match.groups()):
                seconds = sum(int(part or 0) * scale for part, scale in
                              zip(match.groups(), (3600, 60, 1)))
        amount = re.fullmatch(r"\s*Estimated cost:\s*\$(\S+)\s*", line)
        if amount:
            try:
                value = float(amount[1])
            except ValueError:
                continue
            if math.isfinite(value) and value >= 0:
                cost = value
    if seconds is None or cost is None:
        return None
    return {"gpu_seconds": seconds, "cost": cost}


def to_wsl_path(path) -> str:
    match = re.fullmatch(r"([A-Za-z]):[/\\](.*)", os.fspath(path))
    if not match:
        raise ValueError(f"A Windows path with a drive is required: {path!r}")
    return f"/mnt/{match[1].lower()}/" + match[2].replace("\\", "/")


@dataclass(frozen=True)
class Job:
    command: tuple
    max_seconds: float
    label: str
    sku: str | None = None
    remote: bool = False
    region: str | None = None
    pip: tuple = ()
    requirements: str | None = None
    workdir: str | None = None
    allow_interop: bool = False

    def __post_init__(self):
        if not self.command or not self.command[0]:
            raise ValueError("A job needs a command")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if not self.label:
            raise ValueError("A job needs a label")
        if not self.allow_interop and (
                self.command[0].lower().endswith(".exe")
                or self.command[0].startswith("/mnt/")):
            raise ValueError("Windows interop commands require allow_interop=True")


@dataclass
class Receipt:
    label: str
    started: float
    seconds: float
    sku: str | None
    rate: float
    cost: float
    worst_case: float
    exit_code: int | None
    timed_out: bool
    lease: str | None
    gpu: str | None
    released: bool
    release_output: str
    output_tail: str
    error: str = ""
    attached_seconds: float = 0.0
    interrupted: bool = False
    reservation: str = ""


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def default(cls):
        from ultraquant.config import _user_config_dir

        override = os.environ.get("ULTRAQUANT_GPU_LEDGER")
        return cls(override if override else _user_config_dir() / "gpu_ledger.jsonl")

    def _write(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
        with self.path.open("a+b") as handle:
            # A crash can leave a torn final line. Preserve it for inspection,
            # but keep the next durable record on its own parseable line.
            separator = b""
            if handle.seek(0, os.SEEK_END):
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    separator = b"\n"
            handle.write(separator + encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def rows(self) -> list[dict]:
        try:
            handle = self.path.open("r", encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        rows = []
        with handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def receipts(self) -> list[dict]:
        return [row for row in self.rows()
                if row.get("kind", "receipt") == "receipt"]

    def append(self, receipt):
        self._write({**asdict(receipt), "kind": "receipt"})

    def reserve(self, reservation_id, started, worst_case, label):
        self._write({"kind": "reservation", "reservation": reservation_id,
                     "started": started, "worst_case": worst_case, "label": label})

    def reconcile(self, note):
        self._write({"kind": "reconciled", "t": time.time(), "note": note})

    def outstanding(self) -> dict | None:
        pending = None
        for row in self.rows():
            kind = row.get("kind", "receipt")
            if kind == "reconciled":
                pending = None
            elif kind == "receipt":
                pending = None if row.get("released") is True else row
        return pending

    def spent_in_month(self, when=None) -> float:
        current = datetime.fromtimestamp(time.time() if when is None else when,
                                         timezone.utc)
        first = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        after = first.replace(year=first.year + 1, month=1) if first.month == 12 else (
            first.replace(month=first.month + 1))
        month_start, month_end = first.timestamp(), after.timestamp()
        rows = self.rows()
        closed = {row["reservation"] for row in rows
                  if row.get("kind", "receipt") == "receipt"
                  and isinstance(row.get("reservation"), str)}
        total = 0.0
        for row in rows:
            try:
                started = float(row["started"])
                if not math.isfinite(started):
                    continue
                kind = row.get("kind", "receipt")
                if kind == "reservation":
                    if (row.get("reservation") in closed
                            or not month_start <= started < month_end):
                        continue
                    cost = float(row["worst_case"])
                elif kind == "receipt":
                    seconds = max(0.0, float(row.get("seconds", 0.0)),
                                  float(row.get("attached_seconds", 0.0)))
                    if started >= month_end or started + seconds < month_start:
                        continue
                    cost = float(row["cost"])
                else:
                    continue
                if math.isfinite(cost) and cost >= 0:
                    total += cost
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return total


@dataclass(frozen=True)
class Launcher:
    prefix: tuple
    wsl_distro: str | None = None
    python: tuple | None = None
    supervisor: str | None = None

    def _wrap(self, body, workdir):
        if self.wsl_distro is None:
            return body
        result = ["wsl.exe", "-d", self.wsl_distro]
        if workdir:
            result.extend(["--cd", workdir])
        return [*result, "--", *body]

    def argv(self, args, *, workdir=None) -> list[str]:
        return self._wrap([*self.prefix, *args], workdir)

    def supervised(self, run_args, *, workdir=None, deadline, kill_grace,
                   release_attempts, release_timeout) -> list[str]:
        sup = os.fspath(self.supervisor or Path(__file__).with_name("supervisor.py"))
        py = self.python or (("python3",) if self.wsl_distro else (sys.executable,))
        body = [*py, to_wsl_path(sup) if self.wsl_distro else sup,
                "--deadline", f"{deadline:g}", "--kill-grace", f"{kill_grace:g}",
                "--release-attempts", str(release_attempts),
                "--release-timeout", f"{release_timeout:g}"]
        for part in self.prefix:
            body.extend(["--lupine", part])
        return self._wrap([*body, "--", *run_args], workdir)

    def cwd(self, workdir):
        return workdir if self.wsl_distro is None else None

    @classmethod
    def for_this_machine(cls):
        override = os.environ.get("ULTRAQUANT_LUPINE")
        if os.name == "nt":
            distro = os.environ.get("ULTRAQUANT_LUPINE_DISTRO", "Ubuntu")
            if not override:
                with tempfile.TemporaryFile(mode="w+b") as log:
                    process = subprocess.Popen(
                        ["wsl.exe", "-d", distro, "--", "sh", "-c", 'printf %s "$HOME"'],
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, **supervisor._group_options())
                    try:
                        process.wait(timeout=15.0)
                    except BaseException:
                        supervisor.kill_tree(process, 1.0)
                        with contextlib.suppress(BaseException):
                            process.wait(timeout=1.0)
                        raise
                    home = supervisor._read_output(log).strip()
                    if process.returncode or not home:
                        raise FileNotFoundError("WSL did not report its HOME")
                override = home.rstrip("/") + "/.local/bin/lupine"
            return cls((override,), wsl_distro=distro)
        if override:
            return cls((override,))
        executable = shutil.which("lupine")
        if executable:
            return cls((executable,))
        fallback = Path.home() / ".local" / "bin" / "lupine"
        if fallback.exists():
            return cls((str(fallback),))
        raise FileNotFoundError("lupine was not found")


def _clean_output(output):
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return (output or "").replace("\x00", "")


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class OnDemand:
    def __init__(self, launcher, ledger, monthly_cap, *, lock_path=None,
                 lock_wait=600.0, kill_grace=10.0, release_attempts=3,
                 release_timeout=5.0, drain_timeout=5.0):
        if not math.isfinite(monthly_cap) or monthly_cap < 0:
            raise ValueError("monthly_cap must be finite and nonnegative")
        for name, value in (("lock_wait", lock_wait), ("kill_grace", kill_grace),
                            ("release_timeout", release_timeout),
                            ("drain_timeout", drain_timeout)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(release_attempts, int) or release_attempts < 1:
            raise ValueError("release_attempts must be a positive integer")
        self.launcher = launcher
        self.ledger = ledger
        self.monthly_cap = monthly_cap
        if lock_path is None:
            lock_path = os.environ.get("ULTRAQUANT_GPU_LOCK")
        if not lock_path:
            from ultraquant.config import _user_config_dir

            lock_path = _user_config_dir() / "gpu_lease.lock"
        self.lock_path = Path(lock_path).resolve()
        self.lock_wait = lock_wait
        self.kill_grace = kill_grace
        self.release_attempts = release_attempts
        self.release_timeout = release_timeout
        self.drain_timeout = drain_timeout

    def _release_seconds(self):
        n = self.release_attempts
        return n * self.release_timeout + 0.25 * n * (n - 1)

    def grace(self):
        return self.kill_grace + 2 * self.drain_timeout + 2 * self._release_seconds() + 3.0

    @contextlib.contextmanager
    def _exclusive(self):
        key = os.path.normcase(str(self.lock_path))
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.Lock())
        deadline = time.monotonic() + self.lock_wait

        def pause():
            left = deadline - time.monotonic()
            if left <= 0:
                raise LeaseBusy(f"GPU lease lock is busy: {self.lock_path}")
            time.sleep(min(0.1, left))

        while not lock.acquire(blocking=False):
            pause()
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b") as handle:
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                if os.name == "nt":
                    import msvcrt
                else:
                    import fcntl
                while True:
                    try:
                        handle.seek(0)
                        if os.name == "nt":
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        pause()
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock.release()

    def _environment(self) -> dict:
        env = os.environ.copy()
        env["LUPINE_DISABLE_LOCAL"] = "1"
        entries = [entry for entry in env.get("WSLENV", "").split(":")
                   if entry and entry.split("/", 1)[0] != "LUPINE_DISABLE_LOCAL"]
        env["WSLENV"] = ":".join([*entries, "LUPINE_DISABLE_LOCAL/u"])
        return env

    def _stop(self, process, grace, deadline=math.inf):
        deadline = min(deadline, time.monotonic() + self.drain_timeout)
        process._uq_stop_deadline = deadline
        supervisor.kill_tree(process, min(grace, self.drain_timeout))
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, deadline - time.monotonic()))

    def _query(self, args, timeout=10.0, *, deadline=math.inf):
        with tempfile.TemporaryFile(mode="w+b") as log:
            process = subprocess.Popen(
                self.launcher.argv(args), env=self._environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                **supervisor._group_options())
            try:
                remaining = max(0.0, deadline - time.monotonic())
                cleanup = min(self.drain_timeout, 1.0, remaining / 2)
                process.wait(timeout=min(timeout, remaining - cleanup))
            except BaseException as exc:
                self._stop(process, 1.0, deadline)
                if isinstance(exc, subprocess.TimeoutExpired):
                    exc.output = supervisor._read_output(log)
                raise
            return process.returncode, supervisor._read_output(log)

    def rates(self):
        try:
            code, output = self._query(["gpus"])
            if code != 0:
                raise ValueError(output)
            return parse_gpus(output)
        except Exception as exc:
            raise OverBudget("GPU price list is unreadable") from exc

    def estimate(self, job, rates=None):
        rates = self.rates() if rates is None else rates
        return rates.for_sku(job.sku) * (job.max_seconds + self.grace()) / 3600

    def month_spent(self, rates=None):
        try:
            rates = self.rates() if rates is None else rates
            code, output = self._query(["usage"])
            usage = parse_usage(output)
            if code != 0 or usage is None:
                raise ValueError(output)
            return max(self.ledger.spent_in_month(), usage["cost"] + 0.005,
                       usage["gpu_seconds"] * max(rates.by_sku.values()) / 3600)
        except Exception as exc:
            raise OverBudget("Monthly GPU usage is unreadable") from exc

    def _release(self) -> tuple[bool, str]:
        outputs = []
        deadline = time.monotonic() + self._release_seconds()
        for attempt in range(1, self.release_attempts + 1):
            if time.monotonic() >= deadline:
                break
            try:
                code, output = self._query(["end"], timeout=self.release_timeout,
                                           deadline=deadline)
                outputs.append(output)
                if code == 0 and ("ended lease" in output or "no cached lease" in output):
                    return True, "\n".join(outputs)
            except Exception as exc:
                partial = _clean_output(getattr(exc, "output", None))
                outputs.append(f"{partial}\n{type(exc).__name__}: {exc}")
            if attempt < self.release_attempts:
                if time.monotonic() + 0.5 * attempt >= deadline:
                    break
                time.sleep(0.5 * attempt)
        return False, "\n".join(outputs)

    def _reconcile(self):
        row = self.ledger.outstanding()
        if row is not None:
            released, output = self._release()
            if not released:
                raise LeaseNotReleased(row)
            self.ledger.reconcile(output)

    def _wait(self, process, timeout) -> str:
        process.wait(timeout=timeout)
        return supervisor._read_output(process._uq_output)

    @staticmethod
    def _result(output):
        line = next((line for line in reversed(output.splitlines())
                     if line.startswith(supervisor.PREFIX)), None)
        if line is None:
            return {}
        try:
            result = json.loads(line[len(supervisor.PREFIX):])
            if not isinstance(result, dict):
                return {}
            for key in ("seconds", "attached_seconds"):
                value = result[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    return {}
            for key in ("deadline_hit", "interrupted", "released"):
                if type(result[key]) is not bool:
                    return {}
            if result["workload_exit"] is not None and type(result["workload_exit"]) is not int:
                return {}
            for key in ("release_output", "output_tail"):
                if not isinstance(result[key], str):
                    return {}
            for key in ("lease", "gpu"):
                if result[key] is not None and not isinstance(result[key], str):
                    return {}
            return result
        except (KeyError, ValueError, TypeError, OverflowError):
            return {}

    def run(self, job) -> Receipt:
        with self._exclusive():
            self._reconcile()
            rates = self.rates()
            worst = self.estimate(job, rates)
            spent = self.month_spent(rates)
            if not math.isfinite(self.monthly_cap) or spent + worst > self.monthly_cap:
                raise OverBudget(f"Spent ${spent:.6f} + job ${worst:.6f} "
                                 f"> monthly cap ${self.monthly_cap:.6f}")
            rate = rates.for_sku(job.sku)
            reservation = uuid4().hex
            started = time.time()
            process = log = None
            output = ""
            result = {}
            error = None
            released = backstop_fired = interrupted = False
            release_output = ""
            clock_started = time.monotonic()
            self.ledger.reserve(reservation, started, worst, job.label)
            try:
                flags = ["run"]
                if job.remote:
                    flags.append("--remote")
                if job.sku:
                    flags.extend(["--gpu", job.sku])
                if job.region:
                    flags.extend(["--region", job.region])
                for package in job.pip:
                    flags.extend(["--pip", package])
                if job.requirements:
                    flags.extend(["-r", job.requirements])
                flags.extend(["--", *job.command])
                argv = self.launcher.supervised(
                    flags, workdir=job.workdir, deadline=job.max_seconds,
                    kill_grace=self.kill_grace, release_attempts=self.release_attempts,
                    release_timeout=self.release_timeout)
                log = tempfile.TemporaryFile(mode="w+b")
                started = time.time()
                clock_started = time.monotonic()
                process = subprocess.Popen(
                    argv, cwd=self.launcher.cwd(job.workdir), env=self._environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    **supervisor._group_options())
                process._uq_output = log
                backstop = (job.max_seconds + self.kill_grace + self.drain_timeout
                            + self._release_seconds() + 3.0)
                try:
                    output = self._wait(process, backstop)
                except subprocess.TimeoutExpired:
                    backstop_fired = True
            except BaseException as exc:
                error = exc
                interrupted = isinstance(exc, KeyboardInterrupt)
            finally:
                try:
                    if process is not None and (backstop_fired or error is not None):
                        self._stop(process, self.drain_timeout)
                    if log is not None and not output:
                        output = supervisor._read_output(log)
                    result = self._result(output)
                except BaseException as exc:
                    if error is None:
                        error = exc
                    interrupted |= isinstance(exc, KeyboardInterrupt)
                finally:
                    seconds = max(0.0, time.monotonic() - clock_started)
                    released = result.get("released") is True
                    release_output = result.get("release_output", "")
                    try:
                        if not released:
                            released, extra = self._release()
                            release_output = "\n".join(
                                part for part in (release_output, extra) if part)
                    except BaseException as exc:
                        release_output += f"\n{type(exc).__name__}: {exc}"
                        if error is None:
                            error = exc
                        interrupted |= isinstance(exc, KeyboardInterrupt)
                    finally:
                        attached = max(0.0, time.monotonic() - clock_started)
                        receipt = Receipt(
                            label=job.label, started=started,
                            seconds=result.get("seconds", seconds), sku=job.sku,
                            rate=rate, cost=rate * attached / 3600, worst_case=worst,
                            exit_code=result.get("workload_exit"),
                            timed_out=backstop_fired or result.get("deadline_hit", False),
                            lease=result.get("lease"), gpu=result.get("gpu"),
                            released=released, release_output=release_output,
                            output_tail=result.get("output_tail", output[-4000:]),
                            error=f"{type(error).__name__}: {error}" if error else "",
                            attached_seconds=attached,
                            interrupted=interrupted or result.get("interrupted", False),
                            reservation=reservation)
                        try:
                            self.ledger.append(receipt)
                        finally:
                            if log is not None:
                                log.close()
            if not receipt.released:
                raise LeaseNotReleased(receipt) from error
            if error is not None:
                raise error.with_traceback(error.__traceback__)
            return receipt
