"""Paid GPUs must be on demand: the user pays out of pocket.

The user's requirement is "The Usage needs to be on-Demand". lupine bills
each attached GPU per second. Its run command retains a cached lease after
the workload exits; only end releases it. This is the project's entry point
for paid GPUs: refuse jobs that cannot fit the monthly cap, attach for one
job, release on every exit path, and write a durable receipt.

Windows launches lupine through WSL, normally at $HOME/.local/bin/lupine in
Ubuntu. LUPINE_DISABLE_LOCAL=1 hides the user's own GPU, and WSLENV forwards
that setting into WSL so the workload cannot silently use the local GPU
while the rental remains attached.

The lock serialises this process only. Another process's lupine run shares
the cached lease and can interfere with release. Whether killing a --remote
client stops the remote pod is unverified. Each job uses one GPU; multi-GPU
jobs are not supported yet. Receipts price the run process's wall time;
billing can also include time until end finishes, so the budget reserves
grace time and checks the provider's usage as well as the local ledger.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class OverBudget(RuntimeError):
    """A job cannot safely be admitted within the monthly budget."""


class LeaseNotReleased(RuntimeError):
    """Release could not be confirmed; receipt has already been recorded."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(f"Lease release unconfirmed for {receipt.label!r}: "
                         f"{receipt.release_output}")


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
    """Read SKU and hourly price without depending on column widths."""
    prices = {}
    for line in text.replace("\x00", "").splitlines():
        tokens = line.split()
        if (not tokens or tokens[0].upper() == "SKU"
                or line.lstrip().startswith("Run on a type:")):
            continue
        if len(tokens) < 2 or not tokens[-1].startswith("$"):
            continue
        try:
            price = float(tokens[-1][1:])
        except ValueError:
            continue
        if math.isfinite(price) and price >= 0:
            prices[tokens[0]] = price
    if not prices:
        raise ValueError("No GPU prices found")
    return Rates(prices)


def parse_usage(text) -> dict | None:
    """Require both a whole-second duration and a finite dollar amount."""
    seconds = cost = None
    for line in text.replace("\x00", "").splitlines():
        duration = re.fullmatch(r"\s*GPU time:\s*(.*?)\s*", line)
        if duration:
            match = re.fullmatch(
                r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?",
                duration[1])
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

    def __post_init__(self):
        if not self.command:
            raise ValueError("A job needs a command")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if not self.label:
            raise ValueError("A job needs a label")


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


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def default(cls):
        from ultraquant.config import _user_config_dir

        override = os.environ.get("ULTRAQUANT_GPU_LEDGER")
        return cls(override if override else _user_config_dir() / "gpu_ledger.jsonl")

    def append(self, receipt):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(receipt)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def receipts(self) -> list[dict]:
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

    def spent_in_month(self, when=None) -> float:
        current = datetime.fromtimestamp(time.time() if when is None else when,
                                         timezone.utc)
        total = 0.0
        for row in self.receipts():
            try:
                stamp = datetime.fromtimestamp(row["started"], timezone.utc)
                cost = float(row["cost"])
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                continue
            if ((stamp.year, stamp.month) == (current.year, current.month)
                    and math.isfinite(cost)):
                total += cost
        return total


@dataclass(frozen=True)
class Launcher:
    prefix: tuple
    wsl_distro: str | None = None

    def argv(self, args, *, workdir=None, limit=None) -> list[str]:
        if self.wsl_distro is None:
            return [*self.prefix, *args]
        result = ["wsl.exe", "-d", self.wsl_distro]
        if workdir:
            result.extend(["--cd", workdir])
        result.append("--")
        if limit:
            result.extend(["timeout", "--kill-after=10", f"{limit:g}"])
        return [*result, *self.prefix, *args]

    def cwd(self, workdir):
        return workdir if self.wsl_distro is None else None

    @classmethod
    def for_this_machine(cls):
        override = os.environ.get("ULTRAQUANT_LUPINE")
        if os.name == "nt":
            distro = os.environ.get("ULTRAQUANT_LUPINE_DISTRO", "Ubuntu")
            if not override:
                result = subprocess.run(
                    ["wsl.exe", "-d", distro, "--", "sh", "-c", 'printf %s "$HOME"'],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", check=True, timeout=15)
                home = result.stdout.replace("\x00", "").strip()
                if not home:
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
    # TimeoutExpired can carry bytes even when Popen uses text mode.
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return (output or "").replace("\x00", "")


class OnDemand:
    GRACE = 15.0
    RELEASE_ATTEMPTS = 3

    def __init__(self, launcher, ledger, monthly_cap, rates=None):
        if not math.isfinite(monthly_cap) or monthly_cap < 0:
            raise ValueError("monthly_cap must be finite and nonnegative")
        self.launcher = launcher
        self.ledger = ledger
        self.monthly_cap = monthly_cap
        self._rates = rates
        self._lock = threading.Lock()

    def _environment(self) -> dict:
        env = os.environ.copy()
        env["LUPINE_DISABLE_LOCAL"] = "1"
        entries = [entry for entry in env.get("WSLENV", "").split(":")
                   if entry and entry.split("/", 1)[0] != "LUPINE_DISABLE_LOCAL"]
        env["WSLENV"] = ":".join([*entries, "LUPINE_DISABLE_LOCAL/u"])
        return env

    def _launch(self, args, *, workdir=None, limit=None):
        """The single spawn point for all lupine commands."""
        return subprocess.Popen(
            self.launcher.argv(args, workdir=workdir, limit=limit),
            cwd=self.launcher.cwd(workdir), env=self._environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

    def _query(self, args, timeout=10.0):
        process = self._launch(args)
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.returncode, _clean_output(output)
        finally:
            try:
                if process.poll() is None:
                    process.kill()
            finally:
                process.communicate()

    def rates(self):
        if self._rates is None:
            try:
                code, output = self._query(["gpus"])
                if code != 0:
                    raise ValueError(output)
                self._rates = parse_gpus(output)
            except Exception as exc:
                raise OverBudget("GPU price list is unreadable") from exc
        return self._rates

    def estimate(self, job):
        return self.rates().for_sku(job.sku) * (job.max_seconds + self.GRACE) / 3600

    def month_spent(self):
        try:
            code, output = self._query(["usage"])
            usage = parse_usage(output)
            if code != 0 or usage is None:
                raise ValueError(output)
            return max(self.ledger.spent_in_month(), usage["cost"])
        except Exception as exc:
            raise OverBudget("Monthly GPU usage is unreadable") from exc

    def _release(self) -> tuple[bool, str]:
        outputs = []
        for attempt in range(1, self.RELEASE_ATTEMPTS + 1):
            try:
                code, output = self._query(["end"], timeout=4.0)
                outputs.append(output)
                if code == 0 and ("ended lease" in output or "no cached lease" in output):
                    return True, "\n".join(outputs)
            except Exception as exc:
                partial = _clean_output(getattr(exc, "output", None))
                outputs.append(f"{partial}\n{type(exc).__name__}: {exc}")
            if attempt < self.RELEASE_ATTEMPTS:
                time.sleep(0.5 * attempt)
        return False, "\n".join(outputs)

    def run(self, job) -> Receipt:
        with self._lock:
            worst = self.estimate(job)
            spent = self.month_spent()
            if spent + worst > self.monthly_cap:
                raise OverBudget(f"Spent ${spent:.6f} + job ${worst:.6f} "
                                 f"> monthly cap ${self.monthly_cap:.6f}")
            # Resolve pricing before launch so receipt construction needs no CLI.
            rate = self.rates().for_sku(job.sku)
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

            process = None
            output = ""
            timed_out = False
            error = None
            exit_code = None
            started = time.time()
            clock_started = time.monotonic()
            try:
                process = self._launch(flags, workdir=job.workdir, limit=job.max_seconds)
                timeout = job.max_seconds + (12 if self.launcher.wsl_distro is not None else 0)
                try:
                    output, _ = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    output = _clean_output(exc.output)
            except BaseException as exc:
                error = exc
            finally:
                # Cleanup failures must not bypass release or the ledger. A
                # KeyboardInterrupt in communicate follows this same path.
                try:
                    if process is not None:
                        try:
                            # Alive here only after an interruption or a
                            # timeout; timeouts were labelled where raised,
                            # so an interrupted job is not called timed out.
                            if process.poll() is None:
                                process.kill()
                        finally:
                            remaining, _ = process.communicate()
                            output = _clean_output(remaining)
                            exit_code = process.returncode
                except BaseException as exc:
                    if error is None:
                        error = exc
                finally:
                    seconds = max(0.0, time.monotonic() - clock_started)
                    if self.launcher.wsl_distro is not None and exit_code in (124, 137):
                        timed_out = True
                    released = False
                    release_output = ""
                    try:
                        released, release_output = self._release()
                    except BaseException as exc:
                        release_output = f"{type(exc).__name__}: {exc}"
                        if error is None:
                            error = exc
                    finally:
                        output = _clean_output(output)
                        match = re.search(r"^lupine: lease (\S+) on \S+ \(([^\r\n]+)\)\s*$",
                                          output, re.MULTILINE)
                        receipt = Receipt(
                            label=job.label, started=started, seconds=seconds,
                            sku=job.sku, rate=rate, cost=rate * seconds / 3600,
                            worst_case=worst, exit_code=exit_code, timed_out=timed_out,
                            lease=match[1] if match else None,
                            gpu=match[2] if match else None, released=released,
                            release_output=release_output, output_tail=output[-4000:],
                            error=(f"{type(error).__name__}: {error}"
                                   if error is not None else ""))
                        self.ledger.append(receipt)
            if not receipt.released:
                raise LeaseNotReleased(receipt) from error
            if error is not None:
                raise error.with_traceback(error.__traceback__)
            return receipt
