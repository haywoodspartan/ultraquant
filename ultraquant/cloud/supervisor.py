"""Own one lupine job, its deadline, process tree and lease release.

Round three follows the second review of 11.128. An ordinary exit left
background children alive; the Windows descendant snapshot missed orphans
and could select a reused PID. A Job Object now owns Windows processes
before they start, and every exit cleans up that job or the POSIX process
group before release. Linux parent-death signals also cover a lost WSL relay.

The environment flag is configuration, not isolation. The sealed namespace
IS enforcement against Windows interop inside WSL: it covers /run/WSL and
drops capabilities before starting lupine. A failed probe refuses the job
and still attempts release. --remote remains the isolation from the local
GPU; the seal does not hide Linux GPU devices.

This file runs directly inside WSL using only the standard library. Output
goes to regular files, so inherited pipes cannot defeat bounded cleanup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

PREFIX = "uq-supervisor: "


def child_environment(base=None) -> dict:
    env = dict(os.environ if base is None else base)
    env["LUPINE_DISABLE_LOCAL"] = "1"
    env["PATH"] = os.pathsep.join(
        entry for entry in env.get("PATH", "").split(os.pathsep)
        if not entry.startswith("/mnt/"))
    return env


def _group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _read_output(handle, tail=None) -> str:
    # Read a snapshot of a regular file. An inherited descriptor cannot keep
    # this read waiting for EOF, even when a descendant is still alive.
    size = os.fstat(handle.fileno()).st_size
    start = max(0, size - tail * 4) if tail is not None else 0
    handle.seek(start)
    text = handle.read(size - start).decode("utf-8", errors="replace")
    text = text.replace("\x00", "")
    return text[-tail:] if tail is not None else text


def _under_wsl() -> bool:
    if os.name != "posix":
        return False
    if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def sealed(argv) -> list:
    """Prevent even shell-wrapped or copied Windows binaries using interop."""
    if not _under_wsl():
        return list(argv)
    return ["unshare", "-rm", "sh", "-c",
            'mount -t tmpfs none /run/WSL && exec setpriv '
            '--bounding-set=-all --inh-caps=-all --ambient-caps=-all '
            '--no-new-privs -- "$@"', "sh", *argv]


def _windows_job_api():
    """Declare the handle APIs and native layouts, including pointer widths."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int,
                                     ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (argtypes, restype) in signatures.items():
        getattr(kernel, name).argtypes = argtypes
        getattr(kernel, name).restype = restype
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = wintypes.LONG
    return kernel, ntdll, ExtendedLimits


def launch(argv, env, log, *, cwd=None) -> subprocess.Popen:
    """Create a session, or assign a suspended Windows process to its job."""
    options = dict(env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                   stdout=log, stderr=subprocess.STDOUT)
    if os.name != "nt":
        return subprocess.Popen(argv, start_new_session=True, **options)
    import ctypes

    kernel, ntdll, ExtendedLimits = _windows_job_api()
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process = None
    try:
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits),
                                               ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = subprocess.Popen(
            argv, creationflags=0x00000004 | subprocess.CREATE_NEW_PROCESS_GROUP,
            **options)                       # CREATE_SUSPENDED
        if not kernel.AssignProcessToJobObject(job, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())
        process._uq_job = job
        status = ntdll.NtResumeProcess(process._handle)
        if status < 0:
            raise OSError(f"NtResumeProcess failed: NTSTATUS {status & 0xffffffff:#x}")
        return process
    except BaseException:
        if process is not None:
            with contextlib.suppress(BaseException):
                process.kill()
            with contextlib.suppress(BaseException):
                process.wait(timeout=1.0)
            process._uq_job = None
        kernel.CloseHandle(job)
        raise


def close_job(process) -> None:
    """Drop our non-inheritable job handle on every path."""
    with contextlib.suppress(BaseException):
        job = getattr(process, "_uq_job", None)
        if job is not None:
            kernel, _, _ = _windows_job_api()
            if kernel.CloseHandle(job):
                process._uq_job = None


def kill_tree(process, grace) -> None:
    """Best effort, bounded termination, including a surviving process group."""
    try:
        stop_at = min(time.monotonic() + max(0.0, grace),
                      getattr(process, "_uq_stop_deadline", math.inf))
        if os.name == "nt":
            job = getattr(process, "_uq_job", None)
            if job is not None:
                kernel, _, _ = _windows_job_api()
                kernel.TerminateJobObject(job, 1)
            else:
                process.kill()
            with contextlib.suppress(BaseException):
                process.wait(timeout=max(0.0, stop_at - time.monotonic()))
        else:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(BaseException):
                process.wait(timeout=max(0.0, stop_at - time.monotonic()))
            # The leader's exit says nothing about its descendants. Always
            # send KILL to the group, even if wait() already reaped the leader.
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
    except BaseException:
        pass


def release(lupine, env, attempts, timeout) -> tuple[bool, str]:
    outputs = []
    # Shutdown consumes this same retry budget, rather than adding an extra
    # kill and drain allowance to every timed-out attempt.
    end = time.monotonic() + attempts * timeout + 0.25 * attempts * (attempts - 1)
    for attempt in range(1, attempts + 1):
        if time.monotonic() >= end:
            break
        output = ""
        try:
            with tempfile.TemporaryFile(mode="w+b") as log:
                process = None
                try:
                    process = launch([*lupine, "end"], env, log)
                    process._uq_stop_deadline = end
                    remaining = max(0.0, end - time.monotonic())
                    cleanup = min(1.0, remaining / 2)
                    process.wait(timeout=min(timeout, remaining - cleanup))
                finally:
                    if process is not None:
                        kill_tree(process, 1.0)
                        with contextlib.suppress(BaseException):
                            process.wait(timeout=min(1.0, max(0.0, end - time.monotonic())))
                        close_job(process)
                    output = _read_output(log)
                outputs.append(output)
                if process.returncode == 0 and (
                        "ended lease" in output or "no cached lease" in output):
                    return True, "\n".join(outputs)
        except Exception as exc:
            outputs.append(f"{output}\n{type(exc).__name__}: {exc}")
        if attempt < attempts:
            if time.monotonic() + 0.5 * attempt >= end:
                break
            time.sleep(0.5 * attempt)
    return False, "\n".join(outputs)


def parse_lease(text) -> tuple[str | None, str | None]:
    match = re.search(r"^lupine: lease (\S+) on \S+ \(([^\r\n]+)\)\s*$",
                      text, re.MULTILINE)
    return (match[1], match[2]) if match else (None, None)


class _Interrupted(BaseException):
    pass


def _parent_death_signal(parent):
    """Have Linux notify us when the relay dies, including the setup race."""
    if not sys.platform.startswith("linux"):
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return os.getppid() != parent or parent == 1


def main(argv) -> int:
    parent = os.getppid()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--kill-grace", type=float, default=10.0)
    parser.add_argument("--release-attempts", type=int, default=3)
    parser.add_argument("--release-timeout", type=float, default=5.0)
    parser.add_argument("--lupine", action="append", required=True)
    parser.add_argument("run_args", nargs=argparse.REMAINDER)
    # Prefix arguments may themselves be options (for example python -B).
    # Preserve the public, repeated --lupine ARG spelling while preventing
    # argparse from treating those values as supervisor options.
    normalized = []
    arguments = iter(argv)
    for argument in arguments:
        if argument == "--":
            normalized.extend([argument, *arguments])
            break
        if argument == "--lupine":
            try:
                argument = "--lupine=" + next(arguments)
            except StopIteration:
                parser.error("--lupine requires an argument")
        normalized.append(argument)
    args = parser.parse_args(normalized)
    for name in ("deadline", "kill_grace", "release_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.release_attempts < 1:
        parser.error("release_attempts must be positive")
    if not args.run_args or args.run_args[0] != "--":
        parser.error("separate the lupine run arguments with --")

    env = child_environment()
    interrupted = False
    waiting = False
    deadline_hit = False
    process = log = None
    released = False
    release_output = ""
    failure = ""
    output = ""
    seconds = 0.0
    attached_seconds = 0.0
    signals = [signal.SIGINT]
    signals += ([signal.SIGBREAK] if os.name == "nt"
                else [signal.SIGTERM, signal.SIGHUP])
    saved = {}

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        if waiting:
            raise _Interrupted()

    def ignore_signals():
        for signum in saved:
            signal.signal(signum, signal.SIG_IGN)

    started = time.monotonic()
    try:
        try:
            for signum in signals:
                saved[signum] = signal.signal(signum, on_signal)
            interrupted |= _parent_death_signal(parent)
            log = tempfile.TemporaryFile(mode="w+b")
            if _under_wsl():
                # Probe outside the workload namespace. Failure must never
                # fall back to an unsealed launch; release is still harmless.
                try:
                    probe = subprocess.run(
                        ["unshare", "-rm", "sh", "-c",
                         "mount -t tmpfs none /run/WSL"], env=env,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, timeout=2.0)
                    if probe.returncode:
                        raise RuntimeError(f"probe exited {probe.returncode}")
                except Exception as exc:
                    raise RuntimeError(f"WSL sealing failed: {exc}") from exc
            if interrupted:
                raise _Interrupted()
            argv = sealed([*args.lupine, *args.run_args[1:]])
            process = launch(argv, env, log)
            waiting = True
            if interrupted:
                raise _Interrupted()
            process.wait(timeout=max(0.0, args.deadline - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            deadline_hit = True
        except _Interrupted:
            interrupted = True
        except BaseException as exc:
            interrupted |= isinstance(exc, KeyboardInterrupt)
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            waiting = False
            ignore_signals()
            try:
                if process is not None:
                    kill_tree(process, args.kill_grace)
                    with contextlib.suppress(BaseException):
                        process.wait(timeout=1.0)
            finally:
                seconds = max(0.0, time.monotonic() - started)
                try:
                    released, release_output = release(
                        args.lupine, env, args.release_attempts,
                        args.release_timeout)
                except BaseException as exc:
                    release_output = f"{type(exc).__name__}: {exc}"
                attached_seconds = max(0.0, time.monotonic() - started)
        if log is not None:
            output = _read_output(log, 4000)
    finally:
        close_job(process)
        if log is not None:
            log.close()
        for signum, handler in saved.items():
            signal.signal(signum, handler)

    lease, gpu = parse_lease(output)
    result = {
        "workload_exit": process.returncode if process is not None else None,
        "deadline_hit": deadline_hit, "interrupted": interrupted,
        "released": released, "release_output": release_output,
        "lease": lease, "gpu": gpu, "seconds": seconds,
        "attached_seconds": attached_seconds,
        "output_tail": (output + ("\n" + failure if failure else ""))[-4000:],
    }
    print(PREFIX + json.dumps(result), flush=True)
    return 0 if released else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
