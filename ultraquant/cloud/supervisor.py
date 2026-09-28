"""Own one lupine job, its deadline, process tree and lease release.

This file also runs directly inside WSL. It needs only the standard library;
neither importing UltraQuant nor reading a pipe is part of its lifecycle.
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


def _windows_descendants(pid):
    """Hold descendant handles before taskkill can remove their parent.

    Restricted Windows tokens can deny taskkill's process enumeration while
    permitting termination of our own children. Toolhelp needs no WMI service.
    Creation times reject stale parent PIDs; handles prevent PID reuse between
    this snapshot and termination. No unrelated process is selected.
    """
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        "Process32FirstW": ([wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL),
        "Process32NextW": ([wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL),
        "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "GetProcessTimes": ([wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4,
                            wintypes.BOOL),
    }
    for name, (argtypes, restype) in signatures.items():
        getattr(kernel, name).argtypes = argtypes
        getattr(kernel, name).restype = restype

    def created(handle):
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return times[0].dwHighDateTime << 32 | times[0].dwLowDateTime

    root = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not root:
        return kernel, []
    try:
        root_time = created(root)
    finally:
        kernel.CloseHandle(root)
    if root_time is None:
        return kernel, []
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        return kernel, []
    children = {}
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            children.setdefault(entry.parent, []).append(entry.pid)
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    handles, pending, seen = [], [(pid, root_time)], {pid}
    try:
        while pending:
            parent, parent_time = pending.pop()
            for child in children.get(parent, ()):
                if child in seen:
                    continue
                seen.add(child)
                handle = kernel.OpenProcess(0x1001, False, child)  # QUERY | TERMINATE
                if not handle:
                    continue
                child_time = created(handle)
                if child_time is None or child_time < parent_time:
                    kernel.CloseHandle(handle)
                    continue
                handles.append(handle)
                pending.append((child, child_time))
        return kernel, handles
    except BaseException:
        for handle in handles:
            kernel.CloseHandle(handle)
        raise


def kill_tree(process, grace) -> None:
    """Best effort, bounded termination, including a surviving process group."""
    try:
        stop_at = min(time.monotonic() + max(0.0, grace),
                      getattr(process, "_uq_stop_deadline", math.inf))
        if os.name == "nt":
            killer = None
            kernel, handles = None, []
            with contextlib.suppress(BaseException):
                kernel, handles = _windows_descendants(process.pid)
            try:
                killer = subprocess.Popen(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                killer.wait(timeout=max(0.0, stop_at - time.monotonic()))
            except BaseException:
                if killer is not None:
                    with contextlib.suppress(BaseException):
                        killer.kill()
                    with contextlib.suppress(BaseException):
                        killer.wait(timeout=max(0.0, stop_at - time.monotonic()))
            finally:
                for handle in reversed(handles):
                    with contextlib.suppress(BaseException):
                        kernel.TerminateProcess(handle, 1)
                    kernel.CloseHandle(handle)
                if process.poll() is None:
                    process.kill()
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
                    process = subprocess.Popen(
                        [*lupine, "end"], env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, **_group_options())
                    process._uq_stop_deadline = end
                    remaining = max(0.0, end - time.monotonic())
                    cleanup = min(1.0, remaining / 2)
                    process.wait(timeout=min(timeout, remaining - cleanup))
                except BaseException:
                    if process is not None:
                        kill_tree(process, 1.0)
                        with contextlib.suppress(BaseException):
                            process.wait(timeout=min(1.0, max(0.0, end - time.monotonic())))
                    raise
                finally:
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


def main(argv) -> int:
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
    process = None
    released = False
    release_output = ""
    failure = ""
    seconds = 0.0
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
        for signum in signals:
            saved[signum] = signal.signal(signum, on_signal)
        with tempfile.TemporaryFile(mode="w+b") as log:
            try:
                started = time.monotonic()
                process = subprocess.Popen(
                    [*args.lupine, *args.run_args[1:]], env=env,
                    stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, **_group_options())
                waiting = True
                if interrupted:
                    raise _Interrupted()
                process.wait(timeout=args.deadline)
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
                    if process is not None and (
                            deadline_hit or interrupted or failure):
                        kill_tree(process, args.kill_grace)
                        with contextlib.suppress(subprocess.TimeoutExpired):
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
            output = _read_output(log, 4000)
    finally:
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
