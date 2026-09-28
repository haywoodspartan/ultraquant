"""A stand-in lupine CLI, so the on-demand exam never spends money.

The on-demand exam (§11.127, amended in §11.128) drives the runner
through this script instead of the real ``lupine`` binary. It prints
what lupine v0.3.1 prints - the lease line, the gateway line, ``ended
lease`` / ``no cached lease``, the ``gpus`` table and the ``usage``
summary, copied from the real CLI on 2026-09-28 - and it keeps what the
exam believes: its own lease state, its own clock, and a log of every
call. The runner's word that it released a lease is not evidence; this
file's state is.

**Version 2, after adversarial review.** The first version interpreted
``sleep`` inside its own process and stopped its clock when a workload
exited. GPT-6 Astra's review showed what that hid: a killed parent
whose child still holds the output pipe, descendants that outlive a
kill, and billing that runs until ``end`` rather than until the
workload stops. So now:

- ``sleep S`` runs in a real child process, and ``sleep-tree S`` in a
  child that starts a grandchild and waits for it. Every process logs
  its PID, so the exam can check that none survive.
- The attachment clock runs from lease acquisition to ``end``, and
  ``usage`` charges from that clock, rounded to cents as the real CLI
  shows it.

Faults are set in ``state.json`` by the exam:
    garbage          ``run`` prints no lupine lines at all
    provision_delay  seconds ``run`` waits before taking a lease
    end_failures     the next N ``end`` calls fail with exit 1
    end_delay        seconds each ``end`` waits before answering
    end_hang         ``end`` never answers
    usage_cost       extra dollars ``usage`` reports (default 0.0)
    usage_seconds    extra GPU seconds ``usage`` reports (default 0)
    usage_broken     ``usage`` prints nothing parseable
    gpus_text        replaces the price table ``gpus`` prints

State lives in ``$LUPINE_STUB_DIR``; the stub refuses to run without it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
A100_NAME = "NVIDIA A100-SXM4-80GB"
RATES = {"a100": 1.10, "rtx-pro-6000": 1.50}
GPUS_TABLE = (
    "SKU              GPU                                            VRAM     $/HR\n"
    "a100             NVIDIA A100-SXM4-80GB                          80GB    $1.10\n"
    "rtx-pro-6000     NVIDIA RTX PRO 6000 Blackwell Server Edition   96GB    $1.50\n"
    "\n"
    "Run on a type:  lupine run --gpu <SKU> <command>\n")


def _dir() -> Path:
    where = os.environ.get("LUPINE_STUB_DIR")
    if not where:
        sys.stderr.write("lupine-stub: LUPINE_STUB_DIR is not set\n")
        raise SystemExit(2)
    return Path(where)


def _state() -> dict:
    for _ in range(50):
        try:
            return json.loads((_dir() / "state.json").read_text(
                encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):     # mid-replace on Windows: retry
            time.sleep(0.02)
    return {}


def _save(state: dict) -> None:
    path = _dir() / "state.json"
    temp = path.with_name(f"state.{os.getpid()}.tmp")
    temp.write_text(json.dumps(state), encoding="utf-8")
    for _ in range(100):             # another stub may hold the file on Windows
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(temp, path)


def log_event(event: dict, directory: Path | None = None) -> None:
    """Append one event; also used by the stub's own child processes."""
    event = {"t": time.time(), "pid": os.getpid(),
             "disable_local": os.environ.get("LUPINE_DISABLE_LOCAL"),
             "wslenv": os.environ.get("WSLENV"), **event}
    with open((directory or _dir()) / "log.jsonl", "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


_SLEEPER = "import time, sys; time.sleep(float(sys.argv[1]))"
_TREE = (
    "import json, os, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', "
    "'import time, sys; time.sleep(float(sys.argv[1]))', sys.argv[1]])\n"
    "with open(os.path.join(sys.argv[2], 'log.jsonl'), 'a') as handle:\n"
    "    handle.write(json.dumps({'t': time.time(), 'cmd': 'grandchild',\n"
    "                             'pid': grandchild.pid}) + '\\n')\n"
    "raise SystemExit(grandchild.wait())\n")


def _workload(argv: list) -> int:
    if not argv:
        return 0
    head = argv[0]
    if head in ("sleep", "sleep-tree"):
        code = _SLEEPER if head == "sleep" else _TREE
        extra = [] if head == "sleep" else [str(_dir())]
        child = subprocess.Popen([sys.executable, "-c", code, argv[1],
                                  *extra])
        log_event({"cmd": "child", "pid": child.pid})
        return child.wait()
    if head == "exit":
        return int(argv[1])
    if head == "echo":
        _say(" ".join(argv[1:]))
        return 0
    if head == "nvidia-smi":
        _say(f"GPU 0: {GPU_NAME} (via lupine gw-stub) (UUID: GPU-stub)")
        return 0
    _say(f"stub: ran {argv}")
    return 0


def _run(args: list) -> int:
    if "--" in args:
        cut = args.index("--")
        flags, workload = args[:cut], args[cut + 1:]
    else:
        flags, workload = args, []
    sku = flags[flags.index("--gpu") + 1] if "--gpu" in flags else None
    state = _state()
    if state.get("provision_delay"):
        log_event({"cmd": "provisioning"})
        time.sleep(float(state["provision_delay"]))
        state = _state()
    lease = state.get("cached_lease") or str(uuid.uuid4())
    if not state.get("cached_lease"):
        state["attach_start"] = time.time()
        state["lease_rate"] = RATES.get(sku, RATES["rtx-pro-6000"])
    state["cached_lease"] = lease
    _save(state)
    log_event({"cmd": "run", "phase": "start", "flags": flags,
               "workload": workload, "sku": sku, "lease": lease,
               "path": os.environ.get("PATH", "")})
    if not state.get("garbage"):
        name = A100_NAME if sku == "a100" else GPU_NAME
        _say("lupine: closest region: us-east-1")
        _say(f"lupine: lease {lease} on lupine-stub-0 ({name})")
        _say("lupine: HTTPS via https://gw-stub.lupine.sh:9443")
        _say(f"lupine: running: [{' '.join(workload)}]")
    code = _workload(workload)
    log_event({"cmd": "run", "phase": "stop", "lease": lease, "exit": code})
    return code


def _end() -> int:
    state = _state()
    log_event({"cmd": "end", "phase": "start"})
    if state.get("end_hang"):
        time.sleep(3600)
    if state.get("end_delay"):
        time.sleep(float(state["end_delay"]))
        state = _state()
    if state.get("end_failures", 0) > 0:
        state["end_failures"] -= 1
        _save(state)
        log_event({"cmd": "end", "outcome": "failed"})
        _say("lupine: error: dial tcp: i/o timeout")
        return 1
    lease = state.get("cached_lease")
    now = time.time()
    if lease:
        start = state.get("attach_start", now)
        state["attached_total"] = state.get("attached_total", 0.0) + now - start
        state["attached_cost"] = (state.get("attached_cost", 0.0)
                                  + (now - start) * state.get(
                                      "lease_rate", 1.50) / 3600)
        log_event({"cmd": "end", "outcome": "ended", "lease": lease,
                   "attach_start": start, "attach_end": now})
    else:
        log_event({"cmd": "end", "outcome": "none"})
    state["cached_lease"] = None
    state.pop("attach_start", None)
    _save(state)
    _say(f"lupine: ended lease {lease}" if lease else "lupine: no cached lease")
    return 0


def _usage() -> int:
    state = _state()
    log_event({"cmd": "usage"})
    if state.get("usage_broken"):
        _say("lupine: usage is temporarily unavailable")
        return 0
    seconds = int(round(state.get("attached_total", 0.0)
                        + state.get("usage_seconds", 0)))
    cost = state.get("attached_cost", 0.0) + state.get("usage_cost", 0.0)
    _say("Billing period:  Sep 2026 (month to date)")
    _say("Payment method:  on file")
    _say(f"GPU time:        {seconds}s")
    _say("Sessions:        1")
    _say(f"Estimated cost:  ${cost:.2f}")
    _say("Free credit:     $20.00 remaining")
    return 0


def main(argv: list) -> int:
    if not argv:
        _say("usage: lupine <command>")
        return 2
    command, rest = argv[0], argv[1:]
    if command == "run":
        return _run(rest)
    if command == "end":
        return _end()
    if command == "gpus":
        log_event({"cmd": "gpus"})
        sys.stdout.write(_state().get("gpus_text") or GPUS_TABLE)
        return 0
    if command == "usage":
        return _usage()
    log_event({"cmd": command})
    _say(f"lupine-stub: {command} is not simulated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
