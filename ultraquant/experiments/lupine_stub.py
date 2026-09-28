"""A stand-in lupine CLI, so the on-demand exam never spends money.

§11.127's exam drives the on-demand runner through this script instead
of the real ``lupine`` binary. It prints what lupine v0.3.1 prints -
the lease line, the gateway line, ``ended lease`` / ``no cached lease``,
the ``gpus`` table and the ``usage`` summary, copied from the real CLI
on 2026-09-28 - and it keeps what the exam believes: its own lease
state and a log of every call. The runner's word that it released a
lease is not evidence; this file's state is.

Workloads are interpreted, never executed, so killing the stub kills
everything: ``sleep S``, ``exit N``, ``echo ...`` and ``nvidia-smi -L``.

Faults are set in ``state.json`` by the exam:
    garbage       ``run`` prints no lupine lines at all
    end_failures  the next N ``end`` calls fail with exit 1
    usage_cost    the month's cost ``usage`` reports (default 0.0)
    usage_broken  ``usage`` prints nothing parseable

State lives in ``$LUPINE_STUB_DIR``; the stub refuses to run without it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
A100_NAME = "NVIDIA A100-SXM4-80GB"
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
    try:
        return json.loads((_dir() / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    path = _dir() / "state.json"
    temp = path.with_name(f"state.{os.getpid()}.tmp")
    temp.write_text(json.dumps(state), encoding="utf-8")
    for _ in range(50):              # another stub may hold the file on Windows
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(temp, path)


def _log(event: dict) -> None:
    event = {"t": time.time(), "pid": os.getpid(),
             "disable_local": os.environ.get("LUPINE_DISABLE_LOCAL"),
             "wslenv": os.environ.get("WSLENV"), **event}
    with open(_dir() / "log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _workload(argv: list) -> int:
    if not argv:
        return 0
    head = argv[0]
    if head == "sleep":
        time.sleep(float(argv[1]))
        return 0
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
    lease = state.get("cached_lease") or str(uuid.uuid4())
    state["cached_lease"] = lease
    _save(state)
    _log({"cmd": "run", "phase": "start", "flags": flags,
          "workload": workload, "sku": sku, "lease": lease})
    if not state.get("garbage"):
        name = A100_NAME if sku == "a100" else GPU_NAME
        _say("lupine: closest region: us-east-1")
        _say(f"lupine: lease {lease} on lupine-stub-0 ({name})")
        _say("lupine: HTTPS via https://gw-stub.lupine.sh:9443")
        _say(f"lupine: running: [{' '.join(workload)}]")
    started = time.time()
    code = _workload(workload)
    state = _state()
    state["attached_seconds"] = (state.get("attached_seconds", 0.0)
                                 + time.time() - started)
    _save(state)
    _log({"cmd": "run", "phase": "stop", "lease": lease, "exit": code})
    return code


def _end() -> int:
    state = _state()
    if state.get("end_failures", 0) > 0:
        state["end_failures"] -= 1
        _save(state)
        _log({"cmd": "end", "outcome": "failed"})
        _say("lupine: error: dial tcp: i/o timeout")
        return 1
    lease = state.get("cached_lease")
    state["cached_lease"] = None
    _save(state)
    _log({"cmd": "end", "outcome": "ended" if lease else "none",
          "lease": lease})
    _say(f"lupine: ended lease {lease}" if lease else "lupine: no cached lease")
    return 0


def _usage() -> int:
    state = _state()
    _log({"cmd": "usage"})
    if state.get("usage_broken"):
        _say("lupine: usage is temporarily unavailable")
        return 0
    seconds = int(round(state.get("attached_seconds", 0.0)))
    _say("Billing period:  Sep 2026 (month to date)")
    _say("Payment method:  on file")
    _say(f"GPU time:        {seconds}s")
    _say("Sessions:        1")
    _say(f"Estimated cost:  ${state.get('usage_cost', 0.0):.2f}")
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
        _log({"cmd": "gpus"})
        sys.stdout.write(GPUS_TABLE)
        return 0
    if command == "usage":
        return _usage()
    _log({"cmd": command})
    _say(f"lupine-stub: {command} is not simulated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
