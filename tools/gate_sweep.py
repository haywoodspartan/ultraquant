"""Run every gate module in a subprocess and record its verdict from its own text.

Usage:
    python tools/gate_sweep.py OUT.jsonl --repo DIR [--workers N]
                               [--only a,b] [--skip a,b]

§11.151. Sweep a checkout (preferably a clean worktree, with a copy of
uq_home for the gates that read the live library), then diff the result
against ``ultraquant/experiments/records/gate_ledger.json``. That diff is what
"every earlier gate keeps its HEAD verdict" means.

- **Verdicts come from the gate's own text.** A verdict is the first output
  line that STARTS with PASS, FAIL or VOID; failing that, the last line
  containing one of them as a word; failing that, the exit code.
- **Exit codes alone are not verdicts.** The compound and ladder gates print
  FAIL and exit 0.
- **Every gate's full output is kept** beside OUT.jsonl, in OUT.logs/.
- **Distillation gates replay** their recorded samples (--rescore).
- **Gates that need the model hardware are the caller's to skip.**
"""

import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPLAY = {"distill_facts_gate", "distill_hard_gate", "completion_gate", "ownquestions_gate",
          "shape_gate", "roundtrip_gate", "kind_gate", "growth_gate", "second_gate"}
TIMEOUT = 3600
# Hypothesis tests report their own outcome words instead: REJECTED / NOT
# REJECTED (distribution), SUPPORTED / NOT SUPPORTED (load), and planning's
# QUESTIONABLE when its control stops holding.
_OUTCOMES = "PASS|FAIL|VOID|NOT REJECTED|REJECTED|NOT SUPPORTED|SUPPORTED|QUESTIONABLE"
_LEAD = re.compile(rf"^\s*({_OUTCOMES})\b")
_WORD = re.compile(rf"\b({_OUTCOMES})\b")


def verdict(text: str, code) -> tuple[str, str]:
    lines = text.splitlines()
    for line in lines:
        match = _LEAD.match(line)
        if match:
            return match.group(1), line.strip()[:400]
    for line in reversed(lines):
        match = _WORD.search(line)
        if match:
            return match.group(1), line.strip()[:400]
    if code is None:
        return "TIMEOUT", ""
    return ("EXIT0" if code == 0 else f"EXIT{code}"), (lines[0][:400] if lines else "")


def run(repo: Path, logs: Path, name: str) -> dict:
    args = [sys.executable, "-m", f"ultraquant.experiments.{name}"]
    if name in REPLAY:
        args.append("--rescore")
    started = time.monotonic()
    try:
        done = subprocess.run(args, cwd=repo, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=TIMEOUT)
        text, code = done.stdout + "\n" + (done.stderr or ""), done.returncode
    except subprocess.TimeoutExpired as exc:
        text, code = str(exc.stdout or "") + "\nTIMEOUT", None
    (logs / f"{name}.log").write_text(text, encoding="utf-8")
    word, line = verdict(text, code)
    return {"gate": name, "verdict": word, "line": line, "code": code,
            "seconds": round(time.monotonic() - started, 1)}


def main() -> None:
    out = Path(sys.argv[1])
    argv = sys.argv[2:]
    repo = Path(argv[argv.index("--repo") + 1])
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 3
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    skip = set(argv[argv.index("--skip") + 1].split(",")) if "--skip" in argv else set()
    logs = out.with_suffix(".logs")
    logs.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (repo / "ultraquant" / "experiments").glob("*gate*.py"))
    names = [n for n in names if (only is None or n in only) and n not in skip]
    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, repo, logs, name) for name in names]
        for future in as_completed(futures):
            fh.write(json.dumps(future.result()) + "\n")
            fh.flush()
    print("done", len(names))


if __name__ == "__main__":
    main()
