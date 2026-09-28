"""Command-R's next token, computed by our own engine. The forward gate.

The fourth unit of running the user's own Command-R 08-2024, and the
first in which it runs: embedding, 40 blocks, the tied output head, and
logits - in UltraQuant, with every weight product through §11.125's
native kernel. GPT-6 Astra wrote the engine; Claude wrote this exam.

**How close counts as correct.** Not bitwise: llama.cpp quantizes the
activation vector to 8 bits before every dot product, and this engine
does not. So the yardstick is llama.cpp's disagreement with itself.
Its CPU and CUDA backends, on the same GGUF and the same token ids,
agree on the top token every time and differ by 0.048-0.088 nats in
their top-5 log-probabilities - measured before any engine here
existed. Two correct implementations drift that far; ours is held to
the same.

**Tokenization is not under test.** Every engine - ours, llama.cpp CPU
(127.0.0.1:8089) and llama.cpp CUDA (127.0.0.1:8090) - is given the
same id list, produced by §11.124's tokenizer.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 557f14d3...) before the engine existed:

1. **Top-1 agreement** with llama.cpp CPU on every prompt whose CPU
   top-1 margin exceeds that prompt's CPU-vs-GPU spread; near-ties
   under the floor excluded and reported.
2. **Within the noise floor**: per prompt, max |ours - CPU| over CPU's
   top-5 <= max(2 x the prompt's CPU-vs-GPU spread, 0.1 nats).
3. **The exam can fail**: the engine with its RoPE pairing swapped must
   violate 1 or 2 on at least one prompt, or the harness is void.
4. **Deterministic**: the same prompt twice, bit-identical logits.
5. **The full suite is green.** Checked outside this module.

**PASSED on all four measured criteria. Command-R runs in UltraQuant.**

| prompt | ours vs CPU | CPU vs GPU | top-1 |
|---|---:|---:|---|
| The capital of France is | **0.035** | 0.056 | Paris |
| def fibonacci(n): | 0.081 | 0.088 | agrees |
| 2 + 2 = | 0.063 | 0.058 | agrees |
| Le chat est | **0.028** | 0.048 | agrees |
| The meaning of life is | **0.040** | 0.084 | agrees |
| The tower is 300 meters | 0.086 | 0.079 | agrees |
| Hello, my name is | 0.081 | 0.088 | agrees |
| import numpy as | 0.166 | 0.183 | agrees |
| Water boils at | 0.366 | **0.377** | agrees |
| In 1969, humans first | 0.050 | 0.071 | agrees |
| Das ist ein | 0.042 | 0.026 | near-tie, excluded - agrees anyway |
| The quick brown fox | 0.050 | 0.038 | agrees |

(Max |log-probability difference| in nats over llama.cpp CPU's top five.)
Every prompt is inside its bound; on seven of the twelve our engine
sits closer to llama.cpp's CPU backend than llama.cpp's own GPU backend
does. "Water boils at" is the widest - and there llama.cpp disagrees
with itself by 0.377. The planted defect, RoPE pairing halves instead
of adjacent coordinates, failed all 12 prompts: the harness can see a
wrong engine. The same prompt twice gives bit-identical logits.

**The speed, recorded as it came out**: 209 s for the twelve prompts,
about 17 s each - against 5.9 s on llama.cpp's CPU backend and 0.43 s on
its CUDA backend. Correctness first: SIMD decoding, and the CUDA path
with all 17.6 GB resident in the 4090, are where that gap is closed.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["BATTERY", "ForwardReport", "planted_rope", "run_gate"]

COMMAND_R = Path(r"J:\Models\bartowski\c4ai-command-r-08-2024-GGUF"
                 r"\c4ai-command-r-08-2024-Q4_K_S.gguf")
CPU, GPU = "http://127.0.0.1:8089", "http://127.0.0.1:8090"
BATTERY = ("The capital of France is", "def fibonacci(n):", "2 + 2 =",
           "Le chat est", "The meaning of life is", "The tower is 300 meters",
           "Hello, my name is", "import numpy as", "Water boils at",
           "In 1969, humans first", "Das ist ein", "The quick brown fox")


def oracle_top(base: str, ids: list, n: int = 10) -> tuple:
    """{id: logprob} for llama.cpp's top-n next tokens, and seconds."""
    body = {"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": n,
            "cache_prompt": False}
    request = urllib.request.Request(
        base + "/completion", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        out = json.loads(response.read())
    step = out["completion_probabilities"][0]
    tops = step.get("top_logprobs") or step.get("probs") or []
    return ({int(p["id"]): float(p["logprob"]) for p in tops},
            time.perf_counter() - started)


def log_softmax(logits: list) -> list:
    peak = max(logits)
    total = math.fsum(math.exp(v - peak) for v in logits)
    shift = peak + math.log(total)
    return [v - shift for v in logits]


def _ranked(table: dict) -> list:
    return sorted(table, key=table.get, reverse=True)


def score(ours_lp: list, cpu: dict, gpu: dict) -> dict:
    """Criterion 1 and 2 for one prompt."""
    cpu_rank = _ranked(cpu)
    top5 = cpu_rank[:5]
    shared = [t for t in top5 if t in gpu]
    spread = max((abs(cpu[t] - gpu[t]) for t in shared), default=math.inf)
    margin = cpu[cpu_rank[0]] - cpu[cpu_rank[1]]
    ours_top = max(range(len(ours_lp)), key=ours_lp.__getitem__)
    distance = max(abs(ours_lp[t] - cpu[t]) for t in top5)
    bound = max(2 * spread, 0.1)
    near_tie = not margin > spread
    return {"spread": spread, "margin": margin, "near_tie": near_tie,
            "top1_ok": near_tie or ours_top == cpu_rank[0],
            "distance": distance, "bound": bound,
            "within": math.isfinite(distance) and distance <= bound,
            "ours_top": ours_top, "cpu_top": cpu_rank[0]}


class planted_rope:
    """The pre-registered defect: RoPE pairing the wrong coordinates.

    The engine pairs ADJACENT coordinates (2i, 2i+1), as llama.cpp's
    NORM rope does for command-r; this pairs HALVES (i, i + head_dim/2),
    NEOX-style - swapped in only for the length of each call.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    def logits(self, ids: list) -> list:
        from ultraquant.infer import command_r as module

        real = module._rope

        def halves(vector, head_dim, rotations):
            half = head_dim // 2
            for start in range(0, len(vector), head_dim):
                for i, (cosine, sine) in enumerate(rotations):
                    left, right = start + i, start + i + half
                    a, b = vector[left], vector[right]
                    vector[left] = a * cosine - b * sine
                    vector[right] = a * sine + b * cosine

        module._rope = halves
        try:
            return self.engine.logits(ids)
        finally:
            module._rope = real


@dataclass
class ForwardReport:
    """Whether our engine lands where llama.cpp lands on itself.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        rows: Per prompt: spread, margin, distance, bound, outcomes.
        top1_failures: Non-tie prompts whose argmax differs (need 0).
        floor_failures: Prompts outside the noise floor (need 0).
        near_ties: Prompts excluded from criterion 1 as near-ties.
        planted_violations: Prompts the planted defect fails (need > 0).
        deterministic: Same prompt twice, identical logits.
        seconds: ours / cpu / gpu totals over the battery.
        reason: Plain-language verdict.
    """

    passes: bool
    rows: dict = field(default_factory=dict)
    top1_failures: int = 0
    floor_failures: int = 0
    near_ties: int = 0
    planted_violations: int = 0
    deterministic: bool = False
    seconds: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(engine=None, planted=None, tokenizer=None) -> ForwardReport:
    """Every criterion against both llama.cpp backends.

    ``planted`` is the engine with the pre-registered defect; the caller
    builds it from the real one (see ``planted_rope`` once it exists).
    """
    from ultraquant.infer.command_r import CommandR
    from ultraquant.infer.tokenizer import Tokenizer

    tok = tokenizer or Tokenizer.from_gguf(COMMAND_R)
    model = engine or CommandR.from_gguf(COMMAND_R)
    planted = planted if planted is not None else planted_rope(model)
    report = ForwardReport(passes=False)
    totals = {"ours": 0.0, "cpu": 0.0, "gpu": 0.0}
    for prompt in BATTERY:
        ids = tok.encode(prompt, add_special=True)
        started = time.perf_counter()
        ours = log_softmax(model.logits(ids))
        totals["ours"] += time.perf_counter() - started
        cpu, cpu_s = oracle_top(CPU, ids)
        gpu, gpu_s = oracle_top(GPU, ids)
        totals["cpu"] += cpu_s
        totals["gpu"] += gpu_s
        row = score(ours, cpu, gpu)
        if planted is not None:
            bad = score(log_softmax(planted.logits(ids)), cpu, gpu)
            row["planted_fails"] = not (bad["top1_ok"] and bad["within"])
            report.planted_violations += row["planted_fails"]
        report.rows[prompt] = row
        report.near_ties += row["near_tie"]
        report.top1_failures += not row["top1_ok"]
        report.floor_failures += not row["within"]
    first = tok.encode(BATTERY[2], add_special=True)
    report.deterministic = model.logits(first) == model.logits(first)
    report.seconds = totals
    valid = planted is not None and report.planted_violations > 0
    report.passes = (valid and report.top1_failures == 0
                     and report.floor_failures == 0 and report.deterministic)
    if not valid:
        report.reason = (f"VOID: the planted defect violated "
                         f"{report.planted_violations} prompts")
    elif report.passes:
        worst = max(r["distance"] for r in report.rows.values())
        report.reason = (
            f"PASS: {len(BATTERY)} prompts, top-1 agrees on every non-tie "
            f"({report.near_ties} near-ties excluded), all within the noise "
            f"floor (worst {worst:.4f} nats); deterministic; the planted "
            f"defect failed {report.planted_violations}")
    else:
        report.reason = (f"FAIL: top-1 failures {report.top1_failures}, "
                         f"floor failures {report.floor_failures}, "
                         f"deterministic {report.deterministic}")
    return report


if __name__ == "__main__":
    r = run_gate()
    print(r.reason)
    for prompt, row in r.rows.items():
        print(f"  {prompt!r:28} dist {row['distance']:.4f} <= {row['bound']:.4f}"
              f" | spread {row['spread']:.4f} margin {row['margin']:.3f}"
              f"{' (near-tie)' if row['near_tie'] else ''} | top1 "
              f"{'ok' if row['top1_ok'] else 'DIFFERS'} | planted "
              f"{'fails' if row.get('planted_fails') else 'passes'}")
    print(f"  seconds: {r.seconds}")
