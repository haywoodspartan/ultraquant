"""Command-R's matrices, multiplied natively. The matvec gate.

The third unit of running the user's own Command-R 08-2024. §11.123 made
its Q4_K / Q5_K / Q6_K weights readable, bit-exact to ggml - at 0.41 s
per million weights, which is 3.6 hours for one pass over the model.
This is the native kernel: `uq_kq_matvec` in the CPU DLL decodes k-quant
rows exactly as ggml does and multiplies them by a vector, y = W x,
straight from the GGUF bytes. GPT-6 Astra wrote it; Claude wrote this
exam.

**A design fact, fixed before the run.** llama.cpp's CPU path quantizes
the activation vector to Q8_K before every dot product. This kernel does
not: exact float32 weights, float64 accumulation. Its agreement with
llama.cpp end to end can therefore only ever be statistical, and is
judged later; bitwise parity is owed to our own oracle, as SPEC-NATIVE
requires - to 1e-9 in double precision.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 6f806b77...) before the kernel existed:

1. **Parity with the oracle.** Rows first, middle and last of every
   layer-0 matrix and of token_embd, x from a fixed seed: relative
   error <= 1e-9 against the float64 dot of the bit-exact decoded row;
   native block decode bit-identical to the Python decode.
2. **The exam can fail.** The kernel handed Q5_K bytes as Q4_K must
   report parity failures, or the harness is void.
3. **Speed, measured and recorded** - not gated.
4. **Fallback is structural**: with the DLL absent, the Python path
   answers identically.
5. **The full suite is green.** Checked outside this module.

**The exam's own defect came first.** The first run went VOID: the
kernel, handed Q5_K bytes as Q4_K, decoded garbage fp16 scales into NaN,
and the harness compared with ``> 1e-9`` - False for NaN - so it counted
NaN as agreement. The same hole was in criterion 1: ``max(0.0, nan)`` is
0.0, so a kernel returning NaN everywhere would have PASSED parity.
Criterion 2 going void is what stopped the exam vouching for anything
with that hole in it. Non-finite values now always fail.

**PASSED on all four measured criteria.**

| | result |
|---|---|
| parity, 24 rows of 8 matrices | worst relative error **0.0** |
| native decode vs Python decode | bit-identical |
| kernel told the wrong type | **8 of 8** rows fail |
| fallback, DLL "absent" | relative error 0.0 |
| ffn_up, 113 MB, 16 threads | **18.3 ms = 6.2 GB/s** |

Relative error is exactly zero, not merely small: both sides accumulate
each row in float64 in the same order, and threads split rows, not
columns. **The speed is the honest weak point**: 6.2 GB/s against the
~46 GB/s llama.cpp reaches on this CPU - scalar decoding and float64
accumulation, built to be exact first. SIMD decoding and the CUDA path,
with the whole model resident in the 4090's VRAM, are where that gap is
to be closed, each owing this kernel the same parity.
"""

from __future__ import annotations

import math
import random
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["COMMAND_R", "MatvecReport", "run_gate"]

COMMAND_R = Path(r"J:\Models\bartowski\c4ai-command-r-08-2024-GGUF"
                 r"\c4ai-command-r-08-2024-Q4_K_S.gguf")
MATRICES = ("blk.0.attn_q.weight", "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight", "token_embd.weight")
_TYPE = {"Q4_K": 12, "Q5_K": 13, "Q6_K": 14}
_BLOCK = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
_LLAMA_GBPS = 46.0      # llama.cpp on this CPU: 17.5 GB/token x 2.65 tok/s


def _raw_rows(g, info, first: int, count: int) -> bytes:
    size = info.columns // 256 * _BLOCK[info.type_name]
    with g.path.open("rb") as handle:
        handle.seek(g.data_start + info.offset + first * size)
        return handle.read(size * count)


def _oracle(g, info, row: int, x: list) -> float:
    """The float64 dot of the bit-exact decoded row, in order."""
    total = 0.0
    for w, v in zip(g.rows_of(info, row, 1)[0], x):
        total += w * v
    return total


def _relative(a: float, b: float) -> float:
    """Relative error - and infinite for anything non-finite.

    The first run of this exam compared with ``> 1e-9``, and a NaN
    compares False with everything: the kernel, handed the wrong type,
    produced NaN, and the harness counted NaN as agreement. Worse, the
    same hole was in criterion 1 - ``max(0.0, nan)`` is 0.0 - so a kernel
    that returned NaN everywhere would have passed. Caught by criterion
    2 going void before the exam vouched for anything.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(b), 1e-30)


@dataclass
class MatvecReport:
    """Whether the native kernel computes what the oracle computes.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        rows_checked: Rows compared under criterion 1.
        worst_relative: The largest relative error found there.
        dequant_mismatches: Native block decode vs Python, bit for bit.
        wrong_type_failures: Rows that failed parity when the kernel was
            told the wrong type (must be > 0).
        fallback_worst: Largest relative error, fallback vs native.
        gbps_kernel: ffn_up, bytes already in memory, 16 threads.
        seconds_kernel: The same, in seconds.
        reason: Plain-language verdict.
    """

    passes: bool
    rows_checked: int = 0
    worst_relative: float = 0.0
    dequant_mismatches: int = 0
    wrong_type_failures: int = 0
    fallback_worst: float = 0.0
    gbps_kernel: float = 0.0
    seconds_kernel: float = 0.0
    per_matrix: dict = field(default_factory=dict)
    reason: str = ""


def run_gate(path: Path = COMMAND_R) -> MatvecReport:
    """Every criterion, on the user's own file."""
    from ultraquant.convert.gguf import read
    from ultraquant.infer import matvec as M
    from ultraquant.native import accel

    g = read(path)
    rng = random.Random(0)
    report = MatvecReport(passes=False)

    # 1 - parity, and bit-identical decode
    for name in MATRICES:
        info = g.by_name(name)
        x = [rng.uniform(-1.0, 1.0) for _ in range(info.columns)]
        y = M.matvec(g, info, x, threads=16)
        worst = 0.0
        for row in sorted({0, info.rows // 2, info.rows - 1}):
            worst = max(worst, _relative(y[row], _oracle(g, info, row, x)))
            report.rows_checked += 1
            native = accel.kq_dequant_cpu(_TYPE[info.type_name],
                                          _raw_rows(g, info, row, 1))
            python = g.rows_of(info, row, 1)[0]
            report.dequant_mismatches += sum(
                struct.pack("<f", a) != struct.pack("<f", b)
                for a, b in zip(native, python))
        report.per_matrix[name] = worst
        report.worst_relative = max(report.worst_relative, worst)

    # 2 - the exam can fail: Q5_K bytes read as Q4_K
    info = g.by_name("blk.0.attn_v.weight")
    x = [rng.uniform(-1.0, 1.0) for _ in range(info.columns)]
    rows = 8
    as_q4 = _raw_rows(g, info, 0, rows)[:rows * info.columns // 256 * 144]
    wrong = accel.kq_matvec_cpu(12, as_q4, rows, info.columns, x, 16)
    report.wrong_type_failures = sum(
        not _relative(wrong[r], _oracle(g, info, r, x)) <= 1e-9
        for r in range(rows))

    # 3 - speed, bytes already in memory
    info = g.by_name("blk.0.ffn_up.weight")
    raw = _raw_rows(g, info, 0, info.rows)
    x = [rng.uniform(-1.0, 1.0) for _ in range(info.columns)]
    accel.kq_matvec_cpu(12, raw, info.rows, info.columns, x, 16)   # warm
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        accel.kq_matvec_cpu(12, raw, info.rows, info.columns, x, 16)
        best = min(best, time.perf_counter() - started)
    report.seconds_kernel = best
    report.gbps_kernel = len(raw) / best / 1e9

    # 4 - fallback: the DLL "absent"
    info = g.by_name("blk.0.attn_k.weight")
    x = [rng.uniform(-1.0, 1.0) for _ in range(info.columns)]
    native = M.matvec(g, info, x, rows=6, threads=16)
    real = accel.load_cpu
    accel.load_cpu = lambda *a, **k: None
    try:
        fallback = M.matvec(g, info, x, rows=6)
    finally:
        accel.load_cpu = real
    report.fallback_worst = max(_relative(a, b)
                                for a, b in zip(fallback, native))

    valid = report.wrong_type_failures > 0
    report.passes = (valid and math.isfinite(report.worst_relative)
                     and report.worst_relative <= 1e-9
                     and report.dequant_mismatches == 0
                     and report.fallback_worst <= 1e-9)
    if not valid:
        report.reason = "VOID: the wrong type produced no parity failures"
    elif report.passes:
        report.reason = (
            f"PASS: {report.rows_checked} rows, worst relative error "
            f"{report.worst_relative:.1e}; decode bit-identical; the wrong "
            f"type failed {report.wrong_type_failures}/8 rows; fallback "
            f"{report.fallback_worst:.1e}; ffn_up {report.seconds_kernel*1e3:.1f}"
            f" ms = {report.gbps_kernel:.2f} GB/s against llama.cpp's "
            f"~{_LLAMA_GBPS:.0f}")
    else:
        report.reason = (
            f"FAIL: worst {report.worst_relative:.1e}, decode mismatches "
            f"{report.dequant_mismatches}, fallback {report.fallback_worst:.1e}")
    return report


if __name__ == "__main__":
    r = run_gate()
    print(r.reason)
    for name, worst in r.per_matrix.items():
        print(f"  {name:28} worst relative {worst:.1e}")
