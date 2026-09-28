"""K-quant weights, decoded exactly as llama.cpp decodes them.

The first unit of running the user's own Command-R 08-2024 (32B, Q4_K_S,
17.6 GB, already on disk): 271 of its tensors are Q4_K, 9 are Q5_K, and
the tied embedding - which is also the output head - is Q6_K. The
reader refused all three by name. Nothing of this model can run until
they are read, and read *exactly*: a decoder that is merely close
produces a model that is merely plausible.

**The oracle is llama.cpp itself.** LM Studio ships ggml as DLLs, and
`ggml-base.dll` exports `dequantize_row_q4_K`, `_q5_K` and `_q6_K` - the
code every llama.cpp inference on this file runs. This gate calls them
through ctypes on the same bytes. Calibrated before any implementation
existed: ggml's output is reproduced bit-for-bit by float32-emulated
arithmetic, and across 102,400 Q4_K weights fused and two-step rounding
never disagree - every product there is exact in float32.

GPT-6 Astra wrote the decoder; Claude wrote this exam. The product code
stays pure Python - ggml is only ever the examiner.

**The criteria, written before the run** - frozen in a pre-registration
(sha256 ede09a05...) before the decoder existed:

1. **Bit-exact.** First, middle and last block of every Q4_K, Q5_K and
   Q6_K tensor - 281 tensors, 843 blocks, 215,808 weights - equal to
   ggml's float32, bit for bit, and exactly equal as values: 0
   mismatches.
2. **The exam can fail.** The same harness on a decoder with one
   planted defect (the 6-bit scale/min unpacking for sub-blocks 4-7
   reading the wrong byte) must report mismatches, or it is void.
3. **Rows are blocks, joined.** `rows_of()` rows 0, middle and last of
   blk.0.attn_q (Q4_K), blk.0.attn_v (Q5_K) and token_embd (Q6_K),
   bit-exact against ggml's decode of the same bytes.
4. **Nothing else moves.** F32/F16/BF16/Q8_0 unchanged; Q2_K, Q3_K and
   IQ* still refused by name.
5. **The full suite is green.** Checked outside this module.

**PASSED on all four measured criteria.**

| | blocks | weights | mismatches |
|---|---:|---:|---:|
| criterion 1: first, middle, last of every tensor | 843 | 215,808 | **0** |
| extra evidence: every block of every row decoded | 34,656 | 8,871,936 | **0** |

All 281 k-quant tensors - 271 Q4_K, 9 Q5_K, and the Q6_K embedding that
is also the output head - bit-exact against ggml, values and bits both.
Nine whole rows through `rows_of`, bit-exact. The harness was shown able
to fail before it was trusted to pass: an independent decoder matched
ggml exactly, and the same decoder with one planted defect produced 428
mismatches in four blocks. Q2_K, Q3_K and the i-quants are still refused
by name, and the one existing test that asserted Q4_K was refused now
asserts it of Q2_K - its rule unchanged.

**The price of pure Python, measured.** 0.41 s per million weights: all
32 billion of Command-R's would take about 3.6 hours to decode once.
This tier is the oracle every faster one must match, not the one that
runs the model - llama.cpp itself generates 2.65 tokens/s on this CPU.
"""

from __future__ import annotations

import ctypes
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["BLOCK_BYTES", "COMMAND_R", "KQuantReport", "run_gate"]

COMMAND_R = Path(r"J:\Models\bartowski\c4ai-command-r-08-2024-GGUF"
                 r"\c4ai-command-r-08-2024-Q4_K_S.gguf")
GGML_DIR = Path(r"C:\Users\Stephen Hawking\.lmstudio\extensions\backends"
                r"\llama.cpp-win-x86_64-avx2-2.46.0")
BLOCK_BYTES = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}
_KIND = ("Q4_K", "Q5_K", "Q6_K")


class _Ggml:
    """ggml's own decoders, through ctypes. The examiner, never the product."""

    def __init__(self) -> None:
        os.add_dll_directory(str(GGML_DIR))
        lib = ctypes.CDLL(str(GGML_DIR / "ggml-base.dll"))
        self._fn = {}
        for kind in _KIND:
            fn = getattr(lib, f"dequantize_row_{kind[0].lower()}{kind[1]}_K")
            fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
            self._fn[kind] = fn

    def decode(self, kind: str, raw: bytes) -> list:
        n = len(raw) // BLOCK_BYTES[kind] * 256
        out = (ctypes.c_float * n)()
        self._fn[kind](raw, out, n)
        return list(out)


def _same(ours: list, theirs: list) -> int:
    """Mismatches: exact value AND identical float32 bits, both required."""
    bad = 0
    for a, b in zip(ours, theirs):
        if a != b or struct.pack("<f", a) != struct.pack("<f", b):
            bad += 1
    return bad + abs(len(ours) - len(theirs))


def _independent_q4k(raw: bytes, planted: bool) -> list:
    """An independent Q4_K reading - clean, or with ONE planted defect."""
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    half = lambda b: struct.unpack("<e", b)[0]
    out = []
    for at in range(0, len(raw), 144):
        blk = raw[at:at + 144]
        d, dmin = half(blk[0:2]), half(blk[2:4])
        sc, qs = blk[4:16], blk[16:144]

        def unpack(j):
            if j < 4:
                return sc[j] & 63, sc[j + 4] & 63
            low = sc[j] if planted else sc[j + 4]   # the planted defect
            return ((low & 0xF) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4))

        q = sub = 0
        for _ in range(4):
            for part, shift in ((0, 0), (1, 4)):
                s, m = unpack(sub + part)
                d1, m1 = f32(d * s), f32(dmin * m)
                out.extend(f32(f32(d1 * ((qs[q + i] >> shift) & 0xF)) - m1)
                           for i in range(32))
            q += 32
            sub += 2
    return out


def _block_of(g, info, b: int):
    """(row index, slice) of block ``b`` within its row."""
    per_row = info.dims[0] // 256
    row, col = divmod(b, per_row)
    return row, slice(col * 256, col * 256 + 256)


@dataclass
class KQuantReport:
    """Whether the reader now decodes what llama.cpp decodes.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        tensors: k-quant tensors examined, by type.
        blocks: Blocks compared under criterion 1 (843 expected).
        mismatches: Weights that differed from ggml there (must be 0).
        row_blocks: Extra evidence - every block of every decoded row.
        row_mismatches: Mismatches in that extra evidence.
        clean_mismatches: The same independent decoder WITHOUT the defect
            (must be 0, so the defect alone explains what is caught).
        planted_mismatches: The defective decoder's mismatches (must be
            nonzero, or the harness is void).
        rows_checked: Criterion 3 rows compared.
        rows_mismatched: Criterion 3 weights that differed (must be 0).
        refused: Unsupported types still refused by name.
        seconds_per_million: Pure-Python decode speed, for the record.
        reason: Plain-language verdict.
    """

    passes: bool
    tensors: dict = field(default_factory=dict)
    blocks: int = 0
    mismatches: int = 0
    row_blocks: int = 0
    row_mismatches: int = 0
    clean_mismatches: int = 0
    planted_mismatches: int = 0
    rows_checked: int = 0
    rows_mismatched: int = 0
    refused: bool = False
    seconds_per_million: float = 0.0
    reason: str = ""


def run_gate(path: Path = COMMAND_R) -> KQuantReport:
    """Every criterion, against ggml, on the user's own file."""
    import time

    from ultraquant.convert.gguf import TensorInfo, read

    ggml = _Ggml()
    g = read(path)
    report = KQuantReport(passes=False)
    decoded = 0
    started = time.perf_counter()
    with g.path.open("rb") as handle:
        def raw_block(info, b):
            size = BLOCK_BYTES[info.type_name]
            handle.seek(g.data_start + info.offset + b * size)
            return handle.read(size)

        for info in g.tensors:
            if info.type_name not in _KIND:
                continue
            report.tensors[info.type_name] = (
                report.tensors.get(info.type_name, 0) + 1)
            total = info.count // 256
            per_row = info.dims[0] // 256
            for b in sorted({0, total // 2, total - 1}):
                row, cols = _block_of(g, info, b)
                ours_row = g.rows_of(info, row, 1)[0]
                decoded += len(ours_row)
                theirs = ggml.decode(info.type_name, raw_block(info, b))
                report.blocks += 1
                report.mismatches += _same(ours_row[cols], theirs)
                # extra evidence: every block of the row we decoded anyway
                size = BLOCK_BYTES[info.type_name]
                handle.seek(g.data_start + info.offset
                            + row * per_row * size)
                whole = ggml.decode(info.type_name,
                                    handle.read(per_row * size))
                report.row_blocks += per_row
                report.row_mismatches += _same(ours_row, whole)

        # 2 - the harness must be able to fail
        q4 = g.by_name("blk.0.attn_q.weight")
        for b in (0, 1, 2, 3):
            raw = raw_block(q4, b)
            theirs = ggml.decode("Q4_K", raw)
            report.clean_mismatches += _same(_independent_q4k(raw, False),
                                             theirs)
            report.planted_mismatches += _same(_independent_q4k(raw, True),
                                               theirs)

        # 3 - rows_of, whole rows
        for name in ("blk.0.attn_q.weight", "blk.0.attn_v.weight",
                     "token_embd.weight"):
            info = g.by_name(name)
            rows = info.count // info.dims[0]
            size = BLOCK_BYTES[info.type_name]
            per_row = info.dims[0] // 256
            for row in (0, rows // 2, rows - 1):
                ours = g.rows_of(info, row, 1)[0]
                handle.seek(g.data_start + info.offset
                            + row * per_row * size)
                theirs = ggml.decode(info.type_name,
                                     handle.read(per_row * size))
                report.rows_checked += 1
                report.rows_mismatched += _same(ours, theirs)
    elapsed = time.perf_counter() - started
    report.seconds_per_million = elapsed / max(1, decoded) * 1e6

    # 4 - still refused by name
    refused = []
    for kind in ("Q2_K", "Q3_K", "IQ4_XS", "IQ2_XXS"):
        try:
            probe = TensorInfo(name="probe", dims=(256, 1), type_name=kind,
                               offset=0)
            refused.append(not probe.readable)
        except Exception:   # noqa: BLE001 - an unknown name is a refusal too
            refused.append(True)
    report.refused = all(refused)

    valid = report.clean_mismatches == 0 and report.planted_mismatches > 0
    report.passes = (valid and report.blocks == 843
                     and report.mismatches == 0
                     and report.rows_mismatched == 0 and report.refused)
    if not valid:
        report.reason = (f"VOID: clean decoder {report.clean_mismatches} "
                         f"mismatches, planted {report.planted_mismatches}")
    elif report.passes:
        report.reason = (
            f"PASS: {report.blocks} blocks bit-exact against ggml "
            f"({report.row_blocks} blocks of extra evidence, "
            f"{report.row_mismatches} mismatched); {report.rows_checked} "
            f"rows via rows_of bit-exact; the planted defect was caught "
            f"({report.planted_mismatches} mismatches); unsupported types "
            f"still refused")
    else:
        report.reason = (
            f"FAIL: {report.mismatches} mismatches in {report.blocks} "
            f"blocks, {report.rows_mismatched} in rows, refused="
            f"{report.refused}")
    return report


if __name__ == "__main__":
    r = run_gate()
    print(r.reason)
    print(f"  tensors: {r.tensors}")
    print(f"  pure-Python decode: {r.seconds_per_million:.2f} s per "
          f"million weights")
