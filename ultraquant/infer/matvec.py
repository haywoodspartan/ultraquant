"""A matrix-vector product straight from the checkpoint's packed rows.

Command-R's feed-forward matrix has 201 million weights. Expanding it
into Python floats for every token spends the time and memory before
the multiplication even starts. The native kernel reads one 256-weight
block at a time, rounds its weights exactly as the GGUF decoder does,
and accumulates the products in double precision.

The Python decoder remains the oracle. When the DLL or its matvec export
is absent, it supplies one row at a time to the same ordered double sum.
Only the requested prefix of rows is read; the packed native buffer is
released after each call. There is no persistent copy of the model here.
"""

from __future__ import annotations

import operator
from typing import Sequence

from ultraquant.convert.gguf import GgufFile, TensorInfo
from ultraquant.native import accel

__all__ = ["matvec"]

_KQUANTS = {"Q4_K": (12, 144), "Q5_K": (13, 176), "Q6_K": (14, 210)}


def matvec(gguf_file: GgufFile, info: TensorInfo, x: Sequence[float],
           rows: int | None = None, threads: int | None = None) -> list[float]:
    """Return W[:rows] @ x; rows defaults to all, threads to hardware count.

    ``gguf_file`` is the table returned by ``gguf.read`` and ``info`` is
    its tensor record. Vectors are treated as one-row matrices; higher
    dimensions are refused. Other readable types use the Python decoder.
    """
    if len(info.dims) not in (1, 2) or info.columns <= 0:
        raise ValueError("matvec requires a vector or matrix with positive width")
    count = info.rows if rows is None else operator.index(rows)
    if count < 0 or count > info.rows:
        raise ValueError("requested rows are outside the tensor")
    vector = [float(value) for value in x]
    if len(vector) != info.columns:
        raise ValueError("vector length does not match tensor columns")
    nthreads = 0 if threads is None else operator.index(threads)
    if not -(2**31) <= nthreads < 2**31:
        raise ValueError("threads must fit a C int")
    quant = _KQUANTS.get(info.type_name)
    if quant is not None and info.columns % 256:
        raise ValueError("k-quant row width must be a multiple of 256")
    if count == 0:
        return []
    dll = accel.load_cpu() if quant is not None else None
    if dll is not None and hasattr(dll, "uq_kq_matvec"):
        type_number, block_bytes = quant
        size = count * (info.columns // 256) * block_bytes
        with gguf_file.path.open("rb") as handle:
            handle.seek(gguf_file.data_start + info.offset)
            raw = handle.read(size)
        if len(raw) != size:
            raise ValueError(f"truncated {info.type_name} rows")
        return accel.kq_matvec_cpu(type_number, raw, count, info.columns,
                                   vector, nthreads)
    result = []
    for index in range(count):
        row = gguf_file.rows_of(info, first=index, count=1)[0]
        total = 0.0
        for weight, value in zip(row, vector):
            total += weight * value
        result.append(total)
    return result
