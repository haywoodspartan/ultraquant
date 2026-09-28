"""A GGUF reader, pure stdlib.

GGUF is the format the local models on this machine are actually in,
so it is where a conversion kit has to start. The layout is simple
and self-describing: a magic, a version, counts, a metadata table of
typed key/values, one info record per tensor (name, shape, type,
offset), then a padded data section.

**What is supported is stated rather than discovered.** F32, F16,
BF16, Q8_0, Q4_K, Q5_K and Q6_K are read; every other quantisation,
including Q2_K, Q3_K and the i-quants, is refused BY NAME. The three
k-quant decoders follow ggml's reference layout and float32 arithmetic,
returning exactly widened float32 values. Numbers that are wrong in
a format nobody checked are worse than a refusal.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["GgufFile", "TensorInfo", "UnsupportedTensorType", "read"]

#: GGML tensor types, by their wire number. Only the ones this reader
#: can turn into floats are given a decoder; the rest exist here so a
#: refusal can name the type instead of the number.
_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}

_READABLE = {"F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K"}

#: Metadata value types.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64 = 6, 7, 8, 9, 10, 11
_FLOAT64 = 12


class UnsupportedTensorType(RuntimeError):
    """A tensor this reader will not guess at."""


@dataclass
class TensorInfo:
    """One tensor's identity, before any of its data is touched.

    Attributes:
        name: The tensor's name in the checkpoint.
        dims: Shape, fastest-varying first, as GGUF stores it.
        type_name: "F32", "Q4_K", and so on.
        offset: Byte offset within the data section.
    """

    name: str
    dims: tuple
    type_name: str
    offset: int

    @property
    def count(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    @property
    def readable(self) -> bool:
        return self.type_name in _READABLE

    @property
    def rows(self) -> int:
        """Output channels: GGUF stores a matrix as [in, out]."""
        return self.dims[1] if len(self.dims) > 1 else 1

    @property
    def columns(self) -> int:
        return self.dims[0]


@dataclass
class GgufFile:
    """An opened checkpoint: metadata, tensor table, and a data window.

    Nothing is loaded eagerly. A published model is tens of gigabytes
    and this tier reads with the standard library, so tensors are
    seeked to and decoded one at a time - which is also how the
    shard library it feeds wants to be written.
    """

    path: Path
    version: int
    alignment: int
    data_start: int
    metadata: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)

    def by_name(self, name: str) -> TensorInfo:
        for info in self.tensors:
            if info.name == name:
                return info
        raise KeyError(name)

    def type_census(self) -> dict:
        """How many tensors of each type - the honest scope report."""
        census: dict = {}
        for info in self.tensors:
            census[info.type_name] = census.get(info.type_name, 0) + 1
        return census

    def rows_of(self, info: TensorInfo, first: int = 0,
                count: int | None = None) -> list:
        """Decode `count` rows of a tensor as lists of floats.

        Rows rather than the whole tensor, because a conversion is
        per-output-channel and a 4096x4096 tensor in pure Python is
        16 million floats that nobody needs at once.
        """
        if not info.readable:
            raise UnsupportedTensorType(
                f"{info.name} is {info.type_name}; this reader handles "
                + ", ".join(sorted(_READABLE))
                + " and refuses to guess at the rest")
        width = info.columns
        total_rows = info.count // width if width else 0
        if count is None:
            count = total_rows - first
        with self.path.open("rb") as handle:
            return _decode_rows(handle, self.data_start + info.offset,
                                info.type_name, width, first, count)


#: Q8_0 stores 32 weights per block behind one half-precision scale.
_Q8_0_BLOCK = 32


def _float32(value: float) -> float:
    """Round one ggml operation to float32, then widen exactly."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _decode_rows(handle, base: int, type_name: str, width: int,
                 first: int, count: int) -> list:
    """Rows [first, first+count) of a tensor, as lists of floats."""
    rows: list = []
    if type_name == "Q8_0":
        # A row is a whole number of blocks, so a row can be seeked
        # to directly rather than decoded from the beginning.
        blocks_per_row = width // _Q8_0_BLOCK
        row_bytes = blocks_per_row * (2 + _Q8_0_BLOCK)
        handle.seek(base + first * row_bytes)
        raw = handle.read(row_bytes * count)
        at = 0
        for _ in range(count):
            row: list = []
            for _block in range(blocks_per_row):
                scale = struct.unpack_from("<e", raw, at)[0]
                at += 2
                for value in struct.unpack_from("<32b", raw, at):
                    row.append(scale * value)
                at += _Q8_0_BLOCK
            rows.append(row)
        return rows

    if type_name in {"Q4_K", "Q5_K", "Q6_K"}:
        # §11.123: Whole k-quant blocks, with ggml's float32 rounding.
        if width % 256:
            raise ValueError(f"{type_name} row width must be a multiple of 256")
        block_bytes = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}[type_name]
        blocks_per_row = width // 256
        row_bytes = blocks_per_row * block_bytes
        handle.seek(base + first * row_bytes)
        raw = handle.read(row_bytes * count)
        if len(raw) != row_bytes * count:
            raise ValueError(f"truncated {type_name} rows")
        at = 0
        for _ in range(count):
            row = []
            for _block in range(blocks_per_row):
                if type_name == "Q6_K":
                    d = struct.unpack_from("<e", raw, at + 208)[0]
                    scales = struct.unpack_from("<16b", raw, at + 192)
                    for half in range(2):
                        for group in range(4):
                            low = at + half * 64 + (group % 2) * 32
                            high = at + 128 + half * 32
                            for lane in range(32):
                                q = (raw[low + lane] >> (4 * (group // 2))) & 15
                                q |= ((raw[high + lane] >> (2 * group)) & 3) << 4
                                scale = scales[half * 8 + group * 2 + lane // 16]
                                row.append(_float32(_float32(d * scale) * (q - 32)))
                else:
                    d, dmin = struct.unpack_from("<2e", raw, at)
                    scales = raw[at + 4:at + 16]
                    low = at + (48 if type_name == "Q5_K" else 16)
                    for group in range(8):
                        if group < 4:
                            scale = scales[group] & 63
                            minimum = scales[group + 4] & 63
                        else:
                            scale = ((scales[group + 4] & 15)
                                     | ((scales[group - 4] >> 6) << 4))
                            minimum = ((scales[group + 4] >> 4)
                                       | ((scales[group] >> 6) << 4))
                        ds = _float32(d * scale)
                        dm = _float32(dmin * minimum)
                        for lane in range(32):
                            packed = raw[low + (group // 2) * 32 + lane]
                            q = (packed >> (4 * (group % 2))) & 15
                            if type_name == "Q5_K":
                                q |= ((raw[at + 16 + lane] >> group) & 1) << 4
                            row.append(_float32(_float32(ds * q) - dm))
                at += block_bytes
            rows.append(row)
        return rows

    item = {"F32": 4, "F16": 2, "BF16": 2}[type_name]
    handle.seek(base + first * width * item)
    raw = handle.read(width * item * count)
    for index in range(count):
        start = index * width * item
        if type_name == "F32":
            rows.append(list(struct.unpack_from(f"<{width}f", raw, start)))
        elif type_name == "F16":
            rows.append(list(struct.unpack_from(f"<{width}e", raw, start)))
        else:
            # bfloat16 is the top half of a float32, so widening it is
            # a shift rather than a conversion - exact, always.
            halves = struct.unpack_from(f"<{width}H", raw, start)
            rows.append([struct.unpack("<f", struct.pack("<I", h << 16))[0]
                         for h in halves])
    return rows


def _read_string(handle) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    return handle.read(length).decode("utf-8", errors="replace")


def _read_value(handle, value_type: int, keep_strings: bool = False):
    if value_type == _UINT8:
        return struct.unpack("<B", handle.read(1))[0]
    if value_type == _INT8:
        return struct.unpack("<b", handle.read(1))[0]
    if value_type == _UINT16:
        return struct.unpack("<H", handle.read(2))[0]
    if value_type == _INT16:
        return struct.unpack("<h", handle.read(2))[0]
    if value_type == _UINT32:
        return struct.unpack("<I", handle.read(4))[0]
    if value_type == _INT32:
        return struct.unpack("<i", handle.read(4))[0]
    if value_type == _FLOAT32:
        return struct.unpack("<f", handle.read(4))[0]
    if value_type == _BOOL:
        return struct.unpack("<?", handle.read(1))[0]
    if value_type == _STRING:
        return _read_string(handle)
    if value_type == _UINT64:
        return struct.unpack("<Q", handle.read(8))[0]
    if value_type == _INT64:
        return struct.unpack("<q", handle.read(8))[0]
    if value_type == _FLOAT64:
        return struct.unpack("<d", handle.read(8))[0]
    if value_type == _ARRAY:
        (item_type,) = struct.unpack("<I", handle.read(4))
        (length,) = struct.unpack("<Q", handle.read(8))
        # A vocabulary is a million strings and most callers never need
        # it in memory; the length is kept, the contents skipped.
        # §11.124: a caller that does - the tokenizer - names the
        # arrays it wants kept.
        if item_type == _STRING and length > 4096 and not keep_strings:
            for _ in range(length):
                (size,) = struct.unpack("<Q", handle.read(8))
                handle.seek(size, 1)
            return f"<{length} strings, skipped>"
        return [_read_value(handle, item_type, keep_strings)
                for _ in range(length)]
    raise ValueError(f"unknown metadata value type {value_type}")


def read(path: str | Path, *, keep: tuple[str, ...] = ()) -> GgufFile:
    """Read tables, not tensors; retain large string arrays named in `keep`."""
    path = Path(path)
    keep = frozenset(keep)
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path.name} is not a GGUF file")
        version, = struct.unpack("<I", handle.read(4))
        tensor_count, = struct.unpack("<Q", handle.read(8))
        kv_count, = struct.unpack("<Q", handle.read(8))

        metadata: dict = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            (value_type,) = struct.unpack("<I", handle.read(4))
            metadata[key] = _read_value(handle, value_type, key in keep)

        tensors: list = []
        for _ in range(tensor_count):
            name = _read_string(handle)
            (n_dims,) = struct.unpack("<I", handle.read(4))
            dims = struct.unpack(f"<{n_dims}Q", handle.read(8 * n_dims))
            (type_number,) = struct.unpack("<I", handle.read(4))
            (offset,) = struct.unpack("<Q", handle.read(8))
            tensors.append(TensorInfo(
                name=name, dims=dims,
                type_name=_TYPE_NAMES.get(type_number,
                                          f"type{type_number}"),
                offset=offset))

        alignment = int(metadata.get("general.alignment", 32))
        here = handle.tell()
        padding = (alignment - (here % alignment)) % alignment
        data_start = here + padding
    return GgufFile(path=path, version=version, alignment=alignment,
                    data_start=data_start, metadata=metadata,
                    tensors=tensors)
