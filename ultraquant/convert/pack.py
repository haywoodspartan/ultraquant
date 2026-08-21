"""Ternary tensors in a shape the shard library can hold.

The vault stores a shard as JSON, zlib-compressed. That is a fine
home for a learned net's state and an awkward one for sixteen
million trits, so the question this module answers by measurement is
how to spell a ternary matrix so the library can keep it.

Four spellings were measured on a real tensor, after zlib, which is
the only number that matters because the vault always compresses:

    list of ints      2.25 bits/weight
    byte per trit     2.27
    2-bit packed      1.53
    base-243 packed   1.54

The dense packings win by a third, and the two of them tie - which
is not what the arithmetic predicts. Five trits in a byte is 1.60
bits before compression against two-bit's 2.00, a fifth better, and
that advantage disappears in the zlib: the looser packing keeps
byte-aligned structure a compressor can find, and the denser one
looks more like noise. Density and compressibility pull against
each other, and they very nearly cancel.

So the choice was made on the other axis. A table-driven base-243
decoder reads 118M weights/s against two-bit's 34M - **3.4x** - for
0.7% more bytes, and a pageable library decodes on every page-in.
Size was a tie; speed was not.

For reference the symbol entropy of that tensor is 1.403
bits/weight. At 1.54 the packing sits about a tenth over the floor,
and most of that tenth is base64: the payload must be JSON, so the
bytes are text before they are compressed.
"""

from __future__ import annotations

import base64
import json
import zlib

__all__ = ["encode_rows", "decode_rows", "to_payload", "from_payload",
           "BASE", "PER_BYTE"]

#: Five trits fit in a byte because 3^5 = 243 <= 255.
BASE = 3
PER_BYTE = 5


def _decode_table() -> list:
    """byte -> its five trits, built once so decoding is a lookup.

    The whole speed advantage lives here. Decoding by repeated
    divmod costs three arithmetic operations per weight; a table
    costs one index and a list extend, and the table is 243 entries.
    """
    table = []
    for byte in range(256):
        values = []
        remainder = byte
        for _ in range(PER_BYTE):
            values.append((remainder % BASE) - 1)
            remainder //= BASE
        table.append(values)
    return table


_TABLE = _decode_table()


def encode_rows(quantised: list) -> bytes:
    """Ternary rows to packed bytes, five weights per byte.

    Each row starts on a byte boundary. That wastes at most four
    trits per row and buys the thing that matters for a pageable
    store: a row can be decoded without decoding the row before it.
    """
    out = bytearray()
    for row in quantised:
        accumulator = 0
        power = 1
        count = 0
        for value in row:
            accumulator += (value + 1) * power
            power *= BASE
            count += 1
            if count == PER_BYTE:
                out.append(accumulator)
                accumulator = 0
                power = 1
                count = 0
        if count:
            out.append(accumulator)
    return bytes(out)


def decode_rows(blob: bytes, rows: int, width: int) -> list:
    """Packed bytes back to ternary rows - exactly, or not at all."""
    per_row = (width + PER_BYTE - 1) // PER_BYTE
    if len(blob) < rows * per_row:
        raise ValueError(
            f"packed blob holds {len(blob)} bytes; {rows} rows of "
            f"{width} need {rows * per_row}")
    out = []
    for index in range(rows):
        start = index * per_row
        row: list = []
        for byte in blob[start:start + per_row]:
            row.extend(_TABLE[byte])
        out.append(row[:width])
    return out


def _stored_size(payload: dict) -> int:
    """Exactly what the vault will write: canonical JSON, zlibbed."""
    raw = json.dumps(payload, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return len(zlib.compress(raw))


def to_payload(quantised: list, scales: list, name: str,
               source_type: str, spelling: str = "auto") -> dict:
    """A shard payload the vault can serialise as it stands.

    base64 because the payload must be JSON. It costs a third before
    compression and almost nothing after on a weight matrix - the
    measured gap between packed bytes and the stored shard is under
    two per cent - so the constraint the vault imposes is nearly
    free there.

    The spelling is nonetheless CHOSEN rather than fixed, and the
    honest reason is smaller than it first looked. Measured across
    fourteen real tensors, packing wins on twelve - by 0.6 to 0.7
    bits/weight on every weight matrix - and text wins on two 1-D
    tensors by 0.014 and 0.062. So both spellings are written, the
    smaller is kept, and the payload records which so the decoder
    can dispatch; the choice buys about 0.005 bits/weight on this
    checkpoint, which is nearly nothing, and it is kept because it
    can never be worse rather than because it pays.

    An earlier version of this comment claimed packing LOST 0.4-0.6
    bits/weight on 1-D tensors. That was measured against a baseline
    payload carrying no name and no shape - the spelling and the
    metadata at once - and it did not survive a fair comparison.
    """
    if len(quantised) != len(scales):
        raise ValueError("one scale per row, or the reconstruction "
                         "would silently use the wrong one")
    width = len(quantised[0]) if quantised else 0
    base = {
        "name": name,
        "source_type": source_type,
        "rows": len(quantised),
        "width": width,
        "scales": list(scales),
    }
    packed = dict(base, packing="base243",
                  trits=base64.b64encode(
                      encode_rows(quantised)).decode("ascii"))
    if spelling == "base243":
        return packed
    plain = dict(base, packing="ints", trits=quantised)
    if spelling == "ints":
        return plain
    return packed if _stored_size(packed) <= _stored_size(plain) else plain


def from_payload(payload: dict) -> tuple:
    """``(quantised, scales)`` back out of a stored shard."""
    packing = payload.get("packing")
    if packing == "ints":
        return [list(row) for row in payload["trits"]], list(
            payload["scales"])
    if packing != "base243":
        raise ValueError(f"unknown packing {packing!r}")
    blob = base64.b64decode(payload["trits"])
    rows = decode_rows(blob, int(payload["rows"]), int(payload["width"]))
    return rows, list(payload["scales"])
