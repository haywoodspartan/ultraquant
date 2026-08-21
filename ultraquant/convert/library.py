"""Put a converted checkpoint INTO the shard library, not beside it.

Writing bytes to disk is the easy half. The library's whole value is
that it is catalogued - shards carry a category, keyword
associations and access statistics, so a thought pages in what it
needs and nothing else. A converted transformer that lands as an
opaque pile of files has been stored, not packed, and the difference
shows the first time somebody asks for one layer.

So two decisions are made here, and both are visible in the catalog:

**Granularity.** One shard per tensor. A transformer's matmul needs
the whole matrix, so splitting a tensor across shards would page
every piece for every use and buy nothing; keeping several tensors
in one shard would page a layer to read a bias. The tensor is the
unit that gets used, so it is the unit that gets stored.

**Address.** The category is the layer ("blk.0"), because layers are
what get paged together, and the associations are the tensor name's
own tokens.

Those two carry different halves of the address, and it is worth
being exact about which. The keyword index finds a KIND of tensor -
"attn out" scores every category holding an attention output
projection - and it cannot tell layer 3 from layer 9, because a
layer number is a digit and digits are not keywords. Layer identity
lives in the category; kind lives in the index; the shard id names
the tensor exactly. Asking for "layer 3's attention output" is a
question for the catalog and the index together, which is the
division the vault was built with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ultraquant.convert import gguf, glyph, pack, ternary

__all__ = ["PackReport", "pack_checkpoint", "category_of",
           "associations_of", "shape_label"]

_LAYER = re.compile(r"(?:blk|layers?)\.(\d+)")
_SPLIT = re.compile(r"[^a-z0-9]+")


def category_of(name: str) -> str:
    """The layer a tensor belongs to, or its head/tail group.

    Layers are the unit that pages together, so they are the unit
    the catalog groups by. Everything outside a numbered block -
    embeddings, the final norm, the output head - lands in "trunk",
    which is the honest name for "used on every token regardless".
    """
    found = _LAYER.search(name.lower())
    return f"blk.{found.group(1)}" if found else "trunk"


def associations_of(name: str) -> dict:
    """Keyword weights from the tensor's own name.

    No cleverness: the words a person would type to look for this
    tensor are the words already in its name. Numbers are dropped -
    "0" as a keyword would associate every layer-zero tensor with
    every other tensor that happens to contain a zero.
    """
    tokens = [token for token in _SPLIT.split(name.lower())
              if token and not token.isdigit()]
    weights: dict = {}
    for token in tokens:
        weights[token] = weights.get(token, 0.0) + 1.0
    return weights


def shape_label(rows: list, recognizer) -> str:
    """The recognizer's name for a tensor's glyph, or "" without one.

    Kept out of :func:`pack_checkpoint`'s required path on purpose.
    Rendering a glyph is integer arithmetic over trits and needs
    nothing; NAMING one needs a trained classifier, and a converter
    that could not run without a model would be the wrong shape of
    dependency for a file reader.
    """
    if recognizer is None:
        return ""
    label, _confidence = recognizer.recognize(rows)
    return str(label)


@dataclass
class PackReport:
    """What a packing run stored, skipped, and cost.

    Attributes:
        shards: Tensors written as shards.
        labels: Glyph shape label -> how many tensors carry it.
        weights: Ternary weights stored.
        stored_bytes: Bytes the vault's shard files occupy.
        skipped: Tensor type -> count refused by the reader.
        categories: Category -> shard count.
        bits_per_weight: Stored bytes as bits per weight.
    """

    shards: int = 0
    weights: int = 0
    stored_bytes: int = 0
    skipped: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)

    @property
    def bits_per_weight(self) -> float:
        if not self.weights:
            return 0.0
        return 8.0 * self.stored_bytes / self.weights


def pack_checkpoint(model: str | Path, vault, limit: int | None = None,
                    rows_each: int | None = None,
                    recognizer=None) -> PackReport:
    """Convert a checkpoint's tensors into shards in ``vault``.

    Every shard carries a 5x5 glyph rendered from its own trits - a
    fourth address, and the only one that says anything about what is
    IN the tensor rather than what it was called. With a recognizer
    the glyph is also NAMED, and the name is written as an
    association, so "which tensors look like stripes" becomes a
    catalog question like any other.

    Args:
        model: Path to a GGUF checkpoint.
        vault: An open :class:`~ultraquant.shards.vault.ShardVault`.
        limit: Convert at most this many tensors (whole file if None).
        rows_each: Convert at most this many rows per tensor, for
            sampling a large model without reading all of it.
        recognizer: Optional
            :class:`~ultraquant.pattern.recognition.PatternRecognizer`
            used to name each glyph. Without one the glyph is still
            stored - rendering needs nothing, naming needs a model.

    Returns:
        A :class:`PackReport`.
    """
    opened = gguf.read(Path(model))
    report = PackReport()
    written = 0
    for info in opened.tensors:
        if limit is not None and written >= limit:
            break
        if not info.readable:
            report.skipped[info.type_name] = (
                report.skipped.get(info.type_name, 0) + 1)
            continue
        rows_wanted = info.rows if rows_each is None else min(rows_each,
                                                              info.rows)
        rows = opened.rows_of(info, 0, rows_wanted)
        if not rows or not rows[0]:
            continue
        quantised, scales, _rule = ternary.choose_rules(rows)
        payload = pack.to_payload(quantised, scales, info.name,
                                  info.type_name)
        drawn = glyph.glyph_of(quantised)
        payload["glyph"] = list(drawn)
        category = category_of(info.name)
        associations = associations_of(info.name)
        label = shape_label(drawn, recognizer)
        if label:
            # The shape is an address, so it is weighted like one -
            # the same 1.0 a name token carries, because a tensor is
            # no less "that stripey one" than it is "that attn_q one".
            payload["shape"] = label
            associations[label] = associations.get(label, 0.0) + 1.0
            report.labels[label] = report.labels.get(label, 0) + 1
        vault.add_shard(
            shard_id=f"tensor:{info.name}",
            category=category,
            payload=payload,
            kind="ternary-tensor",
            associations=associations,
        )
        written += 1
        report.shards += 1
        report.weights += len(quantised) * len(quantised[0])
        report.categories[category] = report.categories.get(category, 0) + 1
    loose = Path(vault.root) / "loose"
    report.stored_bytes = sum(path.stat().st_size
                              for path in loose.glob("*.uqs"))
    return report
