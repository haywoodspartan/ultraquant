"""Weights that come back OUT of the library and compute.

§11.96 proved tensors go in: packed, catalogued, addressable, and
recovered bit-for-bit through JSON, base64, zlib and sha256. That is
a claim about BYTES. It says nothing about whether a network built
from those bytes computes what the original computed, and "the
weights are imported" is not true until something runs on them.

So this module closes the loop. A shard becomes a
:class:`~ultraquant.model.quantize.TernaryLinear`; a set of shards
becomes an :class:`~ultraquant.model.network.UltraQuantNet`; and the
glyph recognizer - the system's one existing consumer of ternary
weights - is the instrument that says whether the arithmetic
survived the trip.

**Two things are deliberately NOT symmetric with the packer.**

**Biases are stored exactly, never as trits.** §11.95 measured a 1-D
tensor at 0.527 relative error under ternary conversion, and a bias
is added to every output of its layer - the one place a per-element
error cannot cancel. So a bias travels as floats. It is a handful of
numbers against a matrix's millions, so the bytes are free, and
ternarising it would be a measurable loss taken for nothing.

**An imported layer is not re-ternarised on arrival.** Its weights
already ARE ``alpha * q``. Quantising them again would be a second
lossy step on top of a lossy representation - except that it is not
lossy at all, which is the small surprise this unit measured rather
than assumed: for a matrix whose entries are all in
``{-alpha, 0, +alpha}``, the threshold ``0.75 * mean|w|`` sits below
``alpha`` for any non-zero fraction, so every survivor survives and
the recovered scale is ``alpha`` again. Ternary quantisation is
IDEMPOTENT. The layer can be handed back quantised and the forward
pass reproduces the original exactly, which is why the gate above
can ask for bit-for-bit equality rather than a tolerance.
"""

from __future__ import annotations

import random

from ultraquant.convert import glyph, library, pack, ternary
from ultraquant.model.network import UltraQuantNet
from ultraquant.model.quantize import TernaryLinear

__all__ = ["dense_rows", "layer_payload", "layer_from_payload",
           "store_layer", "load_layer", "store_net", "load_net",
           "is_idempotent"]


def dense_rows(payload: dict) -> list[list[float]]:
    """A packed shard back as float rows, each scaled by its own row.

    The scale is per row because the packer stores it that way even
    when one rule produced them all - a per-matrix rule writes the
    same number `rows` times, which costs a few bytes and means the
    reader never has to know which rule was used.
    """
    if payload.get("packing") == "floats":
        return [[float(v) for v in row] for row in payload["weights"]]
    quantised, scales = pack.from_payload(payload)
    return [[scale * value for value in row]
            for row, scale in zip(quantised, scales)]


def layer_payload(layer: TernaryLinear, name: str) -> dict:
    """A shard payload holding one linear layer, weights and bias.

    A TERNARY layer goes through the same codec a converted
    transformer uses, so the library holds one kind of tensor shard
    for the common case rather than two.

    A FULL-PRECISION layer does not, and the first run of the gate is
    why. UltraQuantNet's head is full precision by default, and
    pushing it through the trit codec cost 0.151 in the weights and
    dropped glyph accuracy from 0.900 to 0.792 - the same lesson the
    bias decision above rests on, one level up. Trits are how you
    store a ternary matrix, not how you store a matrix; a layer that
    is not ternary travels as exact floats and says so in the
    payload. Converted transformers are ternary by construction and
    never take this path.
    """
    if not layer.quantized:
        return {
            "name": name,
            "source_type": "float-linear",
            "packing": "floats",
            "rows": len(layer.w),
            "width": len(layer.w[0]) if layer.w else 0,
            "weights": [[float(v) for v in row] for row in layer.w],
            "bias": [float(v) for v in layer.b],
            "quantized": False,
        }
    quantised, scales = ternary.ternarise_rows(layer.w, per_row=False)
    payload = pack.to_payload(quantised, scales, name, "ternary-linear")
    payload["bias"] = [float(v) for v in layer.b]
    payload["quantized"] = True
    payload["glyph"] = glyph.glyph_of(quantised)
    return payload


def layer_from_payload(payload: dict) -> TernaryLinear:
    """One linear layer, rebuilt from a shard.

    The weights arrive already ternary, so they are installed as the
    master weights directly. Re-quantisation on the forward pass is
    a no-op (see the module note), so the layer's own ``quantized``
    flag is restored rather than forced off.
    """
    rows = dense_rows(payload)
    width = len(rows[0]) if rows else 0
    layer = TernaryLinear(width, len(rows), random.Random(0),
                          quantized=bool(payload.get("quantized", True)))
    layer.w = rows
    bias = payload.get("bias")
    layer.b = ([float(v) for v in bias] if bias is not None
               else [0.0] * len(rows))
    layer.gw = [[0.0] * width for _ in rows]
    layer.gb = [0.0] * len(rows)
    return layer


def store_layer(vault, layer: TernaryLinear, shard_id: str,
                category: str, name: str | None = None) -> dict:
    """Write one linear layer into ``vault`` as a tensor shard."""
    tensor_name = name or shard_id
    payload = layer_payload(layer, tensor_name)
    vault.add_shard(shard_id=shard_id, category=category, payload=payload,
                    kind="ternary-tensor",
                    associations=library.associations_of(tensor_name))
    return payload


def load_layer(vault, shard_id: str) -> TernaryLinear:
    """Page one linear layer back out of ``vault``."""
    return layer_from_payload(vault.get(shard_id))


def store_net(vault, net: UltraQuantNet, prefix: str = "net",
              category: str | None = None) -> list[str]:
    """Write a whole network into the library, one shard per layer.

    One shard per LAYER for the same reason §11.96 uses one shard per
    tensor: a matmul needs the whole matrix, so a finer split pages
    every piece for every use and a coarser one pages the network to
    read a layer.

    Returns the shard ids written, in forward order.
    """
    group = category or prefix
    written: list[str] = []
    shape = {
        "input_dim": net.input_dim,
        "hidden_dims": list(net.hidden_dims),
        "num_classes": net.num_classes,
        "act_bits": net.act_bits,
        "quantize_head": bool(net.quantize_head),
        "layers": len(net.hidden) + 1,
    }
    vault.add_shard(shard_id=f"{prefix}:shape", category=group,
                    payload=shape, kind="ternary-tensor",
                    associations=library.associations_of(f"{prefix} shape"))
    written.append(f"{prefix}:shape")
    for index, layer in enumerate(net.hidden):
        shard_id = f"{prefix}:blk.{index}.weight"
        store_layer(vault, layer, shard_id, group, shard_id)
        written.append(shard_id)
    store_layer(vault, net.head, f"{prefix}:head.weight", group,
                f"{prefix}:head.weight")
    written.append(f"{prefix}:head.weight")
    return written


def load_net(vault, prefix: str = "net") -> UltraQuantNet:
    """Page a whole network back out of the library.

    The shape shard is read first, so the network is built at the
    right dimensions and every layer is then REPLACED rather than
    trained into - an imported network that quietly kept one seeded
    layer would still classify, and would be wrong for a reason no
    accuracy number could show.
    """
    shape = vault.get(f"{prefix}:shape")
    net = UltraQuantNet(int(shape["input_dim"]),
                        [int(h) for h in shape["hidden_dims"]],
                        int(shape["num_classes"]),
                        quantize_head=bool(shape.get("quantize_head", False)),
                        act_bits=int(shape.get("act_bits", 8)))
    for index in range(len(net.hidden)):
        net.hidden[index] = load_layer(vault, f"{prefix}:blk.{index}.weight")
    net.head = load_layer(vault, f"{prefix}:head.weight")
    return net


def is_idempotent(layer: TernaryLinear) -> bool:
    """Whether re-quantising this layer's weights changes them.

    The claim the exactness criterion rests on, checkable rather than
    argued: a matrix already in ``{-alpha, 0, +alpha}`` comes back
    from the quantiser unchanged, so an imported layer may be handed
    to the network still flagged quantized.
    """
    from ultraquant.model.quantize import dequantize, quantize_ternary

    before = [list(row) for row in layer.w]
    if not before or not before[0]:
        return True
    q, alpha = quantize_ternary(before)
    return dequantize(q, alpha) == before
