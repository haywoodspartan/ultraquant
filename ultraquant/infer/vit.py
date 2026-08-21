"""The vision encoder, assembled from `ops` and a real checkpoint.

27 blocks, 1152 wide, 16 heads - a SigLIP-shaped tower with a
qwen3vl_merger projector on the end. Every hyperparameter is read
from the file rather than defaulted, because the one that would have
been wrong by default is `layer_norm_epsilon`: this checkpoint says
1e-6 and `ops.layer_norm` defaults to 1e-5. That is precisely the
silent, everywhere, plausible-looking error §11.100 was written to
prevent, and it would have been introduced by trusting a default one
line after building the machinery to avoid it.

**Streaming, because the weights do not fit.** One block is 15.2M
weights; twenty-seven is 402M, which as Python lists is over ten
gigabytes. So a block's weights are read, used, and dropped, and the
residual stream is the only thing that persists. That is not a
concession - it is how the shard library was always meant to be
read, one unit of work at a time.

**What this does NOT do.** It does not decode an image file: the
standard library has no JPEG decoder, and inventing one is a
different project. `patch_embed` takes pixel planes as floats, which
anybody holding a real photo can produce and which a test can
generate. Everything downstream of that is the real encoder on real
weights.

**What cannot be claimed.** There is no reference implementation on
this machine that exposes intermediate activations - LM Studio ships
`llama-server.exe`, which returns text, not the residual stream. So
nothing here is verified against llama.cpp's numbers. What IS
verified is stated in the gate: that the wiring is sensitive to
being wrong, that the invariants a transformer must satisfy hold,
and - the question this was built to answer - how far a ternary
encoder drifts from the same encoder in f16, block by block, where
the f16 tower is its own reference and no third party is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ultraquant.infer import ops

__all__ = ["VisionConfig", "PatchEmbedding", "BlockWeights",
           "load_config", "load_patch_embedding", "load_block",
           "load_tail", "patch_embed", "apply_block", "apply_tail",
           "encode", "normalise"]


@dataclass
class VisionConfig:
    """Every number the encoder needs, all of them from the file."""

    blocks: int
    width: int
    heads: int
    ffn: int
    eps: float
    patch: int
    image_size: int
    merge: int
    use_gelu: bool
    projector: str

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def grid(self) -> int:
        """Patches per side: 768 / 16 = 48, so 2304 positions."""
        return self.image_size // self.patch


def load_config(metadata: dict) -> VisionConfig:
    """Read the encoder's shape out of the checkpoint's metadata.

    Every field is required. A missing key raises rather than
    defaulting, because a default that silently disagrees with the
    file is the failure mode this whole module is arranged against.
    """
    def need(key: str):
        if key not in metadata:
            raise KeyError(f"{key} is not in this checkpoint; refusing to "
                           "guess a value the weights were trained with")
        return metadata[key]

    return VisionConfig(
        blocks=int(need("clip.vision.block_count")),
        width=int(need("clip.vision.embedding_length")),
        heads=int(need("clip.vision.attention.head_count")),
        ffn=int(need("clip.vision.feed_forward_length")),
        eps=float(need("clip.vision.attention.layer_norm_epsilon")),
        patch=int(need("clip.vision.patch_size")),
        image_size=int(need("clip.vision.image_size")),
        merge=int(need("clip.vision.spatial_merge_size")),
        use_gelu=bool(need("clip.use_gelu")),
        projector=str(need("clip.projector_type")),
    )


@dataclass
class PatchEmbedding:
    """The conv that turns pixels into tokens, plus its position table.

    `kernels` is per output channel, flattened as GGUF stores it:
    channel-major over ``[in_channel][ky][kx]``. There are TWO of
    them because this checkpoint patches over two frames - a video
    model reading a still image sees the same frame twice, so both
    kernels apply and their outputs sum.
    """

    kernels: list
    bias: list
    positions: list


@dataclass
class BlockWeights:
    """One transformer block, loaded and about to be dropped."""

    ln1_w: list
    ln1_b: list
    qkv_w: list
    qkv_b: list
    out_w: list
    out_b: list
    ln2_w: list
    ln2_b: list
    up_w: list
    up_b: list
    down_w: list
    down_b: list


def _vector(opened, by: dict, name: str) -> list:
    return opened.rows_of(by[name], 0, by[name].rows)[0]


def _matrix(opened, by: dict, name: str) -> list:
    info = by[name]
    return opened.rows_of(info, 0, info.rows)


def _all_rows(opened, info) -> list:
    """Every row of a tensor, including tensors with more than two
    dimensions.

    `TensorInfo.rows` is `dims[1]`, which is the output count for a
    matrix and is NOT the row count for the 4-D patch kernel - there
    it is 16, the second kernel axis, so asking for `info.rows` rows
    returns 256 of 884,736 values. The size check in
    :func:`load_patch_embedding` caught exactly that on the first
    run; without it the kernel would have been silently built from a
    256-value fragment, and the encoder would have run.
    """
    width = info.columns
    return opened.rows_of(info, 0, info.count // width if width else 0)


def load_patch_embedding(opened, by: dict,
                         config: VisionConfig) -> PatchEmbedding:
    """The patch conv and the learned position table.

    The kernel arrives as a 4-D tensor that the row reader flattens,
    so it is regrouped here into one contiguous kernel per output
    channel. Getting that regrouping wrong would scramble every
    patch identically and produce an encoder that runs, stays
    finite, and means nothing - so the sizes are checked rather than
    assumed.
    """
    per_channel = 3 * config.patch * config.patch
    frames = []
    for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        if name not in by:
            continue
        info = by[name]
        flat = [value for row in _all_rows(opened, info) for value in row]
        if len(flat) != config.width * per_channel:
            raise ValueError(
                f"{name} has {len(flat)} values, expected "
                f"{config.width * per_channel} for {config.width} channels "
                f"of {per_channel}")
        frames.append([flat[i * per_channel:(i + 1) * per_channel]
                       for i in range(config.width)])
    if not frames:
        raise KeyError("no patch embedding in this checkpoint")
    return PatchEmbedding(
        kernels=frames,
        bias=_vector(opened, by, "v.patch_embd.bias"),
        positions=_matrix(opened, by, "v.position_embd.weight"))


def load_block(opened, by: dict, index: int) -> BlockWeights:
    """Read one block. The caller is expected to drop it after use."""
    tag = f"v.blk.{index}"
    return BlockWeights(
        ln1_w=_vector(opened, by, f"{tag}.ln1.weight"),
        ln1_b=_vector(opened, by, f"{tag}.ln1.bias"),
        qkv_w=_matrix(opened, by, f"{tag}.attn_qkv.weight"),
        qkv_b=_vector(opened, by, f"{tag}.attn_qkv.bias"),
        out_w=_matrix(opened, by, f"{tag}.attn_out.weight"),
        out_b=_vector(opened, by, f"{tag}.attn_out.bias"),
        ln2_w=_vector(opened, by, f"{tag}.ln2.weight"),
        ln2_b=_vector(opened, by, f"{tag}.ln2.bias"),
        up_w=_matrix(opened, by, f"{tag}.ffn_up.weight"),
        up_b=_vector(opened, by, f"{tag}.ffn_up.bias"),
        down_w=_matrix(opened, by, f"{tag}.ffn_down.weight"),
        down_b=_vector(opened, by, f"{tag}.ffn_down.bias"))


def load_tail(opened, by: dict) -> dict:
    """The final norm and the two projector layers."""
    tail = {"post_ln_w": _vector(opened, by, "v.post_ln.weight"),
            "post_ln_b": _vector(opened, by, "v.post_ln.bias")}
    for index in (0, 2):
        tail[f"mm{index}_w"] = _matrix(opened, by, f"mm.{index}.weight")
        tail[f"mm{index}_b"] = _vector(opened, by, f"mm.{index}.bias")
    return tail


def patch_embed(planes: list, embedding: PatchEmbedding,
                config: VisionConfig, patches: int | None = None) -> list:
    """Pixels to tokens: the conv, the bias, and the position table.

    Args:
        planes: Frames, each ``[3][H][W]`` of floats already
            normalised the way the checkpoint asks - this model uses
            mean 0.5 and std 0.5 per channel, so ``(x/255 - 0.5) /
            0.5``. Passing raw 0-255 pixels would run, and would be
            wrong by a factor nobody could see downstream.
        embedding: The loaded kernels, bias and positions.
        config: The encoder's shape.
        patches: Stop after this many patches. Present because a
            full 48x48 grid is 2304 tokens and pure Python is not
            the tool for that; the measurement this encoder was
            built for is about DEPTH, and depth is unaffected.

    Returns:
        ``[n_patches][width]`` with position embeddings already added.
    """
    size = config.patch
    grid = config.grid
    wanted = grid * grid if patches is None else min(patches, grid * grid)
    tokens = []
    for index in range(wanted):
        row, column = divmod(index, grid)
        window = []
        for plane in planes[0]:
            for y in range(row * size, row * size + size):
                window.extend(plane[y][column * size:column * size + size])
        value = [sum(k * v for k, v in zip(kernel, window))
                 for kernel in embedding.kernels[0]]
        if len(embedding.kernels) > 1 and len(planes) > 1:
            second = []
            for plane in planes[1]:
                for y in range(row * size, row * size + size):
                    second.extend(plane[y][column * size:column * size + size])
            other = [sum(k * v for k, v in zip(kernel, second))
                     for kernel in embedding.kernels[1]]
            value = ops.add(value, other)
        value = ops.add(value, embedding.bias)
        tokens.append(ops.add(value, embedding.positions[index]))
    return tokens


def apply_block(stream: list, weights: BlockWeights,
                config: VisionConfig) -> list:
    """One pre-norm block: attention, then the feed-forward.

    Pre-norm - `ln1` before attention and `ln2` before the MLP, with
    both residuals bypassing their norm - because that is what a
    SigLIP-shaped tower is and what these tensor names mean. It is
    an ASSUMPTION about this checkpoint, not something measurable
    without a reference, and the gate says so rather than letting
    the code imply otherwise.
    """
    width = config.width
    normed = [ops.layer_norm(token, weights.ln1_w, weights.ln1_b, config.eps)
              for token in stream]
    fused = [ops.linear(weights.qkv_w, weights.qkv_b, token)
             for token in normed]
    queries = [token[:width] for token in fused]
    keys = [token[width:2 * width] for token in fused]
    values = [token[2 * width:] for token in fused]
    attended = ops.multi_head_attention(queries, keys, values, config.heads)
    attended = [ops.linear(weights.out_w, weights.out_b, token)
                for token in attended]
    stream = [ops.add(token, delta) for token, delta in zip(stream, attended)]

    normed = [ops.layer_norm(token, weights.ln2_w, weights.ln2_b, config.eps)
              for token in stream]
    activate = ops.gelu if config.use_gelu else ops.silu
    hidden = [activate(ops.linear(weights.up_w, weights.up_b, token))
              for token in normed]
    hidden = [ops.linear(weights.down_w, weights.down_b, token)
              for token in hidden]
    return [ops.add(token, delta) for token, delta in zip(stream, hidden)]


def apply_tail(stream: list, tail: dict, config: VisionConfig) -> list:
    """Final norm, the 2x2 spatial merge, and the projector.

    The merger is what `qwen3vl_merger` means: `spatial_merge_size`
    is 2, so four neighbouring patches are concatenated into one
    4x1152 vector before the two projection layers - which is why
    `mm.0` is 4608 wide and not 1152. Merging the wrong four patches
    would leave every shape correct.
    """
    normed = [ops.layer_norm(token, tail["post_ln_w"], tail["post_ln_b"],
                             config.eps) for token in stream]
    span = config.merge * config.merge
    merged = [
        [value for token in normed[start:start + span] for value in token]
        for start in range(0, len(normed) - span + 1, span)]
    out = []
    for token in merged:
        hidden = ops.gelu(ops.linear(tail["mm0_w"], tail["mm0_b"], token))
        out.append(ops.linear(tail["mm2_w"], tail["mm2_b"], hidden))
    return out


def encode(opened, planes: list, patches: int | None = None,
           blocks: int | None = None, observer=None) -> list:
    """The whole encoder, one block of weights resident at a time.

    Args:
        opened: An open :class:`~ultraquant.convert.gguf.GgufFile`.
        planes: Frames of normalised pixel planes.
        patches: Stop after this many patches.
        blocks: Run only the first N blocks (all of them if None).
        observer: Called as ``observer(index, weights, stream)`` after
            each block. This is how the gate walks a second tower
            alongside without reading every weight twice - the file
            is 857 MB and a block is 15.2M values.

    Returns:
        The projected tokens.
    """
    by = {tensor.name: tensor for tensor in opened.tensors}
    config = load_config(opened.metadata)
    embedding = load_patch_embedding(opened, by, config)
    stream = patch_embed(planes, embedding, config, patches)
    del embedding
    count = config.blocks if blocks is None else min(blocks, config.blocks)
    for index in range(count):
        weights = load_block(opened, by, index)
        stream = apply_block(stream, weights, config)
        if observer is not None:
            observer(index, weights, stream)
        del weights
    return apply_tail(stream, load_tail(opened, by), config)


def normalise(planes: list, mean: float = 0.5, std: float = 0.5) -> list:
    """Raw 0-255 pixels to what the checkpoint was trained on.

    Separate and named because skipping it is the easiest possible
    mistake: the encoder runs happily on 0-255 input and every
    activation is then roughly 250x too large, which no shape check
    and no finiteness check will notice.
    """
    return [[[[(value / 255.0 - mean) / std for value in row]
              for row in plane] for plane in frame] for frame in planes]
