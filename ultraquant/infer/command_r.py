"""Command-R, from packed weights to the next token's actual logits.

The embedding is a row lookup. Each block normalises the residual stream
once, subtracting its mean as LayerNorm requires. That SAME normalised
vector feeds attention and the gated feed-forward: neither branch sees
the other's answer. The update is x + attention(h) + feed_forward(h).
Putting a second norm between those branches would make a perfectly
well-shaped, entirely different model.

Attention projects queries, keys and values, rotates queries and keys
at their absolute token positions, and keeps the rotated keys and the
unrotated values. Eight neighbouring query heads share one key/value
head in this checkpoint. A token enters the cache before it attends,
so it can see itself and everything before it, and nothing after it.
RoPE pairs ADJACENT coordinates within each head. This is measured on
the user's GGUF: on "The capital of France is", adjacent pairing gives
0.0354 nats maximum error over the CPU oracle's top five; pairing halves
gives 0.6863. The projection weights decide the layout, not a remembered
transformer diagram.

The other branch is down(silu(gate(h)) * up(h)). After the last block,
another bias-free LayerNorm feeds the token embedding matrix again:
there is no separate output matrix. The file's logit_scale is applied
once, to those vocabulary scores. These are logits, not probabilities.

**Weights stay packed.** Every matrix product goes through infer.matvec's
native CPU kernel. Prefill batches a small chunk of positions so a row
of weights can be read and decoded once for several vectors. Generation
then projects just the new token, reusing every layer's KV cache. Only
the small norm vectors and the cache persist as decoded Python floats.

**Where this differs from llama.cpp.** The packed weights decode to the
same float32 values, widened exactly, but activations are never quantised.
Native dots accumulate in ordered float64; norms, rotations, attention,
SiLU and residuals are float64 Python. llama.cpp's activation quantisation
and float32 arithmetic therefore make it a numerical comparison, not a
bit-equality oracle. This is a CPU reference forward pass, not a claim
that Python attention is practical at the file's full context length.
"""

from __future__ import annotations

import math
import operator
import random
from dataclasses import dataclass
from pathlib import Path

from ultraquant.convert import gguf
from ultraquant.infer.matvec import matvec, matvec_batch
from ultraquant.native import accel

__all__ = ["CommandR"]


@dataclass(frozen=True)
class _Config:
    blocks: int
    context: int
    width: int
    ffn: int
    heads: int
    kv_heads: int
    rope_base: float
    eps: float
    logit_scale: float

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @classmethod
    def read(cls, metadata: dict) -> _Config:
        if metadata.get("general.architecture") != "command-r":
            raise ValueError("CommandR requires a command-r GGUF")

        def need(key):
            return metadata["command-r." + key]

        if need("rope.scaling.type") != "none":
            raise ValueError("CommandR supports unscaled RoPE only")
        config = cls(
            blocks=operator.index(need("block_count")),
            context=operator.index(need("context_length")),
            width=operator.index(need("embedding_length")),
            ffn=operator.index(need("feed_forward_length")),
            heads=operator.index(need("attention.head_count")),
            kv_heads=operator.index(need("attention.head_count_kv")),
            rope_base=float(need("rope.freq_base")),
            eps=float(need("attention.layer_norm_epsilon")),
            logit_scale=float(need("logit_scale")),
        )
        if min(config.blocks, config.context, config.width, config.ffn,
               config.heads, config.kv_heads) <= 0:
            raise ValueError("CommandR dimensions must be positive")
        if (config.width % config.heads or config.heads % config.kv_heads
                or config.head_dim % 2):
            raise ValueError("CommandR head dimensions do not divide evenly")
        if any(not math.isfinite(x) or x <= 0 for x in
               (config.rope_base, config.eps, config.logit_scale)):
            raise ValueError("CommandR scales and epsilon must be finite and positive")
        if metadata.get("command-r.rope.dimension_count", config.head_dim) != config.head_dim:
            raise ValueError("CommandR requires RoPE over the whole head")
        return config


def _layer_norm(x: list[float], weight: list[float], eps: float) -> list[float]:
    """Population variance, learned gain, no bias; this is not RMSNorm."""
    mean = sum(x) / len(x)
    centred = [value - mean for value in x]
    scale = 1.0 / math.sqrt(sum(value * value for value in centred) / len(x) + eps)
    return [value * scale * gain for value, gain in zip(centred, weight)]


def _rope(vector: list[float], head_dim: int,
          rotations: list[tuple[float, float]]) -> None:
    """Rotate in place: coordinates 2*i and 2*i+1 form a complex pair."""
    for start in range(0, len(vector), head_dim):
        for i, (cosine, sine) in enumerate(rotations):
            left, right = start + 2 * i, start + 2 * i + 1
            a, b = vector[left], vector[right]
            vector[left] = a * cosine - b * sine
            vector[right] = a * sine + b * cosine


def _silu(value: float) -> float:
    # The negative branch avoids exp(-value) overflowing on a large negative.
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


class CommandR:
    """One checkpoint and one sequence's cache; instances are not thread-safe.

    ``logits(ids)`` starts a fresh sequence and returns the last position's
    scaled vocabulary logits. IDs are used verbatim: tokenisation and BOS
    insertion belong to the caller. ``generate(ids, n)`` also starts fresh
    and returns only the n NEW token IDs, keeping the prefix in its cache.
    It does not stop early at EOS. With greedy=False it samples the full
    softmax at temperature 1 using Python's random module.
    """

    _prefill_chunk = 16

    @classmethod
    def from_gguf(cls, path: str | Path) -> CommandR:
        """Read metadata and norm vectors; leave every matrix on disk."""
        return cls(gguf.read(path))

    def __init__(self, opened: gguf.GgufFile):
        self.gguf = opened
        self.config = c = _Config.read(opened.metadata)
        self._by = {info.name: info for info in opened.tensors}
        if len(self._by) != len(opened.tensors):
            raise ValueError("duplicate tensor names in CommandR checkpoint")
        self._embedding = self._by["token_embd.weight"]
        self.vocab_size = self._embedding.rows
        shapes = {
            "token_embd.weight": (c.width, self.vocab_size),
            "output_norm.weight": (c.width,),
        }
        kv_width = c.kv_heads * c.head_dim
        for layer in range(c.blocks):
            for name, shape in {
                "attn_norm": (c.width,),
                "attn_q": (c.width, c.width),
                "attn_k": (c.width, kv_width),
                "attn_v": (c.width, kv_width),
                "attn_output": (c.width, c.width),
                "ffn_gate": (c.width, c.ffn),
                "ffn_up": (c.width, c.ffn),
                "ffn_down": (c.ffn, c.width),
            }.items():
                shapes[f"blk.{layer}.{name}.weight"] = shape
        if self._by.keys() != shapes.keys():
            raise ValueError("CommandR requires bias-free parallel blocks and a tied head; "
                             f"missing={sorted(shapes.keys() - self._by.keys())}, "
                             f"extra={sorted(self._by.keys() - shapes.keys())}")
        if self.vocab_size <= 0:
            raise ValueError("CommandR vocabulary must be nonempty")
        for name, shape in shapes.items():
            info = self._by[name]
            if info.dims != shape:
                raise ValueError(f"{name} has shape {info.dims}, expected {shape}")
            if len(shape) == 2:
                if info.type_name not in {"Q4_K", "Q5_K", "Q6_K"} or info.columns % 256:
                    raise ValueError(f"{name} requires native Q4_K/Q5_K/Q6_K rows")
            elif not info.readable:
                raise ValueError(f"unreadable norm tensor {name}: {info.type_name}")
        dll = accel.load_cpu()
        if dll is None or not hasattr(dll, "uq_kq_matvec"):
            raise RuntimeError("CommandR requires the native CPU k-quant matvec kernel")
        self._norms = [opened.rows_of(self._by[f"blk.{i}.attn_norm.weight"])[0]
                       for i in range(c.blocks)]
        self._output_norm = opened.rows_of(self._by["output_norm.weight"])[0]
        self._frequencies = [c.rope_base ** (-2.0 * i / c.head_dim)
                             for i in range(c.head_dim // 2)]
        self.reset()

    def reset(self) -> None:
        """Drop the current sequence's keys and values, retaining the checkpoint."""
        self._keys: list[list] = [[] for _ in range(self.config.blocks)]
        self._values: list[list] = [[] for _ in range(self.config.blocks)]
        self._position = 0

    def _ids(self, ids: list[int]) -> list[int]:
        tokens = [operator.index(token) for token in ids]
        if not tokens:
            raise ValueError("CommandR needs at least one prompt token")
        if any(token < 0 or token >= self.vocab_size for token in tokens):
            raise ValueError("token ID is outside the checkpoint vocabulary")
        if len(tokens) > self.config.context:
            raise ValueError("prompt exceeds the checkpoint context length")
        return tokens

    def _project(self, name: str, vectors: list[list[float]]) -> list[list[float]]:
        info = self._by[name]
        if len(vectors) == 1:
            return [matvec(self.gguf, info, vectors[0])]
        return matvec_batch(self.gguf, info, vectors)

    def _attention(self, query: list[float], keys: list, values: list) -> list[float]:
        c = self.config
        dim = c.head_dim
        group = c.heads // c.kv_heads
        scale = 1.0 / math.sqrt(dim)
        result = []
        for head in range(c.heads):
            q = query[head * dim:(head + 1) * dim]
            kv_head = head // group
            scores = [sum(a * b for a, b in zip(q, key[kv_head])) * scale
                      for key in keys]
            peak = max(scores)
            probabilities = [math.exp(score - peak) for score in scores]
            total = sum(probabilities)
            probabilities = [value / total for value in probabilities]
            result.extend(sum(p * value[kv_head][i]
                              for p, value in zip(probabilities, values))
                          for i in range(dim))
        return result

    def _append(self, ids: list[int]) -> list[float]:
        """Append a nonempty chunk; return its last residual, before the head."""
        c = self.config
        if not ids or self._position + len(ids) > c.context:
            raise ValueError("empty chunk or checkpoint context length exceeded")
        stream = [self.gguf.rows_of(self._embedding, first=token, count=1)[0]
                  for token in ids]
        rotations = [[(math.cos(position * freq), math.sin(position * freq))
                      for freq in self._frequencies]
                     for position in range(self._position, self._position + len(ids))]
        try:
            for layer in range(c.blocks):
                tag = f"blk.{layer}."
                h = [_layer_norm(x, self._norms[layer], c.eps) for x in stream]
                queries = self._project(tag + "attn_q.weight", h)
                keys = self._project(tag + "attn_k.weight", h)
                values = self._project(tag + "attn_v.weight", h)
                attended = []
                for q, k, v, angles in zip(queries, keys, values, rotations):
                    _rope(q, c.head_dim, angles)
                    _rope(k, c.head_dim, angles)
                    self._keys[layer].append([k[i:i + c.head_dim]
                                             for i in range(0, len(k), c.head_dim)])
                    self._values[layer].append([v[i:i + c.head_dim]
                                               for i in range(0, len(v), c.head_dim)])
                    attended.append(self._attention(q, self._keys[layer],
                                                     self._values[layer]))
                attention = self._project(tag + "attn_output.weight", attended)
                gate = self._project(tag + "ffn_gate.weight", h)
                up = self._project(tag + "ffn_up.weight", h)
                hidden = [[_silu(g) * u for g, u in zip(gs, us)]
                          for gs, us in zip(gate, up)]
                feed_forward = self._project(tag + "ffn_down.weight", hidden)
                stream = [[x + a + f for x, a, f in zip(xs, ats, fs)]
                          for xs, ats, fs in zip(stream, attention, feed_forward)]
        except BaseException:
            # An interrupted layer must not leave different cache lengths.
            self.reset()
            raise
        self._position += len(ids)
        return stream[-1]

    def _head(self, residual: list[float]) -> list[float]:
        h = _layer_norm(residual, self._output_norm, self.config.eps)
        return [value * self.config.logit_scale
                for value in matvec(self.gguf, self._embedding, h)]

    def logits(self, ids: list[int]) -> list[float]:
        """Prefill a fresh prompt; return its last position's scaled logits."""
        tokens = self._ids(ids)
        self.reset()
        for start in range(0, len(tokens), self._prefill_chunk):
            residual = self._append(tokens[start:start + self._prefill_chunk])
        return self._head(residual)

    def generate(self, ids: list[int], n: int, greedy: bool = True) -> list[int]:
        """Return n new IDs, prefill once, then append one token per step.

        The cache ends at the last token actually evaluated: the final
        returned ID has been selected, but needs no forward pass of its own.
        No EOS stopping, temperature adjustment, or text decoding is implicit.
        """
        tokens = self._ids(ids)
        n = operator.index(n)
        if n < 0:
            raise ValueError("number of generated tokens must be nonnegative")
        if len(tokens) + max(0, n - 1) > self.config.context:
            raise ValueError("generation exceeds the checkpoint context length")
        if n == 0:
            self.reset()
            return []
        scores = self.logits(tokens)
        generated = []
        for step in range(n):
            if greedy:
                token = max(range(self.vocab_size), key=scores.__getitem__)
            else:
                peak = max(scores)
                weights = [math.exp(score - peak) for score in scores]
                token = random.choices(range(self.vocab_size), weights=weights, k=1)[0]
            generated.append(token)
            if step + 1 < n:
                scores = self._head(self._append([token]))
        return generated
