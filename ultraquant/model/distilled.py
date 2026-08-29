"""A shared representation this system owns, distilled from one it borrowed.

§11.109 met Stage 1's criterion — a shared representation buys
few-shot transfer in text, and the transfer is semantic — with a
representation that is **not ours**: `nomic-embed-text-v1.5`,
pretrained and reached over the network on every query.

Two things follow, and only one of them is achievable here.

**Not achievable: learning semantics from nothing.** The teacher's
geometry comes from pretraining on an enormous corpus. This library
has no such corpus — its built-in keyword vocabulary is 25 words, and
the §11.109 test corpus is 79. An encoder trained only on that can
learn the relationships among those words and has nothing whatever to
say about any other. Claiming otherwise would be claiming a lookup
table is a learner, which is exactly the distinction §11.2 drew.

**Achievable, and the one §11.13 already made for the router:** pay
the teacher once, offline, and own the result. §11.13 distilled a
transformer's paraphrase knowledge into the router's associative
weights so that query time stayed where it was. This does the same
for the *geometry* rather than for word pairs: a small from-scratch
network learns to place text where the teacher would place it, and
afterwards **nothing touches the network**.

**What is distilled is structure, not vectors.** The teacher returns
768 dimensions; reproducing them in pure Python would be slow and
pointless, since what the downstream classifier uses is *relative
position*. The teacher's vectors are projected once through a fixed
seeded random projection to `dim` dimensions — the Johnson-
Lindenstrauss argument the library already relies on in
`shards/sketch.py` — and the encoder regresses those. Cosine
structure survives the projection approximately; exact coordinates
were never the point.

**The boundary is real and is measured rather than hidden.** An
encoder over a bag-of-words input can only respond to words it has an
input for. On text containing words it never saw, it has no signal at
all, and the gate beside this module reports what that costs instead
of choosing a corpus where it never happens.
"""

from __future__ import annotations

import math
import random

__all__ = ["DistilledEncoder", "DistilledEmbedder", "random_projection",
           "project"]


def random_projection(source: int, target: int, seed: int = 0) -> list:
    """A fixed seeded projection from `source` to `target` dimensions.

    Deterministic, so an encoder distilled today still agrees with a
    teacher vector projected next month - the same property
    `shards/sketch.py` needs and for the same reason.
    """
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(target)
    return [[rng.gauss(0.0, 1.0) * scale for _ in range(source)]
            for _ in range(target)]


def project(vector: list, planes: list) -> list:
    """Push one teacher vector through the projection."""
    return [sum(p * v for p, v in zip(plane, vector)) for plane in planes]


#: Projection width. Chosen by measurement twice over. Across 630 real
#: teacher pairs the projected cosine's correlation with the true one
#: is 0.374 at 16 dimensions, 0.548 at 32, 0.751 at 64, 0.875 at 128
#: and **0.927 at 256**. Thirty-two would have been the reflex choice
#: and would have discarded nearly half the geometry before the
#: encoder saw any of it.
#:
#: 128 was the first choice and the gate then measured what it cost
#: downstream: the distilled copy kept **42%** of the teacher's
#: advantage at 128 and **58%** at 256, with the semantic control
#: going from uninterpretable to passing. The projection, not the
#: encoder, was the binding constraint - the encoder was already
#: reproducing what it was given at 0.988.
DEFAULT_DIM = 256


class DistilledEncoder:
    """Bag-of-words in, the teacher's geometry out. No network.

    A two-layer network with a tanh hidden layer and a linear head,
    trained by plain SGD to regress the projected teacher vector.
    Deliberately small: the training set is the library's own text,
    which is tens of items, and a larger network would memorise it
    without learning anything transferable.

    Everything here is stdlib and everything runs offline. That is
    the whole point of the exercise - §11.109's advantage came from a
    network round trip on every query, and this is that advantage
    paid for once.
    """

    def __init__(self, input_dim: int, dim: int = DEFAULT_DIM,
                 hidden: int = 64, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.input_dim = input_dim
        self.dim = dim
        self.hidden = hidden
        limit_in = math.sqrt(6.0 / (input_dim + hidden))
        limit_out = math.sqrt(6.0 / (hidden + dim))
        self.w1 = [[rng.uniform(-limit_in, limit_in)
                    for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-limit_out, limit_out)
                    for _ in range(hidden)] for _ in range(dim)]
        self.b2 = [0.0] * dim

    def encode(self, x: list) -> list:
        """The shared representation of one input vector."""
        active = [i for i, v in enumerate(x) if v]
        h = [math.tanh(sum(row[i] * x[i] for i in active) + b)
             for row, b in zip(self.w1, self.b1)]
        return [sum(w * v for w, v in zip(row, h)) + b
                for row, b in zip(self.w2, self.b2)]

    def _step(self, x: list, target: list, lr: float,
              active: list | None = None) -> float:
        """One SGD step of mean squared error against the teacher.

        `active` is the indices of the non-zero inputs, and passing it
        is not an optimisation in the ordinary sense - it is what
        makes a wide vocabulary usable at all. Bag-of-words is
        extremely sparse: a sentence has perhaps eight content words
        against a vocabulary of three thousand. The forward and
        backward passes originally walked the whole width regardless,
        which is a 375x waste at that size and turned a fifty-minute
        measurement into a projected three-hour one before it was
        noticed.
        """
        if active is None:
            active = [i for i, v in enumerate(x) if v]
        pre = [sum(row[i] * x[i] for i in active) + b
               for row, b in zip(self.w1, self.b1)]
        h = [math.tanh(v) for v in pre]
        out = [sum(w * v for w, v in zip(row, h)) + b
               for row, b in zip(self.w2, self.b2)]
        error = [o - t for o, t in zip(out, target)]
        loss = sum(e * e for e in error) / len(error)

        grad_h = [0.0] * self.hidden
        scale = 2.0 / len(error)
        for index, e in enumerate(error):
            g = scale * e
            row = self.w2[index]
            for j in range(self.hidden):
                grad_h[j] += g * row[j]
                row[j] -= lr * g * h[j]
            self.b2[index] -= lr * g
        for j in range(self.hidden):
            g = grad_h[j] * (1.0 - h[j] * h[j])
            row = self.w1[j]
            for i in active:
                row[i] -= lr * g * x[i]
            self.b1[j] -= lr * g
        return loss

    def fit(self, xs: list, targets: list, epochs: int = 200,
            lr: float = 0.05, seed: int = 0) -> float:
        """Distil, and return the final mean loss.

        No labels are involved - only text and where the teacher put
        it - so this can run over every string the library holds,
        which is the property that would let it scale past the corpus
        it was measured on.
        """
        rng = random.Random(seed)
        order = list(range(len(xs)))
        # Computed once, not once per epoch: the sparsity pattern of
        # an input does not change while it is being learned from.
        active = [[i for i, v in enumerate(x) if v] for x in xs]
        loss = 0.0
        for _ in range(epochs):
            rng.shuffle(order)
            loss = sum(self._step(xs[i], targets[i], lr, active[i])
                       for i in order) / len(order)
        return loss


class DistilledEmbedder:
    """A `DistilledEncoder` behind the interface the suggester expects.

    §11.108 to §11.111 established that a shared representation is
    real, that its edge is semantic, that 58% of it survives
    distillation, and that the vocabulary ceiling comes off with the
    repository's own prose. **None of that was used by anything.**
    `reason/semantic.py` still reached LM Studio on every question it
    could not answer lexically.

    This is the adapter that closes that gap: `embed(texts)` returns
    one vector per input, exactly as `LMStudioClient.embed` does, and
    touches nothing. The suggester cannot tell which one it has.

    **What the caller must know.** Cosines here live in the projected
    teacher space, not the teacher's own, so a threshold measured for
    the live embedding does not transfer unexamined - and the gate
    that adopts this has to fit its own floor rather than inherit
    0.75. Words outside the distillation vocabulary contribute
    nothing, which is §11.111's boundary and is why the corpus
    matters.
    """

    def __init__(self, encoder: "DistilledEncoder", vocabulary: list) -> None:
        self.encoder = encoder
        self.vocabulary = list(vocabulary)
        self._index = {word: i for i, word in enumerate(self.vocabulary)}

    def _bag(self, text: str) -> list:
        from ultraquant.experiments.embed_gate import _tokens

        vector = [0.0] * len(self.vocabulary)
        for token in _tokens(text):
            at = self._index.get(token)
            if at is not None:
                vector[at] = 1.0
        return vector

    def embed(self, texts, model=None) -> list:
        """One vector per input, in input order. No network.

        `model` is accepted and ignored so this is a drop-in for the
        LM Studio client's signature; there is only ever one encoder.
        """
        if isinstance(texts, str):
            texts = [texts]
        return [self.encoder.encode(self._bag(text)) for text in texts]

    def available(self) -> bool:
        """Always. That is the point of having distilled it."""
        return True

    def state_dict(self) -> dict:
        """A JSON-safe snapshot, so distillation is paid once ever."""
        return {
            "vocabulary": list(self.vocabulary),
            "dim": self.encoder.dim,
            "hidden": self.encoder.hidden,
            "w1": [list(row) for row in self.encoder.w1],
            "b1": list(self.encoder.b1),
            "w2": [list(row) for row in self.encoder.w2],
            "b2": list(self.encoder.b2),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DistilledEmbedder":
        vocabulary = list(state["vocabulary"])
        encoder = DistilledEncoder(len(vocabulary), dim=int(state["dim"]),
                                   hidden=int(state["hidden"]))
        encoder.w1 = [[float(v) for v in row] for row in state["w1"]]
        encoder.b1 = [float(v) for v in state["b1"]]
        encoder.w2 = [[float(v) for v in row] for row in state["w2"]]
        encoder.b2 = [float(v) for v in state["b2"]]
        return cls(encoder, vocabulary)

    @classmethod
    def distil(cls, texts: list, teacher: dict, dim: int = DEFAULT_DIM,
               hidden: int = 64, seed: int = 0,
               epochs: int = 4000) -> "DistilledEmbedder":
        """Learn to place `texts` where `teacher` puts them.

        Args:
            texts: Every string worth knowing the geometry of - the
                library's own keys and whatever prose is available.
            teacher: text -> the teacher's vector for it.
            epochs: 4000, because §11.110 measured fidelity at 0.683
                after 400 and 0.988 after 4000, and an unfinished copy
                reads as a failed method.
        """
        from ultraquant.experiments.embed_gate import _tokens

        words = set()
        for text in texts:
            words |= _tokens(text)
        vocabulary = sorted(words)
        planes = random_projection(len(teacher[texts[0]]), dim, seed=0)
        embedder = cls(DistilledEncoder(len(vocabulary), dim=dim,
                                        hidden=hidden, seed=seed),
                       vocabulary)
        xs = [embedder._bag(text) for text in texts]
        targets = [project(teacher[text], planes) for text in texts]
        embedder.encoder.fit(xs, targets, epochs=epochs, seed=seed)
        return embedder
