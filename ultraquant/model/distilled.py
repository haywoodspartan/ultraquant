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

__all__ = ["DistilledEncoder", "random_projection", "project"]


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
        h = [math.tanh(sum(w * v for w, v in zip(row, x)) + b)
             for row, b in zip(self.w1, self.b1)]
        return [sum(w * v for w, v in zip(row, h)) + b
                for row, b in zip(self.w2, self.b2)]

    def _step(self, x: list, target: list, lr: float) -> float:
        """One SGD step of mean squared error against the teacher."""
        pre = [sum(w * v for w, v in zip(row, x)) + b
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
            for i, v in enumerate(x):
                if v:
                    row[i] -= lr * g * v
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
        loss = 0.0
        for _ in range(epochs):
            rng.shuffle(order)
            loss = sum(self._step(xs[i], targets[i], lr)
                       for i in order) / len(order)
        return loss
