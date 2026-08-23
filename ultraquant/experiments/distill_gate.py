"""Owning the representation that was borrowed. The distillation gate.

§11.109 met Stage 1's criterion and named exactly what was still
missing: the representation is `nomic-embed-text-v1.5`, pretrained and
reached **over the network on every query**. The criterion was met
with a borrowed representation; the encoder was not built.

**What is not attempted, said first.** Learning that geometry from
nothing. The teacher's semantics come from pretraining on an enormous
corpus; this library's built-in keyword vocabulary is 25 words and the
§11.109 corpus is 79. An encoder trained on that can learn the
relationships among those words and has nothing to say about any
other. §11.5 and §11.11 tried to learn a shared representation from
the data alone and did not succeed, and nothing here contradicts
them.

**What is attempted is the trade §11.13 already made for the
router**: pay the teacher once, offline, and own the result. §11.13
distilled a transformer's paraphrase knowledge into the router's
associative weights so query time stayed where it was. This distils
the *geometry* instead — a small from-scratch network learns where
the teacher puts text, and afterwards nothing touches the network.

**The criteria, written before the run.**

1. **Nothing touches the network at query time** — once distilled,
   encoding must not call LM Studio. Asserted by running the whole
   evaluation with the client removed, not by inspection.
2. **It keeps at least half the borrowed advantage** — §11.109
   measured the pretrained arm at +0.167 over bag-of-words. The
   distilled arm must clear **half of whatever the pretrained arm
   scores in this same run**, by more than the seed sd. Half is a
   bar chosen in advance and is deliberately generous: a distilled
   copy that keeps most of a teacher is a good outcome, and one that
   keeps almost none is the honest failure.
3. **The advantage is still semantic** — the same vocabulary
   permutation, with the teacher and the encoder both re-run inside
   the permuted condition. If the distilled arm's edge survives
   losing the meaning, it is capacity.
4. **The vocabulary boundary is measured, not avoided** — accuracy
   on text containing words the encoder never saw during
   distillation, reported beside the rest. This is the ceiling that
   makes the "not attempted" paragraph above true, and choosing a
   corpus where it never arises would be hiding the one number that
   bounds the whole approach. Reported, not gated.
5. **Both costs are named** — distillation seconds once, and encode
   microseconds per query against the teacher's network round trip.

**FAILED criterion 2 — at 58%, against a bar that turned out to
demand 86%. The trajectory is the result, and both things that were
holding it back were mine.**

| run | epochs | projection | kept of the borrowed margin | semantic? |
|---:|---:|---:|---:|:--|
| 1 | 400 | 128 | **-12%** | inside noise |
| 2 | 4000 | 128 | **+42%** | inside noise |
| 3 | 4000 | 256 | **+58%** | **+0.062 at sd 0.041 — yes** |

**Run one looked like a refutation and was an unfinished copy.** The
distilled encoder scored *below* bag-of-words, which reads as
"distillation does not transfer this". The diagnosis was to ask
whether the copy had been made at all: correlation between the
encoder's pairwise cosines and the teacher's was **0.683** at 400
epochs, 0.921 at 1500 and **0.988** at 4000. It had two thirds of the
geometry and the gate read that as the method failing.

**Run two then showed the second constraint was the projection, not
the encoder.** At 0.988 fidelity to what it was *given*, the copy
still kept only 42% - so the loss was upstream, in the 768-to-128
projection that keeps 0.875 of the structure. At 256 dimensions
(0.927) the kept fraction rose to 58% and **the semantic control went
from uninterpretable to passing**. The encoder was never the binding
constraint.

**Run three, four seeds:**

| | real | permuted |
|---|---:|---:|
| bag-of-words | 0.312 | 0.312 |
| distilled (ours) | **0.417** | 0.354 |
| borrowed (network) | 0.493 | - |

**Criterion 3 passes**: the distilled advantage does not survive the
permutation (+0.062 at sd 0.041), so what was copied is the meaning
and not the capacity.

**Criterion 2 fails, and the bar deserves comment.** It was written
as "must clear half of whatever the pretrained arm scores, by more
than the seed sd" and implemented exactly that way -
`0.5 x 0.181 + 0.066 = 0.157`, which as a share of the borrowed
margin is **86%**, not half. The wording and the implementation
agree; it is the *reading* of "at least half" that was loose. **58%
is the result and it is a fail against the bar as registered**, and
the lesson is about writing criteria rather than about distillation:
an sd term added to a fractional bar changes what the fraction means.

**Criterion 4, the boundary, is real and is the ceiling on all of
this.** On text using any of 19 words withheld from distillation the
encoder scores **0.350**, against **0.500** on text that avoids them.
The encoder knows the geometry of words it was shown and has nothing
whatever for the rest - which is precisely why the opening paragraph
refuses to claim semantics were learned rather than imported.

**Criterion 5, the cost:** 248 seconds once, then **622 microseconds
per encode against the teacher's 8.9 milliseconds** - fourteen times
faster, with the network unplugged for the whole evaluation.

**What this leaves.** 58% of a borrowed advantage, owned outright and
running offline, with the remaining 42% attributable to a projection
that can be widened and a vocabulary that cannot be without a larger
corpus. Two apparent refutations turned out to be a training budget
and a projection width. Neither was a wall.

Run it::

    python -m ultraquant.experiments.distill_gate

Requires LM Studio with an embedding model for the distillation step
only. Without one the gate reports that it did not run.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from ultraquant.experiments.embed_gate import CATEGORIES, _tokens
from ultraquant.experiments.fewshot_gate import (_fold_accuracy, _vocabulary,
                                                 bag_of_words, items_of)
from ultraquant.experiments.permutation_gate import (build_permutation,
                                                     permute,
                                                     permuted_categories)
from ultraquant.model.distilled import (DEFAULT_DIM, DistilledEncoder,
                                        project, random_projection)

__all__ = ["DistillReport", "run_gate", "distil",
           "DEFAULT_EPOCHS"]


#: Distillation epochs. Not a default anybody should reach for
#: casually: run one used 400 and FAILED, and the diagnosis was that
#: the copy had not finished being made. Correlation between the
#: encoder's pairwise cosines and the teacher's, measured over 630
#: real pairs:
#:
#: ===========  ======  =====================
#: epochs       loss    correlation to teacher
#: ===========  ======  =====================
#: 400          0.0028  0.6834
#: 1500         0.0008  0.9213
#: 4000         0.0002  **0.9876**
#: ===========  ======  =====================
#:
#: At 400 the encoder had two thirds of the geometry and the gate
#: read that as distillation not working. It was distillation not
#: being finished.
DEFAULT_EPOCHS = 4000


def distil(spec: dict, vocabulary: list, teacher: dict,
           dim: int = DEFAULT_DIM, seed: int = 0,
           epochs: int = DEFAULT_EPOCHS) -> tuple:
    """Train an encoder to place text where the teacher places it.

    Returns ``(encoder, seconds, final loss)``. No labels are used -
    only text and its teacher vector - so this could run over every
    string the library holds rather than only the ones a gate names.
    """
    texts = [text for entry in spec.values() for text in items_of(entry)]
    planes = random_projection(len(teacher[texts[0]]), dim, seed=0)
    xs = [bag_of_words(text, vocabulary) for text in texts]
    targets = [project(teacher[text], planes) for text in texts]
    encoder = DistilledEncoder(len(vocabulary), dim=dim, seed=seed)
    started = time.perf_counter()
    loss = encoder.fit(xs, targets, epochs=epochs, seed=seed)
    return encoder, time.perf_counter() - started, loss


@dataclass
class DistillReport:
    """What the distilled copy kept, and what it cost.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ran: False when LM Studio had no embedding model.
        model: The teacher.
        seeds: Seeds averaged over.
        bag: Bag-of-words accuracy, real text.
        taught: Distilled-encoder accuracy, real text.
        borrowed: Pretrained-teacher accuracy, real text.
        bag_permuted: Bag-of-words accuracy, permuted.
        taught_permuted: Distilled accuracy, permuted.
        kept: taught margin as a share of the borrowed margin.
        taught_margin: taught - bag.
        borrowed_margin: borrowed - bag.
        taught_sd: Seed sd of the taught margin.
        difference: taught margin minus its permuted counterpart.
        difference_sd: Seed sd of that.
        unseen_words: Vocabulary words held out of distillation.
        unseen_accuracy: Accuracy on text containing them.
        seen_accuracy: Accuracy on text that avoids them.
        distil_seconds: Seconds to distil, once.
        encode_us: Microseconds per encode, after.
        teacher_ms: Milliseconds per teacher call, for contrast.
        offline: Whether evaluation ran with no client at all.
        reason: Plain-language verdict.
    """

    passes: bool
    ran: bool = False
    model: str = ""
    seeds: int = 0
    bag: float = 0.0
    taught: float = 0.0
    borrowed: float = 0.0
    bag_permuted: float = 0.0
    taught_permuted: float = 0.0
    kept: float = 0.0
    taught_margin: float = 0.0
    borrowed_margin: float = 0.0
    taught_sd: float = 0.0
    difference: float = 0.0
    difference_sd: float = 0.0
    unseen_words: int = 0
    unseen_accuracy: float = 0.0
    seen_accuracy: float = 0.0
    distil_seconds: float = 0.0
    encode_us: float = 0.0
    teacher_ms: float = 0.0
    offline: bool = False
    reason: str = ""


def _boundary(spec: dict, vocabulary: list, teacher: dict,
              seed: int) -> tuple:
    """Criterion 4: what the encoder scores on words it never saw.

    A quarter of the vocabulary is withheld from distillation by
    masking those columns to zero in the training inputs - the
    encoder is given no way to learn what they mean. Texts are then
    split by whether they contain any withheld word, and both halves
    are scored.
    """
    import random as _random

    rng = _random.Random(4000 + seed)
    withheld = set(rng.sample(vocabulary, max(1, len(vocabulary) // 4)))
    kept = [word for word in vocabulary if word not in withheld]

    encoder, _seconds, _loss = distil(
        spec, kept, teacher, seed=seed, epochs=DEFAULT_EPOCHS)

    def features(text: str) -> list:
        return encoder.encode(bag_of_words(text, kept))

    names = sorted(spec)
    hits = {True: [0, 0], False: [0, 0]}
    for held in range(6):
        xs, ys = [], []
        for index, name in enumerate(names):
            for position, text in enumerate(items_of(spec[name])):
                if position == held:
                    continue
                xs.append(features(text))
                ys.append(index)
        from ultraquant.model.network import UltraQuantNet

        net = UltraQuantNet(encoder.dim, [24], len(names), seed=seed)
        for _ in range(60):
            net.train_batch(xs, ys, 0.05)
        for index, name in enumerate(names):
            text = items_of(spec[name])[held]
            touched = bool(_tokens(text) & withheld)
            hits[touched][1] += 1
            hits[touched][0] += int(net.predict(features(text))[0] == index)
    unseen = hits[True][0] / hits[True][1] if hits[True][1] else 0.0
    seen = hits[False][0] / hits[False][1] if hits[False][1] else 0.0
    return len(withheld), unseen, seen


def run_gate(seeds: int = 4, model: str | None = None) -> DistillReport:
    """Distil the teacher, then evaluate with the teacher unplugged."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    try:
        client = LMStudioClient()
        chosen = model or (client.embedding_models() or [None])[0]
        if chosen is None:
            raise RuntimeError("no embedding model in LM Studio")

        real = CATEGORIES
        mapping = build_permutation(seed=0)
        permuted = permuted_categories(mapping)
        real_vocab = _vocabulary(real)
        perm_vocab = [mapping[word] for word in real_vocab]

        # Every teacher call happens HERE, before any evaluation.
        texts = [t for s in real.values() for t in items_of(s)]
        texts += [permute(t, mapping) for t in texts]
        started = time.perf_counter()
        vectors = {t: client.embed([t], model=chosen)[0] for t in texts}
        teacher_ms = (time.perf_counter() - started) * 1000.0 / len(texts)
    except Exception as exc:                      # noqa: BLE001
        return DistillReport(
            passes=False, ran=False,
            reason=f"COULD NOT RUN - {type(exc).__name__}: "
                   f"{str(exc)[:140]}. Reported as not run rather than "
                   "passed.")

    # Criterion 1: from here on there is no client. Anything that
    # tried to reach the network would raise rather than quietly work.
    del client

    rows, times, losses = [], [], []
    for seed in range(seeds):
        taught_real, seconds, _loss = distil(real, real_vocab, vectors,
                                             seed=seed)
        taught_perm, _s, _l = distil(permuted, perm_vocab, vectors,
                                     seed=seed)
        times.append(seconds)

        bag_r = _fold_accuracy(real, lambda t: bag_of_words(t, real_vocab),
                               len(real_vocab), seed)
        bag_p = _fold_accuracy(permuted,
                               lambda t: bag_of_words(t, perm_vocab),
                               len(perm_vocab), seed)
        got_r = _fold_accuracy(
            real, lambda t: taught_real.encode(bag_of_words(t, real_vocab)),
            taught_real.dim, seed)
        got_p = _fold_accuracy(
            permuted,
            lambda t: taught_perm.encode(bag_of_words(t, perm_vocab)),
            taught_perm.dim, seed)
        lent = _fold_accuracy(real, lambda t: vectors[t],
                              len(vectors[texts[0]]), seed)
        rows.append((bag_r, got_r, lent, bag_p, got_p))

    withheld, unseen, seen = _boundary(real, real_vocab, vectors, seed=0)

    sample = bag_of_words(texts[0], real_vocab)
    encoder, _s, _l = distil(real, real_vocab, vectors, seed=0, epochs=1)
    started = time.perf_counter()
    for _ in range(200):
        encoder.encode(sample)
    encode_us = (time.perf_counter() - started) * 1e6 / 200

    bag = statistics.fmean(r[0] for r in rows)
    taught = statistics.fmean(r[1] for r in rows)
    borrowed = statistics.fmean(r[2] for r in rows)
    bag_p = statistics.fmean(r[3] for r in rows)
    taught_p = statistics.fmean(r[4] for r in rows)
    taught_margins = [r[1] - r[0] for r in rows]
    differences = [(r[1] - r[0]) - (r[4] - r[3]) for r in rows]
    taught_sd = (statistics.pstdev(taught_margins)
                 if len(taught_margins) > 1 else 0.0)
    difference_sd = (statistics.pstdev(differences)
                     if len(differences) > 1 else 0.0)
    borrowed_margin = borrowed - bag
    taught_margin = taught - bag
    kept = taught_margin / borrowed_margin if borrowed_margin else 0.0

    keeps_half = taught_margin > 0.5 * borrowed_margin + taught_sd
    semantic = statistics.fmean(differences) > difference_sd
    passes = keeps_half and semantic

    report = DistillReport(
        passes=passes, ran=True, model=chosen, seeds=seeds, bag=bag,
        taught=taught, borrowed=borrowed, bag_permuted=bag_p,
        taught_permuted=taught_p, kept=kept, taught_margin=taught_margin,
        borrowed_margin=borrowed_margin, taught_sd=taught_sd,
        difference=statistics.fmean(differences),
        difference_sd=difference_sd, unseen_words=withheld,
        unseen_accuracy=unseen, seen_accuracy=seen,
        distil_seconds=statistics.fmean(times), encode_us=encode_us,
        teacher_ms=teacher_ms, offline=True)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - teacher {chosen}, {seeds} "
        f"seeds; bag {bag:.3f}, distilled {taught:.3f}, borrowed "
        f"{borrowed:.3f}; the distilled copy keeps {kept:.0%} of the "
        f"borrowed margin ({taught_margin:+.3f} of {borrowed_margin:+.3f}) "
        f"at seed sd {taught_sd:.3f} "
        + ("(over half, as registered)" if keeps_half
           else "- UNDER HALF, which is the honest failure")
        + f"; permuted difference {report.difference:+.3f} at sd "
        f"{difference_sd:.3f} "
        + ("(still semantic)" if semantic
           else "- SURVIVES the permutation, so it is capacity")
        + f"; on text using any of {withheld} words held out of "
        f"distillation, {unseen:.3f} against {seen:.3f} on text that "
        f"avoids them; distilled once in {report.distil_seconds:.1f}s, "
        f"then {encode_us:.0f} us per encode against the teacher's "
        f"{teacher_ms:.1f} ms")
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"teacher: {result.model}   seeds: {result.seeds}")
    print()
    print(f"{'':<22}{'real':>9}{'permuted':>11}")
    print(f"{'bag-of-words':<22}{result.bag:>9.3f}"
          f"{result.bag_permuted:>11.3f}")
    print(f"{'distilled (ours)':<22}{result.taught:>9.3f}"
          f"{result.taught_permuted:>11.3f}")
    print(f"{'borrowed (network)':<22}{result.borrowed:>9.3f}{'-':>11}")
    print()
    print(f"kept {result.kept:.0%} of the borrowed margin "
          f"({result.taught_margin:+.3f} of {result.borrowed_margin:+.3f}) "
          f"at seed sd {result.taught_sd:.3f}")
    print(f"permuted difference {result.difference:+.3f} at sd "
          f"{result.difference_sd:.3f}")
    print()
    print(f"vocabulary boundary: {result.unseen_accuracy:.3f} on text using "
          f"any of {result.unseen_words} withheld words, "
          f"{result.seen_accuracy:.3f} on text that avoids them")
    print(f"cost: {result.distil_seconds:.1f}s once, then "
          f"{result.encode_us:.0f} us/encode against "
          f"{result.teacher_ms:.1f} ms/teacher call")
    print()
    print(result.reason)
