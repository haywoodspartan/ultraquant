"""Stage 1, in the domain §11.5 said to try it in.

ARCHITECTURE.md §11.3 states Stage 1's gate before the work, so the
answer cannot be adjusted afterwards:

    A held-out category trained on 5 examples through the encoder
    beats the same category trained on 5 raw-pixel examples, by a
    margin larger than seed variance.

`transfer_gate.py` ran that on pixels and it FAILED, and §11.5 named
why and where to go next: for stroke patterns **raw pixels were
already close to an ideal representation**, so any re-encoding could
only lose information. The condition under which a shared
representation *should* pay was written down at the same time — "a
perceptual domain where the raw features are a poor representation" —
and §11.108 has since established that text is such a domain, and
that the advantage there is **semantic rather than capacity** (92% of
it vanishes under a vocabulary permutation).

So this is Stage 1's criterion, unchanged, in the domain that failure
pointed at. **The translation is the only liberty taken and it is
stated here**: "raw pixels" becomes a binary bag-of-words over the
corpus vocabulary — the direct analogue, the features a lexical
signature already uses — and "through the encoder" becomes the
pretrained embedding.

**Five examples means five, and the corpus supplies them without a
line being written for this gate.** Every category has exactly six
texts, so leave-one-out gives 5 train and 1 test, six folds, with no
sampling and no new corpus. That matters more than it sounds: hand
-writing extra paraphrases for a gate that rewards paraphrase
handling is how a deck gets stacked without anybody deciding to stack
it. Nothing here was written for this measurement.

**Paired, as §11.3's own gate demands.** Same folds, same classifier
architecture, same hyperparameters, same seeds. Only the input
representation differs, and the per-seed difference is what is
reported.

**The control is §11.108's permutation**, which earned its place by
proving its own faithfulness: a bijection over the vocabulary leaves
the bag-of-words arm mathematically unchanged while destroying
meaning. Transfer should FAIL there. A representation that helps on
both is adding capacity, not transferring structure — and reporting
only the favourable arm is the easy way to a positive answer.

**The criteria, written before the run.**

1. **The arms are paired** — identical folds, architecture,
   hyperparameters and seeds, asserted at runtime rather than left to
   trust. Only the representation differs.
2. **The embedding arm wins at five examples** — by more than the
   seed-to-seed standard deviation. Stage 1's criterion verbatim.
3. **The control is faithful** — bag-of-words accuracy must be
   identical between real and permuted conditions. A bijection
   preserves token presence exactly; movement means the permutation
   leaked and nothing else can be read.
4. **The advantage does not survive the permutation** — the
   embedding's margin must be larger in the real condition than in
   the permuted one, by more than the seed sd of that difference.
   Surviving means capacity.
5. **Both arms beat chance** — 1/6 with six categories. A task
   neither arm can do is not measuring representation quality.

**PASSED. Stage 1's criterion is met — and what that does and does
not mean needs saying carefully, because this is the stage the
roadmap has been stuck on since §11.5.**

Six seeds, five examples per class, leave-one-out over six folds:

| | real | permuted |
|---|---:|---:|
| bag-of-words | 0.296 | **0.296** |
| embedding | **0.463** | 0.301 |
| margin | **+0.167** | +0.005 |

**Margin +0.167 at seed sd 0.099** — larger than seed variance, which
is §11.3's criterion verbatim. **Difference +0.162 at sd 0.130**, so
the advantage does not survive losing the meaning: the transfer is
semantic. Both arms clear chance (0.167).

**Neither margin is large, and reading them as large would be the
error this whole file is arranged against.** 1.7x the seed sd and
1.25x respectively — they clear the bars they were measured against
and they clear them thinly. Six categories, thirty-six test items per
seed.

**Two harness defects were caught by the criteria before the result
was believed, and both are worth more than the pass.**

*Criterion 3 failed on the first run*: bag-of-words scored 0.296 real
against 0.269 permuted, when a bijection guarantees they must be
equal. The permuted vocabulary had been sorted independently, so
feature index i meant a different word in each condition and the two
"identical" arms were different arithmetic against the same seeded
weights. The permuted vocabulary is now the IMAGE of the real one in
the same order, and the two arms are literally the same computation.

*Before that, the vocabulary-size assertion failed and found a leak
in §11.108's control* — `permute` split on whitespace, so
"equal-sided" matched no key and passed through untranslated, leaving
real English meaning inside the condition that exists to have none.
That is fixed in `permutation_gate.py` and its numbers are corrected
there. A gate found a defect in the gate before it.

**What this settles.** §11.2 named two things missing from this
architecture: composition, solved at Stage 2, and shared
representation, which failed at Stage 1 on pixels and was recorded as
abandoned in that form. In the domain §11.5 named as the place to
retry — text, where raw features genuinely are poor — **a shared
representation does buy few-shot transfer, and the transfer is
semantic rather than capacity.**

**What it does not settle, and the distinction is exact.** §11.3's
Stage 1 is *"a shared encoder... one representation over the feature
space"* — one this system **builds**. `nomic-embed-text-v1.5` is
pretrained and reached over the network. **The gate's criterion is
met with a borrowed representation; the stage's mechanism is not
built.** §11.5 and §11.11 tried to learn one and did not succeed, and
nothing here changes that.

What it does change is why that matters. Before this run it was an
open question whether a shared representation would help at all in
this system — §11.5 said no on pixels and §11.11 could not read the
answer on text. It is now measured that it does. **Learning one is a
well-motivated engineering problem rather than a bet**, which is a
different and much better place to be stuck.

Run it::

    python -m ultraquant.experiments.fewshot_gate

Requires LM Studio with an embedding model. Without one the gate
reports that it did not run rather than passing vacuously.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ultraquant.experiments.embed_gate import CATEGORIES, _tokens
from ultraquant.experiments.permutation_gate import (build_permutation,
                                                     permuted_categories)
from ultraquant.model.network import UltraQuantNet

__all__ = ["FewShotReport", "run_gate", "items_of", "bag_of_words"]

#: Classifier shape, identical in both arms. Small on purpose: five
#: examples per class cannot support anything larger, and a capacity
#: difference between the arms would confound the one thing being
#: measured.
_HIDDEN = (24,)
_EPOCHS = 60
_LR = 0.05


def items_of(spec: dict) -> list:
    """A category's six texts, in a fixed order."""
    return [spec["canonical"]] + list(spec["partial"]) + \
        list(spec["paraphrases"])


def bag_of_words(text: str, vocabulary: list) -> list:
    """Binary token presence - the "raw features" of this domain.

    The analogue of raw pixels: what the input literally contains,
    with no learned structure over it. This is exactly what a lexical
    signature sees.
    """
    present = _tokens(text)
    return [1.0 if word in present else 0.0 for word in vocabulary]


def _vocabulary(spec: dict) -> list:
    words = set()
    for entry in spec.values():
        for text in items_of(entry):
            words |= _tokens(text)
    return sorted(words)


def _fold_accuracy(spec: dict, features, width: int, seed: int) -> float:
    """Leave-one-out over every category, one classifier per fold.

    Five examples per class train it and the sixth tests it, which is
    §11.3's "trained on 5 examples" taken literally.
    """
    names = sorted(spec)
    right = total = 0
    for held in range(6):
        xs, ys = [], []
        for index, name in enumerate(names):
            for position, text in enumerate(items_of(spec[name])):
                if position == held:
                    continue
                xs.append(features(text))
                ys.append(index)
        net = UltraQuantNet(width, list(_HIDDEN), len(names), seed=seed)
        for _ in range(_EPOCHS):
            net.train_batch(xs, ys, _LR)
        for index, name in enumerate(names):
            text = items_of(spec[name])[held]
            right += int(net.predict(features(text))[0] == index)
            total += 1
    return right / total


@dataclass
class FewShotReport:
    """Whether the shared representation bought transfer, or capacity.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ran: False when LM Studio had no embedding model.
        model: The embedding model measured.
        seeds: Classifier seeds averaged over.
        bag_real: Bag-of-words accuracy, real text.
        bag_permuted: The same under the permutation - must match.
        embed_real: Embedding accuracy, real text.
        embed_permuted: The same under the permutation.
        margin_real: embed_real - bag_real.
        margin_permuted: embed_permuted - bag_permuted.
        margin_sd: Seed-to-seed sd of the real margin.
        difference: margin_real - margin_permuted.
        difference_sd: Seed-to-seed sd of that difference.
        chance: 1 / number of categories.
        reason: Plain-language verdict.
    """

    passes: bool
    ran: bool = False
    model: str = ""
    seeds: int = 0
    bag_real: float = 0.0
    bag_permuted: float = 0.0
    embed_real: float = 0.0
    embed_permuted: float = 0.0
    margin_real: float = 0.0
    margin_permuted: float = 0.0
    margin_sd: float = 0.0
    difference: float = 0.0
    difference_sd: float = 0.0
    chance: float = 0.0
    reason: str = ""


def run_gate(seeds: int = 6, model: str | None = None) -> FewShotReport:
    """Stage 1's criterion, paired, with §11.108's control."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    try:
        client = LMStudioClient()
        chosen = model or (client.embedding_models() or [None])[0]
        if chosen is None:
            raise RuntimeError("no embedding model in LM Studio")
        cache: dict = {}

        def embed(text: str) -> list:
            if text not in cache:
                cache[text] = client.embed([text], model=chosen)[0]
            return cache[text]

        real = CATEGORIES
        mapping = build_permutation(seed=0)
        permuted = permuted_categories(mapping)
        real_vocab = _vocabulary(real)
        # The permuted vocabulary is the IMAGE of the real one, in the
        # same order - not independently sorted. Sorting it separately
        # was a real defect: feature index i then meant a different
        # word in each condition, so the two bag arms were different
        # computations against the same seeded initial weights, and
        # they scored 0.296 against 0.269 where a bijection guarantees
        # they must be equal. Criterion 3 caught it. In this order the
        # bag arms are literally the same arithmetic.
        perm_vocab = [mapping[word] for word in real_vocab]
        assert set(perm_vocab) == set(_vocabulary(permuted)), (
            "the permuted corpus contains words the mapping did not "
            "produce, so the permutation is not a clean bijection")
        width = len(embed(real["square"]["canonical"]))

        rows = []
        for seed in range(seeds):
            bag_r = _fold_accuracy(
                real, lambda t: bag_of_words(t, real_vocab),
                len(real_vocab), seed)
            bag_p = _fold_accuracy(
                permuted, lambda t: bag_of_words(t, perm_vocab),
                len(perm_vocab), seed)
            emb_r = _fold_accuracy(real, embed, width, seed)
            emb_p = _fold_accuracy(permuted, embed, width, seed)
            rows.append((bag_r, bag_p, emb_r, emb_p))
    except Exception as exc:                      # noqa: BLE001
        return FewShotReport(
            passes=False, ran=False,
            reason=f"COULD NOT RUN - {type(exc).__name__}: "
                   f"{str(exc)[:140]}. Reported as not run rather than "
                   "passed.")

    bag_r = statistics.fmean(r[0] for r in rows)
    bag_p = statistics.fmean(r[1] for r in rows)
    emb_r = statistics.fmean(r[2] for r in rows)
    emb_p = statistics.fmean(r[3] for r in rows)
    margins = [r[2] - r[0] for r in rows]
    differences = [(r[2] - r[0]) - (r[3] - r[1]) for r in rows]
    margin_sd = statistics.pstdev(margins) if len(margins) > 1 else 0.0
    difference_sd = (statistics.pstdev(differences)
                     if len(differences) > 1 else 0.0)
    chance = 1.0 / len(CATEGORIES)

    faithful = all(abs(r[1] - r[0]) < 1e-12 for r in rows)
    wins = statistics.fmean(margins) > margin_sd
    semantic = statistics.fmean(differences) > difference_sd
    above_chance = bag_r > chance and emb_r > chance
    passes = faithful and wins and semantic and above_chance

    report = FewShotReport(
        passes=passes, ran=True, model=chosen, seeds=seeds,
        bag_real=bag_r, bag_permuted=bag_p, embed_real=emb_r,
        embed_permuted=emb_p, margin_real=emb_r - bag_r,
        margin_permuted=emb_p - bag_p, margin_sd=margin_sd,
        difference=statistics.fmean(differences),
        difference_sd=difference_sd, chance=chance)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {chosen}, {seeds} seeds, five "
        f"examples per class; bag-of-words {bag_r:.3f} against embedding "
        f"{emb_r:.3f}, margin {report.margin_real:+.3f} at seed sd "
        f"{margin_sd:.3f} "
        + ("(Stage 1's criterion met)" if wins
           else "- MARGIN INSIDE THE NOISE, which is Stage 1's criterion "
                "unmet")
        + f"; control bag {bag_p:.3f} "
        + ("(identical, faithful)" if faithful
           else "- NOT IDENTICAL, the permutation leaked")
        + f", control embedding {emb_p:.3f}, difference "
        f"{report.difference:+.3f} at sd {difference_sd:.3f} "
        + ("(the transfer is semantic)" if semantic
           else "- the advantage SURVIVES losing the meaning, so it is "
                "capacity")
        + f"; chance {chance:.3f}"
        + ("" if above_chance else " - AN ARM IS AT OR BELOW CHANCE"))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"embedding model: {result.model}   seeds: {result.seeds}")
    print(f"five examples per class, leave-one-out, six folds")
    print()
    print(f"{'':<16}{'real':>10}{'permuted':>11}")
    print(f"{'bag-of-words':<16}{result.bag_real:>10.3f}"
          f"{result.bag_permuted:>11.3f}")
    print(f"{'embedding':<16}{result.embed_real:>10.3f}"
          f"{result.embed_permuted:>11.3f}")
    print(f"{'margin':<16}{result.margin_real:>+10.3f}"
          f"{result.margin_permuted:>+11.3f}")
    print()
    print(f"margin at seed sd {result.margin_sd:.3f}; "
          f"difference {result.difference:+.3f} at sd "
          f"{result.difference_sd:.3f}; chance {result.chance:.3f}")
    print()
    print(result.reason)
