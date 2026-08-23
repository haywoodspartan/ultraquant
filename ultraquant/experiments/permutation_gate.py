"""Is the embedding's advantage semantic, or just capacity? A clean control.

§11.2 named two things missing from this architecture: composition and
a shared representation. Composition landed at Stage 2. **Shared
representation is still the open gap**, and it has now failed twice —
§11.5 on pixels, where raw features were already near-ideal, and
§11.11 on text, where the effect was large and the control was not
clean enough to interpret it.

§11.11 is worth restating exactly, because this unit exists to finish
it. It measured **paraphrase +0.533 at 5.66x seed sd** — a large,
real effect clearing §11.5's criterion comfortably — and reported
FAIL because the partial-overlap control also went to embeddings at
+0.087, 2.6x its own noise. Embeddings were therefore *partly just a
better classifier*, and under the criterion written beforehand that
made the paraphrase margin uninterpretable.

It also wrote down the fix and declined to use it:

    "A pure-capacity account predicts a roughly EQUAL edge in both
    conditions, and the paraphrase edge is 6x the control edge. A
    difference-in-differences would read +0.446 and pass easily.
    That is probably the better-specified test — the control exists
    precisely to be subtracted. But it was formulated AFTER seeing
    the numbers, and adopting a criterion because it converts a fail
    into a pass is the goalpost move §11.5 refused."

**So it is registered here, before this run, with a control built for
it.** That is what that note asked for, and it is the only honest way
to use it.

**The control: a consistent bijection onto unrelated real words.**
Every content word in the corpus is replaced by a different, real
English word, the same way everywhere. The point is what this does
and does not disturb:

* **Lexical routing is mathematically unchanged.** Token overlap is
  invariant under a bijection — two texts share a token exactly when
  their images do. So lexical accuracy must come out IDENTICAL, and
  criterion 1 checks it. That is a control with a built-in proof of
  its own faithfulness, which the partial-overlap control had no way
  to offer.
* **Capacity is unchanged.** The embedding model still receives real
  English words and still returns well-formed vectors of the same
  dimension. Nothing about its expressive power moves.
* **Semantics is destroyed.** "a box outline" and "a square shape"
  become unrelated sentences. A representation that routes them
  together because it knows what they MEAN has nothing left to use.

A pure-capacity advantage therefore survives the permutation. A
semantic one does not. That is the whole design.

**The criteria, written before the run.**

1. **The control is faithful** — lexical accuracy must be identical
   between the real and permuted conditions, exactly. A bijection
   preserves token overlap, so any movement means the permutation
   leaked or destroyed information and nothing else here can be
   read. This can fail, and if it does the gate reports a broken
   harness rather than a result.
2. **The embedding wins on paraphrases** — by more than the
   seed-to-seed standard deviation. §11.5's criterion verbatim, the
   one that refused a +0.014 effect whose interval cleared zero.
3. **The advantage is semantic** — the embedding's margin over
   lexical must be LARGER in the real condition than in the permuted
   one, by more than the seed sd of that difference. This is
   §11.11's difference-in-differences, registered in advance.
4. **The permutation must not simply break the embedding** — its
   accuracy in the permuted condition must stay above chance. If the
   permutation destroys capacity as well as semantics, criterion 3
   is measuring the wrong thing and cannot be credited. This is the
   guard against a control that wins by demolition.

**PASSED on all four criteria — and run one had a leak in the
control, found by the NEXT unit's assertion and corrected here.**

§11.109 asserted that the permutation must not change the vocabulary
size, and it did. `permute` had split on whitespace and stripped
punctuation to form its lookup key, so `"equal-sided"` became
`"equalsided"`, matched nothing, and **passed through untranslated** -
real English meaning left sitting inside the control condition.
`_tokens` splits that into two tokens; the substitution now works on
the same alphanumeric runs, so the two agree by construction rather
than by luck.

Criterion 1 did not catch it, and it is worth saying why: the leak
was CONSISTENT, appearing in the permuted category text and the
permuted query alike, so token overlap stayed identical and lexical
still scored 0.222 in both conditions. A faithfulness check on the
lexical arm cannot see a leak that is symmetric. The vocabulary-size
assertion can, and it belongs here too now.

Corrected numbers, eight permutations:

| | real | permuted | run one (leaky) |
|---|---:|---:|---:|
| lexical | 0.222 | **0.222** | 0.222 |
| embedding | **0.778** | 0.306 | 0.264 |
| margin | **+0.556** | +0.083 | +0.042 |

**difference +0.472 at seed sd 0.127 — 3.7x the noise**, against
+0.514 at 4.8x before. The conclusion does not move: **85% of the
embedding's advantage is semantic** (was 92%). The leak flattered the
result by roughly seven points of it, which is the direction a leak
of surviving meaning would be expected to push.

**Criterion 1 held exactly**, which is the part that makes the rest
readable: lexical scored 0.222 in both conditions, identically, over
eight independent permutations. The control did not change what
lexical could see, and that is not asserted from the design — it is
measured, and the design is why it could be.

**The edge is semantic.** Destroy the meaning and hold everything
else constant, and **85% of the embedding's advantage disappears**
(+0.556 to +0.083). A pure-capacity account predicts the margin
survives; it does not.

**Criterion 4 held**, so the control did not win by demolition: the
permuted embedding still routes at 0.306 against a chance floor of
0.167. It is a modest margin and worth reading as one — some capacity
survives, and it is small.

**What this does and does not settle, because the difference is the
whole point.**

It settles §11.11's open question. That unit measured a large effect
and could not interpret it, wrote down the difference-in-differences
that would resolve it, and refused to use a criterion invented after
the numbers. **Registered in advance with a control built for it, the
answer is that the effect was real.** The same form of test, run
honestly.

It does **not** pass Stage 1, and the temptation to say it does
should be named. Stage 1's gate is about **few-shot transfer** — "a
held-out category trained on 5 examples through the encoder beats the
same category on 5 raw-pixel examples". This is about **routing**.
What is established is the PREMISE Stage 1 rests on: that a shared
representation carries structure raw features do not, and that the
structure is semantic. Whether that structure buys transfer is a
different measurement and has not been run.

**And the representation is not ours.** `nomic-embed-text-v1.5` is a
pretrained model reached over the network. §11.5 and §11.11 tried to
LEARN a shared representation and did not succeed; this measures that
a good one exists and helps, which is a weaker and different claim.
A system whose shared representation is an external service has
bought transfer at the cost of the from-scratch principle, and that
trade has not been made here — nothing in the pipeline depends on
this. It is a measurement, not a dependency.

**Small, and to be read as small**: six categories, eighteen
paraphrases, one embedding model.

Run it::

    python -m ultraquant.experiments.permutation_gate

Requires LM Studio with an embedding model. Without one the gate
reports that it did not run rather than passing vacuously.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass, field

from ultraquant.experiments.embed_gate import (CATEGORIES, _cosine,
                                               _embedding_route,
                                               _lexical_route, _tokens)

__all__ = ["SemanticReport", "run_gate", "build_permutation", "permute"]


#: Real English words with no relation to the corpus, used as the
#: image of the permutation. Real rather than nonsense on purpose:
#: an embedding model handed invented strings would return degenerate
#: vectors, and the control would then be destroying capacity as well
#: as meaning - which criterion 4 exists to catch, and which this
#: choice exists to avoid in the first place.
_IMAGE_WORDS = [
    "trumpet", "velvet", "harbour", "lantern", "granite", "cobalt",
    "meadow", "cinder", "pelican", "thicket", "marble", "quiver",
    "beacon", "saffron", "cavern", "trellis", "anchor", "bramble",
    "kettle", "juniper", "ferry", "cypress", "hammock", "obsidian",
    "willow", "brazier", "canvas", "drizzle", "ember", "flint",
    "gable", "heron", "ivory", "jetty", "kelp", "lichen",
    "mantle", "nettle", "orchid", "pewter", "quarry", "rafter",
    "sable", "tundra", "umber", "vellum", "walnut", "yarrow",
    "zephyr", "alcove", "basalt", "chalice", "dovetail", "elmwood",
    "fathom", "gantry", "hollow", "inlet", "jasper", "kindling",
    "lattice", "mortar", "nimbus", "oakum", "plinth", "quill",
    "ridge", "spindle", "tallow", "urchin", "vantage", "wharf",
    "yeoman", "zinc", "arbour", "bellows", "citadel", "dowel",
    "eaves", "furrow", "gorse", "hearth", "ingot", "joist",
    "keystone", "loam", "mullion", "newel", "oriel", "parapet",
]


def corpus_vocabulary() -> list:
    """Every content word the corpus uses, sorted for determinism."""
    words = set()
    for spec in CATEGORIES.values():
        words |= _tokens(spec["canonical"])
        for text in spec["partial"] + spec["paraphrases"]:
            words |= _tokens(text)
    return sorted(words)


def build_permutation(seed: int = 0) -> dict:
    """A bijection from corpus words onto unrelated real words.

    Bijective, so token overlap survives it exactly - that is the
    property criterion 1 checks and the reason this control can prove
    its own faithfulness.
    """
    words = corpus_vocabulary()
    if len(words) > len(_IMAGE_WORDS):
        raise ValueError(
            f"{len(words)} corpus words but only {len(_IMAGE_WORDS)} "
            "images; a non-injective map would merge distinct tokens "
            "and change lexical routing, which is the one thing this "
            "control must not do")
    images = list(_IMAGE_WORDS)
    random.Random(seed).shuffle(images)
    return dict(zip(words, images))


def permute(text: str, mapping: dict) -> str:
    """Rewrite `text`, mapping exactly the runs `_tokens` sees.

    Stopwords stay because they carry no category signal either way -
    `_tokens` drops them, so mapping them would change nothing lexical
    and would only make the permuted text read less like English to
    the embedding model.

    **This substitutes alphanumeric RUNS, not whitespace words**, and
    the difference was a real leak. The first version split on spaces
    and stripped punctuation to look each word up, so "equal-sided"
    became the single key "equalsided", which is in no mapping - and
    the phrase passed through **untranslated**, leaving real English
    meaning inside the control condition. `_tokens` splits it into
    "equal" and "sided"; this now does the same, so the two agree by
    construction rather than by luck.
    """
    return re.sub(r"[a-z0-9]+",
                  lambda m: mapping.get(m.group(), m.group()),
                  text.lower())


def permuted_categories(mapping: dict) -> dict:
    """The whole corpus under the permutation."""
    return {
        name: {
            "canonical": permute(spec["canonical"], mapping),
            "partial": [permute(t, mapping) for t in spec["partial"]],
            "paraphrases": [permute(t, mapping)
                            for t in spec["paraphrases"]],
        }
        for name, spec in CATEGORIES.items()
    }


@dataclass
class SemanticReport:
    """Whether the embedding's edge survived losing its meaning.

    Attributes:
        passes: The verdict under the pre-registered criteria.
        ran: False when LM Studio had no embedding model.
        model: The embedding model measured.
        seeds: Permutations averaged over.
        lexical_real: Lexical accuracy on real paraphrases.
        lexical_permuted: The same under the permutation - must match.
        embedding_real: Embedding accuracy on real paraphrases.
        embedding_permuted: The same under the permutation.
        margin_real: embedding_real - lexical_real.
        margin_permuted: embedding_permuted - lexical_permuted.
        difference: margin_real - margin_permuted, the semantic part.
        difference_sd: Seed-to-seed sd of that difference.
        paraphrase_sd: Seed-to-seed sd of the real margin.
        chance: 1 / number of categories.
        reason: Plain-language verdict.
    """

    passes: bool
    ran: bool = False
    model: str = ""
    seeds: int = 0
    lexical_real: float = 0.0
    lexical_permuted: float = 0.0
    embedding_real: float = 0.0
    embedding_permuted: float = 0.0
    margin_real: float = 0.0
    margin_permuted: float = 0.0
    difference: float = 0.0
    difference_sd: float = 0.0
    paraphrase_sd: float = 0.0
    chance: float = 0.0
    reason: str = ""


def _route_all(spec: dict, canon_tokens: dict, canon_vectors: dict,
               embed) -> tuple:
    """Accuracy of both routes over every paraphrase in a corpus."""
    lex_right = emb_right = total = 0
    for name, entry in spec.items():
        for text in entry["paraphrases"]:
            total += 1
            lex_right += int(_lexical_route(text, canon_tokens) == name)
            emb_right += int(_embedding_route(embed(text),
                                              canon_vectors) == name)
    return (lex_right / total, emb_right / total)


def run_gate(seeds: int = 8, model: str | None = None) -> SemanticReport:
    """Route paraphrases with and without their meaning."""
    from ultraquant.interpreter.lmstudio import LMStudioClient

    try:
        client = LMStudioClient()
        chosen = model or (client.embedding_models() or [None])[0]
        if chosen is None:
            raise RuntimeError("no embedding model in LM Studio")
        cache: dict = {}

        def embed(text: str) -> list:
            # embed() batches and returns one vector PER INPUT, so a
            # single string comes back wrapped in a list of one.
            if text not in cache:
                cache[text] = client.embed([text], model=chosen)[0]
            return cache[text]

        # The real condition is fixed, so it is measured once.
        real_tokens = {n: _tokens(s["canonical"])
                       for n, s in CATEGORIES.items()}
        real_vectors = {n: embed(s["canonical"])
                        for n, s in CATEGORIES.items()}
        lex_real, emb_real = _route_all(CATEGORIES, real_tokens,
                                        real_vectors, embed)

        lex_perms, emb_perms, differences, margins = [], [], [], []
        for seed in range(seeds):
            mapping = build_permutation(seed=seed)
            spec = permuted_categories(mapping)
            tokens = {n: _tokens(s["canonical"]) for n, s in spec.items()}
            vectors = {n: embed(s["canonical"]) for n, s in spec.items()}
            lex_p, emb_p = _route_all(spec, tokens, vectors, embed)
            lex_perms.append(lex_p)
            emb_perms.append(emb_p)
            margins.append(emb_real - lex_real)
            differences.append((emb_real - lex_real) - (emb_p - lex_p))
    except Exception as exc:                      # noqa: BLE001
        return SemanticReport(
            passes=False, ran=False,
            reason=f"COULD NOT RUN - {type(exc).__name__}: "
                   f"{str(exc)[:140]}. Reported as not run rather than "
                   "passed.")

    lex_perm = statistics.fmean(lex_perms)
    emb_perm = statistics.fmean(emb_perms)
    difference = statistics.fmean(differences)
    difference_sd = (statistics.pstdev(differences)
                     if len(differences) > 1 else 0.0)
    permuted_sd = (statistics.pstdev(emb_perms)
                   if len(emb_perms) > 1 else 0.0)
    chance = 1.0 / len(CATEGORIES)

    faithful = all(abs(v - lex_real) < 1e-12 for v in lex_perms)
    beats_lexical = (emb_real - lex_real) > permuted_sd
    semantic = difference > difference_sd
    not_demolished = emb_perm > chance
    passes = faithful and beats_lexical and semantic and not_demolished

    report = SemanticReport(
        passes=passes, ran=True, model=chosen, seeds=seeds,
        lexical_real=lex_real, lexical_permuted=lex_perm,
        embedding_real=emb_real, embedding_permuted=emb_perm,
        margin_real=emb_real - lex_real, margin_permuted=emb_perm - lex_perm,
        difference=difference, difference_sd=difference_sd,
        paraphrase_sd=permuted_sd, chance=chance)
    report.reason = (
        f"{'PASS' if passes else 'FAIL'} - {chosen}, {seeds} permutations; "
        f"lexical {lex_real:.3f} real against {lex_perm:.3f} permuted "
        + ("(identical, so the control is faithful)" if faithful
           else "- NOT IDENTICAL, so the permutation changed what lexical "
                "could see and nothing here can be read")
        + f"; embedding {emb_real:.3f} real against {emb_perm:.3f} "
        f"permuted; margin {report.margin_real:+.3f} against "
        f"{report.margin_permuted:+.3f}, difference {difference:+.3f} at "
        f"seed sd {difference_sd:.3f} "
        + ("(the edge is semantic)" if semantic
           else "- the edge SURVIVES losing the meaning, so it is capacity")
        + f"; permuted embedding {emb_perm:.3f} against chance "
        f"{chance:.3f} "
        + ("(capacity intact)" if not_demolished
           else "- AT OR BELOW CHANCE, so the control destroyed capacity "
                "too and criterion 3 cannot be credited"))
    return report


if __name__ == "__main__":                        # pragma: no cover
    result = run_gate()
    if not result.ran:
        print(result.reason)
        raise SystemExit(1)
    print(f"embedding model: {result.model}   permutations: {result.seeds}")
    print()
    print(f"{'':<14}{'real':>10}{'permuted':>11}")
    print(f"{'lexical':<14}{result.lexical_real:>10.3f}"
          f"{result.lexical_permuted:>11.3f}")
    print(f"{'embedding':<14}{result.embedding_real:>10.3f}"
          f"{result.embedding_permuted:>11.3f}")
    print(f"{'margin':<14}{result.margin_real:>+10.3f}"
          f"{result.margin_permuted:>+11.3f}")
    print()
    print(f"difference (the semantic part): {result.difference:+.3f} "
          f"at seed sd {result.difference_sd:.3f}")
    print(f"chance: {result.chance:.3f}")
    print()
    print(result.reason)
