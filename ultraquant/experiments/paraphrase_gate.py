"""The language wall, measured: paraphrase against the lexical core.

The honest assessment of "could this whole system work" located two
walls; this gate measures the second. Everything the cognitive suite
passed (§11.30–§11.36) rode on token overlap, and the claim to test is
how far surface variation can bend before the machinery stops seeing
the same question. Four form families, same underlying intents:

* **canonical** — "what is the tower melting point?"
* **reorder** — "what is the melting point of the tower?" (token-set
  identical; the machinery is order-blind, so these MUST hold — this is
  the family with a pass/fail bar)
* **verbose** — "could you please tell me the tower melting point" —
  politeness filler that either is or is not treated as content
* **synonym** — "how tall is the tower", "how hard is the spire" —
  different WORDS for the held attribute; the wall proper

Every family carries its own decoys (paraphrased tungsten questions):
robustness that arrives with a fabrication bill is not robustness.

Beside the lexical arm, a COMPARISON arm runs if LM Studio's embedding
model is reachable: embed the question and every fact key, answer from
the top-cosine key above a threshold. §11.11 measured embeddings losing
to lexical signatures at ROUTING; this measures them at question-to-key
matching, with their decoy rate printed next to their synonym gain —
the tradeoff the hybrid argument rests on.

**The claims, written before the run.**

1. **Reorder holds** — reorder-family accuracy must equal canonical
   within one seed sd, with zero additional decoy falls. The machinery
   is order-blind by construction; failing this is a defect, not a wall.
2. **The wall is characterised, not gated** — verbose and synonym
   accuracies are measurements; whatever they are, they go in the book
   with the embedding comparison beside them.
3. **Decoy discipline everywhere** — the lexical arm's decoy falls must
   be zero in every family; the embedding arm's falls are reported at
   each threshold, because that tradeoff is the finding.

**It was run, and it PASSED — with the wall measured and the hybrid
tradeoff quantified.**

| family | accuracy | decoy falls |
|---|---:|---:|
| canonical | 1.000 | 0.000 |
| reorder | 1.000 | 0.000 |
| verbose | 1.000 | 0.000 |
| synonym | **0.000** | 0.000 |

Order-blindness confirmed, politeness filler free, zero fabrications in
every family. The synonym wall stands at full height by the assertion
standard — and its practical shape matters: "how tall is the tower"
still DELIVERS "Nearest I hold: tower height is 417 meters", the right
number hedged rather than asserted, while the chained synonym ("how
hard is the tower") misses entirely and mints a semantically junky
though hygiene-passing curiosity ("steel tall") — the wall leaking into
the ask channel, recorded as a limit.

**The embedding comparison** (text-embedding-nomic, the same catalogue
§11.11 drew from): at cosine>0.75 it buys **0.500 of the synonym wall
with zero decoy falls** — and at 0.65 it fabricates **0.833** of the
ghosts, which is why a threshold is not a tuning detail but the whole
safety argument. The half it buys is exactly the direct-attribute half;
the chained half needs the library's spread, which key-matching cannot
do. That is the hybrid division of labor measured: embeddings translate
surface forms at the boundary, the library chains and refuses at the
core, and neither can do the other's job.

Run it::

    python -m ultraquant.experiments.paraphrase_gate
"""

from __future__ import annotations

import random
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ParaphraseReport", "run_gate"]

_ENTITIES = ["tower", "bridge", "spire", "tunnel", "gate", "dome"]
_MATERIALS = ["steel", "iron", "copper", "granite", "oak", "bronze"]
#: attribute -> the synonym its question uses.
_SYNONYMS = {"height": "tall", "length": "long", "width": "wide"}
_PROPERTIES = {"hardness": "hard", "conductivity": "conductive"}
_GHOSTS = ["obelisk", "viaduct", "citadel"]

_THRESHOLDS = (0.65, 0.75, 0.85)


@dataclass
class ParaphraseReport:
    """Per-family accuracy, decoy discipline, and the comparison arm.

    Attributes:
        passes: Claim 1 (reorder holds) under its bar.
        families: family -> lexical-arm accuracy.
        decoys: family -> lexical-arm decoy fall rate.
        embedding: threshold -> (synonym accuracy, decoy fall rate), or
            empty when no embedding model was reachable.
        embedding_model: The model used, or "".
        reorder_sd: Seed sd of the canonical-reorder contrast.
        reason: Plain-language verdict.
    """

    passes: bool = False
    families: dict = field(default_factory=dict)
    decoys: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    embedding_model: str = ""
    reorder_sd: float = 0.0
    reason: str = ""

    def as_text(self) -> str:
        """The family table, the comparison, then the verdict."""
        lines = [f"{'family':<11} {'accuracy':>9} {'decoy falls':>12}"]
        for family in ("canonical", "reorder", "verbose", "synonym"):
            lines.append(f"{family:<11} {self.families[family]:>9.3f} "
                         f"{self.decoys[family]:>12.3f}")
        if self.embedding:
            lines.append(f"embedding comparison ({self.embedding_model}), "
                         "synonym family:")
            for threshold, (accuracy, falls) in sorted(
                    self.embedding.items()):
                lines.append(f"  cosine>{threshold:.2f}: accuracy "
                             f"{accuracy:.3f}, decoy falls {falls:.3f}")
        else:
            lines.append("embedding arm skipped: no embedding model "
                         "reachable")
        lines.append(("PASS - " if self.passes else "FAIL - ") + self.reason)
        return "\n".join(lines)


def _worlds(rng: random.Random):
    """One seed's facts and its questions per family."""
    entity_a, entity_b = rng.sample(_ENTITIES, 2)
    material = rng.choice(_MATERIALS)
    attr, syn = rng.choice(list(_SYNONYMS.items()))
    prop, prop_syn = rng.choice(list(_PROPERTIES.items()))
    ghost = rng.choice(_GHOSTS)
    value_attr = rng.randrange(50, 950)
    value_prop = rng.randrange(1000, 9999)

    facts = [
        (f"the {entity_a} {attr}", f"{value_attr} meters"),
        (f"the {entity_a} material", material),
        (f"the {material} {prop}", f"{value_prop} units"),
    ]
    questions = {
        "canonical": [
            (f"what is the {entity_a} {attr}?", str(value_attr)),
            (f"what is the {entity_a} {prop}?", str(value_prop)),
        ],
        "reorder": [
            (f"what is the {attr} of the {entity_a}?", str(value_attr)),
            (f"what is the {prop} of the {entity_a}?", str(value_prop)),
        ],
        "verbose": [
            (f"could you please tell me the {entity_a} {attr}?",
             str(value_attr)),
            (f"i would like to know the {entity_a} {prop}?",
             str(value_prop)),
        ],
        "synonym": [
            (f"how {syn} is the {entity_a}?", str(value_attr)),
            (f"how {prop_syn} is the {entity_a}?", str(value_prop)),
        ],
    }
    decoys = {
        "canonical": [f"what is the {ghost} {attr}?"],
        "reorder": [f"what is the {attr} of the {ghost}?"],
        "verbose": [f"could you please tell me the {ghost} {attr}?"],
        "synonym": [f"how {syn} is the {ghost}?"],
    }
    return facts, questions, decoys


def _assertive(response: str) -> bool:
    lowered = response.lower()
    if lowered.startswith(("i don't hold", "nothing believed",
                           "i can't plan", "i couldn't find")):
        return False
    return "(confidence" in lowered or "inferred" in lowered


def _embedding_arm(facts_per_seed, questions_per_seed, decoys_per_seed):
    """Top-cosine key matching at each threshold, or {} when unreachable."""
    try:
        from ultraquant.interpreter.lmstudio import LMStudioClient

        client = LMStudioClient()
        if not client.available():
            return {}, ""
        models = client.embedding_models()
        if not models:
            return {}, ""
        model = models[0]
    except Exception:  # noqa: BLE001 - the comparison arm is optional
        return {}, ""

    def embed(texts):
        return client.embed(list(texts), model=model)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    results = {threshold: [0, 0, 0, 0] for threshold in _THRESHOLDS}
    for facts, questions, decoys in zip(facts_per_seed, questions_per_seed,
                                        decoys_per_seed):
        keys = [key.removeprefix("the ") for key, _v in facts]
        values = {key.removeprefix("the "): value for key, value in facts}
        key_vectors = embed(keys)
        for question, expected in questions["synonym"]:
            q_vec = client.embed(question, model=model)[0]
            scored = sorted(((cosine(q_vec, kv), key) for kv, key in
                             zip(key_vectors, keys)), reverse=True)
            top_sim, top_key = scored[0]
            for threshold in _THRESHOLDS:
                results[threshold][1] += 1
                if top_sim >= threshold and \
                        expected in str(values[top_key]):
                    results[threshold][0] += 1
        for question in decoys["synonym"]:
            q_vec = client.embed(question, model=model)[0]
            scored = sorted(((cosine(q_vec, kv), key) for kv, key in
                             zip(key_vectors, keys)), reverse=True)
            top_sim, _top_key = scored[0]
            for threshold in _THRESHOLDS:
                results[threshold][3] += 1
                if top_sim >= threshold:
                    results[threshold][2] += 1
    return ({threshold: (hits / max(total, 1), falls / max(dtotal, 1))
             for threshold, (hits, total, falls, dtotal)
             in results.items()}, model)


def run_gate(seeds: int = 6) -> ParaphraseReport:
    """Run the lexical arm per family, plus the embedding comparison.

    Args:
        seeds: Fresh worlds.

    Returns:
        The :class:`ParaphraseReport`.
    """
    from ultraquant.interpreter.thoughts import build_session, run_pipeline

    family_acc = {f: [] for f in ("canonical", "reorder", "verbose",
                                  "synonym")}
    family_falls = {f: [] for f in family_acc}
    contrasts = []
    facts_per_seed, questions_per_seed, decoys_per_seed = [], [], []

    for seed in range(seeds):
        rng = random.Random(seed)
        facts, questions, decoys = _worlds(rng)
        facts_per_seed.append(facts)
        questions_per_seed.append(questions)
        decoys_per_seed.append(decoys)
        root = Path(tempfile.mkdtemp(prefix=f"uq_para_{seed}_"))
        try:
            session = build_session(root, seed=seed)
            for key, value in facts:
                run_pipeline(f"{key} is {value}", session)

            per_family = {}
            for family, pairs in questions.items():
                hits = 0
                for question, expected in pairs:
                    response, _trace = run_pipeline(question, session)
                    hits += int(expected in response
                                and _assertive(response))
                per_family[family] = hits / len(pairs)
                family_acc[family].append(per_family[family])
                falls = 0
                for question in decoys[family]:
                    response, _trace = run_pipeline(question, session)
                    falls += int(_assertive(response))
                family_falls[family].append(falls / len(decoys[family]))
            contrasts.append(per_family["reorder"]
                             - per_family["canonical"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    embedding, model = _embedding_arm(facts_per_seed, questions_per_seed,
                                      decoys_per_seed)

    report = ParaphraseReport(
        families={f: statistics.fmean(v) for f, v in family_acc.items()},
        decoys={f: statistics.fmean(v) for f, v in family_falls.items()},
        embedding=embedding,
        embedding_model=model,
        # stdev, not pstdev: the seeds are a sample of generated worlds.
        reorder_sd=(statistics.stdev(contrasts) if seeds > 1 else 0.0),
    )

    lexical_falls = max(report.decoys.values())
    if lexical_falls > 0:
        report.reason = (
            f"decoy discipline failed in the lexical arm "
            f"({lexical_falls:.3f}); paraphrase robustness that "
            "fabricates is not robustness")
        return report
    gap = abs(report.families["reorder"] - report.families["canonical"])
    if gap > max(report.reorder_sd, 1e-9):
        report.reason = (
            f"reorder does not hold: {report.families['canonical']:.3f} "
            f"canonical vs {report.families['reorder']:.3f} reordered - "
            "an order-blind machine failed an order-only change, which "
            "is a defect, not a wall")
        return report
    report.passes = True
    report.reason = (
        f"reorder holds ({report.families['reorder']:.3f} vs "
        f"{report.families['canonical']:.3f} canonical) with zero decoy "
        f"falls; the wall is measured at verbose "
        f"{report.families['verbose']:.3f} and synonym "
        f"{report.families['synonym']:.3f}")
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
