"""Do real embeddings route better than lexical signatures? A pre-registered gate.

§11.5 failed and named why: for stroke patterns **raw pixels were already close
to an ideal representation**, so any learned re-encoding could only lose
information. It also named the condition under which a learned representation
*should* pay — "a perceptual domain where the raw features are a poor
representation". Text routing by lexical overlap is exactly that domain. "A box
outline" and "a square" share no token; a signature built from tokens cannot
connect them in principle, not merely in practice.

So this is a genuinely new mechanism against the same standard, not a retry of
the same one.

**The gate, written before the run.**

1. On *paraphrased* queries — same meaning, disjoint vocabulary — the embedding
   route must beat the lexical route by a margin **larger than the seed-to-seed
   standard deviation**. That is the §11.5 criterion verbatim, and it is the one
   that refused a +0.014 effect whose interval cleared zero. Clearing zero is
   not the test.
2. On *verbatim* queries the lexical route is already at or near ceiling, and
   the embedding route must **not** be credited for winning there.

**The control, and why it decides the result.** A mechanism that wins on both
conditions is not transferring meaning; it is a better classifier, and its
paraphrase win would be explained by capacity rather than semantics. If
embeddings also win the control, the paraphrase result is uninterpretable and
this reports FAIL regardless of margin. §11.5's control was what made its null
meaningful; the same applies to a positive.

**The control had a ceiling, and that is recorded rather than quietly fixed.**
The first version used *verbatim* queries — the query is the stored text — where
lexical scores 1.000. Embeddings tied at 1.000 and the control was called held.
It was not: at ceiling the control **cannot fail in the direction it exists to
detect**, because nothing can score above 1.0. That is precisely the defect that
invalidated §11.5's first run, arriving from the other side, and it would have
let a pure-capacity advantage through unchallenged.

The control is therefore a **partial-overlap** condition: queries sharing some
vocabulary with their category but not all of it. Lexical lands mid-range, with
room above it, so a mechanism that is merely a better classifier has somewhere
to show that — and the control can genuinely fail. Verbatim is still measured
and reported, now as a *sanity check* that lexical reaches ceiling where it must,
not as the control.

**What it reported, once the control could fail.** FAIL. Paraphrase +0.533 at
5.66× seed sd — a large margin that clears the §11.5 criterion comfortably — but
the partial-overlap control also went to embeddings at **+0.087, sd 0.033**, i.e.
2.6× its own noise. Embeddings are therefore *partly just a better classifier*
here, and under the criterion written before the run that makes the paraphrase
margin uninterpretable. Under the ceilinged first control this would have been
reported as a clean pass at 5.66×.

**The tempting rescue, declined.** A pure-capacity account predicts a roughly
*equal* edge in both conditions, and the paraphrase edge is 6× the control edge
(+0.533 vs +0.087); a difference-in-differences would read +0.446 and pass
easily. That is probably the better-specified test — the control exists precisely
to be subtracted. But it was formulated *after* seeing the numbers, and adopting
a criterion because it converts a fail into a pass is the goalpost move §11.5
refused when a +0.014 effect's interval cleared zero. So: this gate failed. The
difference-in-differences criterion is hereby written down **for a future run on
categories not used here**, where it will be pre-registered rather than
retrofitted, and only that run can pass it.

**The cost, which is part of the verdict.** An embedding call is a network round
trip to a neural model: measured at ~2.0 s for three short strings, against
microseconds for a token-set intersection. A win must be large enough to be
worth four orders of magnitude, and the report states the ratio so that
judgement is not quietly skipped.

Requires a running LM Studio server with an embedding model. Absent one, this
reports SKIPPED — never a pass, and never a silent one.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field

from ultraquant.interpreter.lmstudio import LMStudioClient, LMStudioUnavailable

__all__ = ["EmbedGateReport", "run_gate", "CATEGORIES"]


#: Categories, each with a canonical description and paraphrases that share as
#: little vocabulary with it as English allows. The paraphrases are the whole
#: experiment: if they shared tokens, lexical matching would solve them and
#: there would be nothing to measure. The `partial` queries are the control -
#: they share *some* vocabulary, so lexical lands mid-range with room above it
#: and a capacity advantage would have somewhere to show.
CATEGORIES: dict[str, dict] = {
    "square": {
        "canonical": "a square shape with four equal sides",
        "partial": ["a shape with four sides", "sides of equal length"],
        "paraphrases": [
            "a box outline",
            "an equal-sided rectangle",
            "the figure a chessboard cell has",
        ],
    },
    "circle": {
        "canonical": "a circle, a round closed curve",
        "partial": ["a closed round line", "a curve with no corners"],
        "paraphrases": [
            "a ring drawn on paper",
            "the outline of a coin",
            "a wheel seen head on",
        ],
    },
    "addition": {
        "canonical": "addition, summing two numbers together",
        "partial": ["summing values", "adding numbers up"],
        "paraphrases": [
            "putting quantities together to get a total",
            "what plus does to a pair of values",
            "combining amounts into one larger amount",
        ],
    },
    "sorting": {
        "canonical": "sorting a list into order",
        "partial": ["a list put in order", "ordering by size"],
        "paraphrases": [
            "arranging items from smallest to largest",
            "putting things in sequence by value",
            "ordering a collection",
        ],
    },
    "colour": {
        "canonical": "the colour of an object",
        "partial": ["the colour something looks", "the shade an object has"],
        "paraphrases": [
            "what hue something appears",
            "whether a thing looks red or blue",
            "the shade a surface reflects",
        ],
    },
    "counting": {
        "canonical": "counting how many items there are",
        "partial": ["how many items", "counting a group"],
        "paraphrases": [
            "tallying a number of objects",
            "working out the quantity present",
            "enumerating what is in a group",
        ],
    },
}


@dataclass
class EmbedGateReport:
    """What the gate concluded.

    Attributes:
        passes: The gate verdict under the pre-registered criterion.
        skipped: True when no server was reachable — never a pass.
        paraphrase: ``{"lexical": acc, "embedding": acc, "delta": d}``.
        control: The same, for the partial-overlap control - the condition
            that can actually fail.
        verbatim: The same, for the ceilinged sanity check. Reported, never
            decisive.
        seed_sd: Seed-to-seed standard deviation of the paraphrase delta.
        margin_ratio: ``delta / seed_sd`` — the number §11.5 turned on.
        control_violated: True if embeddings also beat lexical on the
            partial-overlap control - a capacity advantage, not meaning.
        cost: Timing for both routes.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    paraphrase: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)
    verbatim: dict = field(default_factory=dict)
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    control_violated: bool = False
    cost: dict = field(default_factory=dict)
    reason: str = ""

    def as_text(self) -> str:
        """A printable report, including the parts that argue against passing."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        lines = [
            f"paraphrased queries   lexical {self.paraphrase['lexical']:.3f}   "
            f"embedding {self.paraphrase['embedding']:.3f}   "
            f"delta {self.paraphrase['delta']:+.3f}",
            f"partial (CONTROL)     lexical {self.control['lexical']:.3f}   "
            f"embedding {self.control['embedding']:.3f}   "
            f"delta {self.control['delta']:+.3f}   <- must not favour embedding",
            f"verbatim (sanity)     lexical {self.verbatim['lexical']:.3f}   "
            f"embedding {self.verbatim['embedding']:.3f}   "
            f"(at ceiling by construction; not the control)",
            f"seed sd {self.seed_sd:.3f}   margin ratio {self.margin_ratio:.2f}x"
            "  (gate needs > 1.0)",
        ]
        if self.cost:
            lines.append(
                f"cost   lexical {self.cost['lexical_ms']:.3f} ms   "
                f"embedding {self.cost['embedding_ms']:.1f} ms   "
                f"({self.cost['ratio']:,.0f}x slower)"
            )
        if self.control_violated:
            lines.append("CONTROL VIOLATED - embeddings also win where lexical "
                         "should be unbeatable; result uninterpretable")
        lines.append(("PASS - " if self.passes else "FAIL - ") + self.reason)
        return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    """Content tokens, the way a lexical signature would take them."""
    stop = {"a", "an", "the", "of", "to", "and", "or", "is", "it", "in", "on",
            "with", "what", "that", "into", "two", "one", "from", "by", "does",
            "has", "there", "how", "many", "something", "things", "thing"}
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w for w in words if w not in stop and len(w) > 1}


def _lexical_route(query: str, canonical: dict[str, set[str]]) -> str:
    """Best category by Jaccard token overlap - the existing mechanism."""
    best, best_score = "", -1.0
    for name, tokens in canonical.items():
        q = _tokens(query)
        union = q | tokens
        score = len(q & tokens) / len(union) if union else 0.0
        if score > best_score:
            best, best_score = name, score
    return best


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def _embedding_route(vector: list[float], canonical: dict[str, list[float]]) -> str:
    return max(canonical, key=lambda name: _cosine(vector, canonical[name]))


def run_gate(client: LMStudioClient | None = None, seeds: int = 8,
             model: str | None = None) -> EmbedGateReport:
    """Run the gate.

    Args:
        client: An :class:`LMStudioClient`; one is constructed if omitted.
        seeds: How many shuffles of the category set to average over. The
            seed-to-seed spread is the denominator of the gate criterion, so
            this must be > 1 to mean anything.
        model: Embedding model id; auto-detected if omitted.

    Returns:
        The :class:`EmbedGateReport`. Unreachable server yields ``skipped``.
    """
    client = client or LMStudioClient()
    try:
        if not client.available():
            return EmbedGateReport(skipped=True,
                                   reason="no LM Studio server reachable")
        chosen = model or client._default_model(embedding=True)
    except LMStudioUnavailable as exc:
        return EmbedGateReport(skipped=True, reason=str(exc))

    names = list(CATEGORIES)
    canonical_text = {n: CATEGORIES[n]["canonical"] for n in names}
    canon_tokens = {n: _tokens(t) for n, t in canonical_text.items()}

    paraphrase_queries = [(p, n) for n in names
                          for p in CATEGORIES[n]["paraphrases"]]
    verbatim_queries = [(canonical_text[n], n) for n in names]
    partial_queries = [(p, n) for n in names for p in CATEGORIES[n]["partial"]]

    # One batched embedding call for everything; per-call latency would
    # otherwise dominate and tell us nothing about the representation.
    to_embed = ([canonical_text[n] for n in names]
                + [q for q, _ in paraphrase_queries]
                + [q for q, _ in verbatim_queries]
                + [q for q, _ in partial_queries])
    started = time.perf_counter()
    try:
        vectors = client.embed(to_embed, model=chosen)
    except LMStudioUnavailable as exc:
        return EmbedGateReport(skipped=True, reason=str(exc))
    embed_seconds = time.perf_counter() - started

    canon_vec = {n: vectors[i] for i, n in enumerate(names)}
    offset = len(names)
    para_vec = vectors[offset:offset + len(paraphrase_queries)]
    verb_start = offset + len(paraphrase_queries)
    verb_vec = vectors[verb_start:verb_start + len(verbatim_queries)]
    part_vec = vectors[verb_start + len(verbatim_queries):]

    def score(queries, query_vectors, subset):
        """Accuracy of both routes over a category subset."""
        lex_ok = emb_ok = 0
        counted = 0
        sub_tokens = {n: canon_tokens[n] for n in subset}
        sub_vecs = {n: canon_vec[n] for n in subset}
        for (query, truth), vector in zip(queries, query_vectors):
            if truth not in subset:
                continue
            counted += 1
            lex_ok += _lexical_route(query, sub_tokens) == truth
            emb_ok += _embedding_route(vector, sub_vecs) == truth
        return (lex_ok / counted, emb_ok / counted) if counted else (0.0, 0.0)

    # Seeds shuffle which categories compete, so the delta gets a spread rather
    # than a single number that cannot be compared against its own noise.
    para_deltas, verb_deltas, ctrl_deltas = [], [], []
    para_lex, para_emb, verb_lex, verb_emb = [], [], [], []
    ctrl_lex, ctrl_emb = [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        subset = set(rng.sample(names, max(3, len(names) - 1)))
        pl, pe = score(paraphrase_queries, para_vec, subset)
        vl, ve = score(verbatim_queries, verb_vec, subset)
        cl, ce = score(partial_queries, part_vec, subset)
        para_lex.append(pl); para_emb.append(pe); para_deltas.append(pe - pl)
        verb_lex.append(vl); verb_emb.append(ve); verb_deltas.append(ve - vl)
        ctrl_lex.append(cl); ctrl_emb.append(ce); ctrl_deltas.append(ce - cl)

    delta = statistics.fmean(para_deltas)
    seed_sd = statistics.pstdev(para_deltas) if len(para_deltas) > 1 else 0.0
    verb_delta = statistics.fmean(verb_deltas)
    ctrl_delta = statistics.fmean(ctrl_deltas)
    ctrl_sd = statistics.pstdev(ctrl_deltas) if len(ctrl_deltas) > 1 else 0.0

    started = time.perf_counter()
    for query, _ in paraphrase_queries:
        _lexical_route(query, canon_tokens)
    lex_ms = (time.perf_counter() - started) * 1000 / len(paraphrase_queries)
    emb_ms = embed_seconds * 1000 / len(to_embed)

    report = EmbedGateReport(
        paraphrase={"lexical": statistics.fmean(para_lex),
                    "embedding": statistics.fmean(para_emb), "delta": delta},
        control={"lexical": statistics.fmean(ctrl_lex),
                 "embedding": statistics.fmean(ctrl_emb), "delta": ctrl_delta},
        verbatim={"lexical": statistics.fmean(verb_lex),
                  "embedding": statistics.fmean(verb_emb), "delta": verb_delta},
        seed_sd=seed_sd,
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else
                     (float("inf") if delta > 0 else 0.0),
        cost={"lexical_ms": lex_ms, "embedding_ms": emb_ms,
              "ratio": emb_ms / lex_ms if lex_ms else float("inf")},
    )
    # The control is checked first: if it fails, the margin is not evidence of
    # anything and the size of the win is irrelevant.
    # The control is the partial-overlap condition, not the ceilinged verbatim
    # one. It is violated when embeddings win there by more than its own seed
    # noise - a real capacity advantage, which would explain the paraphrase
    # margin without any appeal to meaning.
    report.control_violated = ctrl_delta > max(ctrl_sd, 1e-9)
    if report.control_violated:
        report.reason = (
            f"embeddings also beat lexical on the partial-overlap control by "
            f"{ctrl_delta:+.3f} (control sd {ctrl_sd:.3f}); that is a capacity "
            "advantage, so the paraphrase margin cannot be attributed to meaning"
        )
        return report
    if delta <= 0:
        report.reason = f"no paraphrase advantage ({delta:+.3f})"
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (
            f"margin {delta:+.3f} is {report.margin_ratio:.2f}x the seed sd "
            f"({seed_sd:.3f}); the gate asks for larger than seed variance, and "
            "an average that clears zero is not that"
        )
        return report
    report.passes = True
    report.reason = (
        f"paraphrase margin {delta:+.3f} at {report.margin_ratio:.2f}x seed sd, "
        f"and the partial-overlap control held ({ctrl_delta:+.3f}, sd {ctrl_sd:.3f})"
    )
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
