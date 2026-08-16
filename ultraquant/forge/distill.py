"""Distil a transformer's paraphrase knowledge into the lexical router, offline.

This is the answer §11.11 left open. That gate measured a real thing: on
paraphrased queries — same meaning, disjoint vocabulary — lexical routing scored
0.300 and a local embedding model scored 0.833. Lexical matching cannot connect
"a box outline" to "a square" in principle, because they share no token.

The obvious fix is to embed every query at run time. It was measured and it is a
bad trade: **2.9 ms per string against 0.015 ms for a token-set intersection**, a
197x tax on every routing decision forever, plus a model resident to serve it.

The idea here is to pay that cost **once, offline, and never again**. A local LLM
already knows that a box outline is a square. Ask it for the paraphrases, teach
them to the router with :meth:`CategoryRouter.learn`, and the knowledge ends up
in the same token-weight table the router already had. Query time stays at
microseconds. Nothing new is resident. The transformer is a *teacher consulted at
build time*, not a dependency at inference time — which is the whole point when
VRAM is the constraint.

**What this is not.** It does not copy weights, and it could not: a transformer's
representation and a token-weight table have nothing in common to copy between.
It transfers *labelled examples*, which is the only currency these two
architectures share.

**Found is still not believed.** Generated phrasings are training data for a
router, not facts, and they are never promoted to memory. Routing a query to the
wrong category is recoverable and visible; a wrong stored fact is neither. Where
a caller wants the model's *claims* rather than its *phrasings*, that goes
through :func:`~ultraquant.interpreter.llmls.quarantine_answer` instead.

The gate is in :func:`run_gate`, written before the run, and it is held to the
§11.5 standard: the margin must exceed seed variance, and a control that can
actually fail must not move.

**Two mechanisms were run against it, and the first failed.**

*Raw phrasings.* Teaching the generated sentences whole moved held-out
paraphrase routing 0.035 -> 0.396, a margin of **6.31x seed variance** — and
the decoy control **collapsed from 1.000 to 0.133**. Queries like "how do I fix
a leaking tap" were being pulled into the taught categories, because natural
phrasings are dense in function words and ``what``/``how``/``do``/``the`` had
acquired weight for every taught category. The router had not learned meaning;
it had learned to say yes. A ceilinged control would have reported a clean pass
at 6.31x.

*Content tokens only.* Filtering to the words that actually name something
(:func:`content_tokens`) gives **0.028 -> 0.090, +0.062 at 1.79x seed sd, with
the control held exactly at 1.000**. That passes.

**What passing does and does not mean here, stated plainly.** The improvement
is real, controlled and reproducible — and it is *small in absolute terms*.
Routing goes from 2.8% to 9.0% on held-out paraphrases. Both numbers are poor.
§11.11 measured the embedding route at **0.833** on the same kind of task, so
distillation recovers a small fraction of what a resident embedding model
delivers. It is the right trade when VRAM is the binding constraint and query
latency must stay in microseconds; it is **not** a substitute for embeddings
where accuracy is what matters. Anyone reading "PASS" as "routing now works"
would be misreading it.

**And it does not scale — which is the finding that matters most.** The obvious
next move after a small pass is to generate more data. Measured, that breaks it:

===============================  ==========  =========  ========  ===========
volume                           before      after      margin    decoy control
===============================  ==========  =========  ========  ===========
48 phrasings (12 per category)   0.028       0.090      1.79x     **1.000**
117 phrasings (30 per category)  0.000       0.492      8.93x     **0.767**
===============================  ==========  =========  ========  ===========

At the larger volume the margin looks *spectacular* — half the held-out
paraphrases routed, nearly 9x seed variance — and the control has broken again:
roughly a quarter of unrelated decoys are back to being misrouted. Content
filtering delayed the yes-machine effect; it did not remove it. Enough learned
tokens, however carefully chosen, still make every taught category match
everything.

So :data:`SAFE_PHRASINGS_PER_CATEGORY` caps what :func:`run_gate` and callers
request, and the cap is a **measured limit, not a tuning knob**. Raising it will
make the reported margin go up and the result stop meaning anything, which is
exactly the trap the control exists to catch. There is no evidence the ceiling
sits at 48 rather than 60 or 80 — only that 48 holds and 117 does not.

**One further caveat on the numbers.** Generation is not deterministic even at
temperature 0: three identical runs at 48 phrasings gave 0.090, 0.111 and 0.118.
The margin therefore carries run-to-run variance *on top of* the seed variance
the gate measures, and the gate does not currently account for it.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from ultraquant.interpreter.llmls import (
    NO_REASONING,
    LMStudioUnavailable,
    TeacherPanel,
)

__all__ = [
    "DistillReport",
    "GateReport",
    "generate_phrasings",
    "distil_into_router",
    "run_gate",
]

#: Asked of the teacher. Phrasings, not definitions: the router needs to learn
#: what a *query* about this category looks like, and a dictionary definition is
#: not how anyone asks for something.
_PROMPT = (
    "List {n} different short ways someone might ask about or refer to "
    "\"{topic}\". Use everyday wording. Vary the vocabulary as much as "
    "possible - avoid reusing the words in \"{topic}\" itself. "
    "One per line, no numbering, no explanation."
)

#: A phrasing must be usable as a query: long enough to carry tokens, short
#: enough not to be an essay the model wandered into.
_MIN_WORDS, _MAX_WORDS = 2, 14

#: How many phrasings to request per category, per teacher. A **measured
#: ceiling**, not a preference.
#:
#: At 12 (48 phrasings over four categories) the gate passes with the decoy
#: control held exactly at 1.000. At 30 (117 phrasings) the margin rises to
#: 8.93x seed variance and the control **falls to 0.767** — a quarter of
#: unrelated queries misrouted. More data does not make this mechanism better;
#: past a point it re-creates the yes-machine that content filtering was
#: introduced to stop.
#:
#: Raising this will make the reported margin go up and the result stop meaning
#: anything. Nothing here establishes that the true ceiling is 12 rather than
#: 15 or 20 — only that 12 holds and 30 does not.
SAFE_PHRASINGS_PER_CATEGORY = 12


@dataclass
class DistillReport:
    """What a distillation run produced.

    Attributes:
        category: The category taught.
        phrasings: The accepted phrasings, in the order they were learned.
        rejected: Candidates dropped, with the reason - kept so a run that
            produced little can be diagnosed rather than guessed at.
        models: Which teachers were consulted.
    """

    category: str
    phrasings: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        """A printable summary."""
        lines = [f"{self.category}: {len(self.phrasings)} phrasing(s) from "
                 f"{len(self.models)} teacher(s)"]
        lines += [f"    {p}" for p in self.phrasings[:8]]
        if len(self.phrasings) > 8:
            lines.append(f"    ... and {len(self.phrasings) - 8} more")
        if self.rejected:
            lines.append(f"  {len(self.rejected)} rejected "
                         f"(e.g. {self.rejected[0][1]})")
        return "\n".join(lines)


def _clean(line: str) -> str:
    """Strip list markers and punctuation a model adds despite being asked not to."""
    text = line.strip()
    while text[:1] in "-*+0123456789.)• " and text[:1]:
        text = text[1:].lstrip()
        if not text:
            break
    return " ".join(text.strip(" \"'.,;:").split())


def generate_phrasings(
    panel: TeacherPanel,
    category: str,
    topic: str | None = None,
    count: int = SAFE_PHRASINGS_PER_CATEGORY,
    max_tokens: int = 400,
) -> DistillReport:
    """Ask every panel member for query phrasings of one category.

    Each teacher is asked separately and *all* their answers are kept, unlike
    :meth:`TeacherPanel.ask`. Consensus is the wrong instrument here: the point
    of several teachers is more vocabulary coverage, and two models offering
    different phrasings is the desired outcome rather than a disagreement to
    resolve. There is nothing to corroborate — a phrasing is not a claim.

    Args:
        panel: The teachers.
        category: The router category these phrasings should route to.
        topic: What to describe, if the category name is not it.
        count: Phrasings to request per teacher, defaulting to the measured
            :data:`SAFE_PHRASINGS_PER_CATEGORY`. Raising it inflates the
            apparent margin while breaking the decoy control.
        max_tokens: Generous, because reasoning models bill thinking against
            this budget (see ``llmls.DEFAULT_MAX_TOKENS``). Requests are sent
            with ``reasoning_effort="none"``, without which some models never
            emit anything at all.

    Returns:
        A :class:`DistillReport`. An unreachable teacher is recorded and the
        others still contribute.
    """
    subject = topic or category
    prompt = _PROMPT.format(n=count, topic=subject)
    report = DistillReport(category=category)
    seen: set[str] = set()
    for card in panel.cards:
        try:
            # Listing phrasings needs no chain of thought, and some local
            # reasoning models will otherwise spend the entire budget thinking
            # and emit nothing - measured at 400/400 and 1200/1200 tokens on
            # qwen3.6-35b-a3b, empty both times. reasoning_effort="none" is the
            # only remedy that worked.
            answer = panel.client.complete(prompt, model=card.id, system="",
                                           max_tokens=max_tokens,
                                           reasoning_effort=NO_REASONING)
        except LMStudioUnavailable as exc:
            report.rejected.append((card.id, f"unavailable: {exc}"))
            continue
        report.models.append(card.id)
        for line in (answer.text or "").splitlines():
            phrase = _clean(line)
            words = phrase.split()
            key = phrase.lower()
            if not phrase:
                continue
            if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
                report.rejected.append((phrase, "wrong length for a query"))
            elif key in seen:
                report.rejected.append((phrase, "duplicate"))
            else:
                seen.add(key)
                report.phrasings.append(phrase)
    return report


#: Verbs a model reaches for when asked how someone might *ask* about a topic.
#: They carry no topic and belong with the function words, but a general
#: stopword list has no reason to contain them.
_ASKING_VERBS = frozenset({
    "describe", "name", "point", "discuss", "explain", "list", "identify",
    "call", "look", "looking", "talk", "mention", "define", "give", "show",
    "want", "need", "know", "see", "find", "get", "make", "take", "way",
    "ways", "thing", "things", "stuff", "something", "someone",
})


def content_tokens(text: str) -> list[str]:
    """The tokens of ``text`` that actually name a topic.

    Distillation supplies *many* synthetic phrasings at once, and natural
    phrasings are dense in function words — "what are those forms?", "how do
    you call these outlines?". Feeding them wholesale to
    :meth:`CategoryRouter.learn` puts weight on ``what``, ``how``, ``do`` and
    ``the`` for the taught category, and those tokens match **everything**.

    That is not a hypothesis. The first run of :func:`run_gate` taught raw
    phrasings and reported held-out paraphrase routing up 0.035 -> 0.396, a
    margin of 6.31x seed variance — and the decoy control collapsed from
    **1.000 to 0.133**. Unrelated queries like "how do I fix a leaking tap"
    were being pulled into the taught categories. The router had not learned
    meaning; it had learned to say yes.

    A single user typing one query at a time contributes far less generic
    weight, which is why :meth:`CategoryRouter.learn` is left alone and the
    filtering happens here, where the volume is.
    """
    from ultraquant.interpreter.learning import _STOPWORDS

    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return [w for w in words
            if len(w) > 2 and w not in _STOPWORDS and w not in _ASKING_VERBS]


def distil_into_router(router, category: str, phrasings: Sequence[str],
                       delta: float = 0.1) -> int:
    """Teach the *content* of phrasings to the router. Returns how many landed.

    Only tokens that name something are learned; see :func:`content_tokens` for
    the measurement that forced that. A phrasing whose every word is a function
    word teaches nothing and is not counted.

    Nothing here reaches memory or the stash. These are routing hints, and the
    router already learns from real usage the same way — this only supplies
    more examples than one user would produce in a session.
    """
    applied = 0
    for phrase in phrasings:
        content = content_tokens(phrase)
        if content:
            router.learn(" ".join(content), category, delta=delta)
            applied += 1
    return applied


@dataclass
class GateReport:
    """The pre-registered verdict.

    Attributes:
        passes: Whether the gate criterion was met.
        skipped: True when no teacher was reachable. Never a pass.
        before: Mean routing accuracy on held-out paraphrases, pre-distillation.
        after: The same, post-distillation.
        delta: ``after - before``.
        seed_sd: Seed-to-seed standard deviation of the delta.
        margin_ratio: ``delta / seed_sd`` — the number §11.5 turned on.
        control_before: Accuracy on *decoy* queries that belong to no taught
            category, before.
        control_after: The same, after.
        control_violated: True when distillation dragged decoys into the taught
            categories — a router that says yes more often, not one that knows
            more.
        reason: Plain-language verdict.
    """

    passes: bool = False
    skipped: bool = False
    before: float = 0.0
    after: float = 0.0
    delta: float = 0.0
    seed_sd: float = 0.0
    margin_ratio: float = 0.0
    control_before: float = 0.0
    control_after: float = 0.0
    control_violated: bool = False
    reason: str = ""
    phrasings: int = 0

    def as_text(self) -> str:
        """A printable report, including what argues against passing."""
        if self.skipped:
            return f"SKIPPED - {self.reason}"
        return "\n".join([
            f"held-out paraphrases   before {self.before:.3f}   "
            f"after {self.after:.3f}   delta {self.delta:+.3f}",
            f"decoys (CONTROL)       before {self.control_before:.3f}   "
            f"after {self.control_after:.3f}   "
            f"delta {self.control_after - self.control_before:+.3f}"
            "   <- must not FALL (it is decoys correctly left alone)",
            f"seed sd {self.seed_sd:.3f}   margin ratio "
            f"{self.margin_ratio:.2f}x  (gate needs > 1.0)",
            f"{self.phrasings} phrasing(s) learned",
            ("PASS - " if self.passes else "FAIL - ") + self.reason,
        ])


def run_gate(panel: TeacherPanel | None = None, seeds: int = 6,
             holdout: float = 0.5) -> GateReport:
    """Does distillation actually improve routing on unseen paraphrases?

    **The gate, written before the run.** Teach the router half the generated
    phrasings; test on the other half, which it never saw. The margin must
    exceed the seed-to-seed standard deviation — the §11.5 criterion, the one
    that refused a +0.014 effect whose interval cleared zero.

    **The control, and why it can fail.** Decoy queries belonging to none of the
    taught categories are routed before and after. Distillation adds token
    weight, and adding weight makes *every* query score higher against the
    taught categories — a router that has merely become more willing to answer
    would show exactly the same gain on real paraphrases as this gate is
    measuring. If decoy accuracy rises, the improvement is not knowledge and
    the run fails whatever the margin.

    This is where §11.11's ceilinged control was wrong, so this one is chosen to
    be able to move in the direction that would condemn the result.

    Args:
        panel: Teachers; constructed from the settings' saved panel if omitted.
        seeds: Splits to average over. Must exceed 1 or the sd is meaningless.
        holdout: Fraction of phrasings withheld from teaching.

    Returns:
        The :class:`GateReport`. Unreachable teachers yield ``skipped``.
    """
    import tempfile

    from ultraquant.shards.router import CategoryRouter
    from ultraquant.shards.vault import ShardVault

    topics = {
        "geometry": "shapes and geometric figures",
        "arithmetic": "adding and subtracting numbers",
        "colour": "the colour something is",
        "sorting": "putting a list into order",
    }
    decoys = [
        "how do I fix a leaking tap",
        "what time does the train leave",
        "recipe for banana bread",
        "the weather forecast for tomorrow",
        "who won the football match",
    ]

    if panel is None:
        try:
            from ultraquant.config import Settings

            models = Settings.load().get("lmstudio.panel_models") or []
            if not models:
                return GateReport(skipped=True,
                                  reason="no panel models configured; pick "
                                         "some on the Panel tab first")
            panel = TeacherPanel(models)
        except LMStudioUnavailable as exc:
            return GateReport(skipped=True, reason=str(exc))

    generated: dict[str, list[str]] = {}
    for category, topic in topics.items():
        try:
            report = generate_phrasings(panel, category, topic)
        except LMStudioUnavailable as exc:
            return GateReport(skipped=True, reason=str(exc))
        if len(report.phrasings) < 4:
            return GateReport(skipped=True,
                              reason=f"teacher produced only "
                                     f"{len(report.phrasings)} usable "
                                     f"phrasing(s) for {category}")
        generated[category] = report.phrasings

    # An empty vault: routing here is measured on keywords and learned token
    # weights alone, with no stored prototypes to confound the comparison.
    scratch = tempfile.mkdtemp(prefix="uq_distil_")

    def fresh_router():
        router = CategoryRouter(ShardVault(scratch))
        for category in topics:
            # The keywords the category would have had anyway; distillation is
            # measured as what it adds *on top* of an ordinary registration,
            # not against a router that knows nothing.
            router.register(category, [category])
        return router

    def score(router, cases) -> float:
        hit = 0
        for text, truth in cases:
            ranked = router.route(text, top_k=1)
            if ranked and ranked[0][0] == truth:
                hit += 1
        return hit / len(cases) if cases else 0.0

    befores, afters, deltas = [], [], []
    control_befores, control_afters = [], []
    learned = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        teach: dict[str, list[str]] = {}
        test: list[tuple[str, str]] = []
        for category, phrasings in generated.items():
            shuffled = list(phrasings)
            rng.shuffle(shuffled)
            cut = max(1, int(len(shuffled) * (1.0 - holdout)))
            teach[category] = shuffled[:cut]
            test += [(p, category) for p in shuffled[cut:]]
        if not test:
            continue

        # The decoy's "correct" answer is *no* taught category, so accuracy here
        # is the fraction correctly left alone.
        control = [(text, None) for text in decoys]

        def control_score(router) -> float:
            miss = 0
            for text, _ in control:
                ranked = router.route(text, top_k=1)
                if not ranked or ranked[0][0] not in topics:
                    miss += 1
            return miss / len(control)

        before_router = fresh_router()
        befores.append(score(before_router, test))
        control_befores.append(control_score(before_router))

        after_router = fresh_router()
        for category, phrasings in teach.items():
            learned += distil_into_router(after_router, category, phrasings)
        afters.append(score(after_router, test))
        control_afters.append(control_score(after_router))
        deltas.append(afters[-1] - befores[-1])

    if not deltas:
        return GateReport(skipped=True, reason="no usable splits")

    delta = statistics.fmean(deltas)
    # stdev, not pstdev: these six splits are a *sample* of the possible
    # splits, not the population of them. Population sd divides by n rather
    # than n-1 and so understates the spread, which inflates every margin ratio
    # the gate reports against its own threshold.
    seed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    control_before = statistics.fmean(control_befores)
    control_after = statistics.fmean(control_afters)

    report = GateReport(
        before=statistics.fmean(befores), after=statistics.fmean(afters),
        delta=delta, seed_sd=seed_sd,
        # Zero spread is not infinite confidence. Identical deltas across every
        # split means the splits did not vary what they were meant to vary -
        # too few phrasings, or a degenerate holdout - and reporting inf turned
        # that into an automatic pass on a measurement that measured nothing.
        margin_ratio=(delta / seed_sd) if seed_sd > 0 else 0.0,
        control_before=control_before, control_after=control_after,
        phrasings=learned // max(1, seeds),
    )
    # Control first: if decoys got pulled in, the margin means nothing.
    report.control_violated = control_after < control_before
    if report.control_violated:
        report.reason = (
            f"decoy queries were dragged into taught categories "
            f"({control_before:.3f} -> {control_after:.3f}); the router became "
            "more willing to answer, which is not the same as knowing more"
        )
        return report
    if delta <= 0:
        report.reason = f"no improvement on held-out paraphrases ({delta:+.3f})"
        return report
    if seed_sd <= 0.0:
        report.reason = (
            f"every split gave the same delta ({delta:+.3f}), so there is no "
            "seed variance to measure the margin against; the splits did not "
            "vary what they were supposed to vary"
        )
        return report
    if report.margin_ratio <= 1.0:
        report.reason = (
            f"margin {delta:+.3f} is {report.margin_ratio:.2f}x the seed sd "
            f"({seed_sd:.3f}); the gate asks for larger than seed variance"
        )
        return report
    report.passes = True
    report.reason = (
        f"held-out paraphrase routing {report.before:.3f} -> {report.after:.3f} "
        f"({delta:+.3f}, {report.margin_ratio:.2f}x seed sd) with the decoy "
        f"control held ({control_before:.3f} -> {control_after:.3f})"
    )
    return report


if __name__ == "__main__":                        # pragma: no cover
    print(run_gate().as_text())
