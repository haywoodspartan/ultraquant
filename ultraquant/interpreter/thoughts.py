"""The predefined set of thought.

Every input traverses the same fixed pipeline of thoughts, in order:

    Perceive -> Recall -> Route -> Reason -> Respond -> Learn

Each thought reads and writes a shared :class:`ThoughtContext` and appends a line
to the trace, so the reasoning is inspectable after the fact (``:trace`` in the
chat CLI).  This is symbolic, template-driven reasoning over the model's own
stores — it is not a language model and never pretends to be one.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultraquant.archive.artchive import ArTchive
from ultraquant.experts.moe import ExpertPool
from ultraquant.reason.blackboard import (
    Blackboard,
    ExpertContributor,
    run_blackboard,
)
from ultraquant.interpreter.codefunc import CodeError, SafeCodeRunner
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.interpreter.webaccess import WebAccess, WebDisabled
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.pattern.recognition import LABELS, PATTERNS, render, row_means
from ultraquant.shards.budget import ShardCache
from ultraquant.shards.router import CategoryRouter
from ultraquant.shards.vault import ShardVault

__all__ = [
    "Session",
    "ThoughtContext",
    "PIPELINE",
    "build_session",
    "run_pipeline",
]

_URL_RE = re.compile(r"https?://\S+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: A whole-input affirmation. Deliberately strict: only a bare agreement
#: confirms a pending inference - anything longer is a new thought.
_AFFIRMATION_RE = re.compile(
    r"(yes|correct|right|indeed|confirmed|exactly|"
    r"yes[ ,]+(that is|that's) (right|correct)|that is (right|correct)|"
    r"that's (right|correct))[.!]?")
_GLYPH_ROW = re.compile(r"^[#.]{5}$")

#: A text opening with one of these is a question even without a question mark.
#: Kept to unambiguous leads: "is the shop open" is interrogative, but a bare
#: "is" prefix also matches "is short for ..." style fragments rarely typed, and
#: the cost of misreading one of those as a question (an unanswered lookup) is
#: far below the cost of misreading a question as a statement (junk stored as
#: fact, question unanswered) - which is what happened.
_INTERROGATIVE_LEADS = ("what ", "who ", "where ", "when ", "which ", "why ",
                        "how ", "is ", "are ", "does ", "do ", "can ")

#: Below this similarity to every stored prototype, a pattern is reported as
#: unfamiliar rather than forced onto the nearest category. Chosen from measured
#: distributions: noisy copies of known glyphs never fall below 0.866, so this
#: floor costs almost no false alarms. It is a *dissimilarity* test and nothing
#: more — a novel pattern that genuinely resembles a known one still gets
#: recognised as it, which is the right answer more often than not.
UNFAMILIAR_BELOW = 0.85

#: Among *factored* slots, a slot joins the composition only when its score is
#: at least this fraction of the best factored slot's. Whole-vs-factored is a
#: separate rule (see _compose_reading); this margin only trims stray factored
#: slots that merely appeared in the route list.
_COMPOSE_COLEADER = 0.9

#: The slot a category holds when it never declared one — the whole-pattern
#: interpretation, which competes with factored readings and never joins them.
from ultraquant.shards.vault import _DEFAULT_SLOT as _WHOLE_SLOT

#: Where a glyph goes when nothing routes it — the one category whose
#: untrained remedy really is a drawn demonstration.
GLYPH_FALLBACK_CATEGORY = "geometry"

#: Categories the router knows about out of the box.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "geometry": ["shape", "glyph", "square", "diamond", "plus", "cross", "pattern", "symbol"],
    "arithmetic": ["calc", "compute", "number", "sum", "math", "arithmetic", "total"],
    "language": ["word", "text", "sentence", "language", "meaning", "spell"],
    "world": ["fact", "who", "what", "where", "history", "world", "country"],
}


@dataclass
class Session:
    """Everything one interpreter session needs, wired together."""

    root: Path
    memory: SystematicMemory
    vault: ShardVault
    cache: ShardCache
    router: CategoryRouter
    experts: ExpertPool
    web: WebAccess
    stash: ContemporaryStash
    coder: SafeCodeRunner
    archive: ArTchive | None = None
    recognizer: Any | None = None
    rng: random.Random = field(default_factory=random.Random)
    last_trace: list[dict] = field(default_factory=list)
    #: The backend the shard library lives on, if one was configured.
    storage: Any | None = None
    #: The embedding suggester (reason/semantic.py), or None. Sessions
    #: default to None so behavior is deterministic without LM Studio;
    #: the live surfaces enable it from settings, and the gate measured
    #: it against the OFF arm before it counted (§11.39).
    semantic: Any | None = None
    #: Grounded gaps from refused inferences: each entry names the premise
    #: that would have let a question converge. The learn queue drains
    #: these - the system asking to be taught exactly what it needs.
    curiosities: list = field(default_factory=list)
    #: A chain inference from the IMMEDIATELY previous turn, waiting for
    #: the user to confirm it. One turn of freshness only: Perceive clears
    #: it at the start of every run, so repetition can never consolidate -
    #: that is SS11.16's trap, and only an explicit affirmation promotes.
    pending_inference: dict | None = None
    #: Pattern-driven prefetcher, present when the storage has a RAM tier.
    working_set: Any | None = None
    #: The byte-bounded conversation window (§11.14): recent turns resident in
    #: RAM, every turn on disk, a 24-byte reference each. This is the
    #: short-term workable memory; the vault is the permanent store.
    context: Any | None = None

    def save(self) -> None:
        """Persist every store that has one."""
        self.memory.save()
        self.router.save()


def build_session(
    root: str | Path,
    budget_bytes: int = 1_048_576,
    online: bool = False,
    seed: int = 0,
    storage_uri: str | None = None,
    cache: str | int | None = None,
    prefetch: bool = True,
    semantic: bool = False,
) -> Session:
    """Construct a full interpreter session rooted at ``root``.

    Args:
        root: Directory holding the vault, memory, archive and stash.
        budget_bytes: RAM budget for decoded experts in the shard cache.
        online: Whether web access starts enabled.
        seed: Base seed for every stochastic component.
        storage_uri: Where the shard library actually lives — ``nvmeof://X:/uq``,
            ``rados://models/uq``, and so on. Defaults to plain files under
            ``root/vault``.
        cache: RAM tier budget in front of that storage (``"1GB"``, ``"auto"``,
            or None for no tier). The library stays on storage either way; this
            only decides how much of it may be resident.
        prefetch: Let the Route thought pull the shards it expects to need into
            the RAM tier before Reason asks for them.
        semantic: Enable the embedding suggester for questions the lexical
            core cannot assert on. Off by default: without it, behavior is
            byte-identical whether or not LM Studio exists.

    Returns:
        A ready :class:`Session`.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    memory = SystematicMemory(path=root / "memory.json")

    storage = None
    if storage_uri or cache:
        from ultraquant.storage import open_storage

        storage = open_storage(storage_uri or f"local://{(root / 'vault').as_posix()}",
                               cache=cache)

    vault = ShardVault(root / "vault", storage=storage)
    shard_cache = ShardCache(max_bytes=budget_bytes)
    # Facts become catalogued shards like everything else the model learns:
    # paged on demand, reinforced on use, findable through the vault's keyword
    # index, and carried inside a packed library when it is copied elsewhere.
    memory.shards = FactShards(vault, cache=shard_cache)
    if memory._facts:
        memory.shards.migrate(memory._facts)
        memory._facts = {}
    router = CategoryRouter(vault, memory=memory, path=root / "vault" / "router.json")
    for category, keywords in DEFAULT_CATEGORIES.items():
        router.register(category, keywords)
    experts = ExpertPool(vault, shard_cache, input_dim=30, hidden=(32,), seed=seed)

    working_set = None
    if prefetch and storage is not None and hasattr(storage, "pin"):
        from ultraquant.shards.working_set import PatternWorkingSet

        working_set = PatternWorkingSet(vault, router, tier=storage)

    from ultraquant.memory.context import ContextWindow

    session = Session(
        root=root,
        memory=memory,
        vault=vault,
        cache=shard_cache,
        router=router,
        experts=experts,
        web=WebAccess(online=online),
        stash=ContemporaryStash(root / "stash.json"),
        coder=SafeCodeRunner(),
        archive=ArTchive(root / "artchive"),
        rng=random.Random(seed),
        storage=storage,
        working_set=working_set,
        context=ContextWindow(root / "context"),
    )
    if semantic:
        from ultraquant.reason.semantic import SemanticSuggester

        session.semantic = SemanticSuggester()
    # The hot tier (§11.45): bit-exact against the Python path and
    # structurally fallback-safe, so it defaults on wherever a CUDA
    # device exists. Absence costs one probe at build time.
    try:
        from ultraquant.native import accel as _accel

        if _accel.gpu_available():
            from ultraquant.native.vram import VramLayerCache

            session.experts.vram = VramLayerCache()
    except Exception:  # noqa: BLE001 - no GPU, no tier, no change
        pass
    return session


@dataclass
class ThoughtContext:
    """Mutable state carried through one pass of the pipeline."""

    text: str
    session: Session
    trace: list[dict] = field(default_factory=list)
    response_parts: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def note(self, thought: str, summary: str, **extra: Any) -> None:
        """Append one line to the reasoning trace."""
        entry = {"thought": thought, "summary": summary}
        entry.update(extra)
        self.trace.append(entry)

    def say(self, text: str) -> None:
        """Append a fragment to the response."""
        if text:
            self.response_parts.append(text)


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of ``text``."""
    return _TOKEN_RE.findall(text.lower())


def _glyph_rows(text: str) -> list[str] | None:
    """Return five 5-character glyph rows if ``text`` is a glyph, else None."""
    rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(rows) == 5 and all(_GLYPH_ROW.match(r) for r in rows):
        return rows
    return None


class Thought:
    """Base class for a stage of the pipeline."""

    name = "thought"

    def run(self, ctx: ThoughtContext) -> None:  # pragma: no cover - interface
        """Advance the context."""
        raise NotImplementedError


class Perceive(Thought):
    """Tokenize the input and decide what kind of thing it is."""

    name = "Perceive"

    def run(self, ctx: ThoughtContext) -> None:
        text = ctx.text.strip()
        tokens = _tokens(text)
        ctx.data["tokens"] = tokens

        rows = _glyph_rows(text)
        lowered = text.lower()
        # One-turn freshness for a pending inference: whatever this input
        # is, the previous turn's derivation is no longer pending after it.
        pending = getattr(ctx.session, "pending_inference", None)
        ctx.session.pending_inference = None
        if pending is not None and _AFFIRMATION_RE.fullmatch(lowered.strip()):
            ctx.data["intent"] = "affirmation"
            ctx.data["affirmed_inference"] = pending
            ctx.note(self.name, "intent=affirmation, confirms pending "
                                "inference", intent="affirmation")
            return
        if rows is not None:
            intent = "glyph"
            ctx.data["glyph_rows"] = rows
        elif lowered.startswith(("calc:", "code:")):
            intent = "code"
            ctx.data["code"] = text.split(":", 1)[1].strip()
        elif lowered.startswith("goal:"):
            # An explicit goal, so the pipeline stops being one-shot: the
            # planner works out which capabilities to use and in what order.
            # It is a prefix rather than a guess because inferring intent from
            # free text is a different problem, and getting it wrong here would
            # silently route ordinary questions into a search.
            intent = "goal"
            ctx.data["goal_text"] = text.split(":", 1)[1].strip()
        elif _URL_RE.search(text):
            intent = "url"
            ctx.data["url"] = _URL_RE.search(text).group(0)
        elif lowered.startswith("remember") or (
                " is " in lowered and not text.endswith("?")
                and not lowered.startswith(_INTERROGATIVE_LEADS)):
            # An interrogative lead beats the " is " heuristic: "how tall is
            # the tower" contains " is " and was being read as a *statement*,
            # which both refused to answer it and stored junk - the fact
            # "how tall" = "the tower" was really written by this branch.
            intent = "fact_statement"
        elif text.endswith("?") or lowered.startswith(_INTERROGATIVE_LEADS):
            intent = "question"
        else:
            intent = "chat"

        ctx.data["intent"] = intent
        ctx.note(self.name, f"intent={intent}, {len(tokens)} tokens", intent=intent)


class Recall(Thought):
    """Pull anything already known that bears on the input."""

    name = "Recall"

    def run(self, ctx: ThoughtContext) -> None:
        memory = ctx.session.memory
        tokens = ctx.data.get("tokens", [])
        hits: list[tuple[str, dict]] = []
        for key in _candidate_keys(ctx.text, tokens):
            fact = memory.recall_fact(key)
            if fact is not None:
                hits.append((key, fact))
        ctx.data["facts"] = hits
        episodes = memory.recall_episodes(limit=3)
        ctx.data["episodes"] = episodes

        # The context window (§11.14): turns still resident are the working
        # memory and cost nothing to consult; turns evicted from RAM are
        # reachable through their 24-byte references, so a subject buried
        # hundreds of turns ago is one screened seek away rather than gone.
        # Gate measured: buried-fact recall 0.000 -> 0.896 for a 512 B
        # resident budget.
        recovered: list[dict] = []
        window = ctx.session.context
        if window is not None and ctx.text.strip():
            recovered = window.recall(ctx.text, top_k=2)
            ctx.data["recovered_turns"] = recovered
        ctx.note(
            self.name,
            f"{len(hits)} matching fact(s), {len(episodes)} recent episode(s)"
            + (f", {len(recovered)} turn(s) paged back from disk"
               if recovered else ""),
            facts=[k for k, _ in hits],
        )


def _candidate_keys(text: str, tokens: list[str]) -> list[str]:
    """Fact keys worth trying for this input, most specific first."""
    lowered = text.lower().strip().strip("?.!")
    keys: list[str] = []
    for lead in ("what is ", "who is ", "where is ", "what are ", "remember that "):
        if lowered.startswith(lead):
            keys.append(lowered[len(lead):].strip())
    if " is " in lowered:
        keys.append(lowered.split(" is ", 1)[0].strip())
    keys.append(lowered)
    # Keys are stored with leading articles stripped (extract_facts does it),
    # so every candidate gets an article-stripped variant too — without this,
    # "what is the number of glyph pixels?" never tried the key the statement
    # path actually wrote, fell through to the 1-gram "glyph", and answered
    # from the wrong, less specific fact.
    for key in list(keys):
        for article in ("the ", "a ", "an "):
            if key.startswith(article):
                keys.append(key[len(article):])
    # Also try progressively shorter token n-grams (the "sky color" style key),
    # longest first so the most specific stored key wins when several match.
    for size in (5, 4, 3, 2, 1):
        for i in range(len(tokens) - size + 1):
            keys.append(" ".join(tokens[i:i + size]))
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered[:24]


class Route(Thought):
    """Rank categories and note which experts would have to be paged in."""

    name = "Route"

    def run(self, ctx: ThoughtContext) -> None:
        session = ctx.session
        # A glyph has no words in it, so keyword routing finds nothing and every
        # pattern would fall through to one default category — leaving the
        # library's pattern experts unreachable. Patterns route on content.
        ranked: list[tuple[str, float]] = []
        if ctx.data.get("intent") == "glyph" and ctx.data.get("glyph_rows"):
            pixels = render(list(ctx.data["glyph_rows"]))
            ctx.data["pixels"] = pixels
            ranked = session.router.route_pattern(
                pixels, top_k=3, min_similarity=UNFAMILIAR_BELOW,
            )
            if not ranked:
                # Nothing stored looks like this; say so rather than
                # guessing, and let Reason offer to learn it.
                ctx.data['unfamiliar'] = True
                loose = session.router.route_pattern(pixels, top_k=1)
                ctx.data['nearest'] = loose[0] if loose else None
            if ranked:
                ctx.data["routed_by"] = "pattern"
        if not ranked:
            ranked = session.router.route(ctx.text, top_k=3)
        if not ranked and ctx.data.get("intent") == "glyph":
            ranked = [("geometry", 0.0)]
        ctx.data["routes"] = ranked
        resident = set(session.cache.resident())
        would_load = [
            cat for cat, _ in ranked if ExpertPool.shard_id(cat) not in resident
        ]
        ctx.data["would_load"] = would_load

        # This is the point of the whole design: recognition has just named the
        # categories, so the shards those categories need can be pulled off slow
        # storage *now*, while Reason has not asked for them yet. Nothing here
        # scales with the size of the library.
        prefetched = None
        if session.working_set is not None and ranked:
            report = session.working_set.prefetch(
                ctx.text, labels=ctx.data.get("glyph_labels"), limit=len(ranked),
            )
            if report.loaded or report.already_resident:
                prefetched = {
                    "loaded": report.loaded,
                    "hot": report.already_resident,
                    "bytes": report.bytes_loaded,
                    "ms": round(report.seconds * 1000, 2),
                }
                ctx.data["prefetch"] = prefetched

        top = ranked[0][0] if ranked else "(none)"
        summary = f"top={top}; {len(would_load)} expert shard(s) not resident"
        if prefetched:
            summary += (
                f"; prefetched {prefetched['loaded']} "
                f"({prefetched['bytes']:,}b in {prefetched['ms']}ms)"
            )
        ctx.note(
            self.name,
            summary,
            routes=[c for c, _ in ranked],
            would_load=would_load,
            prefetch=prefetched,
        )


class Reason(Thought):
    """Do the actual work implied by the intent."""

    name = "Reason"

    def run(self, ctx: ThoughtContext) -> None:
        intent = ctx.data.get("intent")
        handler = {
            "code": self._code,
            "goal": self._goal,
            "url": self._url,
            "glyph": self._glyph,
            "question": self._question,
            "fact_statement": self._fact,
            "affirmation": self._affirm,
        }.get(intent, self._chat)
        handler(ctx)

    # -- handlers ---------------------------------------------------------

    def _code(self, ctx: ThoughtContext) -> None:
        source = ctx.data.get("code", "")
        try:
            result = ctx.session.coder.run(source)
        except CodeError as exc:
            ctx.say(f"The code function refused or failed: {exc}")
            ctx.note(self.name, f"code error: {exc}")
            return
        if result["stdout"]:
            ctx.say(result["stdout"].rstrip())
        if result["result"] is not None:
            ctx.say(f"= {result['result']}")
        elif not result["stdout"]:
            ctx.say(f"Ran the code; defined {', '.join(result['defined']) or 'nothing'}.")
        ctx.data["code_result"] = result
        ctx.note(self.name, f"code ran, result={result['result']!r}")

    def _url(self, ctx: ThoughtContext) -> None:
        session = ctx.session
        url = ctx.data["url"]
        try:
            page = session.web.fetch(url)
        except WebDisabled:
            ctx.say("Web access is off. Turn it on with ':online on' to fetch that.")
            ctx.note(self.name, "web offline")
            return
        except (ValueError, RuntimeError) as exc:
            ctx.say(f"Could not fetch that: {exc}")
            ctx.note(self.name, f"fetch failed: {exc}")
            return

        ids = session.stash.add_page(url, page["title"], page["text"])
        stats = session.stash.analyze(session.memory)
        staged = [session.stash.get(i) for i in ids]
        by_class: dict[str, int] = {}
        for entry in staged:
            by_class[entry["classification"]] = by_class.get(entry["classification"], 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(by_class.items())) or "nothing usable"
        ctx.say(
            f"Fetched {page['title'] or url} and stashed {len(ids)} claim(s) for analysis "
            f"({summary}). Nothing has been stored as fact - inspect with ':stash', "
            f"then ':promote <id>' or ':reject <id>'."
        )
        ctx.data["stash_ids"] = ids
        ctx.data["stash_stats"] = stats
        ctx.note(self.name, f"stashed {len(ids)} claim(s) from {page['netloc']}", stash_ids=ids)

    def _glyph(self, ctx: ThoughtContext) -> None:
        session = ctx.session
        rows = ctx.data["glyph_rows"]
        pixels = render(rows)
        features = pixels + row_means(pixels)
        ctx.data["features"] = features
        if ctx.data.get("unfamiliar") and session.vault.signatures():
            nearest = ctx.data.get("nearest")
            hint = (f" The closest thing I hold is '{nearest[0]}' at "
                    f"{nearest[1]:.0%} similarity." if nearest else "")
            ctx.say(
                "I do not recognise that pattern." + hint
                + " Teach it to me with ':teach <category> <label>' and I will "
                  "add it to the library."
            )
            ctx.data["prediction"] = None
            ctx.note(self.name, "pattern is unfamiliar; nothing stored matches it")
            return

        category = (ctx.data.get("routes") or [(GLYPH_FALLBACK_CATEGORY, 0.0)])[0][0]

        # Nothing trained, nothing stored, nothing to compare against. Inventing
        # an expert here and reporting its output was measured at 0/8 on the
        # built-in glyphs: a randomly initialised net answering at 0.16-0.35
        # confidence, which reads as a considered answer and is a coin toss.
        # An empty library should say it is empty.
        if session.recognizer is None and not session.experts.has_expert(category):
            ctx.data["unfamiliar"] = True
            ctx.say(
                "I have no pattern library to compare that against yet. Forge one, "
                "or teach me this shape with ':teach <category> <label>' and I will "
                "start one from it."
            )
            ctx.data["prediction"] = None
            ctx.note(self.name, "no trained expert and no signatures; declined to guess")
            return

        if session.recognizer is not None:
            label, confidence = session.recognizer.recognize(rows)
        else:
            if not session.experts.has_expert(category):
                # Only invent an expert when the library genuinely has none;
                # otherwise the trained one for this category is used.
                session.experts.ensure_expert(category, list(LABELS))
            label, confidence, category = self._compose_reading(ctx, features, category)
        ctx.data["prediction"] = (label, confidence)
        ctx.say(f"That pattern reads as '{label}' (confidence {confidence:.2f}, via {category}).")
        ctx.note(self.name, f"glyph -> {label} @ {confidence:.2f}", category=category)

    def _compose_reading(
        self, ctx: ThoughtContext, features: list[float], category: str
    ) -> tuple[str, float, str]:
        """Read a pattern on a blackboard, composing across the slots it fills.

        Which experts are consulted is decided by what the library says its
        categories are *about*. Two categories filling the same slot are
        alternatives, so only the best-routed one is asked and exactly one shard
        is paged — the behaviour every single-slot library has always had.
        Categories filling *different* slots describe different aspects of the
        same input, so each contributes one reading and the answer is their
        composition.

        That keeps the paging discipline honest: one expert is paged per aspect
        of the answer, and never more. A library that has not declared any slots
        pays exactly what it did before.

        An earlier version instead escalated to a runner-up whenever the first
        reading looked weak. It was measured on 200 noisy glyphs, fired 8 times,
        and changed no answer at all, so it was removed rather than kept on the
        argument that it might help somewhere.
        """
        session = ctx.session
        routes = list(ctx.data.get("routes") or [])
        if not routes:
            routes = [(category, 1.0)]

        # Best-routed category per slot: compose across aspects, compete within.
        chosen: dict[str, tuple[str, float]] = {}
        for name, score in routes:
            if not session.experts.has_expert(name):
                continue
            slot = session.vault.slot_of(name)
            if slot not in chosen or score > chosen[slot][1]:
                chosen[slot] = (name, score)
        if not chosen:
            chosen = {session.vault.slot_of(category): (category, 1.0)}

        # The default slot is not an aspect — it is the competing *whole*
        # interpretation. "pattern:cross" does not describe one part of
        # "shape:corners + mark:ex"; it is an alternative reading of the same
        # input, so the two must never be composed. Measured on the deployed
        # library: a plain 'plus' routes crosses 1.000 with border/inner at
        # 0.655 (compounds *contain* a plus), while a plain 'square' pushes
        # border past 0.95 because a square IS the box border — so a margin
        # alone cannot separate the cases and 3 of 8 plain glyphs were being
        # mangled into compositions. The rule instead: whichever
        # interpretation holds the single best route wins outright. A whole
        # reading suppresses composition; a factored reading composes across
        # factored slots only, with the whole slot excluded and factored
        # co-leaders admitted within a margin (box+plus: border 1.000,
        # inner 1.000, nearest whole 'frames' 0.873).
        if len(chosen) > 1:
            whole = _WHOLE_SLOT in chosen
            best_slot = max(chosen, key=lambda slot: chosen[slot][1])
            if whole and best_slot == _WHOLE_SLOT and not any(
                score >= chosen[_WHOLE_SLOT][1]
                for slot, (_name, score) in chosen.items()
                if slot != _WHOLE_SLOT
            ):
                chosen = {_WHOLE_SLOT: chosen[_WHOLE_SLOT]}
            else:
                factored = {slot: value for slot, value in chosen.items()
                            if slot != _WHOLE_SLOT}
                best_score = max(score for _name, score in factored.values())
                chosen = {
                    slot: (name, score)
                    for slot, (name, score) in factored.items()
                    if score >= best_score * _COMPOSE_COLEADER
                }

        board = Blackboard(features, {"category": category})
        run_blackboard(board, [
            ExpertContributor(session.experts, name, slot, weight=score or 1.0)
            for slot, (name, score) in chosen.items()
        ])
        ctx.data["blackboard"] = board.snapshot()

        slots = board.slots()
        if not slots:  # pragma: no cover - the routed expert always answers
            label, confidence = session.experts.predict(category, features)
            return label, confidence, category

        if len(slots) > 1:
            reading = board.reading()
            ctx.data["composed"] = reading
            ctx.note(self.name, "composed across "
                                + ", ".join(f"{s}={reading[s]}" for s in slots))
            # If a language has been learned, say what was seen in it. The
            # sentence is generated from the composed reading by an induced
            # lexicon and frame — there is no template for it anywhere, which
            # is the point of §11.10.
            from ultraquant.reason.language import verbalise_all

            spoken = verbalise_all(session, reading)
            if spoken:
                names = list(spoken)
                ctx.say(f"I see {spoken[names[0]]}.")
                if len(names) > 1:
                    ctx.say("Also: " + "; ".join(
                        f"in '{name}', {spoken[name]}" for name in names[1:]
                    ) + ".")
                ctx.note(self.name, f"verbalised in {len(spoken)} language(s)")
            return (
                " + ".join(f"{slot}:{reading[slot]}" for slot in slots),
                board.confidence(),
                "+".join(chosen[slot][0] for slot in slots),
            )

        best = board.best(slots[0])
        # Whole readings are spoken too, when a stored language covers the
        # label. The reading is the single slot with its predicted label; a
        # language whose lexicon lacks that label simply stays silent, which
        # is how the synthetic families (whose labels are ambiguous) opt out.
        from ultraquant.reason.language import verbalise_all

        spoken = verbalise_all(session, {slots[0]: best.value})
        if spoken:
            names = list(spoken)
            ctx.say(f"I see {spoken[names[0]]}.")
            if len(names) > 1:
                ctx.say("Also: " + "; ".join(
                    f"in '{name}', {spoken[name]}" for name in names[1:]
                ) + ".")
            ctx.note(self.name, f"verbalised in {len(spoken)} language(s)")
        return (best.value,
                float(best.evidence.get("expert_confidence", best.confidence)),
                str(best.evidence.get("category", category)))

    def _goal(self, ctx: ThoughtContext) -> None:
        """Plan a sequence of actions that satisfies a stated goal.

        Every other handler here does one thing. This one decides *what to do*,
        which is the difference between answering and acting. The facts named in
        the goal are the only thing extracted from the text; which capabilities
        to use, and in what order, is searched for.
        """
        from ultraquant.reason.actions import build_actions, goal_derived_from
        from ultraquant.reason.planner import Planner, PlanningError

        session = ctx.session
        text = ctx.data.get("goal_text", "")
        # Through the library index rather than a private dict: this is the
        # same routing path everything else uses, and it pages only the
        # buckets that could hold an answer.
        lowered = text.lower()
        keys = [k for k in session.memory.find_facts(text, top_k=8) if k in lowered]
        if not keys:
            ctx.say("I need a goal that names things I hold. "
                    "Try: goal: the tower height and the bridge length")
            ctx.note(self.name, "goal named nothing known")
            return

        # The parts of the goal that resolved to nothing held. A goal is a
        # conjunction; planning over only the held parts either fails
        # structurally (one key cannot satisfy derived-from-all) or worse,
        # succeeds while silently ignoring what is missing - and "no plan
        # reaches the goal within depth 6" names neither cause.
        segments = [seg.strip() for seg in
                    re.split(r",| and ", lowered) if seg.strip()]
        missing = []
        for seg in segments:
            if any(key in seg for key in keys):
                continue
            cleaned = re.sub(r"^(?:the|a|an)\s+", "", seg).strip("?. ")
            if cleaned and cleaned not in ("goal", "goal:"):
                missing.append(cleaned)

        planner = Planner(build_actions(session))
        state = {"_keys": keys}
        if ctx.data.get("glyph_rows"):
            state["_pattern"] = ctx.data["glyph_rows"]
        try:
            plan = planner.plan(goal_derived_from(*keys), state)
        except PlanningError as exc:
            if missing:
                held = ", ".join(keys)
                wanted = "', '".join(missing)
                ctx.say(f"I can't plan that yet: I hold nothing for "
                        f"'{wanted}' and the goal needs a value derived "
                        f"from everything it names (held: {held}). Tell me "
                        f"directly ('{missing[0]} is ...'), or ':learn' can "
                        "ask.")
            else:
                ctx.say(f"I could not find a way to do that: {exc}")
            ctx.note(self.name, "no plan found")
            return
        if missing:
            wanted = "', '".join(missing)
            ctx.data["plan_caveat"] = (f"Note: I hold nothing for "
                                       f"'{wanted}', so the result ignores "
                                       "it.")

        produced = {
            name: value for name, value in plan.state.items()
            if not name.startswith("_") and name not in keys
        }
        answer = max(produced, key=lambda n: len(n)) if produced else None
        ctx.data["plan"] = plan.describe()
        ctx.data["plan_state"] = produced
        caveat = ctx.data.get("plan_caveat", "")
        ctx.say(
            f"Plan ({len(plan)} steps): " + "; ".join(plan.describe())
            + (f"\nResult: {answer} = {produced[answer]:g}" if answer else "")
            + (f"\n{caveat}" if caveat else "")
        )
        ctx.note(self.name, f"planned {len(plan)} steps over {len(planner.actions)} "
                            f"actions, explored {plan.explored} states")

    def _question(self, ctx: ThoughtContext) -> None:
        compound = self._compound_parts(ctx.text)
        if compound is not None:
            self._answer_compound(ctx, compound)
            return
        if _WHY_ANSWERS and self._why_answer(ctx):
            return
        if _NEGATION_AWARE and self._polar_answer(ctx):
            return
        facts = ctx.data.get("facts", [])
        if facts:
            key, fact = facts[0]
            # §11.29's coverage rule reaches the exact branch: Recall's
            # candidate ladder tries sub-n-grams, so a hit here is not
            # necessarily the whole question. "the high mill sculptor"
            # answers "what is the high mill sculptor?" but must not
            # answer "what is the high mill sculptor workshop
            # population?" - the depth gate caught that question being
            # told who the sculptor IS, while the chain inference that
            # could answer what was ASKED sat unreached below.
            from ultraquant.shards.router import (_informative,
                                                  normalize_token)

            asked = {normalize_token(tok)
                     for tok in _TOKEN_RE.findall(ctx.text.lower())
                     if _informative(tok)}
            held = {normalize_token(tok)
                    for tok in _TOKEN_RE.findall(key.lower())
                    if _informative(tok)}
            if asked <= held:
                ctx.say(f"{key} is {_shown_value(fact)} "
                        f"(confidence {fact['confidence']:.2f}).")
                ctx.note(self.name, f"answered from fact {key!r}")
                return
            ctx.note(self.name,
                     f"sub-key hit {key!r} not asserted; question "
                     "content uncovered - falling through")

        # Exact key-match missed. Two fallbacks, in trust order, before giving
        # up - both were sitting unused while the pipeline said "I don't hold
        # anything" about things it held:
        #
        # 1. Keyword overlap over stored fact keys. "how tall is the tower"
        #    shares no *key phrase* with the stored `tower height`, but shares
        #    the token - and find_facts already ranks exactly that.
        # 2. Turns recovered by the context window (§11.14). A statement made
        #    in conversation and evicted from RAM is reachable through its
        #    24-byte reference; if it parses as "X is Y" and overlaps the
        #    question, it answers - attributed to the conversation, not
        #    presented as a stored fact, because it never earned promotion.
        from ultraquant.shards.router import (_informative,
                                              normalize_token)

        memory = ctx.session.memory
        # Informative tokens only. The first version filtered on length alone,
        # and "what is the melting point of tungsten" matched a recovered turn
        # reading "what is the tower height?" on the token "what" - the
        # fabrication control caught an answer being built out of a stopword.
        # Folded with the router's plural rule on both sides: "what are
        # towers?" holds the token "towers", the stored key holds
        # "tower", and raw comparison missed a fact the session had
        # just been told.
        question_tokens = {normalize_token(tok)
                           for tok in _TOKEN_RE.findall(ctx.text.lower())
                           if _informative(tok)}

        # Derivation runs BEFORE the loose keyword fallback: its coverage
        # rules are strict (every content token accounted for), so when a
        # chain or combination exists it answers the whole question - where
        # the keyword fallback would have answered whichever single premise
        # ranked first and stopped.
        from ultraquant.reason.inference import infer

        derived = infer(ctx.text, memory)
        if derived is not None:
            ctx.say(derived.describe())
            if derived.conclusion is not None:
                key, value = derived.conclusion
                ctx.session.pending_inference = {
                    "key": key, "value": value,
                    "confidence": derived.confidence,
                    "premises": list(derived.premises),
                    "negated": bool(getattr(derived, "negated", False)),
                }
            ctx.note(self.name,
                     f"inferred ({derived.kind}) from "
                     f"{len(derived.premises)} facts: "
                     + ", ".join(repr(k) for k, _v in derived.premises))
            return

        # The metacognitive step the refusal earns: a failed convergence
        # knows which premise was missing, and a grounded gap becomes a
        # learn-queue question instead of a dead end - whatever reply the
        # fallbacks below settle on carries the hint. Fail -> ask -> learn
        # -> infer is the loop; this is the ask.
        from ultraquant.reason.inference import missing_premise

        gap = missing_premise(ctx.text, memory)
        hint = ""
        if gap is not None:
            known = {c["premise_key"] for c in ctx.session.curiosities}
            if gap["premise_key"] not in known:
                ctx.session.curiosities.append(gap)
            hint = (f" If I knew the {gap['premise_key']}, I could work "
                    "this out - ':learn' will ask.")
            ctx.note(self.name,
                     f"curiosity registered: {gap['premise_key']!r} via "
                     f"{gap['via_key']!r}")

        candidates = list(memory.find_facts(ctx.text, top_k=3))
        folded_query = " ".join(sorted(question_tokens))
        for key in memory.find_facts(folded_query, top_k=3):
            if key not in candidates:
                candidates.append(key)
        for key in candidates:
            key_tokens = {normalize_token(tok)
                          for tok in _TOKEN_RE.findall(key.lower())
                          if _informative(tok)}
            if not (key_tokens & question_tokens):
                continue
            fact = memory.recall_fact(key)
            if fact is None:
                continue
            # Assert only when the key covers everything the question asked
            # about. A question holding content the key lacks is asking
            # about something ELSE that happens to share words - "the
            # melting point of tungsten" reached "steel melting point" here
            # and was answered with the wrong metal as if it were the
            # answer. Nearest-held is still worth SAYING; it is not worth
            # asserting as identity.
            if question_tokens <= key_tokens:
                ctx.say(f"{key} is {_shown_value(fact)} "
                        f"(confidence {fact['confidence']:.2f}).")
                ctx.note(self.name, f"answered from keyword fact {key!r}")
                return
            # The embedding suggester gets one shot before the demote:
            # §11.37 measured the synonym family failing lexically while
            # cosine>0.75 plus an anchor token reads it correctly with
            # zero decoy falls. The reading is named so it can be vetoed.
            suggester = getattr(ctx.session, "semantic", None)
            if suggester is not None:
                reading = suggester.suggest(ctx.text, memory)
                if reading is not None:
                    ctx.say(f"Reading that as '{reading.key}': "
                            f"{reading.key} is {reading.value} "
                            f"(confidence {reading.confidence:.2f}, "
                            f"embedding match {reading.similarity:.2f}).")
                    ctx.note(self.name,
                             f"semantic reading {reading.key!r} at "
                             f"{reading.similarity:.2f}")
                    return
            ctx.say(f"I don't hold that exactly. Nearest I hold: {key} is "
                    f"{_shown_value(fact)} (confidence "
                    f"{fact['confidence']:.2f})." + hint)
            ctx.note(self.name,
                     f"nearest-held {key!r}; question content "
                     "not fully covered")
            return

        for turn in ctx.data.get("recovered_turns", []):
            turn_text = str(turn.get("text", "")).strip()
            # A recovered *question* is not a statement, however well it
            # parses: "what is the tower height?" splits into key "what" and
            # value "the tower height?", and answering from it fabricates.
            if turn_text.endswith("?") or turn_text.lower().startswith(
                    _INTERROGATIVE_LEADS):
                continue
            parsed = _parse_statement(turn_text)
            if parsed is None:
                continue
            key, value = parsed
            key_tokens = {tok for tok in _TOKEN_RE.findall(key.lower())
                          if _informative(tok)}
            if key_tokens & question_tokens:
                ctx.say(f"Earlier in this conversation: {key} is {value}.")
                ctx.note(self.name,
                         f"answered from recovered turn {turn.get('id')}")
                return

        # Nothing believed - but the quarantine may already hold claims
        # on exactly this topic, and "I don't hold anything" while
        # sitting on them sends the user to the web for what is one
        # ':analyze' away. Staged text stays unquoted: it has not earned
        # belief.
        staged = 0
        try:
            for entry in ctx.session.stash.entries(status="staged"):
                blob = str(entry.get("claim", entry.get("text", "")))
                entry_tokens = {normalize_token(tok) for tok in
                                _TOKEN_RE.findall(blob.lower())
                                if _informative(tok)}
                if entry_tokens & question_tokens:
                    staged += 1
        except Exception:  # noqa: BLE001 - a stash probe never breaks Respond
            staged = 0
        if staged:
            ctx.say(
                f"Nothing believed on that yet, but {staged} staged "
                "claim(s) in quarantine mention it - ':stash' lists "
                "them, ':analyze' weighs them, ':promote <id>' "
                "believes one." + hint
            )
            ctx.note(self.name,
                     f"no fact; {staged} staged claim(s) surfaced")
            return
        ctx.say(
            "I don't hold anything on that yet. Tell me directly ('X is Y'), "
            "or give me a URL and I'll stash what it claims for analysis."
            + hint
        )
        ctx.note(self.name, "no matching fact")

    def _affirm(self, ctx: ThoughtContext) -> None:
        """Consolidate the derivation the user just confirmed.

        The episodic-to-semantic move, behind the only non-circular signal
        available: the user said so. The derived fact keeps its diluted
        confidence and its provenance, and the provenance is what lets a
        later premise revision retract it (truth maintenance in
        :meth:`SystematicMemory.remember_fact`).
        """
        pending = ctx.data.get("affirmed_inference") or {}
        key = pending.get("key")
        value = pending.get("value")
        if not key or value is None:
            ctx.say("Nothing is pending confirmation.")
            ctx.note(self.name, "affirmation with nothing pending")
            return
        negated = bool(pending.get("negated"))
        ctx.session.memory.consolidate_fact(
            key, value,
            confidence=float(pending.get("confidence", 0.0)),
            premises=pending.get("premises", []),
            negated=negated,
        )
        spoken = f"not {value}" if negated else value
        ctx.say(f"Consolidated: {key} is {spoken} - derived and confirmed, "
                "now recallable, and retracted automatically if a premise "
                "is ever revised.")
        ctx.note(self.name, f"consolidated {key!r} from "
                            f"{len(pending.get('premises', []))} premises")

    @staticmethod
    def _compound_parts(text: str) -> list[str] | None:
        """Split a conjunction question into part phrases, or None.

        Only a real conjunction decomposes: an " and " with substantive
        content on both sides and NO combination word - "the sum of A and
        B" is one arithmetic question about two facts, and decomposing it
        would break the combine path that already answers it.
        """
        from ultraquant.reason.inference import _COMBINE_WORDS
        from ultraquant.shards.router import _informative, normalize_token

        lowered = text.lower().strip().rstrip("?")
        if " and " not in lowered:
            return None
        if any(word in _TOKEN_RE.findall(lowered)
               for word in _COMBINE_WORDS):
            return None
        head = re.sub(r"^(what|which|who|where|when|how)"
                      r"\s+(is|are|was|were)\s+", "", lowered)
        parts = [seg.strip() for seg in head.split(" and ") if seg.strip()]
        if len(parts) < 2:
            return None
        for part in parts:
            tokens = {normalize_token(tok) for tok in
                      _TOKEN_RE.findall(part) if _informative(tok)}
            if len(tokens) < 2:
                return None
        return parts

    def _answer_compound(self, ctx: ThoughtContext, parts: list[str]) -> None:
        """Answer each part of a conjunction through the full single path.

        Sequential attention over sub-questions: each part re-enters the
        machinery a single question gets - exact recall, then the spread,
        then an honest unknown WITH its own well-formed curiosity - and
        the reply names each part's answer beside its part. A half-known
        compound answers the half it can and says which half it cannot,
        instead of demoting the whole question to one nearest-held fact.
        """
        from ultraquant.reason.inference import infer, missing_premise
        from ultraquant.shards.router import _informative, normalize_token

        memory = ctx.session.memory
        pieces, unknown_parts = [], []
        for part in parts:
            sub_question = f"what is the {part}?"
            part_tokens = {normalize_token(tok) for tok in
                           _TOKEN_RE.findall(part) if _informative(tok)}
            answered = False
            for key in memory.find_facts(part, top_k=3):
                key_tokens = {normalize_token(tok) for tok in
                              _TOKEN_RE.findall(key.lower())
                              if _informative(tok)}
                if part_tokens <= key_tokens:
                    record = memory.recall_fact(key)
                    if record is not None:
                        pieces.append(f"{key} is {record['value']}")
                        answered = True
                        break
            if answered:
                continue
            derived = infer(sub_question, memory)
            if derived is not None:
                pieces.append(f"{derived.answer} (inferred)")
                continue
            gap = missing_premise(sub_question, memory)
            if gap is not None:
                known = {c["premise_key"]
                         for c in ctx.session.curiosities}
                if gap["premise_key"] not in known:
                    ctx.session.curiosities.append(gap)
                unknown_parts.append(
                    f"{part} (unknown - ':learn' will ask for the "
                    f"{gap['premise_key']})")
            else:
                unknown_parts.append(f"{part} (unknown)")
        summary = "; ".join(pieces) if pieces else "none of it is held"
        if unknown_parts:
            summary += ". Still missing: " + "; ".join(unknown_parts)
        ctx.say(summary + ".")
        ctx.note(self.name,
                 f"compound: {len(parts)} part(s), {len(pieces)} answered, "
                 f"{len(unknown_parts)} unknown")

    def _polar_answer(self, ctx: ThoughtContext) -> bool:
        """Answer "is X Y?" with yes, no, or an honest don't-know.

        The subject is the longest stored-key prefix of the question;
        the remainder is the claim. Absence is NEVER no: a subject the
        library holds nothing about returns False here and falls
        through to the hedging machinery below - "no" is reserved for
        actual contrary belief, stored either as a different value or
        as a negation. That line - belief-of-absence against
        absence-of-belief - is the §11.48 claim.
        """
        from ultraquant.shards.router import _informative, normalize_token

        lowered = ctx.text.lower().strip().strip("?!. ")
        for lead in ("is ", "are ", "was ", "were "):
            if lowered.startswith(lead):
                rest = lowered[len(lead):].strip()
                break
        else:
            return False
        for article in ("the ", "a ", "an "):
            if rest.startswith(article):
                rest = rest[len(article):]
        words = rest.split()
        if len(words) < 2:
            return False

        if _POLAR_COMPARES and self._polar_compare(ctx, words):
            return True

        memory = ctx.session.memory
        fact = None
        key = ""
        claim_words: list[str] = []
        # One ladder, longest subject first, where "subject" means a
        # stored key OR a derivable one. "dome city climate temperate"
        # must try deriving "dome city climate" BEFORE the stored
        # "dome city" claims the split - the first cut answered about
        # the city when the question asked about the climate.
        for size in range(len(words) - 1, 0, -1):
            candidate = " ".join(words[:size])
            record = memory.recall_fact(candidate)
            if record is not None:
                fact, key = record, candidate
                claim_words = words[size:]
                break
            if len(words) - size <= 2 and size >= 2:
                if self._polar_derive(ctx, words[:size], words[size:]):
                    return True
        if fact is None:
            return False
        claim_negated = any(w in ("not", "never") for w in claim_words)
        claimed = " ".join(w for w in claim_words
                           if w not in ("not", "never", "a", "an", "the"))
        if not claimed:
            return False

        fold = lambda text: {normalize_token(tok) for tok  # noqa: E731
                             in _TOKEN_RE.findall(str(text).lower())
                             if _informative(tok)}
        matches = fold(claimed) == fold(str(fact.get("value", "")))
        held = fact.get("value", "")
        confidence = f"(confidence {fact['confidence']:.2f})"
        if not fact.get("negated"):
            if matches:
                verdict = ("No" if claim_negated else "Yes")
                ctx.say(f"{verdict} - {key} is {held} {confidence}.")
            elif claim_negated:
                ctx.say(f"Yes - {key} is {held}, not {claimed} "
                        f"{confidence}.")
            else:
                ctx.say(f"No - {key} is {held}, not {claimed} "
                        f"{confidence}.")
        else:
            if matches:
                verdict = ("Yes" if claim_negated else "No")
                ctx.say(f"{verdict} - believed not {held} {confidence}.")
            else:
                # A negation of one value says nothing about another:
                # "not steel" cannot answer "is it iron?".
                ctx.say(f"I don't know - I hold only that {key} is "
                        f"not {held} {confidence}.")
        ctx.note(self.name, f"polar question against {key!r}")
        return True

    def _why_answer(self, ctx: ThoughtContext) -> bool:
        """Answer "why is X Y?" from provenance - §11.51.

        The trail IS the answer, finally askable: a consolidated fact
        cites the premises it rests on, a stated fact cites its
        statement, and a derivable one derives and shows the trail.
        The control that matters most: never explain what is not
        believed - "why is the tower material steel?" when iron is
        held answers "It isn't - tower material is iron", because
        rationalising a false premise is the transformer failure this
        architecture exists to refuse.
        """
        from ultraquant.reason.inference import infer
        from ultraquant.shards.router import _informative, normalize_token

        lowered = ctx.text.lower().strip().strip("?!. ")
        for lead in ("why is ", "why are ", "why was ", "why were "):
            if lowered.startswith(lead):
                rest = lowered[len(lead):].strip()
                break
        else:
            return False
        for article in ("the ", "a ", "an "):
            if rest.startswith(article):
                rest = rest[len(article):]
        words = rest.split()
        if len(words) < 2:
            return False

        memory = ctx.session.memory
        fold = lambda text: {normalize_token(tok) for tok  # noqa: E731
                             in _TOKEN_RE.findall(str(text).lower())
                             if _informative(tok)}
        _CLAIM_NOISE = ("not", "never", "believed", "a", "an", "the")

        fact = None
        key = ""
        derived = None
        claim_words: list[str] = []
        for size in range(len(words), 0, -1):
            candidate_words = words[:size]
            remainder = words[size:]
            if len(remainder) <= 2:
                record = memory.recall_fact(" ".join(candidate_words))
                if record is not None:
                    fact, key = record, " ".join(candidate_words)
                    claim_words = remainder
                    break
            if (remainder and len(remainder) <= 2 and size >= 2
                    and candidate_words[-1] not in ("not", "never")):
                attempt = infer(
                    f"what is the {' '.join(candidate_words)}?", memory)
                if attempt is not None \
                        and "as a modifier" not in attempt.answer \
                        and "as modifiers" not in attempt.answer:
                    derived = attempt
                    claim_words = remainder
                    break
        claim_negated = any(w in ("not", "never") for w in claim_words)
        claimed = " ".join(w for w in claim_words
                           if w not in _CLAIM_NOISE)

        if fact is not None:
            shown = _shown_value(fact)
            confidence = f"(confidence {fact['confidence']:.2f})"
            if claimed:
                agree = (fold(claimed) == fold(str(fact.get("value", "")))
                         and claim_negated == bool(fact.get("negated")))
                if not agree:
                    # The anti-rationalisation line: a why-question
                    # carrying a claim the library does not believe is
                    # corrected, never explained.
                    ctx.say(f"It isn't - {key} is {shown} {confidence}.")
                    ctx.note(self.name,
                             f"why-question corrected against {key!r}")
                    return True
            provenance = fact.get("derived_from")
            if provenance:
                trail = "; ".join(f"{p_key} is {p_value}"
                                  for p_key, p_value in provenance)
                ctx.say(f"Because {trail} - consolidated from a "
                        f"confirmed derivation {confidence}.")
            else:
                times = int(fact.get("reinforcements", 0))
                stated = ("stated directly"
                          + (f", reinforced {times} time(s)"
                             if times else ""))
                ctx.say(f"Because it was {stated} {confidence}.")
            ctx.note(self.name, f"why-question answered from {key!r}")
            return True

        if derived is not None:
            suffix = (f"(derived just now, not stored, confidence "
                      f"{derived.confidence:.2f})")
            if claimed:
                value = str(derived.conclusion[1]
                            if derived.conclusion else "")
                agree = (fold(claimed) == fold(value)
                         and claim_negated == bool(derived.negated))
                if not agree:
                    ctx.say(f"It isn't - {derived.answer} {suffix}.")
                    ctx.note(self.name, "why-question corrected against "
                                        "a derivation")
                    return True
            trail = "; ".join(f"{p_key} is {p_value}"
                              for p_key, p_value in derived.premises)
            ctx.say(f"Because {trail} {suffix}.")
            if derived.conclusion is not None:
                d_key, d_value = derived.conclusion
                ctx.session.pending_inference = {
                    "key": d_key, "value": d_value,
                    "confidence": derived.confidence,
                    "premises": list(derived.premises),
                    "negated": bool(derived.negated),
                }
            ctx.note(self.name,
                     f"why-question derived through "
                     f"{len(derived.premises)} premise(s)")
            return True
        return False

    def _polar_compare(self, ctx: ThoughtContext,
                       words: list[str]) -> bool:
        """Answer "is A <taller/heavier/...> than B?" - §11.54.

        The two operands are recalled, converted where §11.42's table
        connects their units, and compared; the verdict names both
        values so it can be checked at a glance. This branch OWNS
        comparative questions: before it existed, the derive path ran
        them through the combine machinery (whose conclusion is None)
        and answered No in both directions, and a missing operand fell
        to the direct matrix and got a fabricated verdict - the
        absence-is-never-no line violated through the comparative
        door. Missing, negated, non-numeric, or unit-incomparable
        operands refuse aloud.
        """
        from ultraquant.reason.inference import (_COMBINE_WORDS,
                                                 _UNIT_TO_FAMILY,
                                                 _convert, _numeric,
                                                 _unit)

        comp_at = None
        for index, word in enumerate(words[1:-1], start=1):
            direction = _COMBINE_WORDS.get(word)
            if (direction in ("larger", "smaller")
                    and words[index + 1] == "than"):
                comp_at = index
                break
        if comp_at is None:
            return False
        direction = _COMBINE_WORDS[words[comp_at]]

        def _side(side_words: list[str]) -> str:
            out = list(side_words)
            for article in ("the", "a", "an"):
                if out and out[0] == article:
                    out = out[1:]
            return " ".join(out)

        left_key = _side(words[:comp_at])
        right_key = _side(words[comp_at + 2:])
        if not left_key or not right_key:
            return False

        from ultraquant.reason.inference import infer

        memory = ctx.session.memory
        sides = {}
        trails = {}
        for name, key in (("left", left_key), ("right", right_key)):
            record = memory.recall_fact(key)
            if record is None and _COMPARE_DERIVES:
                # §11.55: an operand the store does not hold may be
                # derived - with §11.50's split discipline (no
                # modifier-rescued subjects) and §11.48's line (a
                # derived denial holds no number to compare).
                attempt = infer(f"what is the {key}?", memory)
                if (attempt is not None
                        and attempt.conclusion is not None
                        and "as a modifier" not in attempt.answer
                        and "as modifiers" not in attempt.answer):
                    if attempt.negated:
                        ctx.say(f"I can't compare those: I can only "
                                f"derive a denial for '{key}' "
                                f"({key} is not "
                                f"{attempt.conclusion[1]}).")
                        ctx.note(self.name, f"comparative refused; "
                                            f"{key!r} derives negated")
                        return True
                    record = {"value": attempt.conclusion[1],
                              "confidence": attempt.confidence}
                    trails[name] = ", ".join(
                        str(v) for _k, v in attempt.premises[:-1])
            if record is None:
                ctx.say(f"I can't compare those: I hold nothing for "
                        f"'{key}'.")
                ctx.note(self.name,
                         f"comparative refused; {key!r} unheld")
                return True
            if record.get("negated"):
                ctx.say(f"I can't compare those: I hold only a denial "
                        f"for '{key}' ({key} is not "
                        f"{record.get('value', '')}).")
                ctx.note(self.name,
                         f"comparative refused; {key!r} negated")
                return True
            number = _numeric(record.get("value", ""))
            if number is None:
                ctx.say(f"I can't compare those: '{key}' holds no "
                        f"number ({record.get('value', '')}).")
                ctx.note(self.name,
                         f"comparative refused; {key!r} non-numeric")
                return True
            sides[name] = (key, record, number,
                           _unit(record.get("value", "")))

        (l_key, l_rec, l_num, l_unit) = sides["left"]
        (r_key, r_rec, r_num, r_unit) = sides["right"]
        converted = ""
        if l_unit != r_unit or bool(l_unit) != bool(r_unit):
            family = _UNIT_TO_FAMILY.get(l_unit)
            if (not l_unit or not r_unit or family is None
                    or _UNIT_TO_FAMILY.get(r_unit) != family):
                ctx.say(f"I can't compare those: '{l_key}' is in "
                        f"{l_unit or 'no unit'} and '{r_key}' in "
                        f"{r_unit or 'no unit'}, and no definition "
                        "connects them.")
                ctx.note(self.name, "comparative refused; units "
                                    "incomparable")
                return True
            r_in_l = _convert(r_num, r_unit, l_unit)
            wins = (l_num > r_in_l if direction == "larger"
                    else l_num < r_in_l)
            equal = l_num == r_in_l
            converted = " (units converted)"
        else:
            wins = (l_num > r_num if direction == "larger"
                    else l_num < r_num)
            equal = l_num == r_num
        confidence = min(float(l_rec.get("confidence", 0.0)),
                         float(r_rec.get("confidence", 0.0)))
        l_mark = (f" (derived via {trails['left']})"
                  if "left" in trails else "")
        r_mark = (f" (derived via {trails['right']})"
                  if "right" in trails else "")
        both = (f"{l_key} is {l_rec.get('value', '')}{l_mark}, "
                f"{r_key} is {r_rec.get('value', '')}{r_mark}")
        if equal:
            ctx.say(f"No - they are equal: {both}{converted} "
                    f"(confidence {confidence:.2f}).")
        elif wins:
            ctx.say(f"Yes - {both}{converted} "
                    f"(confidence {confidence:.2f}).")
        else:
            ctx.say(f"No - {both}{converted} "
                    f"(confidence {confidence:.2f}).")
        ctx.note(self.name,
                 f"comparative {l_key!r} vs {r_key!r} ({direction})")
        return True

    def _polar_derive(self, ctx: ThoughtContext, subject_words: list[str],
                      claim_words: list[str]) -> bool:
        """Answer a polar question by DERIVING the subject - §11.50.

        "is the dome city climate temperate?" with only "dome city is
        york" and "york climate is not temperate" held: the chain
        machinery answers "what is the dome city climate?" and the
        derived value meets the claim under the same matrix direct
        facts use - polarity included, absence still never no, and the
        reply marked derived with its trail so the verdict can be
        vetoed premise by premise.
        """
        if not _POLAR_DERIVES:
            return False
        if subject_words and subject_words[-1] in ("not", "never"):
            # The split landed mid-claim: "dome city climate not" /
            # "temperate" absorbed the polarity word into the subject
            # and answered No where the claim agreed. The negator
            # belongs to the claim; let the next size put it there.
            return False
        from ultraquant.reason.inference import infer
        from ultraquant.shards.router import _informative, normalize_token

        claim_negated = any(w in ("not", "never") for w in claim_words)
        claimed = " ".join(w for w in claim_words
                           if w not in ("not", "never", "a", "an", "the"))
        if not claimed:
            return False
        subject = " ".join(subject_words)
        derived = infer(f"what is the {subject}?", ctx.session.memory)
        if derived is None:
            return False
        if "as a modifier" in derived.answer \
                or "as modifiers" in derived.answer:
            # The derivation only converged by reading part of THIS
            # subject away - which means the split was wrong (the
            # dropped word belongs to the claim: "tower hardness 490"
            # derived by dropping '490'). Let a shorter subject try.
            return False
        if derived.conclusion is not None:
            d_key, d_value = derived.conclusion
            ctx.session.pending_inference = {
                "key": d_key, "value": d_value,
                "confidence": derived.confidence,
                "premises": list(derived.premises),
                "negated": bool(derived.negated),
            }
        fold = lambda text: {normalize_token(tok) for tok  # noqa: E731
                             in _TOKEN_RE.findall(str(text).lower())
                             if _informative(tok)}
        matches = (fold(claimed)
                   == fold(str(derived.conclusion[1]
                               if derived.conclusion else "")))
        suffix = f" (derived, confidence {derived.confidence:.2f})."
        if not derived.negated:
            if matches:
                verdict = "No" if claim_negated else "Yes"
            else:
                verdict = "Yes" if claim_negated else "No"
            ctx.say(f"{verdict} - {derived.answer}{suffix}")
        elif matches:
            verdict = "Yes" if claim_negated else "No"
            ctx.say(f"{verdict} - {derived.answer}{suffix}")
        else:
            # A derived negation of one value says nothing about
            # another - same line as the stored case.
            ctx.say(f"I don't know - I can only derive that "
                    f"{derived.answer}{suffix}")
        ctx.note(self.name,
                 f"polar question derived through "
                 f"{len(derived.premises)} premise(s)")
        return True

    def _fact(self, ctx: ThoughtContext) -> None:
        parsed = _parse_statement(ctx.text)
        if parsed is None:
            ctx.say("I couldn't find a clear 'X is Y' in that.")
            ctx.note(self.name, "unparsed statement")
            return
        ctx.data["statement"] = parsed
        ctx.say(f"Noted: {parsed[0]} is {parsed[1]}.")
        ctx.note(self.name, f"statement {parsed[0]!r} -> {parsed[1]!r}")

    def _chat(self, ctx: ThoughtContext) -> None:
        routes = ctx.data.get("routes", [])
        facts = ctx.data.get("facts", [])
        if facts:
            key, fact = facts[0]
            ctx.say(f"That lands near '{key}', which I hold as: "
                    f"{_shown_value(fact)}.")
        elif routes:
            ctx.say(f"That reads as {routes[0][0]}. I have nothing stored on it yet.")
        else:
            ctx.say("I have nothing on that yet. ':help' lists what I can do.")
        ctx.note(self.name, "conversational reply")


#: The §11.48 polarity switch: the negation gate's baseline arm turns
#: it off to measure shipped behavior ("not steel" stored as a value,
#: no polar questions). Sessions run with it on.
_NEGATION_AWARE = True

#: The §11.50 rung: polar questions with no direct fact may DERIVE
#: their verdict through the chain machinery. The polar-derive gate's
#: baseline arm turns it off.
_POLAR_DERIVES = True

#: The §11.51 rung: "why is X Y?" answers from provenance - stated
#: facts cite their statement, consolidated facts cite their premises,
#: derivable ones derive and show the trail. The why gate's baseline
#: arm turns it off.
_WHY_ANSWERS = True

#: The §11.53 rung: a statement that conflicts with held belief is
#: revised ALOUD - the old belief named, the retracted derivatives
#: counted - instead of behind a bare "Noted:". The revision gate's
#: baseline arm turns it off.
_REVISION_ALOUD = True

#: The §11.54 rung: "is A taller than B?" compares the two held
#: numerics (units converted where a definition connects them) and
#: answers with both values named. The comparative gate's baseline arm
#: turns it off - and its baseline is not merely featureless: without
#: this branch the polar machinery answered comparatives with a coin
#: that always said No, and fabricated verdicts over missing operands.
_POLAR_COMPARES = True

#: The §11.55 rung: a comparative operand the store does not hold may
#: be DERIVED through the chain machinery (never a denial, never a
#: modifier-rescued split), the verdict marking which side was derived
#: and its trail. The compderive gate's baseline arm turns it off.
_COMPARE_DERIVES = True


def _split_polarity(value: str) -> tuple[str, bool]:
    """Split a statement value into (value, negated).

    "not steel" and "never steel" are belief-of-absence; the bare value
    is stored and the polarity travels as a flag, so the fold can never
    again bridge a denial as an assertion ("via not steel").
    """
    if not _NEGATION_AWARE:
        return value, False
    lowered = value.lower()
    for lead in ("not ", "never "):
        if lowered.startswith(lead):
            rest = value[len(lead):].strip()
            if rest:
                return rest, True
    return value, False


def _shown_value(fact: dict) -> str:
    """A record's value as speech, polarity included."""
    value = fact.get("value", "")
    return f"not {value}" if fact.get("negated") else str(value)


def _parse_statement(text: str) -> tuple[str, str] | None:
    """Parse a user statement into ``(key, value)``."""
    cleaned = text.strip().rstrip(".")
    lowered = cleaned.lower()
    for lead in ("remember that ", "remember, ", "remember: ", "remember "):
        if lowered.startswith(lead):
            cleaned = cleaned[len(lead):]
            lowered = cleaned.lower()
            break
    if "=" in cleaned:
        key, value = cleaned.split("=", 1)
        return key.strip().lower(), value.strip()
    idx = lowered.find(" is ")
    if idx > 0:
        key = cleaned[:idx].strip().lower()
        value = cleaned[idx + 4:].strip()
        for article in ("the ", "a ", "an "):
            if key.startswith(article):
                key = key[len(article):]
        if key and value:
            return key, value
    return None


class Respond(Thought):
    """Assemble the reply."""

    name = "Respond"

    def run(self, ctx: ThoughtContext) -> None:
        if not ctx.response_parts:
            ctx.say("I have nothing to say about that yet. Try ':help'.")
        ctx.data["response"] = " ".join(part.strip() for part in ctx.response_parts if part.strip())
        ctx.note(self.name, f"{len(ctx.data['response'])} chars")


def _route_confirmation(ctx) -> str | None:
    """Why *this route* should be believed, or None if nothing confirms it.

    The pipeline cannot know whether a route was *correct* — the router's own
    choice is the only signal available at reinforcement time, and learning from
    it is the circularity §11.16 measured: accuracy held at 0.400 while the
    wrong answer's margin grew 0.60 -> 2.33, making every error 3.9x harder to
    overturn.

    What it *can* know is whether the machinery the route reached did anything.
    Only two signals qualify, and narrowing to them was not the first attempt:

    * ``prediction`` — an expert from the routed category placed the input, and
      it was not flagged unfamiliar.
    * ``composed`` — the blackboard built a reading from the routed experts.

    **The signals that had to be removed.** The first version also accepted a
    recalled fact, a code result, and an advanced plan. None of those depends on
    the route: ``Recall`` runs *before* ``Route`` in the pipeline, so a fact is
    found regardless of where the text routes, and code and planning are driven
    by intent rather than category. Caught by running it — "what shape is this
    glyph" reinforced ``crosses`` because a fact happened to be recalled, which
    is the original defect wearing a confirmation signal that confirmed the
    wrong thing.

    Returns:
        A short reason, recorded in the trace, or None.
    """
    if ctx.data.get("prediction") and not ctx.data.get("unfamiliar"):
        return "expert placed it"
    if ctx.data.get("composed"):
        return "blackboard composed"
    return None


class Learn(Thought):
    """Build off what just happened: store, reinforce, and consolidate truth."""

    name = "Learn"

    def run(self, ctx: ThoughtContext) -> None:
        session = ctx.session
        intent = ctx.data.get("intent", "chat")
        learned: list[str] = []

        # A statement the *user* typed is testimony we accept directly.
        if intent == "fact_statement" and ctx.data.get("statement"):
            key, value = ctx.data["statement"]
            value, negated = _split_polarity(value)
            result = session.memory.remember_fact(key, value,
                                                  confidence=0.6,
                                                  negated=negated)
            learned.append(f"fact {key!r}")
            if (_REVISION_ALOUD and isinstance(result, dict)
                    and result.get("outcome") == "revised"):
                # Honest aloud reaches belief CHANGE: the episode log
                # always recorded revisions, but the reply said the
                # same bare "Noted:" for a change of mind as for news.
                # The old belief is named so it can be defended, and
                # the retractions are counted so the cost is visible.
                notice = (f"That revises what I held: {key} was "
                          f"{result['was']}.")
                retracted = result.get("retracted") or []
                if retracted:
                    names = ", ".join(repr(k) for k in retracted)
                    notice += (f" {len(retracted)} derived fact(s) "
                               f"rested on it and were retracted: "
                               f"{names}.")
                ctx.say(notice)
                ctx.data["response"] = " ".join(
                    part.strip() for part in ctx.response_parts
                    if part.strip())

        # Web claims only cross into memory once independent sources agree.
        promoted: list[int] = []
        for entry_id in ctx.data.get("stash_ids", []):
            entry = session.stash.get(entry_id)
            if entry["status"] == "corroborated" and entry["classification"] == "factual-claim":
                try:
                    session.stash.promote(entry_id, session.memory)
                    promoted.append(entry_id)
                except Exception:  # noqa: BLE001 - promotion is best-effort
                    continue
        if promoted:
            learned.append(f"{len(promoted)} corroborated claim(s)")
            ctx.say(
                f"Corroborated by independent sources, so promoted to fact: "
                f"{', '.join(str(i) for i in promoted)}."
            )
            ctx.data["response"] = " ".join(
                part.strip() for part in ctx.response_parts if part.strip()
            )

        # Reinforce the associative catalog — but only for a route something
        # downstream actually confirmed.
        #
        # This used to reinforce the top route unconditionally, which made the
        # router learn from its own guesses. Measured (§11.16): accuracy did
        # not fall, but the margin by which a wrong category beat the right one
        # grew 0.60 -> 2.33 over 60 turns, so every error became ~3.9x harder to
        # overturn. That is why the plural fix in §11.15 could give `arithmetic`
        # a full point of base overlap and still lose to a `crosses` that had
        # entrenched past it.
        #
        # There is no ground truth here — the router's own choice is the only
        # thing available, which is the circularity. What there *is* is evidence
        # the route led somewhere: an expert placed the input, a fact was
        # recalled, code ran, a plan advanced. Absent any of that the route
        # produced nothing, and reinforcing it teaches a guess.
        tokens = ctx.data.get("tokens", [])[:8]
        confirmation = _route_confirmation(ctx)
        for category, _score in ctx.data.get("routes", [])[:1]:
            if not confirmation:
                learned.append(f"not reinforced ({category}): unconfirmed route")
                continue
            session.router.learn(ctx.text, category, delta=0.05)
            shard_id = ExpertPool.shard_id(category)
            if session.vault.has(shard_id) and tokens:
                session.vault.reinforce(shard_id, tokens, delta=0.05)
            learned.append(f"reinforced {category} ({confirmation})")

        if intent == "glyph" and ctx.data.get("unfamiliar"):
            session.memory.remember_episode(
                "unknown-pattern",
                {"rows": list(ctx.data.get("glyph_rows") or []),
                 "nearest": ctx.data.get("nearest")},
                tags=["pattern", "unknown"],
            )
            learned.append("recorded an unfamiliar pattern")

        if intent == "glyph" and ctx.data.get("prediction"):
            label, _confidence = ctx.data["prediction"]
            bits = [int(v) for v in ctx.data["features"][:25]]
            session.memory.store_signature(label, bits)
            learned.append(f"signature for {label!r}")

        session.memory.remember_episode(
            "interaction",
            {
                "text": ctx.text,
                "intent": intent,
                "response": ctx.data.get("response", ""),
                "routes": [c for c, _ in ctx.data.get("routes", [])],
            },
            tags=["chat", intent],
        )
        session.save()
        ctx.note(self.name, "; ".join(learned) if learned else "episode only")


#: The predefined set of thought, in fixed order.
PIPELINE: tuple[Thought, ...] = (
    Perceive(),
    Recall(),
    Route(),
    Reason(),
    Respond(),
    Learn(),
)


def run_pipeline(text: str, session: Session) -> tuple[str, list[dict]]:
    """Run one input through every thought in order.

    Args:
        text: The user's input.
        session: The session providing all stores.

    Returns:
        ``(response, trace)``.
    """
    ctx = ThoughtContext(text=text, session=session)
    for thought in PIPELINE:
        thought.run(ctx)
    session.last_trace = ctx.trace
    # Appended after the pipeline, not before: Recall consults the window, and
    # a turn that could match itself would score perfectly on every query it
    # contains while teaching nothing.
    if session.context is not None and text.strip():
        session.context.add(text, intent=ctx.data.get("intent", "chat"))
    return ctx.data.get("response", ""), ctx.trace
