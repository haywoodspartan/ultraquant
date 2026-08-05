"""Learning mode: the model finds its own gaps and asks about them.

Everything else in the interpreter is reactive — you say something, it answers.
This is the other direction. It inspects what it holds, works out where it is
weakest, and asks *you* the question whose answer would help most.

The gaps it can see are the ones its own stores record:

* **shaky** — a fact held at low confidence that keeps getting recalled.
* **disputed** — two sources contradict each other, or a claim contradicts
  something already believed. It cannot resolve this alone; someone has to say
  which is right.
* **unclassified** — a stashed web claim it could not judge as fact or opinion.
* **unknown-term** — a word that keeps appearing in conversation with nothing
  stored against it. Repetition is the signal: asked once, ignore; asked
  repeatedly, it matters.
* **weak-pattern** — a category whose expert recognises its own training data
  poorly, or has none at all. The remedy is a demonstration, so it asks for one.
* **untrained-category** — a routed category with nothing behind it yet.
  Pattern categories ask for a drawn demonstration; topic categories ask
  for knowledge, because a glyph teaches 'arithmetic' nothing.

Questions are ranked by a crude expected-gain score — how uncertain the model is,
multiplied by how often the thing comes up — so the first question asked is the
one worth answering. Answers feed straight back: facts are written, stash entries
promoted or rejected, glyphs trained into the relevant expert.

Nothing here invents knowledge. Answers come from the user, or - through
:meth:`LearningSession.research` - from the web via the contemporary stash's
quarantine, where found is still not believed. The model only decides what is
worth asking, and asks the human second.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ultraquant.experts.moe import ExpertPool
from ultraquant.interpreter.stash import StashError

__all__ = ["Question", "Answer", "LearningSession"]

#: Where research looks a term up. Templates, not code: tests point these at a
#: local server, and a deployment can swap in any source it trusts. Everything
#: fetched still goes through the contemporary stash - the source being
#: configurable does not soften the quarantine.
RESEARCH_SOURCES: tuple[str, ...] = (
    "https://en.wikipedia.org/wiki/{term}",
    # A second location lets the stash's corroboration fire: the same claim
    # from two netlocs is held at 0.8 instead of 0.55. The independence of
    # these two is genuinely weak - a known limitation of netloc-counting,
    # recorded in the roadmap - but two accounts of a definition agreeing is
    # still more than one.
    "https://simple.wikipedia.org/wiki/{term}",
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")

#: Words too common to be worth asking about.
_STOPWORDS = frozenset("""
the a an and or but if then than that this these those there here what when where
who whom which why how is are was were be been being do does did done have has had
of to in on at by for with from into about as it its it's you your yours i me my
mine we us our ours they them their can could should would may might must will
just not no yes so very more most much many some any all each every other same
show tell give make take get got let see look know think want need use used using
""".split())


@dataclass
class Question:
    """Something the model wants to know, and what it will do with the answer."""

    id: int
    kind: str
    prompt: str
    subject: str
    score: float
    options: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    #: Free text, a chosen option, or five glyph rows, depending on ``kind``.
    expects: str = "text"

    def as_dict(self) -> dict:
        """JSON-safe form."""
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "subject": self.subject,
            "score": round(self.score, 4),
            "options": list(self.options),
            "expects": self.expects,
        }


@dataclass
class Answer:
    """What came of answering a question."""

    question: Question
    accepted: bool
    detail: str
    learned: dict = field(default_factory=dict)


class LearningSession:
    """Finds knowledge gaps, asks about them, and applies the answers."""

    def __init__(self, session: Any, max_questions: int = 12) -> None:
        """Bind to an interpreter session.

        Args:
            session: The :class:`~ultraquant.interpreter.thoughts.Session`.
            max_questions: Cap on the queue length.
        """
        self.session = session
        self.max_questions = int(max_questions)
        self.pending: list[Question] = []
        self.answered: list[Answer] = []
        self.skipped: set[int] = set()
        self._next_id = 1
        self.started = time.time()

    # ------------------------------------------------------------ discovery

    def _new(self, kind: str, prompt: str, subject: str, score: float, **kwargs) -> Question:
        """Mint a question."""
        question = Question(
            id=self._next_id, kind=kind, prompt=prompt, subject=subject,
            score=score, **kwargs,
        )
        self._next_id += 1
        return question

    def _shaky_facts(self) -> list[Question]:
        """Facts held weakly enough to be worth confirming."""
        memory = self.session.memory
        # Through the public API: facts may live in the shard library rather
        # than in a dict on the memory object, and reaching for the dict was
        # how learning mode went blind the day they moved.
        out = []
        for key in memory.fact_keys():
            record = memory.recall_fact(key) or {}
            confidence = float(record.get("confidence", 0.5))
            if confidence >= 0.75:
                continue
            reinforcements = int(record.get("reinforcements", 0))
            # Uncertain *and* used is what makes a gap worth closing.
            score = (0.75 - confidence) * (1.0 + reinforcements)
            out.append(self._new(
                "shaky",
                f"I hold '{key}' as {record.get('value')!r}, but only at "
                f"{confidence:.0%} confidence. Is that right?",
                subject=key,
                score=score,
                options=["yes", "no"],
                context={"value": record.get("value"), "confidence": confidence},
                expects="choice",
            ))
        return out

    def _stash_gaps(self) -> list[Question]:
        """Disputed and unclassifiable web claims."""
        stash = getattr(self.session, "stash", None)
        if stash is None:
            return []
        out = []
        for entry in stash.entries():
            if entry["status"] == "disputed":
                known = self.session.memory.recall_fact(
                    entry["claim"].split(" is ")[0].strip().lower()
                )
                out.append(self._new(
                    "disputed",
                    f"A source claims: \"{entry['claim'][:150]}\". That contradicts "
                    f"what I hold"
                    + (f" ({known.get('value')!r})" if known else "")
                    + ". Which should I believe?",
                    subject=str(entry["id"]),
                    score=2.0,
                    options=["keep what I have", "use the new claim", "neither"],
                    context={"entry": entry["id"], "url": entry["url"]},
                    expects="choice",
                ))
            elif entry["classification"] == "unclassified" and entry["status"] == "staged":
                out.append(self._new(
                    "unclassified",
                    f"I could not judge this claim: \"{entry['claim'][:150]}\". "
                    f"Is it a fact, an opinion, or should I drop it?",
                    subject=str(entry["id"]),
                    score=0.8,
                    options=["fact", "opinion", "drop"],
                    context={"entry": entry["id"], "url": entry["url"]},
                    expects="choice",
                ))
        return out

    def _unknown_terms(self) -> list[Question]:
        """Words that keep coming up with nothing stored against them."""
        memory = self.session.memory
        counts: dict[str, int] = {}
        for episode in memory.recall_episodes(limit=200):
            content = episode.get("content") or {}
            text = str(content.get("text", ""))
            for token in set(_TOKEN_RE.findall(text.lower())):
                if token in _STOPWORDS:
                    continue
                counts[token] = counts.get(token, 0) + 1

        out = []
        for token, seen in sorted(counts.items(), key=lambda kv: -kv[1]):
            if seen < 3:
                continue
            if memory.recall_fact(token) is not None:
                continue
            out.append(self._new(
                "unknown-term",
                f"You have mentioned '{token}' {seen} times and I hold nothing "
                f"about it. What is {token}?",
                subject=token,
                score=0.4 * seen,
                context={"occurrences": seen},
                expects="text",
            ))
            if len(out) >= 4:
                break
        return out

    def _unknown_patterns(self) -> list[Question]:
        """Patterns the model met and could not place.

        This is the loop closing: recognition reports what it cannot identify,
        and the thing it could not identify becomes the question worth asking.
        Answering it teaches the pattern into the library, after which the same
        input routes correctly.
        """
        out = []
        seen: set[str] = set()
        for episode in self.session.memory.recall_episodes(
            kind="unknown-pattern", limit=40
        ):
            content = episode.get("content") or {}
            rows = content.get("rows") or []
            if len(rows) != 5:
                continue
            key = "".join(rows)
            if key in seen:
                continue
            seen.add(key)
            nearest = content.get("nearest")
            hint = (f" The closest thing I hold is '{nearest[0]}'."
                    if nearest else "")
            out.append(self._new(
                "unknown-pattern",
                "I met this pattern and could not place it:\n  "
                + "\n  ".join(rows)
                + hint
                + " What is it? Answer as '<category> <label>'.",
                subject=key,
                score=1.8,
                context={"rows": list(rows)},
                expects="text",
            ))
            if len(out) >= 3:
                break
        return out

    def _proposed_concepts(self) -> list[Question]:
        """Groups the model found by itself and wants a name for.

        Every other question here asks the user to fill in something the model
        knows it is missing. This one is different in kind: the model decided
        that a set of things it could not place belong *together*, and asks only
        what to call them. That is the difference between being taught a
        vocabulary and proposing one.

        It stays quiet unless the grouping is well separated -- k-means always
        returns clusters, so without a floor a system that proposes its own
        categories will confidently invent them out of noise.
        """
        from ultraquant.reason.discovery import ConceptDiscovery

        out = []
        for proposal in ConceptDiscovery(self.session).propose()[:2]:
            preview = "\n  ".join(
                "  ".join(rows) for rows in proposal["rows"][:3]
            )
            out.append(self._new(
                "proposed-concept",
                f"I have met {proposal['size']} patterns that group together and "
                f"I have no name for them. A few of them:\n  {preview}\n"
                f"What should I call this category? Answer with one word.",
                subject=f"cluster-{proposal['cluster']}",
                # Above unknown-pattern: naming a group settles several
                # unplaceable inputs at once.
                score=2.2,
                context={"rows": proposal["rows"], "size": proposal["size"],
                         "silhouette": proposal["silhouette"]},
                expects="text",
            ))
        return out

    def _answer_proposed_concept(self, question: Question, reply: str) -> Answer:
        """Name a discovered group, and train it from every member."""
        from ultraquant.interpreter.selflearn import SelfLearner

        category = reply.split()[0].strip().lower() if reply.split() else ""
        if not category:
            return Answer(question, False, "give the category one word")

        rows_list = question.context["rows"]
        labels = sorted({category, f"not_{category}"})
        self.session.experts.ensure_expert(category, labels)
        learner = SelfLearner(self.session)
        report: dict = {}
        for rows in rows_list:
            report = learner.teach_glyph(category, labels, list(rows), category)
        return Answer(
            question, True,
            f"named the group '{category}' and trained it on "
            f"{len(rows_list)} examples: accuracy {report.get('accuracy', 0.0):.2f}",
            {"category": category, "examples": len(rows_list), **report},
        )

    def _weak_patterns(self) -> list[Question]:
        """Categories whose expert is missing or unconvincing."""
        vault = self.session.vault
        experts = self.session.experts
        out = []
        for entry in vault.catalog():
            if entry.get("kind") != "expert-net":
                continue
            category = entry["category"]
            try:
                payload = vault.get(entry["shard_id"])
            except Exception:  # noqa: BLE001 - an unreadable shard is its own problem
                continue
            labels = payload.get("labels") or []
            prototypes = payload.get("prototypes") or {}
            if not prototypes:
                continue
            worst_label, worst_confidence = None, 1.0
            for label, rows in prototypes.items():
                try:
                    from ultraquant.pattern.recognition import render, row_means

                    pixels = render(list(rows))
                    predicted, confidence = experts.predict(category, pixels + row_means(pixels))
                except Exception:  # noqa: BLE001
                    continue
                if predicted != label or confidence < worst_confidence:
                    worst_label, worst_confidence = label, confidence
            if worst_label is not None and worst_confidence < 0.6:
                out.append(self._new(
                    "weak-pattern",
                    f"I recognise '{worst_label}' in {category} at only "
                    f"{worst_confidence:.0%} confidence. Show me one - five rows "
                    f"of five using # and . - and I will train on it.",
                    subject=f"{category}/{worst_label}",
                    score=1.5 * (0.6 - worst_confidence) * 10,
                    context={"category": category, "label": worst_label,
                             "labels": labels},
                    expects="glyph",
                ))
        return out

    def _untrained_categories(self) -> list[Question]:
        """Categories the router knows but nothing covers.

        The remedy depends on what kind of category it is, and asking for the
        wrong one is worse than useless. A *pattern* category — the glyph
        fallback, or anything holding content signatures — is remedied by a
        drawn demonstration. A *topic* category (``arithmetic``, ``world``) is
        text the router steers by keywords; a glyph teaches it nothing, and
        the first end-to-end drive of the loop showed exactly that absurdity:
        the model asking for five rows of pixels to learn about arithmetic.
        Those ask for knowledge instead.
        """
        from ultraquant.interpreter.thoughts import GLYPH_FALLBACK_CATEGORY

        router = self.session.router
        vault = self.session.vault
        signatures = vault.signatures()
        out = []
        state = router.state() if hasattr(router, "state") else {}
        categories = list(state.get("base", {}) or state.get("categories", {}) or {})
        for category in categories:
            if vault.has(ExpertPool.shard_id(category)):
                continue
            pattern_like = (category == GLYPH_FALLBACK_CATEGORY
                            or category in signatures)
            if pattern_like:
                out.append(self._new(
                    "untrained-category",
                    f"I route patterns to '{category}' but have no expert for "
                    f"it. Show me an example - five rows of five using # and . "
                    f"- and I will start one.",
                    subject=category,
                    score=0.9,
                    context={"category": category, "remedy": "glyph"},
                    expects="glyph",
                ))
            else:
                out.append(self._new(
                    "untrained-category",
                    f"I route things to '{category}' but hold no knowledge "
                    f"about it. Tell me something about {category} - a "
                    f"statement like 'X is Y' - and I will remember it.",
                    subject=category,
                    score=0.9,
                    context={"category": category, "remedy": "fact"},
                    expects="text",
                ))
        return out

    def research(self, sources: list[str] | None = None,
                 max_questions: int = 4) -> list[dict]:
        """Try to answer the model's own questions from the web, first.

        The point of self-learning is that the model works out what it wants
        to know; the point of this is that a human is the *second* resort. For
        each researchable pending question the subject is fetched from the
        configured sources, and everything found goes through the contemporary
        stash exactly as a user-pasted URL would — quarantined, classified,
        corroborated — because found is not believed
        ([ARCHITECTURE.md §4.2]). Only a claim the stash judges a factual
        claim about the subject is promoted; everything else stays staged for
        ordinary triage, and the question stays open.

        Args:
            sources: URL templates with a ``{term}`` placeholder. The default
                looks the term up on Wikipedia. Tests point this at a local
                server; nothing here hard-requires any particular site.
            max_questions: How many pending questions to research this pass.

        Returns:
            One report per question attempted:
            ``{"subject", "kind", "outcome", "detail"}`` where outcome is
            ``resolved`` (a fact was promoted and the question closed),
            ``stashed`` (claims staged, nothing promotable), ``nothing``
            (fetch found no usable claims), ``offline`` or ``error``.
        """
        from ultraquant.interpreter.webaccess import WebDisabled

        sources = list(sources or RESEARCH_SOURCES)
        reports: list[dict] = []
        researchable = [
            question for question in self.pending
            if question.id not in self.skipped and (
                question.kind == "unknown-term"
                or (question.kind == "untrained-category"
                    and question.context.get("remedy") == "fact")
            )
        ]
        for question in researchable[:max_questions]:
            term = question.subject
            report = {"subject": term, "kind": question.kind,
                      "outcome": "nothing", "detail": ""}
            reports.append(report)
            staged_ids: list[int] = []
            for template in sources:
                url = template.format(term=term.replace(" ", "_"))
                try:
                    page = self.session.web.fetch(url)
                except WebDisabled:
                    report["outcome"] = "offline"
                    report["detail"] = "web access is off - ':online on' to allow it"
                    break
                except Exception as exc:  # noqa: BLE001 - a dead source is data
                    report["outcome"] = "error"
                    report["detail"] = f"{type(exc).__name__}: {exc}"
                    continue
                staged_ids.extend(
                    self.session.stash.add_page(url, page["title"], page["text"])
                )
            if report["outcome"] == "offline":
                break
            if not staged_ids:
                continue
            self.session.stash.analyze(self.session.memory)

            promoted_key = None
            candidate = self._best_definition(term, staged_ids)
            if candidate is not None:
                try:
                    promoted_key = self.session.stash.promote(
                        candidate, self.session.memory
                    )
                except StashError:
                    promoted_key = None
            if promoted_key is not None:
                # A question about 'code' is only resolved if 'code' can now
                # be recalled. The first live run closed a question with the
                # fact stored under 'early example of a code' - true, stored,
                # and unreachable by the very question it supposedly answered.
                # The promoted key keeps its provenance; the bare term gets
                # the same value so "what is X?" actually works.
                promoted = self.session.memory.recall_fact(promoted_key) or {}
                confidence = float(promoted.get("confidence", 0.55))

                # Graded belief from subject-level agreement: what the sources
                # jointly say about the subject, accumulated, lifts confidence
                # on a measured scale. The floor (0.08) sits above the largest
                # cross-subject agreement ever measured (0.048); full credit
                # (0.30) is where the strongest live same-subject agreement
                # sat (0.327). Between them, belief rises linearly - agreement
                # is evidence by degree, not a switch.
                evidence = self.session.stash.subject_support(term)
                support = evidence["support"]
                if support > 0.08:
                    scaled = min(1.0, (support - 0.08) / (0.30 - 0.08))
                    confidence = max(confidence, min(0.85, 0.55 + 0.25 * scaled))
                    for key in {promoted_key, term}:
                        fact = self.session.memory.recall_fact(key)
                        if fact is not None:
                            self.session.memory.confirm_fact(key, confidence)
                    report["support"] = support
                if self.session.memory.recall_fact(term) is None and promoted:
                    self.session.memory.remember_fact(
                        term, promoted.get("value"), confidence=confidence,
                    )
                report["outcome"] = "resolved"
                report["detail"] = (
                    f"stored '{promoted_key}' from the web"
                    + (f"; cross-site support {support:.2f} -> "
                       f"confidence {confidence:.2f}" if support > 0.08 else "")
                )
                self.answered.append(Answer(
                    question, True,
                    f"researched: {report['detail']}",
                    {"fact": promoted_key, "source": "web"},
                ))
                self.pending = [q for q in self.pending
                                if q.id != question.id]
            else:
                report["outcome"] = "stashed"
                report["detail"] = (f"{len(staged_ids)} claim(s) quarantined; "
                                    "none promotable - see ':stash'")
        if reports and any(r["outcome"] == "resolved" for r in reports):
            self.session.save()
        return reports

    def _best_definition(self, term: str, staged_ids: list[int]) -> int | None:
        """The stashed claim that actually *defines* the term, if any.

        The first live run promoted whatever factual claim merely mentioned
        the subject — which on a real Wikipedia page meant a hatnote ("For the
        1703 Russian textbook, see ...") stored under a junk key. A claim only
        answers "what is X?" if it *says what X is*: it must split as
        ``key is value`` with the term in the key. Exact key matches beat
        prefix/suffix matches beat long keys that merely contain the term;
        singular/plural is tolerated. No qualifying claim means the question
        stays open — everything fetched remains quarantined for triage either
        way, including definitional claims the stash marked *disputed* because
        they contradict something already held. Those become disputed
        questions, which is the quarantine doing its job, not a failure here.
        """
        wanted = term.lower().strip()
        stems = {wanted, wanted.rstrip("s")}
        best_id = None
        best_score = 0
        for entry_id in staged_ids:
            entry = self.session.stash.get(entry_id)
            if (entry["classification"] != "factual-claim"
                    or entry["status"] not in ("staged", "corroborated")):
                continue
            claim = entry["claim"]
            if " is " not in claim.lower():
                continue
            key = claim.lower().split(" is ", 1)[0].strip().strip('"')
            # Encyclopedia leads front-load a qualifier clause before the
            # subject - "In communications and information processing, code is
            # a system of rules". Measured live, the length cap excluded
            # exactly that sentence and a worse claim ("An early example of a
            # code is language...") won, so "what is code?" answered that code
            # IS language. The part after the final comma is the subject the
            # sentence defines; both forms are scored and the better one
            # counts.
            variants = {key}
            if ", " in key:
                variants.add(key.rsplit(", ", 1)[1].strip())
            score = 0
            for variant in variants:
                for article in ("the ", "a ", "an "):
                    if variant.startswith(article):
                        variant = variant[len(article):]
                if variant in stems:
                    score = max(score, 3)
                elif any(variant.startswith(stem) or variant.endswith(stem)
                         for stem in stems) and len(variant) <= len(wanted) + 30:
                    score = max(score, 2)
                elif any(stem in variant for stem in stems) and len(variant) <= 45:
                    score = max(score, 1)
            if score > best_score:
                best_id, best_score = entry_id, score
        return best_id

    def survey(self) -> list[Question]:
        """Look for gaps and build the question queue, best first.

        Returns:
            The pending questions.
        """
        found: list[Question] = []
        for finder in (
            self._stash_gaps,
            self._unknown_patterns,
            self._proposed_concepts,
            self._weak_patterns,
            self._shaky_facts,
            self._unknown_terms,
            self._untrained_categories,
        ):
            try:
                found.extend(finder())
            except Exception:  # noqa: BLE001 - one bad probe must not stop the survey
                continue
        found.sort(key=lambda q: -q.score)
        self.pending = found[: self.max_questions]
        return self.pending

    # ------------------------------------------------------------- asking

    def next_question(self) -> Question | None:
        """The most valuable unanswered question, or None.

        With a whimsy well on the session, curiosity occasionally lifts a
        lower-ranked question instead — every pending question is *worth
        asking* (acceptable-equals, which is what makes the draw admissible),
        the ranking is a value estimate, and always asking the top of a crude
        estimate never wanders. The draw is receipted and replayable; without
        a well, behaviour is exactly the deterministic ranking.
        """
        open_questions = [
            question for question in self.pending
            if question.id not in self.skipped and not any(
                a.question.id == question.id for a in self.answered
            )
        ]
        if not open_questions:
            return None
        well = getattr(self.session, "whimsy", None)
        if (well is not None and len(open_questions) > 1
                and well.occasionally("learning", "curiosity", out_of=6)):
            return well.choose("learning", "curiosity-pick",
                               open_questions[1:])
        return open_questions[0]

    def skip(self, question_id: int) -> None:
        """Set a question aside for this session."""
        self.skipped.add(int(question_id))

    # ------------------------------------------------------------ answering

    def answer(self, question: Question, reply: str) -> Answer:
        """Apply an answer, updating whichever store it belongs to.

        Args:
            question: The question being answered.
            reply: The user's answer — free text, an option, or glyph rows.

        Returns:
            An :class:`Answer` describing what changed.
        """
        reply = (reply or "").strip()
        if not reply:
            self.skip(question.id)
            return Answer(question, False, "skipped (no answer given)")

        handler = {
            "shaky": self._answer_shaky,
            "disputed": self._answer_disputed,
            "unclassified": self._answer_unclassified,
            "unknown-term": self._answer_unknown_term,
            "unknown-pattern": self._answer_unknown_pattern,
            "proposed-concept": self._answer_proposed_concept,
            "weak-pattern": self._answer_glyph,
            "untrained-category": self._answer_untrained,
        }.get(question.kind)
        if handler is None:
            return Answer(question, False, f"no handler for {question.kind!r}")

        try:
            result = handler(question, reply)
        except Exception as exc:  # noqa: BLE001 - report, never crash the session
            result = Answer(question, False, f"{type(exc).__name__}: {exc}")

        self.answered.append(result)
        self.session.memory.remember_episode(
            "learning",
            {
                "kind": question.kind,
                "subject": question.subject,
                "prompt": question.prompt,
                "reply": reply[:200],
                "accepted": result.accepted,
                "detail": result.detail,
            },
            tags=["learning", question.kind],
        )
        self.session.save()
        return result

    def _answer_shaky(self, question: Question, reply: str) -> Answer:
        """Confirm, correct, or reject a weakly held fact."""
        memory = self.session.memory
        lowered = reply.lower()
        if lowered in ("yes", "y", "correct", "right", "confirm"):
            # Direct testimony, not another incidental mention: assert the
            # confidence rather than nudging it.
            memory.confirm_fact(question.subject, confidence=0.9)
            return Answer(question, True,
                          f"confirmed '{question.subject}' at higher confidence",
                          {"fact": question.subject})
        if lowered in ("no", "n", "wrong", "incorrect"):
            memory.remember_fact(question.subject, "(unconfirmed)", confidence=0.2)
            return Answer(question, True,
                          f"marked '{question.subject}' unconfirmed - tell me the "
                          f"right value and I will store it",
                          {"fact": question.subject})
        # Anything else is taken as the corrected value.
        memory.remember_fact(question.subject, reply, confidence=0.85)
        return Answer(question, True, f"corrected '{question.subject}' to {reply!r}",
                      {"fact": question.subject})

    def _answer_disputed(self, question: Question, reply: str) -> Answer:
        """Resolve a contradiction between a source and stored belief."""
        stash = self.session.stash
        entry_id = int(question.context["entry"])
        lowered = reply.lower()
        if lowered.startswith("use") or lowered in ("new", "2", "b"):
            key = stash.promote(entry_id, self.session.memory, force=True)
            return Answer(question, True, f"adopted the new claim as '{key}'",
                          {"fact": key})
        if lowered.startswith("keep") or lowered in ("old", "1", "a"):
            stash.reject(entry_id, "user kept the existing fact")
            return Answer(question, True, "kept what I had; rejected the claim",
                          {"stash": entry_id})
        stash.reject(entry_id, "user rejected both")
        return Answer(question, True, "dropped the claim; existing fact left alone",
                      {"stash": entry_id})

    def _answer_unclassified(self, question: Question, reply: str) -> Answer:
        """Classify a stashed claim the model could not judge."""
        stash = self.session.stash
        entry_id = int(question.context["entry"])
        lowered = reply.lower()
        if lowered.startswith("fact"):
            try:
                key = stash.promote(entry_id, self.session.memory, force=True)
            except StashError as exc:
                return Answer(question, False, str(exc))
            return Answer(question, True, f"stored as fact '{key}'", {"fact": key})
        if lowered.startswith("opinion"):
            entry = stash.get(entry_id)
            entry["classification"] = "opinion"
            stash.reject(entry_id, "classified as opinion by the user")
            return Answer(question, True, "marked as opinion and left out of memory",
                          {"stash": entry_id})
        stash.reject(entry_id, "dropped by the user")
        return Answer(question, True, "dropped", {"stash": entry_id})

    def _answer_unknown_term(self, question: Question, reply: str) -> Answer:
        """Store what a repeatedly-seen term means."""
        self.session.memory.remember_fact(question.subject, reply, confidence=0.8)
        return Answer(question, True, f"learned '{question.subject}' = {reply!r}",
                      {"fact": question.subject})

    def _answer_unknown_pattern(self, question: Question, reply: str) -> Answer:
        """Learn a pattern the model could not place, from '<category> <label>'."""
        from ultraquant.interpreter.selflearn import SelfLearner

        parts = reply.split()
        if len(parts) == 1:
            category, label = parts[0], parts[0]
        elif len(parts) >= 2:
            category, label = parts[0], parts[1]
        else:
            return Answer(question, False, "answer as '<category> <label>'")

        rows = list(question.context["rows"])
        labels = [label]
        shard_id = ExpertPool.shard_id(category)
        if self.session.vault.has(shard_id):
            stored = self.session.vault.get(shard_id).get("labels") or []
            labels = sorted(set(list(stored) + [label]))
        if len(labels) < 2:
            labels = sorted(set(labels + [f"not_{label}"]))

        self.session.experts.ensure_expert(category, labels)
        report = SelfLearner(self.session).teach_glyph(category, labels, rows, label)
        return Answer(
            question, True,
            f"learned '{label}' into {category}: accuracy {report['accuracy']:.2f}",
            {"category": category, "label": label, **report},
        )

    def _answer_untrained(self, question: Question, reply: str) -> Answer:
        """Apply whichever remedy the category actually needs."""
        if question.context.get("remedy") == "fact":
            return self._answer_topic_fact(question, reply)
        return self._answer_glyph(question, reply)

    def _answer_topic_fact(self, question: Question, reply: str) -> Answer:
        """Learn a statement about a topic category, and route towards it."""
        from ultraquant.interpreter.selflearn import SelfLearner

        category = question.context["category"]
        parsed = SelfLearner.extract_facts(reply)
        if not parsed:
            return Answer(question, False,
                          "tell me as a statement: 'X is Y'")
        for key, value in parsed:
            self.session.memory.remember_fact(key, value, confidence=0.7)
        # The statement's words now pull towards the category, so the next
        # question about this topic routes where the knowledge lives.
        self.session.router.learn(reply, category, delta=0.2)
        self.session.memory.remember_episode(
            "topic-teaching",
            {"category": category, "facts": [list(pair) for pair in parsed]},
            tags=["learn", category],
        )
        self.session.save()
        keys = ", ".join(key for key, _value in parsed)
        return Answer(question, True,
                      f"remembered {len(parsed)} fact(s) about {category}: {keys}",
                      {"category": category, "facts": len(parsed)})

    def _answer_glyph(self, question: Question, reply: str) -> Answer:
        """Train a demonstrated glyph into the relevant expert."""
        from ultraquant.interpreter.selflearn import SelfLearner

        rows = [line.strip() for line in reply.splitlines() if line.strip()]
        if len(rows) != 5 or not all(
            len(r) == 5 and set(r) <= {"#", "."} for r in rows
        ):
            return Answer(question, False,
                          "I need exactly five rows of five characters using # and .")

        category = question.context["category"]
        label = question.context.get("label") or f"{category}_1"
        labels = list(question.context.get("labels") or [])
        if label not in labels:
            labels = sorted(set(labels + [label])) or [label]
        if len(labels) < 2:
            # A classifier needs something to discriminate against.
            labels = sorted(set(labels + [f"not_{label}"]))

        self.session.experts.ensure_expert(category, labels)
        report = SelfLearner(self.session).teach_glyph(category, labels, rows, label)
        return Answer(
            question, True,
            f"trained '{label}' in {category}: accuracy {report['accuracy']:.2f}",
            {"category": category, "label": label, **report},
        )

    # ------------------------------------------------------------- reporting

    def stats(self) -> dict:
        """How the session is going."""
        accepted = sum(1 for a in self.answered if a.accepted)
        by_kind: dict[str, int] = {}
        for question in self.pending:
            by_kind[question.kind] = by_kind.get(question.kind, 0) + 1
        return {
            "pending": len(self.pending),
            "asked": len(self.answered),
            "accepted": accepted,
            "skipped": len(self.skipped),
            "by_kind": by_kind,
            "seconds": round(time.time() - self.started, 1),
        }
