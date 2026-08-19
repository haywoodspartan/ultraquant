"""LLMLS: a panel of teacher models, and an accounting of what their agreement is worth.

The user picks which open-source models sit on the panel; LM Studio loads them on
demand. Each is asked the *same* question in isolation, and what they agree on
enters the contemporary stash — quarantined, like anything else the system did
not derive itself.

**Why a panel rather than a model.** A single model's answer is one source, and
this system's founding rule is that a source is not a fact. Asking one model
twice produces agreement carrying no evidence whatsoever; so does asking a model
and its own fine-tune. A panel is the only arrangement in which agreement can
mean something — and only to the extent its members are genuinely independent.
That "only to the extent" is the whole design problem, and it is where most of
this module's care goes.

**Independence cannot be proved from metadata, so it is bounded instead.** The
LM Studio catalogue on the machine this was written against makes the problem
concrete:

===========================================  ==============  ============
model                                        ``arch``        ``publisher``
===========================================  ==============  ============
``qwen/qwen3-coder-30b``                     qwen3moe        qwen
``qwen/qwen3-coder-next``                    qwen3next       qwen
``openai/gpt-oss-20b``                       gpt-oss         openai
``openai-gpt-oss-20b-abliterated-...``       gpt-oss         DavidAU
``deepseek-coder-33b-instruct``              llama           TheBloke
``codellama-34b``                            llama           TheBloke
===========================================  ==============  ============

Neither field works alone. The two Qwens have *different* ``arch`` strings and
are obviously one lineage — publisher catches them. The gpt-oss pair has
different publishers and is one lineage — an abliterated fine-tune of the other —
and ``arch`` catches that. Meanwhile ``arch=llama`` and ``publisher=TheBloke``
both lump DeepSeek-Coder with CodeLlama, which are genuinely separate trainings
that merely share an architecture and a quantizer.

**A second catalogue showed the failure in the other, worse direction.** After
the user swapped their models, three Qwen derivatives defeated all three signals
at once:

===================================================  ===========  ==============
model                                                ``arch``     ``publisher``
===================================================  ===========  ==============
``qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive``   qwen35moe    HauhauCS
``qwen3-48b-a4b-savant-commander-distill-12x-...``   qwen3moe     DavidAU
``katarau-9b-ru-rp-nsfw``                            qwen35       mradermacher
===================================================  ===========  ==============

Different arch strings, different publishers, name stems mangled past
recognition — and the panel reported **three independent voices** for what is
one base family. That is the direction that matters, because under-grouping
*manufactures evidence*. :func:`_arch_family` now reduces an architecture to its
leading alphabetic run, so all three collapse to ``qwen``, and that catalogue
went from six voices to four.

So :func:`lineage_key` deliberately **over-groups**: any shared architecture
family, publisher, or name stem collapses models into one voice. It will
sometimes call two independent models correlated and cost the panel some
corroboration it had earned. That direction is chosen on purpose — the opposite
error *manufactures evidence*, which is the failure this system exists to avoid. The returned
:class:`Consensus` therefore reports independent-lineage counts as a **lower
bound on correlation**, never as a proof of independence, and
:attr:`Consensus.caveat` says so in the output rather than in a comment.

**Disagreement is the useful output, not a failure.** A claim two independent
lineages contradict each other on is exactly the claim that should stay
quarantined, and a single-model integration cannot produce that signal at all.
:meth:`TeacherPanel.ask` reports splits as first-class results.

**Where a panel is the wrong instrument.** Consensus only measures anything when
the question has a determinate answer *and a canonical form to state it in*.
Three failure modes were found by running it, and they are different problems:

1. **Open-ended prompts.** "Tell me something about arithmetic" drew "branch of
   mathematics dealing with numbers" and "commutative under addition" — both
   correct, no corroboration. A question with many right answers cannot be
   settled by agreement, and reporting none is the honest reading.
2. **Under-specified prompts.** "What is code?" drew "set of instructions for a
   computer" and "secret communication system" — both correct, about different
   senses of a polysemous word. This one is *fixable*, and
   :func:`grounded_prompt` fixes it: the system asked because it had seen the
   term repeatedly, and those sentences say which sense it meant. Passing them
   moved both models onto the programming sense.
3. **Definitional answers, which remain unsolved.** Grounded, the same two
   models returned "sequence instructions executed computer" and "computer
   instructions written programming language". Same claim, different words,
   scored as a split — because positions are compared exactly, and deciding
   that two phrasings mean the same thing is the semantic-equivalence problem
   this module refuses to guess at.

So the panel corroborates cleanly where an answer has a canonical form —
``au``, ``canberra``, ``2``, ``frank herbert``, ``o log n`` — and poorly on
definitions, which legitimately admit many wordings. That is a real limit, not a
tuning problem, and :attr:`Consensus.overlap_note` states it in the output so a
reader does not mistake "not corroborated" for "the models disagree". The note
is **reported and never credited**, exactly as monobit is in
:mod:`ultraquant.quantum.entropy_blackbox`: crediting the test that a
coincidence would pass is how manufactured consensus gets in.

**What the panel is never allowed to do** is speak for the system. The models
here are *sources being interrogated*, in the same position as a scraped web
page. The system's own answers still come from its induced grammar (§11.10);
routing this through a chat model would make every claim in §11 unfalsifiable
while making the demo look better. See
:mod:`ultraquant.interpreter.lmstudio` for the longer form of that refusal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ultraquant.interpreter.lmstudio import (
    MODEL_SOURCE_HOST,
    Answer,
    LMStudioClient,
    LMStudioUnavailable,
)

__all__ = [
    "grounded_prompt",
    "ModelCard",
    "TeacherPanel",
    "Consensus",
    "lineage_key",
    "catalogue",
    "find_cli",
]

#: LM Studio's richer, non-OpenAI endpoint. Carries arch/publisher/state, which
#: the OpenAI-compatible /v1/models does not.
_NATIVE_MODELS = "/api/v0/models"

#: Idle seconds before LM Studio unloads a panel model. A seven-teacher panel
#: must not need seven models resident; without a TTL it would.
DEFAULT_TTL = 900

#: Words a model reaches for when it does not know, which must never become a
#: quarantined claim. Checked as a prefix on the stripped reply.
_REFUSALS = (
    "i don't know", "i do not know", "unknown", "i'm not sure", "i am not sure",
    "as an ai", "i cannot", "i can't", "there is no", "insufficient",
)

#: The default system message, and a load-bearing part of the method rather
#: than a nicety.
#:
#: Positions are compared by exact string match after normalisation, because
#: judging two differently-worded sentences to "mean the same thing" is a hard
#: problem and a heuristic that guessed at it would *manufacture consensus* —
#: the one error this system must not make. Exact matching is only usable if
#: replies are short and bare, so the panel asks for that explicitly.
#:
#: This was not a design instinct; it was measured. An early live run over three
#: independent models asked "what is the time complexity of binary search?"
#: without this constraint. All three answered O(log n) — unanimous — but two
#: wrapped it in different prose and the panel reported 2 of 3 voices. On "best
#: language for beginners" all three said Python and it reported 1 of 3. An
#: agreement measure that misses unanimity is not conservative, it is broken.
TERSE = (
    "Answer with the bare fact only: at most eight words, no sentence, no "
    "explanation, no restatement of the question. If you do not know, reply "
    "exactly: unknown"
)

#: Above this many words, a reply is prose and exact matching will under-report
#: agreement. Reported, never silently corrected.
_PROSE_WORDS = 12

#: Completion ceiling per model. Generous on purpose: answer *length* is bounded
#: by :data:`TERSE` in the system prompt, and this is only a safety stop.
#:
#: It has to be generous because a reasoning model's thinking tokens are billed
#: against the same budget. Measured on gpt-oss-20b with a grounded prompt: at
#: max_tokens=60 the reply came back **empty** with
#: ``reasoning_tokens=51`` — 51 of the 60 spent thinking, 9 left, nothing
#: emitted. At 200 the identical request answered "A sequence of instructions
#: executed by a computer." The panel had been recording that truncation as the
#: model having nothing to say, which is a measurement error, not a quiet model.
DEFAULT_MAX_TOKENS = 400

#: An empty reply with at least this fraction of the budget spent on reasoning
#: is starvation, not abstention, and is reported as such.
_STARVED_FRACTION = 0.7

#: At or above this fraction, the model consumed the *entire* budget thinking.
#: That is a different failure from starvation and needs a different fix, so it
#: gets a different message.
#:
#: Measured on ``qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive``, asked for six
#: short phrasings: 400 of 400 completion tokens spent reasoning, then 1200 of
#: 1200 at triple the budget — output empty both times. Raising ``max_tokens``
#: does not help, because the model simply reasons into whatever it is given.
#: ``reasoning_effort="none"`` returned 42 tokens with zero reasoning and the
#: answer; ``"low"``, a ``/no_think`` suffix and
#: ``chat_template_kwargs={"enable_thinking": False}`` each had no effect.
_RUNAWAY_FRACTION = 0.99

#: Passed to models that will otherwise reason forever. Not a global default:
#: a model that reasons *usefully* should be allowed to, and this is applied
#: where the task genuinely needs none (listing phrasings) or where runaway has
#: actually been observed.
NO_REASONING = "none"

#: Content-token overlap above which two positions are *reported* as probably
#: the same claim differently worded — and, deliberately, still not credited.
#:
#: This is the entropy assessor's monobit rule applied here: a signal worth
#: showing the reader, never worth buying agreement with. Measured on the
#: grounded "what is code?" run, where the two independent voices returned
#: "sequence instructions executed computer" and "computer instructions written
#: programming language". Those are the same claim. Deciding so automatically
#: is the semantic-equivalence problem, and a threshold that credited it would
#: manufacture consensus on the day two genuinely different answers happened to
#: share vocabulary. So it is surfaced as an observation and the verdict stands.
#:
#: Calibrated on the six splits observed so far, which separate cleanly::
#:
#:     0.286  same claim   "sequence instructions executed computer"
#:                     vs  "computer instructions written programming language"
#:     0.200  same claim   "python" vs "python simplest syntax beginner friendly"
#:     0.100  contested    "vim widely regarded..." vs "vim emacs vs code..."
#:     0.000  other sense  "set instructions computer" vs "secret communication system"
#:     0.000  other aspect "branch of mathematics..." vs "commutative under addition"
#:     0.000  other answer "canberra" vs "sydney"
#:
#: Six observations is a small calibration and this will misfire in both
#: directions. It is tolerable *only* because the note decides nothing: no
#: verdict, no credit, no stored belief moves on it.
_OVERLAP_NOTE = 0.15


@dataclass(frozen=True)
class ModelCard:
    """One model in the LM Studio catalogue.

    Attributes:
        id: The id to pass as ``model`` in API calls.
        arch: Architecture family as LM Studio reports it (``qwen3moe``,
            ``llama``, ``gpt-oss``...). Catches a fine-tune of another model.
        publisher: Who published this artifact. Catches same-vendor siblings
            with different arch strings. Unreliable on its own — a quantizer
            such as TheBloke publishes many unrelated models.
        kind: ``llm``, ``vlm``, or ``embeddings``.
        loaded: Whether it is currently resident in memory.
        quantization: Reported quantization, for the record.
        context: Maximum context length reported.
    """

    id: str
    arch: str = ""
    publisher: str = ""
    kind: str = "llm"
    loaded: bool = False
    quantization: str = ""
    context: int = 0

    @property
    def is_chat(self) -> bool:
        """Whether this model can answer questions (excludes embedders)."""
        return self.kind in ("llm", "vlm")


def _name_stem(model_id: str) -> str:
    """The family part of a model id, with vendor prefix and size suffix cut.

    ``qwen/qwen3-coder-30b`` and ``qwen3-coder-next`` both reduce to
    ``qwen3-coder``; ``codellama-34b`` and ``codellama-13b-instruct`` both to
    ``codellama``. Crude on purpose — it is one of three over-grouping signals,
    not a classifier.
    """
    name = model_id.split("/")[-1].lower()
    drop = {"instruct", "chat", "next", "coder", "v2", "abliterated",
            "uncensored", "neo", "imatrix", "gguf"}
    parts = []
    for part in name.replace("_", "-").split("-"):
        # A size token (7b, 30b, 3.1) ends the family name.
        if part and (part[0].isdigit() and part[-1] in "bB") or part in drop:
            continue
        parts.append(part)
    return "-".join(parts[:2]) if parts else name


def _arch_family(arch: str) -> str:
    """The base family of an architecture string, ignoring version and variant.

    ``qwen3moe``, ``qwen35moe`` and ``qwen35`` are all Qwen and all report
    *different* ``arch`` strings, so comparing them literally counted one base
    family as three independent voices. That is the failure direction that
    matters: over-grouping costs corroboration the panel had earned, but
    under-grouping **manufactures evidence**, which is the error this module
    exists to prevent.

    Found on a real catalogue where the community had renamed everything:
    ``qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive`` (arch ``qwen35moe``,
    publisher HauhauCS), ``qwen3-48b-a4b-savant-commander-distill-...`` (arch
    ``qwen3moe``, publisher DavidAU) and ``katarau-9b-ru-rp-nsfw`` (arch
    ``qwen35``, publisher mradermacher) defeated all three signals at once —
    different arch strings, different publishers, unrecognisable name stems —
    and were reported as three voices.

    Taking the leading alphabetic run collapses them to ``qwen``. It is blunt
    and deliberately so: ``gemma4`` -> ``gemma``, ``command-r`` -> ``command``,
    ``gpt-oss`` -> ``gpt``. Anything it merges wrongly costs only credit.
    """
    letters = []
    for char in arch.lower().strip():
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters) or arch.lower().strip()


def lineage_key(card: ModelCard) -> tuple[str, str, str]:
    """A conservative identity for "probably the same model family".

    Two cards sharing *any* component are treated as one voice. This
    over-groups — DeepSeek-Coder and CodeLlama both report ``arch=llama`` and
    will be merged though they are separate trainings — and that is the
    intended direction. Under-grouping would let a model and its own fine-tune
    corroborate each other, inventing evidence out of nothing.

    Returns:
        ``(arch_family, publisher, name_stem)``, lowercased. The architecture
        is reduced to its family by :func:`_arch_family` first - comparing raw
        arch strings let three Qwen derivatives pass as three voices.
    """
    return (_arch_family(card.arch),
            card.publisher.lower().strip(),
            _name_stem(card.id))


def _correlated(a: ModelCard, b: ModelCard) -> bool:
    """Whether two models share any lineage signal at all."""
    ka, kb = lineage_key(a), lineage_key(b)
    return any(x and x == y for x, y in zip(ka, kb))


def independent_groups(cards: Sequence[ModelCard]) -> list[list[ModelCard]]:
    """Partition models into suspected-independent voices.

    Transitive: if A shares an arch with B and B shares a publisher with C, all
    three become one voice. Union-find over the pairwise relation, kept
    transitive because correlation is not confined to pairs — a base model, its
    fine-tune, and a requantization of the fine-tune form a chain.

    Returns:
        Groups of cards. ``len(result)`` is the number of independent voices,
        and is a **lower bound on correlation**: the true count may be smaller
        if two apparently-unrelated models share undisclosed training data.
    """
    parent = list(range(len(cards)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if _correlated(cards[i], cards[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[ModelCard]] = {}
    for index, card in enumerate(cards):
        groups.setdefault(find(index), []).append(card)
    return list(groups.values())


@dataclass
class Consensus:
    """What a panel concluded, and what that is worth.

    Attributes:
        question: What was asked.
        answers: Every model's reply, keyed by model id. Kept in full so a
            stored claim can be traced to the exact source that produced it.
        voices: Independent-lineage count behind the majority position.
        panel_voices: Independent-lineage count on the panel overall.
        agreed: Whether a majority of *voices* — not of models — converged.
        split: Distinct positions found, each with its supporting model ids.
        corroborated: True only when at least two independent voices agree.
            One voice is one source, however many models it contains.
        caveat: The standing warning about what independence accounting can
            and cannot establish. Always populated.
        errors: Models that failed to answer, with the reason.
        prose_warning: Set when replies were long enough that exact
            matching will under-report agreement. Never silently applied
            as a correction - the reader is told the number is a floor.
        overlap_note: Set when distinct positions share enough content to
            look like one claim worded differently. Reported so a reader does
            not misread "not corroborated" as "the models disagree"; never
            credited, because crediting it is the manufactured-consensus
            error this module exists to avoid.
    """

    question: str
    answers: dict = field(default_factory=dict)
    voices: int = 0
    panel_voices: int = 0
    agreed: bool = False
    split: dict = field(default_factory=dict)
    corroborated: bool = False
    caveat: str = ""
    errors: dict = field(default_factory=dict)
    prose_warning: str = ""
    overlap_note: str = ""

    def as_text(self) -> str:
        """A printable summary, disagreement included."""
        lines = [f"Q: {self.question}",
                 f"panel: {len(self.answers)} model(s) in {self.panel_voices} "
                 f"independent voice(s)"]
        for position, backers in sorted(self.split.items(),
                                        key=lambda kv: -len(kv[1])):
            lines.append(f"  [{len(backers)}] {position[:90]}")
            for backer in backers:
                lines.append(f"        - {backer}")
        if self.errors:
            for model, reason in self.errors.items():
                lines.append(f"  !! {model}: {reason[:70]}")
        lines.append(
            f"corroborated: {self.corroborated} "
            f"({self.voices}/{self.panel_voices} voices agree)")
        lines.append(f"caveat: {self.caveat}")
        return "\n".join(lines)


def find_cli() -> str | None:
    """Locate the ``lms`` CLI, which is what can load and unload models.

    The HTTP API can *use* a model and will JIT-load it on first request, but
    cannot pin a TTL or unload on demand. Returns None when the CLI is absent,
    in which case the panel still works through JIT loading.
    """
    candidate = os.environ.get("LMS_CLI")
    if candidate and os.path.exists(candidate):
        return candidate
    home = os.path.expanduser("~")
    for relative in (r".lmstudio\bin\lms.exe", ".lmstudio/bin/lms"):
        path = os.path.join(home, relative)
        if os.path.exists(path):
            return path
    from shutil import which
    return which("lms")


def catalogue(base_url: str | None = None, timeout: float = 10.0) -> list[ModelCard]:
    """Every model LM Studio has on disk, with lineage metadata.

    Uses LM Studio's native ``/api/v0/models`` rather than the
    OpenAI-compatible route, because only the native one reports ``arch`` and
    ``publisher`` — without which independence cannot be assessed at all.

    Raises:
        LMStudioUnavailable: If the server is unreachable.
    """
    try:
        root = (base_url or LMStudioClient().base_url).rstrip("/")
    except ValueError as exc:
        # LMStudioClient refuses a non-loopback host with ValueError, which is
        # right - but $LMSTUDIO_BASE_URL pointing off-machine is precisely the
        # inherited-environment case the guard exists for, and this function
        # documents LMStudioUnavailable as its only failure. Callers catch that
        # and nothing else, so the refusal has to arrive in the expected shape.
        raise LMStudioUnavailable(str(exc)) from exc
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(f"{root}{_NATIVE_MODELS}",
                                    timeout=timeout) as response:
            payload = json.loads(response.read())
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("no model list in the response")
        cards = []
        for entry in rows:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            try:
                context = int(entry.get("max_context_length") or 0)
            except (TypeError, ValueError):
                context = 0        # a bad length is not worth losing the model
            cards.append(ModelCard(
                id=str(entry["id"]),
                arch=str(entry.get("arch") or ""),
                publisher=str(entry.get("publisher") or ""),
                kind=str(entry.get("type") or "llm"),
                loaded=str(entry.get("state") or "") == "loaded",
                quantization=str(entry.get("quantization") or ""),
                context=context,
            ))
    except (urllib.error.URLError, OSError, ValueError, TypeError,
            AttributeError) as exc:
        # Parsing moved inside the guard: it used to sit after it, so a
        # well-formed HTTP 200 carrying an unexpected shape escaped as
        # AttributeError past every caller's `except LMStudioUnavailable`.
        raise LMStudioUnavailable(
            f"cannot read the LM Studio catalogue at {root}: {exc}"
        ) from exc
    return cards


class TeacherPanel:
    """A user-chosen set of models, asked in isolation and counted by lineage."""

    def __init__(
        self,
        model_ids: Iterable[str],
        client: LMStudioClient | None = None,
        ttl: int = DEFAULT_TTL,
        gpu: str | None = None,
    ) -> None:
        """
        Args:
            model_ids: Which models sit on the panel. The user's choice — this
                module never picks teachers on its own, because which models
                are trusted is a judgement about sources, not a default.
            client: An :class:`LMStudioClient`; constructed if omitted.
            ttl: Idle seconds before LM Studio unloads a model. Applies only
                when the CLI is available.
            gpu: GPU offload ratio for :meth:`load` — ``"max"``, ``"off"``, or
                a fraction. None lets LM Studio decide.

        Raises:
            LMStudioUnavailable: If the catalogue cannot be read, or a
                requested model is not in it.
        """
        try:
            self.client = client or LMStudioClient()
        except ValueError as exc:
            raise LMStudioUnavailable(str(exc)) from exc
        self.ttl = int(ttl)
        self.gpu = gpu
        self.cli = find_cli()

        available = {card.id: card for card in catalogue(self.client.base_url)}
        wanted = list(model_ids)
        missing = [name for name in wanted if name not in available]
        if missing:
            raise LMStudioUnavailable(
                f"not in the LM Studio catalogue: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(available))}"
            )
        self.cards = [available[name] for name in wanted]
        chat_only = [card for card in self.cards if not card.is_chat]
        if chat_only:
            raise LMStudioUnavailable(
                "these are not chat models and cannot sit on a panel: "
                + ", ".join(card.id for card in chat_only)
            )

    # ---- lineage --------------------------------------------------------

    def voices(self) -> list[list[ModelCard]]:
        """The panel partitioned into suspected-independent voices."""
        return independent_groups(self.cards)

    def independence_report(self) -> str:
        """A printable account of which panel members count as one voice.

        Worth reading before trusting a panel: a five-model panel that resolves
        to two voices provides two sources of evidence, not five, and the
        difference is invisible without this.
        """
        groups = self.voices()
        lines = [f"{len(self.cards)} model(s) -> {len(groups)} independent "
                 f"voice(s) (lower bound on correlation)"]
        for index, group in enumerate(groups, 1):
            head = group[0]
            lines.append(f"  voice {index}: arch={head.arch or '?'} "
                         f"publisher={head.publisher or '?'}")
            for card in group:
                lines.append(f"      {card.id}")
            if len(group) > 1:
                lines.append("      ^ these agree as ONE source, not "
                             f"{len(group)}")
        return "\n".join(lines)

    # ---- model management -----------------------------------------------

    def is_loaded(self, model_id: str) -> bool:
        """Whether the model is resident right now, per a fresh catalogue read."""
        try:
            return any(card.id == model_id and card.loaded
                       for card in catalogue(self.client.base_url))
        except LMStudioUnavailable:
            return False

    def load(self, model_id: str) -> bool:
        """Load one model into memory, with a TTL so it unloads when idle.

        Already-resident models are skipped. ``lms load`` on a loaded model
        does not no-op — it loads a *second instance* under a ``:2`` identifier,
        and a panel re-run across a few 30B models will quietly fill the card
        with duplicates. Found by checking the catalogue after a couple of runs
        and seeing ``gpt-oss-20b:2`` and ``qwen3-coder-30b:2`` resident
        alongside the originals.

        The guard is **best-effort and not atomic**: two processes calling this
        at the same time can both see "not loaded" and both load. Observed
        exactly that way, with a second ``google/gemma-4-31b:2`` appearing when
        two panel runs overlapped. Locking it properly would mean coordinating
        across processes for a convenience feature; the TTL clears the
        duplicate, and ``lms ps`` shows it. Worth knowing before running panels
        concurrently on a card that is already tight.

        Returns:
            True if the model is resident afterwards. False when the CLI is
            absent — not an error: the HTTP API JIT-loads on first request, so
            the panel still works, just without TTL control.
        """
        if self.is_loaded(model_id):
            return True
        if not self.cli:
            return False
        command = [self.cli, "load", model_id, "--ttl", str(self.ttl), "--yes"]
        if self.gpu:
            command += ["--gpu", self.gpu]
        try:
            # encoding is explicit: lms writes UTF-8 progress output that
            # the Windows default (cp1252) cannot decode, which killed the
            # reader thread mid-load in an early run.
            done = subprocess.run(command, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=600)
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0

    def unload(self, model_id: str) -> bool:
        """Unload one model, freeing its memory immediately."""
        if not self.cli:
            return False
        try:
            done = subprocess.run([self.cli, "unload", model_id],
                                  capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=120)
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0

    def unload_all(self) -> None:
        """Unload every panel model. Call when done with a long session."""
        for card in self.cards:
            self.unload(card.id)

    # ---- interrogation --------------------------------------------------

    def ask(
        self,
        question: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sequential: bool = True,
        usages: Sequence[str] | None = None,
    ) -> Consensus:
        """Put one question to every panel member, in isolation.

        Each model sees only the question — never another model's answer.
        Showing them each other's replies would produce agreement by
        contagion, which looks exactly like corroboration and is not.

        Args:
            question: What to ask.
            system: Shared system message, defaulting to :data:`TERSE`. Shared
                on purpose: varying it per model would confound
                disagreement-about-the-answer with
                disagreement-about-the-instruction. Pass ``system=""`` to send
                none, accepting that prose replies will under-report agreement.
            max_tokens: Completion ceiling per model, defaulting to
                :data:`DEFAULT_MAX_TOKENS`. Not the length control - that is
                :data:`TERSE`. Setting this low starves reasoning models,
                whose thinking is billed against it.
            sequential: Ask one at a time. The default, because concurrent
                requests to several large local models contend for the same
                GPU and will usually be slower, not faster.

        Returns:
            The :class:`Consensus`, including any split.
        """
        instruction = TERSE if system is None else (system or None)
        # Asked with context; recorded without it. The Consensus carries
        # the plain question because that is what a stored claim is about.
        asked = grounded_prompt(question, usages)
        answers: dict[str, Answer] = {}
        errors: dict[str, str] = {}
        for card in self.cards:
            try:
                answers[card.id] = self.client.complete(
                    asked, model=card.id, system=instruction,
                    max_tokens=max_tokens, temperature=0.0,
                    # The one measured remedy for runaway reasoners
                    # (see NO_REASONING above). The distillation path has
                    # always passed it; this path forgot, and gemma spent
                    # all 400 tokens thinking on every panel question in
                    # the first library-training run.
                    reasoning_effort=NO_REASONING)
            except LMStudioUnavailable as exc:
                errors[card.id] = str(exc)
        return self._tally(question, answers, errors)

    def _tally(self, question: str, answers: dict, errors: dict) -> Consensus:
        """Group answers into positions and count *voices*, not models."""
        by_card = {card.id: card for card in self.cards}
        panel_voices = len(self.voices())

        split: dict[str, list[str]] = {}
        for model_id, answer in answers.items():
            text = (answer.text or "").strip()
            if not text or text.lower().startswith(_REFUSALS):
                # A refusal is not a position. Counting it as one would let
                # several models "agree" on knowing nothing.
                #
                # An *empty* reply is not a refusal at all, though, and saying
                # "no usable answer" for it hides a bug in the caller: a
                # reasoning model whose thinking consumed the token budget was
                # cut off, not quiet. The two need different fixes, so they get
                # different messages.
                errors.setdefault(model_id, _no_answer_reason(answer))
                continue
            split.setdefault(_position(text, question), []).append(model_id)

        prose = [m for m, a in answers.items()
                 if len((a.text or "").split()) > _PROSE_WORDS]

        overlap = _overlap_note(list(split))

        best_voices, best_position = 0, ""
        for position, backers in split.items():
            cards = [by_card[m] for m in backers if m in by_card]
            count = len(independent_groups(cards)) if cards else 0
            if count > best_voices:
                best_voices, best_position = count, position

        agreed = bool(best_position) and len(split) == 1
        corroborated = best_voices >= 2
        caveat = (
            "independent-voice counts are a LOWER BOUND on correlation: shared "
            "training data between apparently unrelated models cannot be seen "
            "from metadata. Agreement here is evidence, never proof, and "
            "nothing is believed until the stash promotes it."
        )
        return Consensus(
            question=question,
            answers={k: v.text for k, v in answers.items()},
            voices=best_voices,
            panel_voices=panel_voices,
            agreed=agreed,
            split=split,
            corroborated=corroborated,
            caveat=caveat,
            errors=errors,
            prose_warning=(
                f"{len(prose)} model(s) replied in prose (>{_PROSE_WORDS} "
                "words); positions are matched exactly, so the agreement "
                "reported here is a FLOOR - models saying the same thing in "
                "different words are counted as disagreeing"
                if prose else ""),
            overlap_note=overlap,
        )

    # ---- quarantine -----------------------------------------------------

    def teach(self, question: str, stash: Any, max_claims: int = 4,
              system: str | None = None,
              usages: Sequence[str] | None = None) -> dict:
        """Ask the panel and file the result in the contemporary stash.

        Only positions backed by **at least two independent voices** are
        quarantined. A single-voice answer is filed as well but marked, because
        the stash's own corroboration machinery must not mistake one model's
        opinion for a corroborated claim. Nothing here is promoted to fact —
        that decision stays with the stash, exactly as it does for a scraped
        page.

        Args:
            question: What to ask the panel.
            stash: A :class:`~ultraquant.interpreter.stash.ContemporaryStash`.
            max_claims: Ceiling on claims taken from each position.
            system: Optional shared system message.

        Returns:
            ``{"consensus": Consensus, "entry_ids": [...], "filed": n}``.
        """
        consensus = self.ask(question, system=system, usages=usages)
        entry_ids: list[int] = []
        by_card = {card.id: card for card in self.cards}
        for position, backers in consensus.split.items():
            cards = [by_card[m] for m in backers if m in by_card]
            groups = independent_groups(cards) if cards else []
            # One URL per *voice group*, so two models of one lineage cannot
            # enter as two sources and corroborate each other in the stash.
            head = groups[0][0] if groups else None
            if head is None:
                continue
            url = (f"lmstudio://{MODEL_SOURCE_HOST}/panel/{len(groups)}-voice/"
                   f"{head.arch or head.id}")
            title = f"LLMLS panel ({len(groups)} voice): {question[:50]}"
            entry_ids.extend(stash.add_page(url, title,
                                            _as_claim(question, position),
                                            max_claims=max_claims))
        return {"consensus": consensus, "entry_ids": entry_ids,
                "filed": len(entry_ids)}


def grounded_prompt(question: str, usages: Sequence[str] | None = None,
                    limit: int = 3) -> str:
    """Attach the contexts a term was seen in, so the sense is not guessed.

    A bare question can be *under-specified* rather than hard, and the panel
    cannot tell the difference — it reports disagreement either way. Measured:
    asked "what is code?" cold, gpt-oss answered "set of instructions for a
    computer" and Qwen answered "secret communication system". **Both are
    correct**; they answered about different senses of a polysemous word, and
    the panel recorded a split that looked like unreliability.

    The system already held what settles it. It asked because it had seen the
    term repeatedly, and those sentences say which sense is meant. Passing them
    turns an ambiguous question into a determinate one, which is the only kind
    a consensus panel can measure.

    Args:
        question: The question as the system phrased it.
        usages: Sentences the term appeared in. Empty or None returns the
            question unchanged, so nothing depends on context existing.
        limit: How many usages to include; more is context the models will
            summarise rather than use.

    Returns:
        The prompt to send. The plain ``question`` is still what gets recorded
        against any resulting claim.
    """
    examples = [u for u in (usages or []) if u.strip()][:limit]
    if not examples:
        return question
    lines = "\n".join(f"- {u}" for u in examples)
    return (f"{question}\n\nThe term was used in these contexts, so answer "
            f"about this sense of it:\n{lines}")


def _overlap_note(positions: Sequence[str]) -> str:
    """Flag positions that look like one claim worded two ways. Never credit it.

    Grounding a question fixes which *sense* is being asked about; it cannot
    make two models phrase the same definition identically. Measured on the
    grounded "what is code?" run: two independent voices returned "sequence
    instructions executed computer" and "computer instructions written
    programming language", which are the same claim and are counted as a split.

    That is the correct verdict under exact matching and the verdict stands.
    What was wrong was leaving a reader to read "not corroborated" as "the
    models disagree", when the cause is that **definitions have no canonical
    form** — unlike ``au``, ``canberra`` or ``2``, where the panel corroborates
    cleanly. So the overlap is stated and nothing is bought with it.

    Returns:
        A note, or an empty string when no pair overlaps enough.
    """
    if len(positions) < 2:
        return ""
    best = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a = {w for w in positions[i].split() if w not in _FILLER}
            b = {w for w in positions[j].split() if w not in _FILLER}
            union = a | b
            if union:
                best = max(best, len(a & b) / len(union))
    if best < _OVERLAP_NOTE:
        return ""
    return (f"distinct positions share {best:.0%} of their content words - "
            "this may be one claim worded differently rather than a "
            "disagreement. NOT counted as agreement: judging that is the "
            "semantic-equivalence problem, and definitional questions have no "
            "canonical answer for exact matching to find")


def _no_answer_reason(answer: Answer) -> str:
    """Why a reply was unusable - truncation and abstention are not the same.

    A reasoning model that spent its whole budget thinking emitted nothing
    because it was cut off. Reporting that as "no usable answer" points the
    reader at the model when the fix is in the request.
    """
    text = (answer.text or "").strip()
    if text:
        return "declined to answer"
    details = (answer.usage or {}).get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    completion = int((answer.usage or {}).get("completion_tokens") or 0)
    if completion and reasoning >= completion * _RUNAWAY_FRACTION:
        # The whole budget went to thinking. Raising it just buys more
        # thinking - measured at 400/400 and again at 1200/1200 - so telling
        # the reader to raise max_tokens would send them down a dead end.
        return (f"empty reply - the model spent ALL {completion} completion "
                "tokens reasoning and emitted nothing; raising max_tokens will "
                "not help, pass reasoning_effort=\"none\"")
    if completion and reasoning >= completion * _STARVED_FRACTION:
        return (f"empty reply - {reasoning} of {completion} completion tokens "
                "went to reasoning; raise max_tokens")
    return "empty reply"


def _as_claim(question: str, position: str) -> str:
    """Pair a question with its answer so the stash can hold it.

    :data:`TERSE` and the stash pull in opposite directions, and the conflict
    was found by running them together rather than reasoned about: the panel
    agreed 3-of-3 that gold is ``au``, and **nothing reached the stash**.
    ``add_page`` requires 20-420 characters, on the sound grounds that a
    two-character fragment of a web page is not a claim — and a terse answer is
    exactly that fragment. Constraining replies to make them comparable had
    silently disabled the quarantine path.

    The answer alone cannot be stored, so the question is stored with it. What
    is deliberately *not* done is asking a model to phrase the fact as a
    sentence: that would put wording nothing corroborated into a stash entry
    carrying a corroborated claim's provenance. Both halves here are verbatim.
    """
    stem = question.strip().rstrip("?").strip()
    return f"{stem}: {position}".strip()


#: Number words, so "two", "Two" and "2" are one token rather than three
#: positions. Small on purpose: this covers the range short factual answers
#: actually use, and an open-ended numeric parser would be scope creep.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
    "fifty": "50", "hundred": "100", "thousand": "1000", "million": "1000000",
}

#: Words carrying no position, stripped so "the answer is X" and "X" agree.
_FILLER = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "it", "its", "of", "to",
    "and", "or", "in", "on", "at", "by", "for", "with", "that", "this",
    "there", "has", "have", "had", "answer", "about", "approximately",
    "roughly", "around", "exactly",
})


def _position(text: str, question: str = "") -> str:
    """Normalise a reply to a comparable position.

    Comparison is exact after normalisation, and normalisation is deliberately
    confined to **orthography** — lowercasing, punctuation, number spellings,
    filler words, and words the question itself supplied. It never judges that
    two *different* words mean the same thing. That is the hard problem, and a
    heuristic guessing at it would manufacture consensus, which is the one
    error this system must not make.

    The line matters and is worth stating exactly: rewriting ``two`` to ``2``
    is the same class of operation as lowercasing — two spellings of one token.
    Deciding that ``fast`` and ``efficient`` are one position would be a claim
    about meaning, and is refused.

    Question words are dropped because every model echoes the prompt, and an
    echo is not a position. Measured: asked how many moons Mars has, three
    models answered ``two moons``, ``two`` and ``2`` — unanimous, and reported
    as a three-way split until this normalisation existed.

    Args:
        text: The model's reply.
        question: The prompt, whose content words are treated as echo.

    Returns:
        A normalised position string, empty if nothing survived.
    """
    # Chat-template artifacts are not words: command-r closes every reply
    # with <|END_OF_TURN_TOKEN|>, which lowercase-and-despace turned into
    # the position suffix "end turn token" - the SS11.19 leak in a second
    # pipeline, and it broke exact-match corroboration ("earth round end
    # turn token" can never equal another voice's "earth round").
    text = re.sub(r"<\|[^|>]{1,64}\|>", " ", text)
    stripped = text.strip()
    # A numbered-list reply ("1. Paramiko is ...") must not have its list
    # marker read as the sentence: the first training run recorded a
    # model's entire position as "1".
    stripped = re.sub(r"^\s*\d{1,3}[.)]\s*", "", stripped)
    first = stripped.split(". ")[0].strip()
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " "
                      for c in first)
    echo = {word for word in
            "".join(c.lower() if c.isalnum() or c.isspace() else " "
                    for c in question).split()}
    words = []
    for word in cleaned.split():
        word = _NUMBER_WORDS.get(word, word)
        if word in _FILLER or word in echo:
            continue
        words.append(word)
    # Falling back to the unstripped form keeps an answer that was *entirely*
    # question words from collapsing to the empty string and colliding with
    # every other such answer.
    return " ".join(words)[:200] or " ".join(cleaned.split())[:200]
