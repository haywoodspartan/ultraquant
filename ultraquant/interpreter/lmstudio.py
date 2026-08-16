"""LM Studio: a local LLM as a *source*, never as the system's voice.

LM Studio (https://lmstudio.ai) serves local GGUF models over an
OpenAI-compatible HTTP API, by default on ``http://127.0.0.1:1234/v1``. That
makes it reachable with nothing but :mod:`urllib`, and it runs on the user's own
machine — no key, no account, no data leaving the box. It is an *optional* tier
in the same sense as :class:`~ultraquant.quantum.bluequbit.BlueQubitBackend` and
:class:`~ultraquant.quantum.backend.QPUBackend`: its absence is the normal case,
every other path still works, and nothing here is load bearing.

**What this integration deliberately is not.** It would be trivial to route the
interpreter's replies through a chat model and produce something that sounds
far better than this system does. That is precisely why it is refused. Every
claim in [§11](../ARCHITECTURE.md) — the grounded grammar in §11.10, the
composition in §11.6, the failed encoder in §11.5 — is a claim about *this*
system's own machinery. An LLM in the answer path would make all of it
unfalsifiable while making the demo look better, which is the worst possible
trade. :func:`explain` and friends do not exist here. The system speaks with the
grammar it induced.

**The two roles it does take.**

*Embeddings* are the substantive one. The library routes text by lexical
signature, which cannot see that "a box outline" and "a square" are the same
request. §11.5 failed because raw pixels were already close to an ideal
representation and compressing them could only lose; the lexical case is the
opposite — the raw representation is genuinely poor for meaning, which is the
condition §11.5 named as the one where a learned representation should pay.
:meth:`LMStudioClient.embed` supplies that vector, and
``ultraquant/experiments/embed_gate.py`` measures whether it actually helps
rather than assuming so.

**A model is not independent of the web, and the stash now knows it.** An
answer used to enter as an ordinary source, so a Wikipedia page and a local
model agreeing were counted as two sources and the claim promoted to
*corroborated* — when the model was very likely trained on that page.
Model-derived sources now collapse to one pseudo-source in
:func:`~ultraquant.interpreter.stash._independent_sources`: a model may add its
voice to a claim, but can never be the second source that makes one believed.

*Research* is the guarded one. A model's answer enters through
:meth:`~ultraquant.interpreter.stash.ContemporaryStash.add_page` with an
``lmstudio://model`` URL — the same quarantine as any scraped web page, subject
to the same corroboration before anything is believed. This matters more here
than for the web: a language model is a *fluent* source, and fluency is exactly
what the quarantine exists to discount. It is also a **single** source and
cannot corroborate itself; asking the same model twice, or two models sharing a
base, produces agreement that means nothing. :func:`quarantine_answer` records
provenance so that stays visible.

**Localhost by default.** ``base_url`` is checked against a loopback host unless
``allow_remote=True`` is passed explicitly. A remote endpoint would silently
ship every prompt — and every quarantined claim — off the machine, and that
should never happen by typo or by an inherited environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlparse

__all__ = [
    "LMStudioUnavailable",
    "LMStudioClient",
    "Answer",
    "quarantine_answer",
    "DEFAULT_BASE_URL",
]

#: Where LM Studio serves its OpenAI-compatible API out of the box.
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

#: The host every model-derived source URL carries, so the stash can recognise
#: one by netloc alone — which is all it records.
#:
#: ``.invalid`` is the reserved TLD for exactly this: it can never resolve, so
#: the marker cannot be mistaken for a real origin. The model id lives in the
#: path, keeping provenance readable while every model collapses to a single
#: source host.
MODEL_SOURCE_HOST = "lm-studio.invalid"

#: Hosts considered on-machine. Anything else needs ``allow_remote=True``.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"})


class LMStudioUnavailable(RuntimeError):
    """LM Studio is not reachable, or refused the request.

    Raised rather than returned so an optional tier cannot be mistaken for a
    working one. Callers that want graceful degradation use
    :meth:`LMStudioClient.available`, which never raises.
    """


@dataclass
class Answer:
    """One model response, with the provenance the quarantine needs.

    Attributes:
        text: The reply.
        model: Which model produced it, as the server reported it.
        prompt: The exact prompt sent, kept so a stored claim can be traced
            back to the question that produced it.
        usage: Token counts the server volunteered, if any.
    """

    text: str
    model: str
    prompt: str
    usage: dict = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        """A provenance URL for the stash.

        Not a fetchable address — a stable identifier that makes the origin of
        a quarantined claim unmistakable in the stash listing, and keeps two
        answers from the same model recognisable as one source.
        """
        return f"lmstudio://{MODEL_SOURCE_HOST}/{self.model}"


class LMStudioClient:
    """Talk to a local LM Studio server over its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
        allow_remote: bool = False,
    ) -> None:
        """
        Args:
            base_url: API root. Defaults to ``$LMSTUDIO_BASE_URL`` then
                :data:`DEFAULT_BASE_URL`.
            timeout: Per-request timeout in seconds. Local models are slow —
                a 30B model took 17 s for a one-sentence answer on the machine
                this was written on — so the default is generous.
            allow_remote: Permit a non-loopback host. Off by default so
                prompts cannot leave the machine unintentionally.

        Raises:
            ValueError: If ``base_url`` is not loopback and ``allow_remote`` is
                False.
        """
        url = base_url or os.environ.get("LMSTUDIO_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = url.rstrip("/")
        self.timeout = float(timeout)
        host = urlparse(self.base_url).hostname or ""
        if not allow_remote and host not in _LOOPBACK:
            raise ValueError(
                f"refusing a non-local LM Studio endpoint ({host!r}): prompts and "
                "quarantined claims would leave this machine. Pass "
                "allow_remote=True if that is genuinely intended."
            )
        self.is_remote = host not in _LOOPBACK

    # ---- transport ------------------------------------------------------

    def _request(self, path: str, payload: dict | None = None) -> dict:
        """One HTTP round trip, with every failure mode named.

        Raises:
            LMStudioUnavailable: For a refused connection, a timeout, an HTTP
                error, or a body that is not JSON. The message says which,
                because "LM Studio is not running" and "the model is not
                loaded" need different fixes from the user.
        """
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:                      # pragma: no cover - best effort
                pass
            raise LMStudioUnavailable(
                f"LM Studio returned HTTP {exc.code} for {path}"
                + (f": {detail}" if detail else "")
            ) from exc
        except urllib.error.URLError as exc:
            raise LMStudioUnavailable(
                f"cannot reach LM Studio at {self.base_url} ({exc.reason}). "
                "Is the server started? LM Studio: Developer tab -> Start Server."
            ) from exc
        except OSError as exc:                     # socket timeouts land here
            raise LMStudioUnavailable(
                f"LM Studio timed out after {self.timeout:.0f}s: {exc}"
            ) from exc
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise LMStudioUnavailable(
                f"LM Studio sent a non-JSON reply to {path}"
            ) from exc

    # ---- capability -----------------------------------------------------

    def available(self) -> bool:
        """Whether the server answers, without raising. Cheap; safe to poll."""
        try:
            self.models()
        except LMStudioUnavailable:
            return False
        return True

    def models(self) -> list[str]:
        """Model ids the server currently exposes.

        Returns:
            Ids in the order reported. LM Studio lists everything downloaded,
            not only what is loaded, so a model appearing here may still take
            a load pause on first use.
        """
        payload = self._request("/models")
        # Valid JSON of the wrong shape is not the same as an unreachable
        # server, and it used to be worse: something else listening on 1234 (a
        # stale dev server, a proxy) answering with an array made available() -
        # documented "never raises" - raise AttributeError instead of
        # returning False. A present-but-null "data" did the same via
        # .get(k, default) returning None rather than the default.
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise LMStudioUnavailable(
                f"{self.base_url}/models did not return a model list; "
                "is something other than LM Studio listening on this port?")
        return [str(entry.get("id", "")) for entry in rows
                if isinstance(entry, dict) and entry.get("id")]

    def embedding_models(self) -> list[str]:
        """Models whose id marks them as embedding models.

        The API does not distinguish them, so this is a name heuristic and is
        allowed to be wrong; :meth:`embed` names the model explicitly in real
        use. Kept for the convenience of picking a default.
        """
        return [name for name in self.models() if "embed" in name.lower()]

    def _default_model(self, embedding: bool = False) -> str:
        candidates = self.embedding_models() if embedding else [
            name for name in self.models() if "embed" not in name.lower()
        ]
        if not candidates:
            kind = "embedding" if embedding else "chat"
            raise LMStudioUnavailable(f"LM Studio exposes no {kind} model")
        return candidates[0]

    # ---- generation -----------------------------------------------------

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Answer:
        """Ask a chat model one question.

        Args:
            prompt: The question.
            model: Model id; the first non-embedding model if omitted.
            system: Optional system message.
            max_tokens: Reply ceiling.
            temperature: Defaults to 0. Deterministic output is the right
                default for a source whose claims get quarantined and compared
                — sampling noise would make two runs disagree with themselves
                and look like a contradiction between sources.
            reasoning_effort: Passed through when set. ``"none"`` is the only
                thing measured to stop a runaway reasoning model here: asked
                for six phrasings, ``qwen3.6-35b-a3b`` spent **400 of 400** and
                then **1200 of 1200** completion tokens thinking and emitted
                nothing, at every budget. ``"none"`` returned 42 tokens with
                zero reasoning and the answer. ``"low"``, a ``/no_think``
                suffix and ``chat_template_kwargs={"enable_thinking": False}``
                were each tried and had **no effect at all**.

        Returns:
            The :class:`Answer`, carrying provenance.
        """
        chosen = model or self._default_model()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": chosen,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if reasoning_effort:
            body["reasoning_effort"] = str(reasoning_effort)
        payload = self._request("/chat/completions", body)
        choices = payload.get("choices") or []
        if not choices:
            raise LMStudioUnavailable("LM Studio returned no choices")
        text = str(choices[0].get("message", {}).get("content", "")).strip()
        return Answer(text=text, model=str(payload.get("model", chosen)),
                      prompt=prompt, usage=dict(payload.get("usage") or {}))

    def embed(
        self,
        texts: Sequence[str] | str,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed one or more strings.

        Args:
            texts: A string or a sequence of them. Batched in one request,
                which is much faster than one call each.
            model: Embedding model id; the first id containing "embed" if
                omitted.

        Returns:
            One vector per input, in input order. The server is trusted for
            order — it returns an ``index`` field, and this sorts by it rather
            than assuming, because a reordered batch would silently mislabel
            every vector.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        chosen = model or self._default_model(embedding=True)
        payload = self._request("/embeddings", {"model": chosen, "input": items})
        rows = payload.get("data") or []
        if len(rows) != len(items):
            raise LMStudioUnavailable(
                f"asked for {len(items)} embeddings, got {len(rows)}"
            )
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        return [[float(x) for x in row.get("embedding", [])] for row in ordered]

    def describe(self) -> dict:
        """A summary safe to print. Never includes prompts or replies."""
        try:
            models = self.models()
            reachable = True
        except LMStudioUnavailable:
            models, reachable = [], False
        return {
            "base_url": self.base_url,
            "reachable": reachable,
            "remote": self.is_remote,
            "models": models,
            "embedding_models": [m for m in models if "embed" in m.lower()],
        }


def quarantine_answer(
    answer: Answer,
    stash: Any,
    max_claims: int = 6,
) -> list[int]:
    """Put a model's answer into the contemporary stash, unbelieved.

    A language model is a fluent source, and fluency is what the quarantine
    exists to discount — a confident wrong sentence and a confident right one
    are indistinguishable at the point of storage. So the answer is filed like
    any scraped page: split into claims, classified, and left pending until
    something *independent* corroborates it.

    The single-source limit is the important one and cannot be engineered
    around here: asking the same model twice, or asking two models that share a
    base, yields agreement carrying no evidence. Because
    :attr:`Answer.source_url` is keyed on the model id, repeated answers from
    one model collapse to one source in the stash's own accounting rather than
    accumulating fake support.

    Args:
        answer: What the model said.
        stash: A :class:`~ultraquant.interpreter.stash.ContemporaryStash`.
        max_claims: Ceiling on claims extracted from the reply.

    Returns:
        The ids of the quarantined entries.
    """
    title = f"LM Studio answer: {answer.prompt[:60]}"
    return stash.add_page(answer.source_url, title, answer.text,
                          max_claims=max_claims)
