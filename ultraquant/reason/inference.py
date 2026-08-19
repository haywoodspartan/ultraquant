"""Inference by spreading activation: answers the store holds implicitly.

**The architectural claim this module carries.** A transformer infers by one
dense forward pass: every parameter participates, the "reasoning" is
implicit in attention weights, and nothing about the answer says which
memories produced it. This is the other architecture — the brain-shaped
one, and the difference is structural, not cosmetic:

* **Activation, not attention.** A question injects activation at its
  concept tokens. Activation spreads along stored associations — a fact's
  key tokens link to its value tokens, and the value tokens are other
  facts' keys. Nothing participates except what the spread reaches: the
  library stays on disk and only touched facts page in, exactly the
  sparse-recall economics the shard architecture exists for.
* **Convergence, not completion.** An answer forms where activations from
  *different parts of the question* meet. "what is the tower melting
  point?" spreads from {tower} and from {melting, point}; the fact
  ``steel melting point`` is reached by both — through ``tower material →
  steel`` on one side and directly on the other — and that convergence IS
  the answer. A fact reached from only part of the question is a pun, not
  a proof, and stays silent.
* **The trail is the answer.** Every inference names the facts it crossed.
  Confidence is the **minimum** along the path — spreading dilutes belief,
  never concentrates it — and the reply is marked inferred, never
  presented as stored.
* **Bounded depth.** Activation decays; two bridge hops is the measured
  limit of honesty for 2-4 token keys. Longer chains multiply pun risk
  faster than they add reach.

What this deliberately does NOT do: free association. The fabrication
family this codebase has been burned by three times (§11.14's stopword
answer, the tungsten question, the keyword fallback asserting the wrong
metal) is exactly an activation spread without the convergence rule.
Every guard here exists because its absence already produced a wrong
answer somewhere in the book.

Arithmetic combination (sum, difference, larger/smaller over two numeric
facts) sits beside the spread as a separate operation: convergence finds
WHICH facts the question relates; it cannot add. Mismatched units refuse —
300 meters + 2 kilometers is not 302 anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ultraquant.shards.router import _informative, normalize_token

__all__ = ["Inference", "infer"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words that ask for a numeric combination, mapped to the relation.
_COMBINE_WORDS = {
    "sum": "sum", "total": "sum", "together": "sum", "combined": "sum",
    "difference": "difference", "gap": "difference",
    "taller": "larger", "longer": "larger", "bigger": "larger",
    "larger": "larger", "higher": "larger", "heavier": "larger",
    "shorter": "smaller", "smaller": "smaller", "lower": "smaller",
    "lighter": "smaller",
}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: Activation lost per bridge hop. Direct token contact is 1.0; one hop
#: through a fact's value arrives at 0.5; a second hop at 0.25. Below the
#: floor, activation is noise and stops spreading.
_DECAY = 0.5
_FLOOR = 0.2
_MAX_HOPS = 2


@dataclass
class Inference:
    """One derived answer and the evidence trail that produced it.

    Attributes:
        answer: The plain-language answer text.
        premises: ``(key, value)`` pairs the activation crossed, in order.
        confidence: min over the premises' confidences.
        kind: ``"chain"`` or ``"combine"``.
    """

    answer: str
    premises: list = field(default_factory=list)
    confidence: float = 0.0
    kind: str = ""
    #: ``(key, value)`` a chain's conclusion would be stored under if it
    #: earns consolidation. Combines carry None: arithmetic re-derives.
    conclusion: tuple | None = None

    def describe(self) -> str:
        """The answer with its premises named - inferred, never asserted."""
        trail = "; ".join(f"{k} is {v}" for k, v in self.premises)
        return (f"{self.answer} - inferred, not stored: {trail} "
                f"(confidence {self.confidence:.2f})")


def _fold(text: str) -> set[str]:
    return {normalize_token(tok) for tok in _TOKEN_RE.findall(str(text).lower())
            if _informative(tok)}


def _reachable_facts(memory, probes: list[str], top_k: int = 8) -> dict:
    """Candidate facts touched by any probe text, key -> record.

    This is the paging boundary: only facts the index surfaces for a probe
    are ever considered, so the spread's working set stays a handful of
    records however large the library is.
    """
    found: dict = {}
    for probe in probes:
        for key in memory.find_facts(probe, top_k=top_k):
            if key in found:
                continue
            record = memory.recall_fact(key)
            if record is not None:
                found[key] = record
    return found


@dataclass
class _Node:
    """One activated fact during a spread."""

    key: str
    record: dict
    #: question token -> activation strength that reached this fact for it.
    sources: dict = field(default_factory=dict)
    #: Facts crossed to get here, nearest last (empty for direct contact).
    path: list = field(default_factory=list)


def _spread(question_tokens: set[str], memory) -> Inference | None:
    """Spreading activation with a convergence requirement.

    Round 0 activates every fact sharing a token with the question, each
    source token at strength 1.0. Each later round lets an activated
    fact's VALUE tokens carry its accumulated sources (decayed) into facts
    they name. A fact whose sources come to cover the WHOLE question - some
    directly, some over bridges - is a convergence, and the best one
    answers.
    """
    probes = [" ".join(sorted(question_tokens))]
    probes += sorted(question_tokens)
    facts = _reachable_facts(memory, probes)
    if not facts:
        return None

    nodes: dict[str, _Node] = {}
    for key, record in facts.items():
        key_tokens = _fold(key)
        contact = key_tokens & question_tokens
        if not contact:
            continue
        node = _Node(key=key, record=record)
        for token in contact:
            node.sources[token] = 1.0
        nodes[key] = node

    for _hop in range(_MAX_HOPS):
        grown = False
        for node in list(nodes.values()):
            strength = max(node.sources.values()) * _DECAY
            if strength < _FLOOR:
                continue
            value = str(node.record.get("value", ""))
            bridge_tokens = _fold(value)
            if not bridge_tokens:
                continue
            targets = _reachable_facts(memory, [value])
            for t_key, t_record in targets.items():
                if t_key == node.key:
                    continue
                if t_key in {p_key for p_key, _v in node.path} :
                    continue
                t_tokens = _fold(t_key)
                if not (t_tokens & bridge_tokens):
                    continue
                target = nodes.get(t_key)
                if target is None:
                    target = _Node(key=t_key, record=t_record)
                    direct = t_tokens & question_tokens
                    for token in direct:
                        target.sources[token] = 1.0
                    nodes[t_key] = target
                carried = node.path + [(node.key,
                                        node.record.get("value", ""))]
                for token, s in node.sources.items():
                    arriving = s * _DECAY
                    if arriving >= _FLOOR and \
                            arriving > target.sources.get(token, 0.0):
                        target.sources[token] = arriving
                        target.path = carried
                        grown = True
        if not grown:
            break

    # Convergence: sources must cover the whole question, and at least one
    # source must have arrived over a bridge (all-direct coverage is plain
    # recall, which already had its turn upstream).
    best: _Node | None = None
    for node in nodes.values():
        if set(node.sources) != question_tokens:
            continue
        if not node.path:
            continue
        if best is None:
            best = node
            continue
        if (min(node.sources.values()), -len(node.path)) > \
                (min(best.sources.values()), -len(best.path)):
            best = node
    if best is None:
        return None

    premises = best.path + [(best.key, best.record.get("value", ""))]
    seen_keys = [p_key for p_key, _v in premises]
    confidences = [float(facts[p_key].get("confidence", 0.0))
                   if p_key in facts else
                   float(memory.recall_fact(p_key).get("confidence", 0.0))
                   for p_key in seen_keys]
    subject_key = premises[0][0]
    subject = " ".join(tok for tok in _TOKEN_RE.findall(subject_key.lower())
                       if normalize_token(tok) in question_tokens)
    asked = " ".join(sorted(question_tokens - _fold(subject_key)))
    via = ", ".join(str(v) for _k, v in premises[:-1])
    return Inference(
        answer=(f"the {subject} {asked} is "
                f"{best.record.get('value', '')}, via {via}"),
        premises=premises,
        confidence=min(confidences) if confidences else 0.0,
        kind="chain",
        conclusion=(f"{subject} {asked}", str(best.record.get("value", ""))),
    )


def _numeric(value: str) -> float | None:
    match = _NUMBER_RE.search(str(value))
    return float(match.group()) if match else None


def _unit(value: str) -> str:
    """The unit word of a stored value, singular-folded ('meters'->'meter')."""
    tokens = re.findall(r"[a-z]+", str(value).lower())
    return normalize_token(tokens[-1]) if tokens else ""


def _combine(question_tokens: set[str], text: str,
             memory) -> Inference | None:
    """An arithmetic relation between exactly two held numeric facts."""
    relations = {_COMBINE_WORDS[tok] for tok in
                 _TOKEN_RE.findall(text.lower()) if tok in _COMBINE_WORDS}
    if len(relations) != 1:
        return None
    relation = relations.pop()

    content = question_tokens - {normalize_token(w) for w in _COMBINE_WORDS}
    facts = _reachable_facts(memory, [" ".join(sorted(content))])
    matched = []
    for key, record in facts.items():
        key_tokens = _fold(key)
        # Participation requires the key be FULLY asked about: "the sum of
        # the tower height and the bridge height" names {tower, bridge,
        # height}, and 'spire height' shares the attribute token without
        # being asked - four sharers used to make the whole question
        # unanswerable, and any laxer rule would add an unasked number.
        if not key_tokens or not key_tokens <= content:
            continue
        number = _numeric(record.get("value", ""))
        if number is None:
            continue
        matched.append((key, record, number))
    if len(matched) != 2:
        return None
    (ka, ra, na), (kb, rb, nb) = matched
    # 300 meters + 2 kilometers is not 302 anything: a combination across
    # different units is wrong, not approximate. Unit conversion is a
    # mechanism this module does not have, so mismatched units refuse.
    unit_a, unit_b = _unit(ra.get("value", "")), _unit(rb.get("value", ""))
    if unit_a and unit_b and unit_a != unit_b:
        return None
    confidence = min(float(ra.get("confidence", 0.0)),
                     float(rb.get("confidence", 0.0)))
    premises = [(ka, ra.get("value", "")), (kb, rb.get("value", ""))]
    if relation == "sum":
        answer = f"{ka} + {kb} = {na + nb:g}"
    elif relation == "difference":
        answer = f"the difference between {ka} and {kb} is {abs(na - nb):g}"
    else:
        bigger = ka if na >= nb else kb
        smaller = kb if na >= nb else ka
        chosen = bigger if relation == "larger" else smaller
        answer = (f"{chosen} "
                  f"({max(na, nb) if chosen == bigger else min(na, nb):g}) "
                  f"is the {'larger' if relation == 'larger' else 'smaller'} "
                  f"of the two")
    return Inference(answer=answer, premises=premises,
                     confidence=confidence, kind="combine")


def missing_premise(text: str, memory) -> dict | None:
    """The fact that would let a refused question converge, or None.

    The metacognitive half of the spread: when convergence fails, the
    partial activations say exactly what was missing. Asked "what is the
    tower conductivity?" while holding only ``tower material -> steel``,
    the tower half activates and the conductivity half reaches nothing —
    so the question worth ASKING is "what is the steel conductivity?":
    the held bridge's value plus the uncovered remainder.

    Curiosity only flows FORWARD along a held bridge, and that constraint
    is the spam guard: the bridging value must name a thing (non-numeric,
    at most two informative tokens). "What is the obelisk melting point?"
    touches ``steel melting point`` through its attribute tokens, but that
    fact's value is "1370 degrees" — a quantity, not a subject — so no
    question is generated. A system that asks about everything it fails
    to answer is §11.16's noise wearing a curious face; a system that
    asks only what a held fact points at is doing what a reader does.

    Args:
        text: The question that failed to converge.
        memory: The session's :class:`SystematicMemory`.

    Returns:
        ``{"premise_key", "via_key", "via_value", "original"}`` for the
        best grounded gap, or None when nothing held points anywhere.
    """
    question_tokens = _fold(text)
    if len(question_tokens) < 2:
        return None
    probes = [" ".join(sorted(question_tokens))] + sorted(question_tokens)
    facts = _reachable_facts(memory, probes)
    best: tuple | None = None
    for key, record in facts.items():
        key_tokens = _fold(key)
        covered = key_tokens & question_tokens
        remainder = question_tokens - key_tokens
        if not covered or not remainder:
            continue
        value = str(record.get("value", ""))
        if _numeric(value) is not None:
            continue
        value_tokens = _fold(value)
        if not value_tokens or len(value_tokens) > 2:
            continue
        candidate = (len(covered), -len(value_tokens), key)
        if best is None or candidate > best[0]:
            ordered_remainder = [tok for tok in
                                 _TOKEN_RE.findall(text.lower())
                                 if normalize_token(tok) in remainder]
            premise_key = " ".join([value.strip().lower()]
                                   + ordered_remainder)
            best = (candidate, {
                "premise_key": premise_key,
                "via_key": key,
                "via_value": value,
                "original": text,
            })
    return best[1] if best is not None else None


def infer(text: str, memory) -> Inference | None:
    """Derive an answer from stored facts, or None.

    Args:
        text: The question as asked.
        memory: The session's :class:`SystematicMemory`.

    Returns:
        An :class:`Inference` with premises and diluted confidence, or None
        when no honest convergence over held facts exists.
    """
    question_tokens = _fold(text)
    if len(question_tokens) < 2:
        return None
    combined = _combine(question_tokens, text, memory)
    if combined is not None:
        return combined
    return _spread(question_tokens, memory)
