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
* **Bounded depth.** Activation decays over a floor, and the hop cap
  moves only by measurement: two bridges originally, three after
  §11.52, four after §11.61 — each move gated on zero decoy
  assertions at the new depth, because the pun resistance lives in
  the structural rules, not the decay floor.

What this deliberately does NOT do: free association. The fabrication
family this codebase has been burned by three times (§11.14's stopword
answer, the tungsten question, the keyword fallback asserting the wrong
metal) is exactly an activation spread without the convergence rule.
Every guard here exists because its absence already produced a wrong
answer somewhere in the book.

Arithmetic combination (sum, difference, larger/smaller over two numeric
facts) sits beside the spread as a separate operation: convergence finds
WHICH facts the question relates; it cannot add. Units the definition
table connects convert (300 meters + 2 kilometers is 2.3 kilometers,
never 302); units it cannot connect refuse, exactly as they always did.
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
#: through a fact's value arrives at 0.5, a second at 0.25, a third at
#: 0.125, a fourth at 0.0625. Below the floor, activation is noise and
#: stops spreading — a fifth bridge would arrive at 0.03125 and is
#: structurally refused. §11.31 capped this at two hops with a pun
#: warning; §11.52 measured the warning no longer binding at three
#: bridges, and §11.61 at four (zero decoy assertions both times,
#: shallower chains intact both times): the pun resistance lives in
#: the structural rules — single-origin, origin coverage,
#: order-as-evidence, whole-value bridging, synchronous frontiers —
#: not in the decay floor. Each next limit stays a measurement away.
_DECAY = 0.5
_FLOOR = 0.05
_MAX_HOPS = 4

#: §11.49's door: a negated fact may be reached as a chain TERMINAL and
#: answer "believed not X". False restores §11.48's phase one (denials
#: fully inert) - the negchain gate's baseline arm.
_NEGATED_TERMINALS = True

#: §11.59's clock: relay only the frontier (nodes whose origins changed
#: last round - equal-strength re-arrivals are discarded by the strict
#: comparison anyway) and memoise bridge-probe retrieval within one
#: spread (the probes depend only on the value and the fixed question).
#: Both are pure call-elimination: the convergence set is bit-identical,
#: and the clock gate holds parity absolute before reading the clock.
#: False restores the every-node-every-round loop - the gate's arm.
_FRONTIER_SPREAD = True


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
    #: True when the conclusion is belief-of-absence ("believed not X"):
    #: the terminal premise was a negation, reached but never bridged.
    negated: bool = False

    def describe(self) -> str:
        """The answer with its premises named - inferred, never asserted."""
        trail = "; ".join(f"{k} is {v}" for k, v in self.premises)
        return (f"{self.answer} - inferred, not stored: {trail} "
                f"(confidence {self.confidence:.2f})")


def _fold(text: str) -> set[str]:
    return {normalize_token(tok) for tok in _TOKEN_RE.findall(str(text).lower())
            if _informative(tok)}


def _probe_phrases(text_tokens: list[str]) -> list[str]:
    """Consecutive bigrams and trigrams, the question's phrase chunks.

    Single-token probes drown at density: "little" matches hundreds of
    keys and top-k sampling misses the one that matters. The subject of a
    question is a PHRASE, and probing "little wall arch" ranks that
    entity's facts above every single-word sharer - activation injected
    at chunks, the way subjects are actually named.
    """
    phrases = []
    for size in (3, 2):
        for start in range(len(text_tokens) - size + 1):
            phrases.append(" ".join(text_tokens[start:start + size]))
    return phrases


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
    """One activated fact during a spread.

    ``sources`` is DIRECT contact only. Bridged evidence lives in
    ``origins``: origin fact key -> {"sources": ..., "path": ...}, one
    entry per lineage, never pooled - the single-origin rule that keeps
    a crowded library from assembling puns out of unrelated subjects.
    """

    key: str
    record: dict
    #: question token -> DIRECT activation (the fact's own key contact).
    sources: dict = field(default_factory=dict)
    #: origin key -> bridged contribution, kept separate per lineage.
    origins: dict = field(default_factory=dict)


def _ordered_fold(text: str) -> list[str]:
    """The question's informative tokens, folded, in question order."""
    out, seen = [], set()
    for tok in _TOKEN_RE.findall(str(text).lower()):
        folded = normalize_token(tok)
        if folded in seen or not _informative(tok):
            continue
        seen.add(folded)
        out.append(folded)
    return out


def _spread(question_tokens: set[str], memory,
            ordered: list[str] | None = None) -> Inference | None:
    """Spreading activation with a convergence-and-consistency requirement.

    Round 0 activates every fact sharing a token with the question, each
    source token at strength 1.0. Each later round lets an activated
    fact's VALUE tokens carry its sources (decayed) into facts they name.
    A fact whose DIRECT contact plus the contribution of ONE bridge origin
    covers the whole question is a convergence, and the best one answers.

    The single-origin rule is the density lesson: in a crowded library,
    "old tower material is steel" and "north keep material is steel" both
    bridge into "steel hardness", and pooling their activations completes
    the coverage for "the north tower hardness" out of two UNRELATED
    subjects - a pun assembled from fragments, measured at 0.500
    fabrication the first time token-colliding worlds were tried. Bridged
    evidence must trace to one origin fact; direct contact (the attribute
    side) may come from the target itself.
    """
    if ordered is None:
        ordered = sorted(question_tokens)
    probes = [" ".join(sorted(question_tokens))]
    probes += _probe_phrases(ordered)
    probes += sorted(question_tokens)
    facts = _reachable_facts(memory, probes)
    if not facts:
        return None

    nodes: dict[str, _Node] = {}
    for key, record in facts.items():
        # A negation is inhibitory knowledge: "the dome material is not
        # steel" says what may NOT be believed, and letting it excite the
        # network bridged a denial into an assertion - the dome's melting
        # point derived "via not steel", the value's polarity invisible
        # to the fold. Negated facts neither seed nor relay.
        if record.get("negated"):
            continue
        key_tokens = _fold(key)
        contact = key_tokens & question_tokens
        if not contact:
            continue
        node = _Node(key=key, record=record)
        for token in contact:
            node.sources[token] = 1.0
        nodes[key] = node

    frontier: set[str] | None = None
    probe_memo: dict[str, dict] = {}
    for _hop in range(_MAX_HOPS):
        grown = False
        # Synchronous rounds: updates collect during the pass and apply
        # after it. The first form mutated origins in place, and a lucky
        # dict order cascaded origin -> middle -> target inside ONE
        # round - the depth gate measured a "one-hop" arm answering
        # three-fact chains at 0.667, meaning hop depth depended on
        # iteration order. Cognition must not depend on dict order;
        # a round is a frontier, not a free-for-all.
        pending: list[tuple[str, dict, str, dict, list]] = []
        changed: set[str] = set()
        for node in list(nodes.values()):
            # §11.59: after round one, only nodes that gained or
            # strengthened an origin last round have anything NEW to
            # relay - a node's own direct sources never change after
            # seeding, and re-relaying them yields equal-strength
            # arrivals the strict comparison below discards unread.
            if (_FRONTIER_SPREAD and frontier is not None
                    and node.key not in frontier):
                continue
            # §11.49: a negated fact may be reached and ANSWER ("believed
            # not temperate", via the chain that reached it) but its value
            # names what is NOT believed - it never relays. The denial
            # stays inert as a bridge, exactly as §11.48 measured.
            if node.record.get("negated"):
                continue
            # A node relays each origin lineage separately. Its own direct
            # contact starts a lineage rooted at itself; every inherited
            # lineage keeps its root.
            lineages = []
            if node.sources:
                lineages.append((node.key, dict(node.sources), []))
            for origin, entry in node.origins.items():
                merged = dict(entry["sources"])
                for token, s in node.sources.items():
                    merged[token] = max(merged.get(token, 0.0), s)
                lineages.append((origin, merged, entry["path"]))
            value = str(node.record.get("value", ""))
            bridge_tokens = _fold(value)
            if not bridge_tokens:
                continue
            # Addressed bridge probes: "iron" alone is also a PREFIX word
            # in a dense store, and its ranked buckets drown the three
            # "iron <property>" keys. Pairing the bridge value with each
            # question token addresses the target's own bucket directly.
            bridge_probes = [value] + [f"{value} {tok}"
                                       for tok in sorted(question_tokens)]
            if _FRONTIER_SPREAD:
                targets = probe_memo.get(value)
                if targets is None:
                    targets = _reachable_facts(memory, bridge_probes)
                    probe_memo[value] = targets
            else:
                targets = _reachable_facts(memory, bridge_probes)
            for t_key, t_record in targets.items():
                if t_key == node.key:
                    continue
                if t_record.get("negated") and not _NEGATED_TERMINALS:
                    continue
                t_tokens = _fold(t_key)
                # Whole-value bridging, both directions. A bridge that
                # carries on ANY shared token lets "wren the younger"
                # hop through plain wren's birthplace - the depth pun,
                # asserted at 100% of epithet decoys the first time it
                # was measured. And the mirror: value "wren" must not
                # flow into "wren the younger birthplace", whose extra
                # qualifier the question never asked for. Every token
                # of a chain fact is accounted for - by the incoming
                # value, by the question, or by ONE terminal relation
                # slot ("bronzite BASE" traverses; the slot trails the
                # subject the way "material" trails "tower" in origin
                # coverage) - or the hop refuses. An unaccounted token
                # INSIDE the subject is a qualifier being dropped, and
                # dropping qualifiers is how one entity substitutes
                # for another.
                if not bridge_tokens <= t_tokens:
                    continue
                extra = t_tokens - bridge_tokens - question_tokens
                if extra:
                    key_order = _ordered_fold(t_key)
                    if (len(extra) > 1 or not key_order
                            or key_order[-1] not in extra):
                        continue
                for origin, sources, path in lineages:
                    if any(p_key == t_key for p_key, _v in path):
                        continue
                    carried_path = path + [(node.key,
                                            node.record.get("value", ""))]
                    arriving = {}
                    for token, s in sources.items():
                        decayed = s * _DECAY
                        if decayed >= _FLOOR:
                            arriving[token] = decayed
                    if not arriving:
                        continue
                    pending.append((t_key, t_record, origin, arriving,
                                    carried_path))
        for t_key, t_record, origin, arriving, carried_path in pending:
            target = nodes.get(t_key)
            if target is None:
                target = _Node(key=t_key, record=t_record)
                for token in _fold(t_key) & question_tokens:
                    target.sources[token] = 1.0
                nodes[t_key] = target
            entry = target.origins.get(origin)
            if entry is None or                     sum(arriving.values()) > sum(
                        entry["sources"].values()):
                target.origins[origin] = {"sources": arriving,
                                          "path": carried_path}
                changed.add(t_key)
                grown = True
        frontier = changed
        if not grown:
            break

    # Convergence: the target's DIRECT contact plus ONE origin's bridged
    # contribution must cover the whole question, and the origin's path
    # must be non-empty (all-direct coverage is plain recall's turn).
    # Order is EVIDENCE at density: "river wall barn" and "river barn
    # wall" are different entities with identical token sets, and the
    # first 10,000-fact world contained both. Shared adjacent bigrams
    # with the question rank the right origin above its anagram; a tie
    # even there is genuine ambiguity and refuses below.
    question_bigrams = {f"{a} {b}" for a, b in zip(ordered, ordered[1:])}

    def _order_overlap(key: str) -> int:
        key_tokens = [normalize_token(tok)
                      for tok in _TOKEN_RE.findall(key.lower())]
        key_bigrams = {f"{a} {b}" for a, b in
                       zip(key_tokens, key_tokens[1:])}
        return len(key_bigrams & question_bigrams)

    best = None
    contenders = []
    for node in nodes.values():
        for origin, entry in node.origins.items():
            if not entry["path"]:
                continue
            combined = dict(entry["sources"])
            for token, s in node.sources.items():
                combined[token] = max(combined.get(token, 0.0), s)
            if set(combined) != question_tokens:
                continue
            # Origin coverage: the first fact of the chain may have at
            # most ONE token the question did not say - the relation slot
            # ("material" in "tower material"). Two absent tokens means
            # the origin names a MORE SPECIFIC subject than asked: "low
            # mill material" answering for "the royal mill" is the
            # density fabrication where a dropped or missing qualifier
            # silently substitutes one entity for another.
            origin_key = entry["path"][0][0]
            if len(_fold(origin_key) - question_tokens) > 1:
                continue
            origin_key = entry["path"][0][0]
            order_score = _order_overlap(origin_key)
            # The absolute form of the order rule: a multi-word subject
            # must share at least one adjacent bigram with the question.
            # "great BARN SPIRE material" answering for a ghost "great
            # SPIRE BARN" was the density fabrication that survived the
            # comparative tie-break, because an anagram with no
            # competitor wins unopposed. Two-token origins are exempt -
            # a single-word subject has no order to check.
            if len(_TOKEN_RE.findall(origin_key.lower())) >= 3                     and order_score < 1:
                continue
            candidate = (order_score,
                          min(combined.values()), -len(entry["path"]),
                          node, entry)
            contenders.append(candidate)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None
    # Anagram ambiguity: two origins with DIFFERENT subjects but the same
    # rank cannot be told apart, and guessing between them is the pun the
    # whole rule stack exists to refuse.
    top = [c for c in contenders if c[:3] == best[:3]]
    if len({c[4]["path"][0][0] for c in top}) > 1:
        return None
    _order, _strength, _neg, node, entry = best

    premises = entry["path"] + [(node.key, node.record.get("value", ""))]
    seen_keys = [p_key for p_key, _v in premises]
    confidences = [float(facts[p_key].get("confidence", 0.0))
                   if p_key in facts else
                   float((memory.recall_fact(p_key) or {}).get(
                       "confidence", 0.0))
                   for p_key in seen_keys]
    subject_key = premises[0][0]
    subject = " ".join(tok for tok in _TOKEN_RE.findall(subject_key.lower())
                       if normalize_token(tok) in question_tokens)
    asked = " ".join(sorted(question_tokens - _fold(subject_key)))
    via = ", ".join(str(v) for _k, v in premises[:-1])
    negated = bool(node.record.get("negated"))
    value = str(node.record.get("value", ""))
    if negated:
        # The terminal is a denial: the chain earned "believed not X",
        # and the premise line reads the polarity too.
        premises[-1] = (node.key, f"not {value}")
    spoken = f"believed not {value}" if negated else value
    return Inference(
        answer=f"the {subject} {asked} is {spoken}, via {via}",
        premises=premises,
        confidence=min(confidences) if confidences else 0.0,
        kind="chain",
        conclusion=(f"{subject} {asked}", value),
        negated=negated,
    )

def _numeric(value: str) -> float | None:
    match = _NUMBER_RE.search(str(value))
    return float(match.group()) if match else None


def _unit(value: str) -> str:
    """The unit word of a stored value, singular-folded ('meters'->'meter')."""
    tokens = re.findall(r"[a-z]+", str(value).lower())
    return normalize_token(tokens[-1]) if tokens else ""


#: Unit families with exact factors to a family base. Small on purpose:
#: every factor here is a definition, not an approximation, and a unit
#: outside the table stays outside the arithmetic - refusing "florins"
#: is correct, converting them would be invention.
_UNIT_FAMILIES: dict[str, dict[str, float]] = {
    "length": {"millimeter": 0.001, "centimeter": 0.01, "meter": 1.0,
               "kilometer": 1000.0, "foot": 0.3048, "mile": 1609.344},
    "mass": {"milligram": 1e-6, "gram": 0.001, "kilogram": 1.0,
             "tonne": 1000.0},
    "time": {"second": 1.0, "minute": 60.0, "hour": 3600.0,
             "day": 86400.0},
}

_UNIT_TO_FAMILY = {unit: family
                   for family, units in _UNIT_FAMILIES.items()
                   for unit in units}


def _convert(number: float, unit: str, target_unit: str) -> float | None:
    """``number unit`` expressed in ``target_unit``, or None.

    Only within one family, only through the exact table. 300 meters
    into kilometers is 0.3; 300 meters into degrees is a refusal, and
    always was (§11.30's guard) - conversion narrows that refusal to
    the pairs a definition connects.
    """
    family = _UNIT_TO_FAMILY.get(unit)
    if family is None or _UNIT_TO_FAMILY.get(target_unit) != family:
        return None
    factors = _UNIT_FAMILIES[family]
    return number * factors[unit] / factors[target_unit]


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
        # "the bridge height is not 120 meters" holds no number to add -
        # a negated numeric summed 120 anyway the first time this was
        # probed. Denials never participate in arithmetic.
        if record.get("negated"):
            continue
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
    # 300 meters + 2 kilometers is not 302 anything - but it IS 2.3
    # kilometers when a definition connects the units (§11.42's table).
    # Pairs the table cannot connect still refuse, exactly as §11.30
    # required, and a bare number beside a united one refuses too:
    # assuming its unit would be invention.
    unit_a, unit_b = _unit(ra.get("value", "")), _unit(rb.get("value", ""))
    converted_note = ""
    result_unit = ""
    if unit_a != unit_b or bool(unit_a) != bool(unit_b):
        if not unit_a or not unit_b:
            return None
        family = _UNIT_TO_FAMILY.get(unit_a)
        if family is None or _UNIT_TO_FAMILY.get(unit_b) != family:
            return None
        factors = _UNIT_FAMILIES[family]
        # The result reads in the larger unit of the two.
        result_unit = (unit_a if factors[unit_a] >= factors[unit_b]
                       else unit_b)
        na = _convert(na, unit_a, result_unit)
        nb = _convert(nb, unit_b, result_unit)
        converted_note = " (units converted)"
    confidence = min(float(ra.get("confidence", 0.0)),
                     float(rb.get("confidence", 0.0)))
    premises = [(ka, ra.get("value", "")), (kb, rb.get("value", ""))]
    suffix = f" {result_unit}s" if converted_note else ""
    if relation == "sum":
        answer = f"{ka} + {kb} = {na + nb:g}{suffix}{converted_note}"
    elif relation == "difference":
        answer = (f"the difference between {ka} and {kb} is "
                  f"{abs(na - nb):g}{suffix}{converted_note}")
    else:
        bigger = ka if na >= nb else kb
        smaller = kb if na >= nb else ka
        chosen = bigger if relation == "larger" else smaller
        shown = max(na, nb) if chosen == bigger else min(na, nb)
        answer = (f"{chosen} ({shown:g}{suffix}) "
                  f"is the {'larger' if relation == 'larger' else 'smaller'} "
                  f"of the two{converted_note}")
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
    # A combination question's gaps belong to its PARTS: "the sum of the
    # tower height and the bridge height" with both facts missing needs
    # two separate premises, and one minted key ("steel sum height bridge
    # height", then "steel height bridge height" after word-stripping)
    # answers neither. Combines re-derive from parts; the parts ask.
    if any(tok in _COMBINE_WORDS for tok in _TOKEN_RE.findall(text.lower())):
        return None
    question_tokens = _fold(text)
    if len(question_tokens) < 2:
        return None
    # The same modifier tolerance the inference path has, but the
    # one-unknown rule cannot work here: in the missing-premise scenario
    # the asked-about attribute is ITSELF library-unknown (that is why it
    # is missing), so "weathered tower conductivity" holds two unknowns.
    # The discriminator is positional - an adjective PRECEDES a key-known
    # token ("weathered tower"), while the asked attribute precedes
    # nothing known. "obelisk melting point" drops its ghost the same way
    # and then dies at the empty-remainder guard below.
    raw_sequence = [normalize_token(tok)
                    for tok in _TOKEN_RE.findall(text.lower())]
    droppable = set()
    for index, folded in enumerate(raw_sequence):
        if folded not in question_tokens:
            continue
        if not _library_unknown(folded, memory):
            continue
        following = next((later for later in raw_sequence[index + 1:]
                          if later in question_tokens), None)
        if following is not None and not _library_unknown(following,
                                                          memory):
            droppable.add(folded)
    if len(droppable) == 1 and len(question_tokens) >= 3:
        question_tokens = question_tokens - droppable
    probes = [" ".join(sorted(question_tokens))] + sorted(question_tokens)
    facts = _reachable_facts(memory, probes)
    best: tuple | None = None
    for key, record in facts.items():
        if record.get("negated"):
            # "not steel" once minted the premise key "not steel steel";
            # a denial names no bridge to ask through.
            continue
        key_tokens = _fold(key)
        covered = key_tokens & question_tokens
        remainder = question_tokens - key_tokens
        if not covered or not remainder:
            continue
        # A remainder wider than two tokens is a clause, not a fact key:
        # the compound question "the tower melting point and the bridge
        # length" once minted "steel melting point bridge length" - a
        # question no fact could ever answer. Decomposition owns
        # compounds; curiosity owns single gaps.
        if len(remainder) > 2:
            continue
        value = str(record.get("value", ""))
        if _numeric(value) is not None:
            continue
        value_tokens = _fold(value)
        if not value_tokens or len(value_tokens) > 2:
            continue
        # The same origin-coverage rule the spread enforces: a via-fact
        # with two tokens the question never said names a more specific
        # subject than asked, and asking for ITS continuation manufactures
        # curiosity about somebody else's property.
        if len(key_tokens - covered) > 1:
            continue
        candidate = (len(covered), -len(value_tokens), key)
        if best is None or candidate > best[0]:
            ordered_remainder, seen_folds = [], set()
            for tok in _TOKEN_RE.findall(text.lower()):
                folded = normalize_token(tok)
                if folded in remainder and folded not in seen_folds:
                    ordered_remainder.append(tok)
                    seen_folds.add(folded)
            premise_key = " ".join([value.strip().lower()]
                                   + ordered_remainder)
            best = (candidate, {
                "premise_key": premise_key,
                "via_key": key,
                "via_value": value,
                "original": text,
            })
    return best[1] if best is not None else None


def _library_unknown(token: str, memory) -> bool:
    """True when no held fact key contains this token."""
    for key in memory.find_facts(token, top_k=3):
        if token in _fold(key):
            return False
    return True


#: How many library-unknown tokens the inference path may read as
#: decoration. §11.36 shipped at one, §11.46 moved it to two, §11.62
#: to three - each behind the unmoved residual floor (at least two
#: known informative tokens must survive the drop, and convergence
#: still demands a bridge). The floor, not the cap, is the safety
#: line, measured three times now.
_MAX_MODIFIERS = 3


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
    ordered = _ordered_fold(text)
    direct = _spread(question_tokens, memory, ordered=ordered)
    if direct is not None:
        return direct

    # Adjective tolerance, on the INFERENCE path only (§11.35's registered
    # claim). "the weathered tower conductivity" fails whole because no
    # key holds "weathered"; dropping library-unknown tokens and retrying
    # is safe HERE because convergence still demands a bridge - "the
    # melting point of tungsten" minus "tungsten" leaves tokens whose
    # coverage is all-direct, which the spread already refuses as plain
    # recall's territory. The same drop on the exact/keyword path would
    # assert steel's number for tungsten, which is exactly the fabrication
    # the demotion rule exists to prevent - so recall never gets it.
    # Up to _MAX_MODIFIERS tokens may decorate, behind the residual
    # floor: at least two known informative tokens must survive, so a
    # question that is mostly unknown refuses rather than being read
    # away ("the ancient obelisk density" thins to one token and dies).
    unknown_set = {tok for tok in question_tokens
                   if _library_unknown(tok, memory)}
    residual = question_tokens - unknown_set
    if (unknown_set and len(unknown_set) <= _MAX_MODIFIERS
            and len(residual) >= 2):
        tolerated = _spread(residual, memory,
                            ordered=[tok for tok in ordered
                                     if tok not in unknown_set])
        if tolerated is not None:
            named = [tok for tok in dict.fromkeys(ordered)
                     if tok in unknown_set]
            named += sorted(unknown_set - set(named))
            if len(named) == 1:
                tolerated.answer += (f" (reading '{named[0]}' as a "
                                     "modifier)")
            else:
                joined = " and ".join(f"'{tok}'" for tok in named)
                tolerated.answer += f" (reading {joined} as modifiers)"
            return tolerated
    return None
