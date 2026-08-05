"""Language induced from examples, grounded in what the library perceives.

The interpreter's responses are templates — fixed strings with values dropped in.
A template can only say what someone anticipated. This module learns to say
things from paired examples instead: given ``(utterance, meaning)`` pairs, it
induces which words realise which concepts, what order the language puts them
in, and which function words frame them. Nothing about any particular language
appears in this file, and a test asserts that precisely: the hidden languages
used to evaluate it have words that occur in no string literal and no
identifier here — a template would have to live in one of those.

**Grounded** means the meanings bottom out in the pattern library rather than in
other words. A meaning here is the same structure the blackboard produces when
it reads a compound glyph ([§11.6]) — containers outermost-first with the
innermost mark last — so a learned language lets the system *say what it sees*,
and :func:`verbalise` is exactly that bridge.

What is induced, and from what:

* **Lexicon** — which word realises which ``(role, value)`` atom, from
  co-occurrence alone. Scored by ``P(word|atom) * P(atom|word)``, so a function
  word that appears everywhere scores near an atom's base rate and loses to the
  word that appears *iff* the atom does. Words below the exclusivity floor are
  frame, not content.
* **Order** — whether the mark is uttered before or after its containers, and
  whether containers go inner-first or outer-first, by majority vote over the
  training pairs. Both directions are represented; neither is preferred.
* **Frame** — the tokens before the first content word and the tokens between
  each adjacent *role pair*, keyed by role pair rather than by position. That
  keying is what lets a language learned at nesting depth 2-3 be spoken at
  depth 4: the gap between two container words is the same gap however deep the
  recursion goes.

The trained model serialises to a JSON-safe dict and is stored in the shard
library like everything else the system learns — paged on demand, shipped
inside a packed ``.uql``.

Whether this is real is measured, not asserted:
``ultraquant/experiments/language_gate.py``.

Pure Python standard library.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["GroundedLanguage", "store_language", "list_languages",
           "verbalise", "verbalise_all", "LANGUAGE_SHARD"]

#: Where a session's learned language lives in the vault.
LANGUAGE_SHARD = "language:model"

#: Below this exclusivity score a word is frame, not the realisation of an atom.
#: A perfectly exclusive word scores 1.0; a function word that appears in every
#: sentence scores the atom's base rate (~1/values). 0.5 sits well clear of
#: both.
_EXCLUSIVITY_FLOOR = 0.5

Atom = tuple[str, str]  # (role, value)


def _atoms(levels: list[str]) -> list[Atom]:
    """A meaning's atoms: containers outermost-first, the mark last."""
    return [("container", value) for value in levels[:-1]] + [("mark", levels[-1])]


class GroundedLanguage:
    """A language induced from ``(utterance, meaning)`` pairs."""

    def __init__(self) -> None:
        self.lexicon: dict[Atom, str] = {}
        self.reverse: dict[str, Atom] = {}
        self.mark_first: bool = True
        self.inner_first: bool = True
        self.prefix: tuple[str, ...] = ()
        self.gaps: dict[tuple[str, str], tuple[str, ...]] = {}
        self.suffix: tuple[str, ...] = ()
        self.trained_pairs: int = 0

    # ------------------------------------------------------------------ #
    # induction
    # ------------------------------------------------------------------ #

    def fit(self, pairs: list[tuple[str, list[str]]]) -> dict:
        """Induce lexicon, order and frame from paired examples.

        Args:
            pairs: ``(utterance, levels)`` where ``levels`` is containers
                outermost-first with the mark last — the blackboard's reading.

        Returns:
            ``{"lexicon": n, "coverage": fraction of atoms that found a word}``.
        """
        word_atom: Counter = Counter()
        word_count: Counter = Counter()
        atom_count: Counter = Counter()
        for utterance, levels in pairs:
            tokens = utterance.split()
            atoms = _atoms(levels)
            for word in set(tokens):
                word_count[word] += 1
                for atom in set(atoms):
                    word_atom[(word, atom)] += 1
            for atom in set(atoms):
                atom_count[atom] += 1

        # -- lexicon: the word that appears iff the atom does ----------------
        #
        # Assignment is greedy and exclusive, in a deterministic order. The
        # first version took a bare argmax per atom, and a tiny corpus exposed
        # two bugs at once: ties between candidate words were broken by set
        # iteration order — nondeterministic under hash randomisation, which
        # this project forbids — and two atoms could claim the same word,
        # yielding sentences like "a box box". Scoring ties now break by joint
        # count then alphabetically, and a claimed word is off the table.
        self.lexicon = {}
        self.reverse = {}
        candidates: list[tuple[float, int, str, Atom]] = []
        for atom in sorted(atom_count):
            for word in sorted(word_count):
                joint = word_atom[(word, atom)]
                if not joint:
                    continue
                score = (joint / atom_count[atom]) * (joint / word_count[word])
                if score >= _EXCLUSIVITY_FLOOR:
                    candidates.append((score, joint, word, atom))
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
        claimed: set[str] = set()
        for _score, _joint, word, atom in candidates:
            if atom in self.lexicon or word in claimed:
                continue
            self.lexicon[atom] = word
            claimed.add(word)
        for atom, word in self.lexicon.items():
            self.reverse[word] = atom

        # -- order: majority vote over observed utterance positions ----------
        mark_votes = 0
        mark_total = 0
        inner_votes = 0
        inner_total = 0
        observed: list[tuple[list[Atom], list[str]]] = []
        for utterance, levels in pairs:
            tokens = utterance.split()
            atoms = _atoms(levels)
            position = {}
            for atom in atoms:
                word = self.lexicon.get(atom)
                if word is not None and word in tokens:
                    position[atom] = tokens.index(word)
            if len(position) != len(atoms):
                continue  # a pair the lexicon cannot fully explain teaches order nothing
            observed.append((atoms, tokens))
            mark = atoms[-1]
            for container in atoms[:-1]:
                mark_total += 1
                mark_votes += position[mark] < position[container]
            for outer_index in range(len(atoms) - 2):
                inner_total += 1
                # levels are outermost-first, so index+1 is the deeper container
                inner_votes += (position[atoms[outer_index + 1]]
                                < position[atoms[outer_index]])
        self.mark_first = mark_votes * 2 >= mark_total if mark_total else True
        self.inner_first = inner_votes * 2 >= inner_total if inner_total else True

        # -- frame: prefix, per-role-pair gaps, suffix ------------------------
        prefixes: Counter = Counter()
        suffixes: Counter = Counter()
        gap_votes: dict[tuple[str, str], Counter] = {}
        for atoms, tokens in observed:
            ordered = self._utterance_order(atoms)
            spans = [tokens.index(self.lexicon[atom]) for atom in ordered]
            prefixes[tuple(tokens[:spans[0]])] += 1
            suffixes[tuple(tokens[spans[-1] + 1:])] += 1
            for left, right in zip(ordered, ordered[1:]):
                key = (left[0], right[0])
                gap = tuple(tokens[tokens.index(self.lexicon[left]) + 1:
                                   tokens.index(self.lexicon[right])])
                gap_votes.setdefault(key, Counter())[gap] += 1
        self.prefix = prefixes.most_common(1)[0][0] if prefixes else ()
        self.suffix = suffixes.most_common(1)[0][0] if suffixes else ()
        self.gaps = {key: votes.most_common(1)[0][0]
                     for key, votes in gap_votes.items()}
        self.trained_pairs = len(pairs)

        return {
            "lexicon": len(self.lexicon),
            "coverage": len(self.lexicon) / max(1, len(atom_count)),
        }

    # ------------------------------------------------------------------ #
    # speaking and understanding
    # ------------------------------------------------------------------ #

    def _utterance_order(self, atoms: list[Atom]) -> list[Atom]:
        """Structural atoms arranged in the learned spoken order."""
        containers = atoms[:-1]
        if self.inner_first:
            containers = list(reversed(containers))
        if self.mark_first:
            return [atoms[-1]] + containers
        return containers + [atoms[-1]]

    def generate(self, levels: list[str]) -> str | None:
        """Say a meaning, in the induced language.

        Args:
            levels: Containers outermost-first, mark last.

        Returns:
            The utterance, or ``None`` if the lexicon lacks a word for any atom
            — an honest refusal rather than a sentence with a hole in it.
        """
        atoms = _atoms(levels)
        if any(atom not in self.lexicon for atom in atoms):
            return None
        ordered = self._utterance_order(atoms)
        tokens: list[str] = list(self.prefix)
        for index, atom in enumerate(ordered):
            if index:
                key = (ordered[index - 1][0], atom[0])
                tokens.extend(self.gaps.get(key, ()))
            tokens.append(self.lexicon[atom])
        tokens.extend(self.suffix)
        return " ".join(tokens)

    def parse(self, utterance: str) -> list[str] | None:
        """Understand an utterance back into a meaning.

        Returns:
            Levels outermost-first with the mark last, or ``None`` when the
            content words do not amount to one mark plus containers.
        """
        scanned = [self.reverse[token] for token in utterance.split()
                   if token in self.reverse]
        marks = [value for role, value in scanned if role == "mark"]
        containers = [value for role, value in scanned if role == "container"]
        if len(marks) != 1:
            return None
        if self.inner_first:
            containers = list(reversed(containers))
        return containers + marks

    # ------------------------------------------------------------------ #
    # persistence — the language is library data like everything else
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        """JSON-safe form, storable as a shard."""
        return {
            "lexicon": [[list(atom), word] for atom, word in
                        sorted(self.lexicon.items())],
            "mark_first": self.mark_first,
            "inner_first": self.inner_first,
            "prefix": list(self.prefix),
            "suffix": list(self.suffix),
            "gaps": [[list(key), list(gap)] for key, gap in
                     sorted(self.gaps.items())],
            "trained_pairs": self.trained_pairs,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "GroundedLanguage":
        """Rebuild a language saved by :meth:`state_dict`."""
        model = cls()
        model.lexicon = {(role, value): word
                         for (role, value), word in
                         ((tuple(atom), word) for atom, word in state["lexicon"])}
        model.reverse = {word: atom for atom, word in model.lexicon.items()}
        model.mark_first = bool(state["mark_first"])
        model.inner_first = bool(state["inner_first"])
        model.prefix = tuple(state["prefix"])
        model.suffix = tuple(state["suffix"])
        model.gaps = {tuple(key): tuple(gap) for key, gap in state["gaps"]}
        model.trained_pairs = int(state.get("trained_pairs", 0))
        return model


def store_language(session: Any, model: GroundedLanguage,
                   slot_roles: dict[str, str], name: str = "model") -> None:
    """Put a learned language into the library.

    The language is learned data, so it lives where learned data lives: as a
    catalogued shard, paged on demand and carried inside a packed ``.uql`` —
    the same lesson the fact store taught ([§11.9]).

    A library may hold several languages at once — each is its own shard, so
    speaking one pages one.

    Args:
        session: The interpreter session.
        model: The trained language.
        slot_roles: How the blackboard's slot names map onto language roles,
            e.g. ``{"shape": "container", "mark": "mark"}``. Stored beside the
            model because the mapping is part of what was learned about this
            library, not a property of the language itself.
        name: Which language this is. The default keeps the single-language
            shard id unchanged.
    """
    shard_id = f"language:{name}"
    session.vault.add_shard(
        shard_id, "language",
        {"model": model.state_dict(), "slot_roles": dict(slot_roles)},
        kind="language-model",
    )
    session.cache.invalidate(shard_id)


def list_languages(session: Any) -> list[str]:
    """Names of every language the library holds, primary first.

    ``plain`` (the library's own vocabulary) leads when present, then
    ``model`` (the single-language default), then the rest alphabetically —
    so chat speaks the most immediately intelligible one and offers the others.
    """
    names = sorted(
        entry["shard_id"].split(":", 1)[1]
        for entry in session.vault.catalog()
        if entry.get("kind") == "language-model"
    )
    for preferred in ("model", "plain"):
        if preferred in names:
            names.remove(preferred)
            names.insert(0, preferred)
    return names


def _load_language(session: Any, name: str) -> tuple[GroundedLanguage, dict] | None:
    """A stored language and its slot-role mapping, through the cache."""
    shard_id = f"language:{name}"
    if not session.vault.has(shard_id):
        return None
    payload = session.cache.get(
        shard_id,
        lambda: (session.vault.get(shard_id),
                 int(session.vault.entry(shard_id)["nbytes"])),
    )
    return (GroundedLanguage.from_state_dict(payload["model"]),
            payload.get("slot_roles", {}))


def verbalise_all(session: Any, reading: dict[str, str]) -> dict[str, str]:
    """The reading spoken in every language the library holds.

    Returns:
        ``{language name: sentence}``, empty when nothing is stored or no
        stored language covers the reading.
    """
    out: dict[str, str] = {}
    for name in list_languages(session):
        sentence = verbalise(session, reading, language=name)
        if sentence:
            out[name] = sentence
    return out


def verbalise(session: Any, reading: dict[str, str],
              language: str | None = None) -> str | None:
    """Say what the blackboard read, if a language has been learned.

    This is the grounding: the meaning comes from perception — the composed
    reading of an actual glyph — and the words come from a language induced
    from examples. Neither direction involves a template.

    Args:
        session: The interpreter session; languages live in its vault.
        reading: The blackboard's ``{slot: value}`` reading.
        language: Which stored language to speak; ``None`` uses the primary.

    Returns:
        A sentence, or ``None`` when no language is stored, the stored roles do
        not cover the reading, or the lexicon lacks a word.
    """
    if language is None:
        names = list_languages(session)
        if not names:
            return None
        language = names[0]
    loaded = _load_language(session, language)
    if loaded is None:
        return None
    model, slot_roles = loaded
    containers = [value for slot, value in reading.items()
                  if slot_roles.get(slot) == "container"]
    marks = [value for slot, value in reading.items()
             if slot_roles.get(slot) == "mark"]
    if len(marks) != 1:
        return None
    return model.generate(containers + marks)
