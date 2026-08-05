"""The deployment languages, and the starter knowledge the model learns from.

`ultraquant/reason/language.py` is the induction machinery and deliberately
contains no words of any language — a test parses its AST to prove it. The
*vocabularies* have to live somewhere, and for the deployed library they live
here: three languages covering both kinds of reading the interpreter produces,
plus a small body of starter facts that gives self-learning something to work
on from the first session.

**Coverage.** Each language covers the factored vocabulary (shapes and marks,
spoken as scenes) and the eight built-in whole-pattern labels (spoken as bare
things: "a square"). The 24 synthetic families are *excluded* from whole-label
speech for an honest reason: their labels (``sym_0``, ``sym_1``) repeat across
every family, so as meaning atoms they are ambiguous — "a sym_0" names 24
different things. A language should not be taught to say something it cannot
mean.

**One homonym dodged, deliberately.** The hidden languages already realise the
mark ``ex`` as "saltire". The whole-pattern label ``cross`` is a different atom
and must not reuse that word — one word for two atoms was measured to cut both
generation and parsing to 0.75 (§11.10), so ``cross`` becomes "crux".

The whole-label ``plus`` is the *same* atom as the factored mark ``plus``
(both are ``("mark", "plus")``), so it shares that word by identity rather
than by coincidence.
"""

from __future__ import annotations

from typing import Any

from ultraquant.reason.language import GroundedLanguage, store_language

__all__ = ["WHOLE_LABEL_WORDS", "BASIC_FACTS", "deploy_languages", "seed_knowledge"]

#: How each language names the built-in whole-pattern labels.
#: 'plain' speaks the library's own label names; the hidden languages coin
#: their own nouns, consistent with their existing vocabulary.
WHOLE_LABEL_WORDS: dict[str, dict[str, str]] = {
    "plain": {
        "square": "square", "diamond": "diamond", "plus": "plus",
        "cross": "cross", "arrow_up": "arrow_up", "arrow_down": "arrow_down",
        "stripes_h": "stripes_h", "stripes_v": "stripes_v",
    },
    "coined": {  # shared by 'nested' and 'holding'; frames and order differ
        "square": "quad", "diamond": "lozenge", "plus": "plus",
        "cross": "crux", "arrow_up": "riser", "arrow_down": "sinker",
        "stripes_h": "lanes", "stripes_v": "pales",
    },
}

#: Starter knowledge, taught through the normal pipeline so it lands in the
#: sharded fact store with ordinary confidences. Deliberately *mixed*: most of
#: it is solid, and a few entries are marked shaky so learning mode has real
#: gaps to ask about from the first survey — self-learning needs something to
#: doubt, not only something to know.
BASIC_FACTS: list[tuple[str, str, bool]] = [
    # (key phrase, value, solid)
    ("a square", "a shape with four equal sides", True),
    ("a diamond", "a square rotated forty-five degrees", True),
    ("a plus", "two bars crossing at the centre", True),
    ("a cross", "two diagonals crossing at the centre", True),
    ("an arrow", "a shape that points in a direction", True),
    ("stripes", "parallel bars that repeat", True),
    ("a pattern", "an arrangement of pixels with structure", True),
    ("a glyph", "a five by five pattern of pixels", True),
    ("the library", "the sharded store of everything learned", True),
    ("a shard", "one catalogued piece of the library", True),
    ("a category", "a group of patterns that belong together", True),
    ("a fact", "a key and value held with a confidence", True),
    ("a container", "a border shape that holds a mark inside it", True),
    ("a mark", "a small shape drawn inside a container", True),
    ("the number of glyph pixels", "25", True),
    ("the number of glyph rows", "5", True),
    # Shaky on purpose: plausible, unconfirmed, worth asking about.
    ("the tallest builtin pattern", "stripes_v", False),
    ("the densest builtin pattern", "square", False),
    ("the rarest mark", "dot", False),
]


def deploy_languages(session: Any, shapes: list[str], marks: list[str]) -> list[str]:
    """Train and store the three deployment languages.

    Each is induced from pairs generated over the full factored vocabulary plus
    the built-in whole labels, then stored as its own shard with slot roles
    covering both the factored slots and the whole-pattern slot.

    Args:
        session: The interpreter session to store into.
        shapes: The factored container vocabulary.
        marks: The factored mark vocabulary.

    Returns:
        The names of the stored languages.
    """
    # Surface words for the factored atoms, per language family.
    plain_shape = {shape: shape for shape in shapes}
    plain_mark = {mark: mark for mark in marks}
    coined_shape = {"box": "box", "rails": "grill", "caps": "caps",
                    "corners": "nooks"}
    coined_mark = {"plus": "plus", "ex": "saltire", "dot": "speck",
                   "bar": "dash"}

    def pairs(shape_words, mark_words, whole_words, frame, mark_side):
        out = []
        for shape in shapes:
            for mark in marks:
                inner, outer = mark_words[mark], shape_words[shape]
                if mark_side == "first":
                    utterance = f"a {inner} {frame} a {outer}"
                else:
                    utterance = f"a {outer} {frame} a {inner}"
                out.append((utterance, [shape, mark]))
        for label, word in whole_words.items():
            out.append((f"a {word}", [label]))
        return out

    roles = {"shape": "container", "mark": "mark", "pattern": "mark"}
    recipes = {
        "plain": pairs(plain_shape, plain_mark, WHOLE_LABEL_WORDS["plain"],
                       "inside", "first"),
        "nested": pairs(coined_shape, coined_mark, WHOLE_LABEL_WORDS["coined"],
                        "inside", "first"),
        "holding": pairs(coined_shape, coined_mark, WHOLE_LABEL_WORDS["coined"],
                         "holding", "last"),
    }
    for name, training in recipes.items():
        model = GroundedLanguage()
        model.fit(training)
        store_language(session, model, roles, name=name)
    return sorted(recipes)


def seed_knowledge(session: Any) -> dict:
    """Teach the starter facts through the normal pipeline.

    Facts go in as statements so they take the same path as anything a user
    says: extracted, stored in the sharded fact store, and reinforced on use.
    The shaky entries then have their confidence pulled down so the first
    learning-mode survey has genuine questions to ask — the point is to leave
    the model with material to *doubt*, which is where self-learning starts.

    Returns:
        ``{"taught": n, "shaky": n}``.
    """
    from ultraquant.interpreter.thoughts import run_pipeline

    taught = 0
    shaky = 0
    for key, value, solid in BASIC_FACTS:
        run_pipeline(f"{key} is {value}", session)
        taught += 1
        if not solid:
            # The statement path may normalise the key; look it up as stored.
            stored_key = None
            for candidate in session.memory.find_facts(key, top_k=3):
                if session.memory.recall_fact(candidate):
                    stored_key = candidate
                    break
            if stored_key:
                fact = session.memory.recall_fact(stored_key)
                fact["confidence"] = 0.4
                session.memory._put_fact(stored_key, fact)
                shaky += 1
    session.save()
    return {"taught": taught, "shaky": shaky}
