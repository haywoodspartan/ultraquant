"""The repository's own documentation, as a distillation corpus.

§11.110 distilled a pretrained embedding into a from-scratch encoder
and measured the ceiling on how far that can go: on text using words
withheld from distillation the encoder scored **0.350 against 0.500**
on text avoiding them. It knows the geometry of words it was shown
and has nothing whatever for the rest.

That ceiling is a corpus problem. The library's built-in keyword
vocabulary is 25 words and the §11.109 test corpus is 79, and no
amount of training fixes a vocabulary that small.

**But the repository is full of real English prose.** ARCHITECTURE.md,
JOURNAL.md and README.md are thousands of sentences of ordinary
argumentative writing - **1,221 sentences over 3,245 distinct content
words**, forty-one times the test corpus. It is not a pretraining
corpus and nobody should call it one, but it is real text, it is
already here, and the teacher will embed any of it for free.

**Why this is not circular.** The prose is used only to give the
encoder more of the teacher's geometry to imitate. No label, no
category and no test item comes from it - the §11.109 task is
untouched. What the extra text can buy is robustness: an encoder that
has seen three thousand words placed by the teacher may still route a
sentence correctly when one of its words is missing, because the rest
of the sentence is in a space it understands.

Whether it does buy that is the question §11.111 measures, and it can
come out no.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["SOURCES", "sentences", "vocabulary"]

#: Documents mined for prose. Markdown rather than source, because
#: docstrings are dense with identifiers and the point is ordinary
#: English.
SOURCES = ("ARCHITECTURE.md", "JOURNAL.md", "README.md")

#: Fenced code, tables and markup are stripped: an encoder learning
#: the geometry of "|---|---:|" has learned nothing about language.
_FENCE = re.compile(r"```.*?```", re.S)
_MARKUP = re.compile(r"[|#*`\[\]()_>-]+")
_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(root: Path | None = None, low: int = 30,
              high: int = 160) -> list:
    """Ordinary English sentences from the repository's documentation.

    Bounded in length at both ends: fragments carry no structure to
    learn and very long sentences in this corpus are usually tables
    that survived the markup strip. Deterministic - the same files
    give the same list in the same order, so a distillation run is
    reproducible.
    """
    base = Path(root) if root is not None else Path(__file__).resolve(
        ).parents[2]
    out: list = []
    for name in SOURCES:
        path = base / name
        if not path.exists():
            continue
        text = _MARKUP.sub(" ", _FENCE.sub(" ", path.read_text(
            encoding="utf-8")))
        for raw in _SPLIT.split(text):
            line = " ".join(raw.split())
            if low <= len(line) <= high and line[:1].isupper():
                out.append(line)
    return out


def vocabulary(texts: list) -> list:
    """Content words across `texts`, sorted for determinism."""
    from ultraquant.experiments.embed_gate import _tokens

    words = set()
    for text in texts:
        words |= _tokens(text)
    return sorted(words)
