"""A dictionary in the library, read a page at a time.

The user asked for a dictionary, and chose Open English WordNet (§11.167). It
is a lexicon of its own, beside the vault rather than inside it: every word the
release lists and every synset it defines is kept in hash-addressed pages,
packed into one library file, ``<home>/lexicon/wordnet.uql``.

* **Word pages.** A normalized form hashes to one of :data:`PAGES` word pages,
  which holds that form's senses in release order.
* **Synset pages.** A synset id hashes to one of :data:`PAGES` synset pages,
  which holds the release's record for it, plus the links that point at it.
* **One about page** says what was imported and how it must be credited.

A lookup computes its page and reads it through the vault, so defining a word
reads its word page and the synset pages of its senses, and nothing else. The
lexicon only answers what words mean. It never writes a fact, and a lookup
never writes anything at all.

Pure Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

from ultraquant.shards.vault import ShardVault

__all__ = ["Lexicon", "normalize_word", "page_of", "source_record", "PAGES"]

#: How many word pages, and separately how many synset pages, there are.
#:
#: Sized so a page stays worth reading as a unit: 151,779 forms and 120,564
#: synsets come to about 300 forms or 235 synsets a page. One page per word
#: would make the catalog larger than the dictionary; one page for all of them
#: would read the whole dictionary to define one word.
PAGES = 512

#: Decoded pages kept between lookups, least recently used first out.
CACHE_PAGES = 64

#: The packed library, inside ``<home>/lexicon``.
LIBRARY = "wordnet.uql"

#: The page saying what was imported, and how it must be credited.
ABOUT_ID = "lexicon:about"

#: The pinned release: what is imported, and the line that credits it.
SOURCE_PATH = Path(__file__).with_name("data") / "source.json"


def source_record() -> dict:
    """The pinned release record, ``data/source.json``."""
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


# §11.167: not normalize_subject, which drops a leading article - "A" is a word.
def normalize_word(text: str) -> str:
    """Fold case, accents and punctuation; no word is ever dropped."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in bare).split())


def page_of(key: str) -> int:
    """The page a normalized form, or a synset id, is kept on."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % PAGES


def word_page(form: str) -> str:
    """The shard id of the word page holding a normalized form."""
    return f"lexicon:word:{page_of(form):03d}"


def synset_page(synset_id: str) -> str:
    """The shard id of the synset page holding a synset."""
    return f"lexicon:synset:{page_of(synset_id):03d}"


class Lexicon:
    """Read-only word lookups over a built lexicon.

    Opening one reads the catalog and nothing else; pages are read when a word
    is asked about. Build one with ``python -m ultraquant.lexicon.build``.

    Args:
        home: The library home. The lexicon is ``home/lexicon``.

    Attributes:
        home: The library home, as given.
        library: This home's packed lexicon, ``home/lexicon/wordnet.uql``.
        vault: The lexicon's own :class:`ShardVault`.
    """

    def __init__(self, home: str | Path) -> None:
        self.home = Path(home)
        self.library = self.home.absolute() / "lexicon" / LIBRARY
        self.vault = ShardVault(Path(home) / "lexicon")
        self._pages: OrderedDict[str, dict] = OrderedDict()
        # A copied library still names the file it was packed into, which may
        # be gone or, worse, a different lexicon. Attaching this home's own
        # file re-points every page at it, reading only the library's index.
        target = self.library.resolve()
        if any(Path(path).resolve() != target for path in self.vault.libraries()):
            self.vault.attach(self.library)

    @classmethod
    def open(cls, home: str | Path) -> Lexicon | None:
        """The home's lexicon, or None when it has none; creates nothing."""
        if not (Path(home) / "lexicon" / LIBRARY).exists():
            return None
        return cls(home)

    # ------------------------------------------------------------------ #
    # pages
    # ------------------------------------------------------------------ #

    def _page(self, shard_id: str, held: dict[str, dict]) -> dict:
        """One decoded page, read from the vault at most once per lookup.

        ``held`` is the lookup's own set of pages. The cache alone would
        re-read a page that a long lookup (the ancestors of a word with many
        senses) pushed out of it; holding them for the lookup's length is
        what makes "each page at most once" true rather than likely.
        """
        page = held.get(shard_id)
        if page is not None:
            return page
        page = self._pages.get(shard_id)
        if page is not None:
            self._pages.move_to_end(shard_id)
        else:
            # Only get(): it counts the access in memory and writes nothing.
            page = self.vault.get(shard_id) if self.vault.has(shard_id) else {}
            self._pages[shard_id] = page
            if len(self._pages) > CACHE_PAGES:
                self._pages.popitem(last=False)
        held[shard_id] = page
        return page

    def _senses(self, word: str, held: dict[str, dict]) -> list[dict]:
        form = normalize_word(word)
        if not form:
            return []
        rows = self._page(word_page(form), held).get("words", {}).get(form, [])
        return [{"lemma": lemma, "pos": pos, "synset": synset, "sense_key": key}
                for lemma, pos, synset, key in rows]

    def _record(self, synset_id: str, held: dict[str, dict]) -> dict | None:
        synsets = self._page(synset_page(synset_id), held).get("synsets", {})
        return synsets.get(synset_id)

    # ------------------------------------------------------------------ #
    # lookups
    # ------------------------------------------------------------------ #

    def senses(self, word: str) -> list[dict]:
        """``{"lemma", "pos", "synset", "sense_key"}`` per sense, release order."""
        return self._senses(word, {})

    def synset(self, synset_id: str) -> dict | None:
        """A copy of the stored record for ``synset_id``, or None."""
        record = self._record(synset_id, {})
        return deepcopy(record) if record is not None else None

    def define(self, word: str) -> list[dict]:
        """``{"lemma", "pos", "synset", "definition"}`` for each sense, in order."""
        held: dict[str, dict] = {}
        out = []
        for sense in self._senses(word, held):
            record = self._record(sense["synset"], held) or {}
            out.append({"lemma": sense["lemma"], "pos": sense["pos"],
                        "synset": sense["synset"],
                        "definition": list(record.get("definition", []))})
        return out

    def synonyms(self, word: str) -> list[str]:
        """Members of the word's synsets, in order, other than the word itself."""
        held: dict[str, dict] = {}
        own = normalize_word(word)
        found: dict[str, None] = {}
        for sense in self._senses(word, held):
            record = self._record(sense["synset"], held) or {}
            for member in record.get("members", []):
                if normalize_word(member) != own:
                    found.setdefault(member)
        return list(found)

    def kinds(self, word: str) -> list[list[str]]:
        """Per sense, the ids of its ancestors, breadth-first up to the roots.

        An instance ("The Hague") climbs through ``instance_hypernym`` to its
        class, and a class through ``hypernym``; both are followed.
        """
        held: dict[str, dict] = {}
        out = []
        for sense in self._senses(word, held):
            seen = {sense["synset"]}
            ancestors: list[str] = []
            frontier = deque([sense["synset"]])
            while frontier:
                record = self._record(frontier.popleft(), held) or {}
                for parent in (*record.get("instance_hypernym", ()),
                               *record.get("hypernym", ())):
                    if parent not in seen:
                        seen.add(parent)
                        ancestors.append(parent)
                        frontier.append(parent)
            out.append(ancestors)
        return out

    def about(self) -> dict[str, Any]:
        """What was imported, when, how much, and how it must be credited."""
        return deepcopy(self._page(ABOUT_ID, {}).get("about", {}))
