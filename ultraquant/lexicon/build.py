"""Import Open English WordNet into a library as its dictionary (§11.167).

Run it::

    python -m ultraquant.lexicon.build --home uq_home
    python -m ultraquant.lexicon.build --home DIR --source other.zip

The release is hashed against its pin before anything is written. It is then
read in release order and written as hash-addressed pages into a fresh vault at
``<home>/lexicon.building``, packed into one ``wordnet.uql``. Only a complete
build replaces ``<home>/lexicon``, so a failed one leaves the old dictionary as
it was.

Nothing else in the home is opened: not the memory, the stash, the approvals
journal or the vault. The dictionary is a lexicon, not a source of facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ultraquant.lexicon.lexicon import (
    ABOUT_ID, LIBRARY, PAGES, normalize_word, page_of, source_record,
)
from ultraquant.shards.vault import ShardVault

__all__ = ["build", "read_release", "main"]

#: The repository root, which source.json's "path" is relative to.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Each inverse link a synset page carries, and the release link it inverts.
_INVERSES = (("hyponym", "hypernym"), ("instance", "instance_hypernym"))


def _sha256(path: str | os.PathLike) -> str:
    """The sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_release(path: str | os.PathLike) -> tuple[dict, dict]:
    """Every sense and every synset in a release zip, in release order.

    Release order is zip name order, then JSON object order, then
    part-of-speech order, then sense list order. ``frames.json`` (verb frames)
    is skipped. No inverse link is computed here.

    Returns:
        ``(forms, synsets)``. ``forms`` maps each normalized form to its
        ``{"lemma", "pos", "synset", "sense_key"}`` senses; ``synsets`` maps
        each synset id to the release's record, unchanged, plus ``"lexfile"``,
        the name of the file it came from without ``.json``.
    """
    forms: dict[str, list[dict]] = {}
    synsets: dict[str, dict] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            base = PurePosixPath(name).name
            if not base.endswith(".json") or base == "frames.json":
                continue
            data = json.loads(archive.read(name).decode("utf-8"))
            if base.startswith("entries-"):
                for lemma, by_pos in data.items():
                    form = normalize_word(lemma)
                    for key, entry in by_pos.items():
                        # "n-1" and "n-2" are one spelling with two words
                        # behind it (bass the fish, bass the voice). Both are
                        # nouns: the part of speech is what precedes the dash.
                        pos = key.split("-", 1)[0]
                        for sense in entry.get("sense", ()):
                            forms.setdefault(form, []).append({
                                "lemma": lemma, "pos": pos,
                                "synset": sense["synset"],
                                "sense_key": sense["id"],
                            })
            else:
                lexfile = base[:-len(".json")]
                for synset_id, record in data.items():
                    record["lexfile"] = lexfile
                    synsets[synset_id] = record
    return forms, synsets


def _add_inverses(synsets: dict[str, dict]) -> None:
    """Record on each synset what points at it: its hyponyms and instances.

    The release links upward only. A page cannot be searched for who names it,
    so "which kinds of city are there" needs the inverse written down at build
    time. Each list is sorted ascending and present only when non-empty.
    """
    for inverse, relation in _INVERSES:
        pointing: dict[str, set[str]] = {}
        for synset_id, record in synsets.items():
            for target in record.get(relation, ()):
                pointing.setdefault(target, set()).add(synset_id)
        for target, ids in pointing.items():
            if target in synsets:
                synsets[target][inverse] = sorted(ids)


def build(home: str | os.PathLike, source: str | os.PathLike | None = None, *,
          expected_sha256: str | None = None) -> dict:
    """Import a release into ``<home>/lexicon``, replacing any lexicon there.

    Args:
        home: The library home.
        source: The release zip; defaults to source.json's ``"path"``, relative
            to the repository root.
        expected_sha256: What the zip must hash to; defaults to source.json's
            ``"sha256"``.

    Returns:
        ``{"forms", "synsets", "senses", "pages": {"word", "synset"}, "bytes",
        "sha256"}``; ``bytes`` and ``sha256`` are the packed library's.

    Raises:
        ValueError: If the zip is not the release expected. Nothing is
            written, not even a directory.
    """
    pinned = source_record()
    home = Path(home).absolute()
    source = Path(source) if source is not None else REPO_ROOT / pinned["path"]
    expected = pinned["sha256"] if expected_sha256 is None else expected_sha256
    actual = _sha256(source)
    if actual != expected.lower():
        raise ValueError(f"{source} has sha256 {actual}, not the expected "
                         f"{expected}; nothing was written")

    forms, synsets = read_release(source)
    _add_inverses(synsets)
    word_pages: list[dict] = [{} for _ in range(PAGES)]
    senses = 0
    for form, items in forms.items():
        word_pages[page_of(form)][form] = [
            [item["lemma"], item["pos"], item["synset"], item["sense_key"]]
            for item in items]
        senses += len(items)
    synset_pages: list[dict] = [{} for _ in range(PAGES)]
    for synset_id, record in synsets.items():
        synset_pages[page_of(synset_id)][synset_id] = record
    about = {**pinned, "forms": len(forms), "synsets": len(synsets),
             "senses": senses,
             "built": datetime.now(timezone.utc).isoformat()}

    building = home / "lexicon.building"
    final = home / "lexicon"
    aside = home / "lexicon.old"
    if building.exists():
        shutil.rmtree(building)  # an interrupted build is never resumed
    try:
        vault = ShardVault(building)
        # One catalog write for 1,025 pages, not one per page. Every page is
        # its own category and carries no associations: a word is found by
        # hashing it, never by routing.
        with vault.batch():
            for number, page in enumerate(word_pages):
                shard_id = f"lexicon:word:{number:03d}"
                vault.add_shard(shard_id, shard_id, {"words": page},
                                kind="lexicon-word")
            for number, page in enumerate(synset_pages):
                shard_id = f"lexicon:synset:{number:03d}"
                vault.add_shard(shard_id, shard_id, {"synsets": page},
                                kind="lexicon-synset")
            vault.add_shard(ABOUT_ID, ABOUT_ID, {"about": about},
                            kind="lexicon-about")
        vault.pack(building / LIBRARY, prune_loose=True)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise

    # The old lexicon stands until the new one is complete, then steps aside.
    if aside.exists():
        shutil.rmtree(aside)
    if final.exists():
        os.replace(final, aside)
    try:
        os.replace(building, final)
    except OSError:
        if aside.exists() and not final.exists():
            os.replace(aside, final)
        raise
    if aside.exists():
        shutil.rmtree(aside)

    # The pack recorded lexicon.building as the library's path; re-attaching
    # reads only the index and points every page at where it now lives.
    library = final / LIBRARY
    ShardVault(final).attach(library)
    return {
        "forms": len(forms),
        "synsets": len(synsets),
        "senses": senses,
        "pages": {"word": PAGES, "synset": PAGES},
        "bytes": library.stat().st_size,
        "sha256": _sha256(library),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m ultraquant.lexicon.build``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Import Open English WordNet into a library as its "
                    "dictionary")
    parser.add_argument("--home", required=True,
                        help="the library home; the lexicon goes in HOME/lexicon")
    parser.add_argument("--source",
                        help="the release zip (default: the pinned release in "
                             "ultraquant/lexicon/data/source.json)")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.home, args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
