"""A dictionary. The gate.

The user asked "Give it a Dictionary" and chose Open English WordNet (2025+
edition, CC BY 4.0 over the WordNet License). The library knew facts about
subjects but not the words themselves: what "scandium" means, that it is a
metallic element, that "atomic weight" is "atomic mass". The release is
imported into the library as a lexicon of its own - hash-addressed word and
synset pages in one packed library file, read a page at a time - and the chat
answers word questions from it. It never writes a fact.

**The criteria, written before any code** - frozen in a pre-registration
(sha256 ad34f6dd...) before the implementation existed; Amendment A (sha256
9d1219a6..., POST-RUN): the release keys homographs "n-1"/"n-2", and the part of
speech is the key before the dash. Run 1, as frozen, FAILED criterion 1 on
exactly those 51 forms and nothing else. The world: a copy of
the library with its lexicon built from the tracked release.
1. **Faithful**: every normalized form's senses, and every synset's record
   (every field, plus its lexicographer file), are exactly the release's; the
   inverse links (hyponym, instance) are exactly the inverse of hypernym and
   instance_hypernym, sorted.
2. **Paged, not loaded**: opening reads no page; ``define`` of a form reads
   exactly its word page and its senses' synset pages (1,000 random forms,
   each on a freshly opened lexicon).
3. **The chat answers from it**: "What does W mean?", "Define W" and
   ":define W" reply exactly the release's definitions for W with the
   attribution (200 random forms); "what does this word mean" and "what is the
   meaning of this word" reply as they did without a lexicon; so do 40 fact
   questions from the world's structured facts.
4. **No fact written**: the build leaves every file outside ``lexicon/``
   byte-identical; 200 word questions leave every fact, every stash entry and
   ``approvals.jsonl`` unchanged.
5. **Only the approved file**: a copy of the release with one byte changed is
   refused, and nothing is written.
6. **The exam can fail**: P115 (instance_hypernym not stored) breaches 1;
   P116 (every page read on open) breaches 2; P117 (the chat's lexicon
   missing) breaches 3; P118 (the build writes a definition as a fact)
   breaches 4.
7. **Nothing regresses**: a sweep matches the ledger. Checked outside this
   module.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import random
import re
import shutil
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

__all__ = ["DictionaryReport", "run_gate"]

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data" / "wordnet" / "english-wordnet-2025-plus-json.zip"
SHA256 = "8832b8fa26a14c0ba8c99bb1ef8db6f9e122a6d9193b65ca9e0ef572580fee7e"
PAGES = 512
POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
_RUN: dict = {}


def _normalize_word(text: str) -> str:
    """The brief's word normalization: case, accents and punctuation folded; no word dropped."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in bare).split())


def _page(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(), "big") % PAGES


def _build_module():
    # By module path: a package attribute named "build" must not shadow it.
    return importlib.import_module("ultraquant.lexicon.build")


def _lexicon_class():
    return importlib.import_module("ultraquant.lexicon.lexicon").Lexicon


def _word_page(form: str) -> str:
    return f"lexicon:word:{_page(form):03d}"


def _synset_page(synset_id: str) -> str:
    return f"lexicon:synset:{_page(synset_id):03d}"


# -- the release, read independently of the implementation ---------------------------------

def _release() -> dict:
    """forms -> senses and synset -> record, straight from the zip."""
    if "release" not in _RUN:
        forms: dict[str, list[dict]] = {}
        lemmas: dict[str, str] = {}
        synsets: dict[str, dict] = {}
        with zipfile.ZipFile(SOURCE) as archive:
            names = archive.namelist()
            for name in names:
                data = json.loads(archive.read(name))
                if name.startswith("entries-"):
                    for lemma, by_pos in data.items():
                        form = _normalize_word(lemma)
                        lemmas.setdefault(form, lemma)
                        for pos, entry in by_pos.items():
                            for sense in entry.get("sense", []):
                                # Amendment A (post-run): "n-1"/"n-2" key two words
                                # spelt alike; the part of speech precedes the dash.
                                forms.setdefault(form, []).append(
                                    {"lemma": lemma, "pos": pos.split("-", 1)[0],
                                     "synset": sense["synset"], "sense_key": sense["id"]})
                elif name != "frames.json":
                    for synset_id, record in data.items():
                        synsets[synset_id] = {**record, "lexfile": name[:-len(".json")]}
        hyponyms: dict[str, list[str]] = {}
        instances: dict[str, list[str]] = {}
        for synset_id, record in synsets.items():
            for parent in record.get("hypernym", []):
                hyponyms.setdefault(parent, []).append(synset_id)
            for parent in record.get("instance_hypernym", []):
                instances.setdefault(parent, []).append(synset_id)
        for synset_id, children in hyponyms.items():
            synsets[synset_id]["hyponym"] = sorted(children)
        for synset_id, members in instances.items():
            synsets[synset_id]["instance"] = sorted(members)
        _RUN["release"] = {"forms": forms, "lemmas": lemmas, "synsets": synsets}
    return _RUN["release"]


def _reply(word: str, form: str) -> str:
    """The exact reply the brief fixes for a word question."""
    release = _release()
    lines = [f"{word}:"]
    for number, sense in enumerate(release["forms"][form], 1):
        definition = "; ".join(release["synsets"][sense["synset"]]["definition"])
        lines.append(f"{number}. ({POS[sense['pos']]}) {definition}")
    lines.append(_about()["attribution_line"])
    return "\n".join(lines)


def _about() -> dict:
    return json.loads((REPO / "ultraquant" / "lexicon" / "data" / "source.json")
                      .read_text(encoding="utf-8"))


_ASKABLE = re.compile(r"[A-Za-z][A-Za-z '\-]*[A-Za-z]")


def _sample(n: int, seed: int) -> list[tuple[str, str]]:
    """(word as written, form) pairs a user could type in a question."""
    release = _release()
    candidates = sorted(form for form, lemma in release["lemmas"].items()
                        if form in release["forms"] and _ASKABLE.fullmatch(lemma)
                        and " mean" not in f" {lemma.lower()}")
    rng = random.Random(seed)
    return [(release["lemmas"][form], form) for form in rng.sample(candidates, n)]


# -- the world -----------------------------------------------------------------------------

@contextlib.contextmanager
def _home(with_lexicon: bool = True):
    """A copy of the library, with its lexicon built from the tracked release."""
    from ultraquant.experiments.catalogue_gate import LIVE_HOME
    scratch = Path(tempfile.mkdtemp(prefix="uq_dictionary_"))
    try:
        home = scratch / "uq_home"
        shutil.copytree(LIVE_HOME, home, ignore=shutil.ignore_patterns("lexicon"))
        if with_lexicon:
            _build_module().build(home)
        yield home
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _hashes(root: Path, skip: str = "lexicon") -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).parts[0] != skip}


def _session(home: Path):
    from ultraquant.interpreter.thoughts import build_session
    return build_session(home, budget_bytes=1024 * 1024, seed=0, semantic=False,
                         auto_approve=False)


def _ask(session, question: str) -> str:
    from ultraquant.interpreter.thoughts import run_pipeline
    reply, _trace = run_pipeline(question, session)
    return reply


def _command(session, line: str) -> str:
    from ultraquant.interpreter.chat import ChatCLI
    out = io.StringIO()
    ChatCLI(session, out=out).handle(line)
    return out.getvalue().rstrip("\n")


def _facts(home: Path) -> dict:
    from ultraquant.memory.factshards import FactShards
    from ultraquant.memory.systematic import SystematicMemory
    from ultraquant.shards.vault import ShardVault
    memory = SystematicMemory(path=home / "memory.json")
    if (home / "vault").exists():
        memory.shards = FactShards(ShardVault(home / "vault"))
    out = {}
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        out[key] = tuple(record.get(k) for k in ("value", "confidence", "subject", "attribute"))
    return out


def _stash(home: Path) -> str:
    path = home / "stash.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _approvals(home: Path) -> str:
    path = home / "approvals.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


# -- 1: faithful ---------------------------------------------------------------------------

def faithful() -> bool:
    Lexicon = _lexicon_class()
    release = _release()
    with _home() as home:
        lexicon = Lexicon(home)
        wrong_forms = [form for form in sorted(release["forms"], key=_word_page)
                       if lexicon.senses(form) != release["forms"][form]]
        wrong_synsets = [sid for sid in sorted(release["synsets"], key=_synset_page)
                         if lexicon.synset(sid) != release["synsets"][sid]]
        about = lexicon.about()
    _RUN["faithful"] = {"forms": len(release["forms"]), "synsets": len(release["synsets"]),
                        "wrong forms": wrong_forms[:5], "wrong synsets": wrong_synsets[:5],
                        "about counts": {k: about.get(k) for k in ("forms", "synsets")}}
    return (not wrong_forms and not wrong_synsets and len(release["forms"]) == 151779
            and len(release["synsets"]) == 120564
            and about.get("forms") == len(release["forms"])
            and about.get("synsets") == len(release["synsets"])
            and about.get("sha256") == SHA256)


# -- 2: paged, not loaded -----------------------------------------------------------------

def paged() -> bool:
    from ultraquant.shards.vault import ShardVault
    Lexicon = _lexicon_class()
    release = _release()
    forms = [form for _word, form in _sample(1000, seed=167)]
    wrong = []
    with _home() as home:
        reads: list[str] = []
        real = ShardVault.load_bytes

        def counting(self, shard_id):
            reads.append(shard_id)
            return real(self, shard_id)
        with mock.patch.object(ShardVault, "load_bytes", counting):
            Lexicon(home)
            opened = list(reads)
            for form in forms:
                reads.clear()
                Lexicon(home).define(form)
                want = {_word_page(form)} | {_synset_page(s["synset"])
                                             for s in release["forms"][form]}
                if set(reads) != want or len(reads) != len(want):
                    wrong.append((form, sorted(set(reads) ^ want)[:3], len(reads)))
    _RUN["paged"] = {"read on open": len(opened), "wrong": wrong[:5], "forms": len(forms)}
    return not opened and not wrong


# -- 3: the chat answers from it ----------------------------------------------------------

def chat_answers() -> bool:
    sample = _sample(200, seed=1670)
    wrong, unchanged = [], []
    with _home() as home:
        session = _session(home)
        for word, form in sample:
            want = _reply(word, form)
            for got in (_ask(session, f"What does {word} mean?"), _ask(session, f"Define {word}"),
                        _command(session, f":define {word}")):
                if got != want:
                    wrong.append((word, got[:160]))
                    break
    # Unchanged without a word to define, and for fact questions.
    questions = ["what does this word mean", "what is the meaning of this word"] + _fact_questions(40)
    replies = {}
    for with_lexicon in (True, False):
        with _home(with_lexicon=with_lexicon) as home:
            session = _session(home)
            replies[with_lexicon] = [_ask(session, q) for q in questions]
    unchanged = [q for q, a, b in zip(questions, replies[True], replies[False]) if a != b]
    _RUN["chat"] = {"wrong": wrong[:5], "changed": unchanged[:5], "asked": len(sample),
                    "fact questions": len(questions) - 2}
    return not wrong and not unchanged and len(questions) == 42


def _fact_questions(n: int) -> list[str]:
    with _home(with_lexicon=False) as home:
        facts = _facts(home)
    structured = sorted(k for k, (_v, _c, subject, attribute) in facts.items() if subject and attribute)
    rng = random.Random(1671)
    chosen = rng.sample(structured, min(n, len(structured)))
    return [f"What is the {facts[k][3]} of {facts[k][2]}?" for k in chosen]


# -- 4: no fact written --------------------------------------------------------------------

def no_fact_written() -> bool:
    with _home(with_lexicon=False) as home:
        before = _hashes(home)
        _build_module().build(home)
        after_build = _hashes(home)
        facts, stash, approvals = _facts(home), _stash(home), _approvals(home)
        session = _session(home)
        for word, _form in _sample(200, seed=1672):
            _ask(session, f"What does {word} mean?")
        session.save()
        changed = {"files by the build": sorted(k for k in set(before) | set(after_build)
                                                if before.get(k) != after_build.get(k))[:5],
                   "facts": sorted(k for k in set(facts) | set(_facts(home))
                                   if facts.get(k) != _facts(home).get(k))[:5],
                   "stash": _stash(home) != stash, "approvals": _approvals(home) != approvals}
    _RUN["no fact"] = changed
    return not any(changed.values())


# -- 5: only the approved file ------------------------------------------------------------

def only_the_approved_file() -> bool:
    build = _build_module()
    with _home(with_lexicon=False) as home:
        bad = home.parent / "tampered.zip"
        data = bytearray(SOURCE.read_bytes())
        data[len(data) // 2] ^= 0x01
        bad.write_bytes(bytes(data))
        before = _hashes(home, skip="")
        refused = False
        try:
            build.build(home, source=bad)
        except ValueError:
            refused = True
        after = _hashes(home, skip="")
        no_lexicon = not (home / "lexicon").exists()
        good_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA256
    _RUN["approved"] = {"refused": refused, "wrote": sorted(set(after) - set(before))[:5],
                        "lexicon left": not no_lexicon}
    return refused and before == after and no_lexicon and good_sha


CASES = {"1 faithful": faithful, "2 paged": paged, "3 chat": chat_answers,
         "4 no fact": no_fact_written, "5 approved file": only_the_approved_file}


# -- plants --------------------------------------------------------------------------------

def _plants():
    B = _build_module()
    L = importlib.import_module("ultraquant.lexicon.lexicon")

    real_read = B.read_release

    def no_instances(*args, **kwargs):
        forms, synsets = real_read(*args, **kwargs)
        return forms, {sid: {k: v for k, v in record.items() if k != "instance_hypernym"}
                       for sid, record in synsets.items()}

    real_init = L.Lexicon.__init__

    def eager(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        for entry in self.vault.catalog():
            self.vault.get(entry["shard_id"])

    real_build = B.build

    def build_and_file(home, *args, **kwargs):
        result = real_build(home, *args, **kwargs)
        from ultraquant.memory.systematic import SystematicMemory
        from ultraquant.memory.factshards import FactShards
        from ultraquant.shards.vault import ShardVault
        memory = SystematicMemory(path=Path(home) / "memory.json")
        if (Path(home) / "vault").exists():
            memory.shards = FactShards(ShardVault(Path(home) / "vault"))
        memory.remember_fact("meaning of scandium", "a white trivalent metallic element", 0.9,
                             subject="scandium", attribute="meaning")
        memory.save()
        return result

    return [
        ("P115 instance_hypernym not stored", "1 faithful",
         [mock.patch.object(B, "read_release", no_instances)]),
        ("P116 every page read on open", "2 paged",
         [mock.patch.object(L.Lexicon, "__init__", eager)]),
        ("P117 the chat's lexicon missing", "3 chat",
         [mock.patch.object(L.Lexicon, "open", staticmethod(lambda home: None))]),
        ("P118 the build files a definition as a fact", "4 no fact",
         [mock.patch.object(B, "build", build_and_file)]),
    ]


@dataclass
class DictionaryReport:
    passes: bool
    cases: dict = field(default_factory=dict)
    planted: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    reason: str = ""


def run_gate() -> DictionaryReport:
    report = DictionaryReport(passes=False)
    for name, case in CASES.items():
        try:
            report.cases[name] = bool(case())
        except Exception as exc:        # a crash is a failure
            report.cases[name] = False
            report.errors[name] = repr(exc)
    report.measured = {k: _RUN.get(k) for k in ("faithful", "paged", "chat", "no fact", "approved")}
    release = _RUN.get("release")
    for name, target, patches in _plants():
        _RUN.clear()
        if release is not None:
            _RUN["release"] = release
        try:
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                outcome = bool(CASES[target]())
            report.planted[name] = outcome is False
        except Exception as exc:        # a crash proves nothing
            report.planted[name] = False
            report.errors[name] = repr(exc)
    valid = len(report.planted) == 4 and all(report.planted.values())
    met = len(report.cases) == 5 and all(report.cases.values())
    report.passes = valid and met
    if not valid:
        report.reason = f"VOID: plants not caught: {[n for n, c in report.planted.items() if not c]}"
    elif not met:
        report.reason = f"FAIL: {[n for n, ok in report.cases.items() if not ok]}"
    else:
        report.reason = "PASS: 5 cases; 4 of 4 plants caught"
    return report


def main() -> int:
    report = run_gate()
    print(report.reason)
    print(json.dumps({"cases": report.cases, "planted": report.planted,
                      "measured": report.measured, "errors": report.errors},
                     indent=1, default=str, ensure_ascii=False))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
