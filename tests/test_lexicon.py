"""The dictionary (§11.167): Open English WordNet as a paged lexicon.

Every test runs on a tiny release written here, never the real 11 MB zip. A
build writes 1,025 fsynced pages, which costs seconds even for a tiny release,
so the release is built once for the module and the built home is copied
wherever a test needs one of its own. A copy is also exactly the relocation
case a lexicon has to survive.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import types
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import ultraquant.lexicon
import ultraquant.lexicon.build as build_module
from ultraquant.interpreter.chat import HELP, ChatCLI
from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.lexicon import Lexicon, normalize_word
from ultraquant.lexicon.build import build, read_release
from ultraquant.lexicon.lexicon import (
    CACHE_PAGES, SOURCE_PATH, page_of, synset_page, word_page,
)
from ultraquant.memory.factshards import normalize_subject
from ultraquant.shards.vault import ShardVault

CREDIT = ("Source: Open English WordNet 2025+ (CC BY 4.0), derived from "
          "Princeton WordNet (WordNet License).")

ENTITY = "00001740-n"
CITY = "08524735-n"
CAPITAL_CITY = "08518505-n"
HAGUE = "08970180-n"
AMSTERDAM = "08950407-n"
ASSETS = "13356402-n"
FIRST_RATE = "02226162-s"
LETTER_A = "06840000-n"
ANGSTROM = "13653000-n"
ELAN = "07522000-n"
MILNE = "11174624-n"
WRITER = "10794014-n"
BASS_RANGE = "04994045-n"
BASS_FISH = "07793921-n"
WORD = "06286395-n"
FAST_N = "07000001-n"
FAST_V = "01200001-v"
FAST_A = "00976508-a"
FAST_S = "01446000-s"
FAST_R = "00085811-r"

#: A release in miniature, in the real one's shape and archive order: an
#: adjective file, the entries files, frames.json, then the noun files with
#: noun.Tops last. Ids are listed out of sorted order where order is tested.
RELEASE_MEMBERS: list[tuple[str, dict]] = [
    ("adj.all.json", {
        FIRST_RATE: {"definition": ["first-rate"],
                     "members": ["capital", "Capital", "excellent"],
                     "partOfSpeech": "s", "ili": "i11002"},
        FAST_A: {"definition": ["acting or moving or capable of acting or "
                                "moving quickly"],
                 "members": ["fast"], "partOfSpeech": "a"},
        FAST_S: {"definition": ["securely fixed in place"],
                 "members": ["fast", "firm"], "partOfSpeech": "s"},
    }),
    ("adv.all.json", {
        FAST_R: {"definition": ["quickly or rapidly"],
                 "members": ["fast"], "partOfSpeech": "r"},
    }),
    ("entries-0.json", {
        "élan": {"n": {"pronunciation": [{"value": "eɪˈlɑːn"}],
                       "sense": [{"id": "élan%1:07:00::", "synset": ELAN}]}},
    }),
    ("entries-a.json", {
        "A": {"n": {"sense": [{"id": "a%1:23:00::", "synset": ANGSTROM}]}},
        "a": {"n": {"sense": [{"id": "a%1:10:00::", "synset": LETTER_A}]}},
        "A. A. Milne": {"n": {"sense": [{"id": "milne%1:18:00::",
                                         "synset": MILNE}]}},
        "Amsterdam": {"n": {"sense": [{"id": "amsterdam%1:15:00::",
                                       "synset": AMSTERDAM}]}},
        "angstrom": {"n": {"sense": [{"id": "angstrom%1:23:00::",
                                      "synset": ANGSTROM}]}},
    }),
    ("entries-b.json", {
        "bass": {"n-1": {"sense": [{"id": "bass%1:07:01::",
                                    "synset": BASS_RANGE}]},
                 "n-2": {"sense": [{"id": "bass%1:13:02::",
                                    "synset": BASS_FISH}]}},
    }),
    ("entries-c.json", {
        "capital": {
            "a": {"sense": [{"id": "capital%5:00:00:first-rate:00",
                             "synset": FIRST_RATE}]},
            "n": {"sense": [{"id": "capital%1:21:01::", "synset": ASSETS},
                            {"id": "capital%1:15:00::",
                             "synset": CAPITAL_CITY}]}},
        "city": {"n": {"sense": [{"id": "city%1:15:00::", "synset": CITY}]}},
    }),
    ("entries-e.json", {
        "elan": {"n": {"sense": [{"id": "elan%1:07:00::", "synset": ELAN}]}},
        "entity": {"n": {"sense": [{"id": "entity%1:03:00::",
                                    "synset": ENTITY}]}},
    }),
    ("entries-f.json", {
        "fast": {"n": {"sense": [{"id": "fast%1:04:00::", "synset": FAST_N}]},
                 "v": {"sense": [{"id": "fast%2:34:00::", "synset": FAST_V}]},
                 "a": {"sense": [{"id": "fast%3:00:01::", "synset": FAST_A}]},
                 "s": {"sense": [{"id": "fast%5:00:00:fixed:00",
                                  "synset": FAST_S}]},
                 "r": {"sense": [{"id": "fast%4:02:00::",
                                  "synset": FAST_R}]}},
    }),
    ("entries-t.json", {
        "The Hague": {"n": {"sense": [{"id": "the_hague%1:15:00::",
                                       "synset": HAGUE}]}},
    }),
    ("entries-w.json", {
        "word": {"n": {"sense": [{"id": "word%1:10:00::", "synset": WORD}]}},
        "working capital": {"n": {"sense": [{"id": "working_capital%1:21:00::",
                                             "synset": ASSETS}]}},
        "writer": {"n": {"sense": [{"id": "writer%1:18:00::",
                                    "synset": WRITER}]}},
    }),
    ("frames.json", {"1": "Something ----s", "2": "Somebody ----s"}),
    ("noun.act.json", {
        FAST_N: {"definition": ["abstaining from food"],
                 "members": ["fast", "fasting"], "partOfSpeech": "n",
                 "hypernym": [ENTITY]},
    }),
    ("noun.attribute.json", {
        BASS_RANGE: {"definition": ["the lowest part of the musical range"],
                     "members": ["bass"], "partOfSpeech": "n",
                     "hypernym": [ENTITY]},
    }),
    ("noun.communication.json", {
        LETTER_A: {"definition": ["the 1st letter of the Roman alphabet"],
                   "members": ["A", "a"], "partOfSpeech": "n",
                   "hypernym": [ENTITY]},
        WORD: {"definition": ["a unit of language that native speakers can "
                              "identify"],
               "members": ["word"], "partOfSpeech": "n",
               "hypernym": [ENTITY]},
    }),
    ("noun.feeling.json", {
        ELAN: {"definition": ["enthusiastic and assured vigor and liveliness"],
               "members": ["élan", "dash", "elan"], "partOfSpeech": "n",
               "hypernym": [ENTITY]},
    }),
    ("noun.food.json", {
        BASS_FISH: {"definition": ["the lean flesh of a saltwater fish"],
                    "members": ["bass"], "partOfSpeech": "n",
                    "hypernym": [ENTITY]},
    }),
    ("noun.location.json", {
        HAGUE: {"definition": ["the seat of government of the Netherlands"],
                "members": ["The Hague", "'s Gravenhage", "Den Haag"],
                "partOfSpeech": "n", "instance_hypernym": [CITY],
                "wikidata": "Q36600"},
        AMSTERDAM: {"definition": ["the capital and largest city of the "
                                   "Netherlands"],
                    "members": ["Amsterdam", "Dutch capital"],
                    "partOfSpeech": "n", "instance_hypernym": [CITY]},
        CITY: {"definition": ["a large and densely populated urban area",
                              "may include several independent "
                              "administrative districts"],
               "members": ["city", "metropolis", "urban center"],
               "partOfSpeech": "n", "hypernym": [ENTITY],
               "example": ["Ancient Troy was a great city"]},
        CAPITAL_CITY: {"definition": ["a seat of government"],
                       "members": ["capital"], "partOfSpeech": "n",
                       "hypernym": [CITY]},
    }),
    ("noun.person.json", {
        MILNE: {"definition": ["English writer of stories for children"],
                "members": ["Milne", "A. A. Milne", "Alan Alexander Milne"],
                "partOfSpeech": "n", "instance_hypernym": [WRITER]},
        WRITER: {"definition": ["writes books or stories professionally"],
                 "members": ["writer", "author"], "partOfSpeech": "n",
                 "hypernym": [ENTITY]},
    }),
    ("noun.possession.json", {
        ASSETS: {"definition": ["assets available for use in the production "
                                "of further assets"],
                 "members": ["capital", "working capital"],
                 "partOfSpeech": "n", "hypernym": [ENTITY]},
    }),
    ("noun.quantity.json", {
        ANGSTROM: {"definition": ["a metric unit of length equal to one ten "
                                  "billionth of a meter"],
                   "members": ["angstrom", "angstrom unit", "A"],
                   "partOfSpeech": "n", "hypernym": [ENTITY]},
    }),
    ("noun.Tops.json", {
        ENTITY: {"definition": ["that which is perceived or known or "
                                "inferred to have its own distinct existence"],
                 "members": ["entity"], "partOfSpeech": "n"},
    }),
    ("verb.consumption.json", {
        FAST_V: {"definition": ["abstain from certain foods"],
                 "members": ["fast"], "partOfSpeech": "v"},
    }),
]

#: What every sense of "capital" defines, in release order.
CAPITAL = [("adjective", "first-rate"),
           ("noun", "assets available for use in the production of further "
                    "assets"),
           ("noun", "a seat of government")]

_DIR = Path()
RELEASE = Path()
SHA = ""
TEMPLATE = Path()
RESULT: dict = {}


def write_release(path: Path, members: list[tuple[str, dict]]) -> str:
    """Write ``members`` into a zip in the order given; return its sha256."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, json.dumps(data, ensure_ascii=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setUpModule() -> None:
    global _DIR, RELEASE, SHA, TEMPLATE, RESULT
    _DIR = Path(tempfile.mkdtemp(prefix="uq_lexicon_"))
    RELEASE = _DIR / "release.zip"
    SHA = write_release(RELEASE, RELEASE_MEMBERS)
    TEMPLATE = _DIR / "template"
    RESULT = build(TEMPLATE, RELEASE, expected_sha256=SHA)


def tearDownModule() -> None:
    shutil.rmtree(_DIR, ignore_errors=True)


def expected(word: str, *senses: tuple[str, str]) -> str:
    """The dictionary answer's exact text, written out independently."""
    lines = [f"{word}:"]
    lines += [f"{number}. ({pos}) {text}"
              for number, (pos, text) in enumerate(senses, start=1)]
    return "\n".join(lines + [CREDIT])


@contextlib.contextmanager
def counting_reads():
    """Record every stored page read, through the one method that reads."""
    reads: list[str] = []
    original = ShardVault.load_bytes

    def counting(vault, shard_id):
        reads.append(shard_id)
        return original(vault, shard_id)

    with mock.patch.object(ShardVault, "load_bytes", counting):
        yield reads


def snapshot(root: Path, skip: str | None = None) -> dict[str, tuple]:
    """Every file under ``root``, with its bytes and modification time."""
    out = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if skip is not None and relative.parts[0] == skip:
            continue
        if path.is_file():
            out[relative.as_posix()] = (path.read_bytes(),
                                        path.stat().st_mtime_ns)
    return out


class _Temp(unittest.TestCase):
    """A scratch directory per test, and copies of the built home."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_lexicon_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def copy_home(self, name: str = "home") -> Path:
        home = self.dir / name
        shutil.copytree(TEMPLATE, home)
        return home


class NormalizeWordTests(unittest.TestCase):
    """Case, accents and punctuation fold; no word is ever dropped."""

    def test_a_leading_article_is_a_word(self) -> None:
        self.assertEqual(normalize_word("A"), "a")
        self.assertEqual(normalize_word("The Hague"), "the hague")
        # The reason it is not normalize_subject.
        self.assertEqual(normalize_subject("A"), "")

    def test_initials_become_words(self) -> None:
        self.assertEqual(normalize_word("A. A. Milne"), "a a milne")

    def test_accents_case_and_punctuation_fold(self) -> None:
        self.assertEqual(normalize_word("Élan"), "elan")
        self.assertEqual(normalize_word("café"), "cafe")
        self.assertEqual(normalize_word("Ångström"), "angstrom")
        self.assertEqual(normalize_word("  Rock-and-Roll! "), "rock and roll")
        self.assertEqual(normalize_word("?!"), "")


class ReadReleaseTests(unittest.TestCase):
    """Senses in release order; records as released, plus their lexfile."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.forms, cls.synsets = read_release(RELEASE)

    def test_senses_come_in_release_order(self) -> None:
        # JSON object order: "A" is written before "a".
        self.assertEqual(self.forms["a"], [
            {"lemma": "A", "pos": "n", "synset": ANGSTROM,
             "sense_key": "a%1:23:00::"},
            {"lemma": "a", "pos": "n", "synset": LETTER_A,
             "sense_key": "a%1:10:00::"},
        ])
        # Part-of-speech order as written, then sense list order.
        self.assertEqual(
            [(sense["pos"], sense["synset"]) for sense in self.forms["capital"]],
            [("a", FIRST_RATE), ("n", ASSETS), ("n", CAPITAL_CITY)])
        # Zip name order: entries-0.json comes before entries-e.json.
        self.assertEqual([sense["lemma"] for sense in self.forms["elan"]],
                         ["élan", "elan"])

    def test_a_homograph_key_is_its_part_of_speech(self) -> None:
        self.assertEqual(
            [(sense["pos"], sense["synset"]) for sense in self.forms["bass"]],
            [("n", BASS_RANGE), ("n", BASS_FISH)])

    def test_records_are_unchanged_plus_their_lexfile(self) -> None:
        for name, data in RELEASE_MEMBERS:
            if name.startswith("entries-") or name == "frames.json":
                continue
            for synset_id, record in data.items():
                with self.subTest(synset=synset_id):
                    self.assertEqual(self.synsets[synset_id],
                                     {**record, "lexfile": name[:-5]})
        self.assertEqual(self.synsets[ENTITY]["lexfile"], "noun.Tops")

    def test_frames_are_skipped_and_nothing_is_inverted(self) -> None:
        self.assertEqual(len(self.synsets), 20)
        self.assertNotIn("1", self.synsets)
        for record in self.synsets.values():
            self.assertNotIn("hyponym", record)
            self.assertNotIn("instance", record)

    def test_the_forms(self) -> None:
        self.assertEqual(sorted(self.forms), [
            "a", "a a milne", "amsterdam", "angstrom", "bass", "capital",
            "city", "elan", "entity", "fast", "the hague", "word",
            "working capital", "writer"])
        self.assertEqual(sum(len(senses) for senses in self.forms.values()), 23)


class BuiltLibraryTests(unittest.TestCase):
    """What one build writes: the pages, the about page, the inverse links."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vault = ShardVault(TEMPLATE / "lexicon")

    def record(self, synset_id: str) -> dict:
        return self.vault.get(synset_page(synset_id))["synsets"][synset_id]

    def test_the_result(self) -> None:
        library = TEMPLATE / "lexicon" / "wordnet.uql"
        self.assertEqual(RESULT, {
            "forms": 14, "synsets": 20, "senses": 23,
            "pages": {"word": 512, "synset": 512},
            "bytes": library.stat().st_size,
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        })

    def test_every_page_is_packed_in_one_library(self) -> None:
        entries = {entry["shard_id"]: entry for entry in self.vault.catalog()}
        self.assertEqual(set(entries), (
            {f"lexicon:word:{number:03d}" for number in range(512)}
            | {f"lexicon:synset:{number:03d}" for number in range(512)}
            | {"lexicon:about"}))
        library = (TEMPLATE / "lexicon" / "wordnet.uql").resolve()
        for shard_id, entry in entries.items():
            kind = ("lexicon-about" if shard_id == "lexicon:about"
                    else "lexicon-" + shard_id.split(":")[1])
            self.assertEqual(entry["kind"], kind)
            self.assertEqual(entry["category"], shard_id)
            self.assertEqual(entry["associations"], {})
            self.assertEqual(entry["location"], "library")
            self.assertEqual(Path(entry["library_path"]).resolve(), library)

    def test_every_form_and_synset_is_on_its_hashed_page(self) -> None:
        words: dict[str, list] = {}
        synsets: dict[str, dict] = {}
        for number in range(512):
            page = self.vault.get(f"lexicon:word:{number:03d}")
            for form, rows in page["words"].items():
                self.assertEqual(page_of(form), number)
                words[form] = rows
            page = self.vault.get(f"lexicon:synset:{number:03d}")
            for synset_id, record in page["synsets"].items():
                self.assertEqual(page_of(synset_id), number)
                synsets[synset_id] = record
        self.assertEqual(len(words), 14)
        self.assertEqual(len(synsets), 20)
        self.assertEqual(words["capital"], [
            ["capital", "a", FIRST_RATE, "capital%5:00:00:first-rate:00"],
            ["capital", "n", ASSETS, "capital%1:21:01::"],
            ["capital", "n", CAPITAL_CITY, "capital%1:15:00::"],
        ])
        released = dict(RELEASE_MEMBERS)["noun.location.json"][HAGUE]
        self.assertEqual(synsets[HAGUE], {**released,
                                          "lexfile": "noun.location"})

    def test_inverse_links_are_sorted_and_only_where_something_points(self):
        # The Hague is written before Amsterdam; the list is sorted anyway.
        self.assertEqual(self.record(CITY)["instance"], [AMSTERDAM, HAGUE])
        self.assertEqual(self.record(CITY)["hyponym"], [CAPITAL_CITY])
        self.assertEqual(self.record(ENTITY)["hyponym"], sorted([
            BASS_RANGE, LETTER_A, ELAN, BASS_FISH, CITY, WRITER, ASSETS,
            ANGSTROM, WORD, FAST_N]))
        self.assertEqual(self.record(WRITER)["instance"], [MILNE])
        self.assertNotIn("instance", self.record(ENTITY))
        self.assertNotIn("hyponym", self.record(WRITER))
        for leaf in (CAPITAL_CITY, HAGUE, FIRST_RATE, MILNE):
            self.assertNotIn("hyponym", self.record(leaf))
            self.assertNotIn("instance", self.record(leaf))

    def test_the_about_page(self) -> None:
        about = self.vault.get("lexicon:about")["about"]
        pinned = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        built = about.pop("built")
        self.assertEqual(about, {**pinned, "forms": 14, "synsets": 20,
                                 "senses": 23})
        self.assertIsNotNone(datetime.fromisoformat(built).tzinfo)

    def test_the_build_leaves_nothing_but_the_lexicon(self) -> None:
        self.assertEqual([path.name for path in TEMPLATE.iterdir()],
                         ["lexicon"])
        self.assertEqual(
            sorted(path.name for path in (TEMPLATE / "lexicon").iterdir()),
            ["catalog.json", "loose", "wordnet.uql"])
        self.assertEqual(list((TEMPLATE / "lexicon" / "loose").iterdir()), [])


class PinnedSourceTests(_Temp):
    """source.json pins the release, and build checks it before writing."""

    def test_source_json_pins_the_release(self) -> None:
        pinned = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(list(pinned), [
            "file", "path", "bytes", "sha256", "edition", "release",
            "published", "url", "license", "attribution", "attribution_line"])
        self.assertEqual(pinned["path"],
                         "data/wordnet/english-wordnet-2025-plus-json.zip")
        self.assertEqual(pinned["bytes"], 11298794)
        self.assertEqual(pinned["sha256"], "8832b8fa26a14c0ba8c99bb1ef8db6f9"
                                           "e122a6d9193b65ca9e0ef572580fee7e")
        self.assertEqual(pinned["attribution_line"], CREDIT)
        release = build_module.REPO_ROOT / pinned["path"]
        if release.exists():  # its size is checked; it is never read here
            self.assertEqual(release.stat().st_size, pinned["bytes"])

    def test_the_default_source_and_sha_are_the_pinned_ones(self) -> None:
        pinned = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        hashed = []

        def fake_sha(path):
            hashed.append(Path(path))
            return "0" * 64

        home = self.dir / "home"
        with mock.patch.object(build_module, "_sha256", fake_sha):
            with self.assertRaises(ValueError) as caught:
                build(home)
        self.assertEqual(hashed, [build_module.REPO_ROOT / pinned["path"]])
        self.assertIn(pinned["sha256"], str(caught.exception))
        self.assertFalse(home.exists())

    def test_a_wrong_sha_writes_nothing(self) -> None:
        home = self.dir / "fresh"
        with self.assertRaises(ValueError):
            build(home, RELEASE, expected_sha256="0" * 64)
        self.assertFalse(home.exists())
        # Without an expected sha, the pinned release's is expected.
        with self.assertRaises(ValueError):
            build(home, RELEASE)
        self.assertFalse(home.exists())

    def test_a_wrong_sha_leaves_the_old_lexicon_alone(self) -> None:
        home = self.copy_home()
        before = snapshot(home)
        with self.assertRaises(ValueError):
            build(home, RELEASE, expected_sha256="0" * 64)
        self.assertEqual(snapshot(home), before)
        self.assertEqual([path.name for path in home.iterdir()], ["lexicon"])

    def test_the_release_is_read_through_the_module_name(self) -> None:
        read = []

        class Stop(Exception):
            pass

        def fake(path):
            read.append(Path(path))
            raise Stop

        home = self.dir / "home"
        with mock.patch.object(build_module, "read_release", fake):
            with self.assertRaises(Stop):
                build(home, RELEASE, expected_sha256=SHA)
        self.assertEqual(read, [RELEASE])
        self.assertFalse(home.exists(), "the release is read before writing")

    def test_the_command_line_prints_the_result_as_json(self) -> None:
        calls = []

        def fake_build(home, source=None, **options):
            calls.append((home, source, options))
            return {"forms": 1, "pages": {"word": 512, "synset": 512}}

        out = io.StringIO()
        with mock.patch.object(build_module, "build", fake_build), \
                contextlib.redirect_stdout(out):
            code = build_module.main(["--home", "somewhere",
                                      "--source", "release.zip"])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [("somewhere", "release.zip", {})])
        self.assertEqual(json.loads(out.getvalue()),
                         {"forms": 1, "pages": {"word": 512, "synset": 512}})

    def test_the_package_keeps_build_a_module(self) -> None:
        self.assertIs(ultraquant.lexicon.Lexicon, Lexicon)
        self.assertIs(ultraquant.lexicon.normalize_word, normalize_word)
        self.assertIsInstance(ultraquant.lexicon.build, types.ModuleType)
        self.assertIs(ultraquant.lexicon.build, build_module)


class RebuildTests(_Temp):
    """A rebuild replaces the lexicon whole and touches nothing else."""

    def test_a_rebuild_replaces_the_lexicon_and_touches_nothing_else(self):
        home = self.copy_home()
        (home / "memory.json").write_text('{"facts": {}}', encoding="utf-8")
        (home / "stash.json").write_text('{"entries": []}', encoding="utf-8")
        (home / "approvals.jsonl").write_text("", encoding="utf-8")
        (home / "vault").mkdir()
        (home / "vault" / "catalog.json").write_text('{"shards": []}',
                                                     encoding="utf-8")
        library = snapshot(home, skip="lexicon")
        second = self.dir / "second.zip"
        sha = write_release(second, [
            ("entries-c.json", {"capital": {"n": {"sense": [
                {"id": "capital%1:21:01::", "synset": ASSETS}]}}}),
            ("entries-z.json", {"zebra": {"n": {"sense": [
                {"id": "zebra%1:05:00::", "synset": "02391049-n"}]}}}),
            ("noun.animal.json", {"02391049-n": {
                "definition": ["any of several fleet black-and-white striped "
                               "African equines"],
                "members": ["zebra"], "partOfSpeech": "n"}}),
            ("noun.possession.json", {ASSETS: {
                "definition": ["wealth in the form of money or property"],
                "members": ["capital"], "partOfSpeech": "n"}}),
        ])
        roots: list[Path] = []

        class Recording(ShardVault):
            def __init__(self, root, *args, **kwargs):
                roots.append(Path(root))
                super().__init__(root, *args, **kwargs)

        with mock.patch.object(build_module, "ShardVault", Recording):
            result = build(home, second, expected_sha256=sha)

        self.assertEqual((result["forms"], result["synsets"],
                          result["senses"]), (2, 2, 2))
        lexicon = Lexicon.open(home)
        self.assertEqual(lexicon.senses("The Hague"), [])
        self.assertEqual(lexicon.define("capital"), [{
            "lemma": "capital", "pos": "n", "synset": ASSETS,
            "definition": ["wealth in the form of money or property"]}])
        self.assertEqual(lexicon.about()["forms"], 2)
        self.assertEqual([Path(path).resolve()
                          for path in lexicon.vault.libraries()],
                         [(home / "lexicon" / "wordnet.uql").resolve()])
        # No leftovers: no building directory, no old lexicon, no loose page.
        self.assertEqual(sorted(path.name for path in home.iterdir()), [
            "approvals.jsonl", "lexicon", "memory.json", "stash.json",
            "vault"])
        self.assertEqual(
            sorted(path.name for path in (home / "lexicon").iterdir()),
            ["catalog.json", "loose", "wordnet.uql"])
        self.assertEqual(list((home / "lexicon").rglob("*.uqs")), [])
        # The rest of the library was never opened, let alone changed.
        self.assertEqual(snapshot(home, skip="lexicon"), library)
        self.assertTrue(roots)
        for root in roots:
            self.assertEqual(root.parent, home.absolute())
            self.assertIn(root.name, ("lexicon", "lexicon.building"))


class LexiconTests(_Temp):
    """Lookups read their own pages, at most once each, and write nothing."""

    def test_open_is_none_without_a_lexicon_and_creates_nothing(self) -> None:
        self.assertIsNone(Lexicon.open(self.dir))
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_opening_reads_no_page_and_writes_nothing(self) -> None:
        catalog = TEMPLATE / "lexicon" / "catalog.json"
        before = catalog.read_bytes()
        with counting_reads() as reads, mock.patch.object(
                ShardVault, "attach", side_effect=AssertionError("attached")):
            lexicon = Lexicon.open(TEMPLATE)
        self.assertIsInstance(lexicon, Lexicon)
        self.assertEqual(reads, [])
        self.assertEqual(catalog.read_bytes(), before)

    def test_senses(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        self.assertEqual(lexicon.senses("A"), [
            {"lemma": "A", "pos": "n", "synset": ANGSTROM,
             "sense_key": "a%1:23:00::"},
            {"lemma": "a", "pos": "n", "synset": LETTER_A,
             "sense_key": "a%1:10:00::"},
        ])
        self.assertEqual(lexicon.senses("CAPITAL"), lexicon.senses("capital"))
        self.assertEqual([sense["lemma"] for sense in lexicon.senses("Élan")],
                         ["élan", "elan"])
        for unknown in ("xyzzy", "this word", "Hague", "?!", ""):
            with self.subTest(word=unknown):
                self.assertEqual(lexicon.senses(unknown), [])

    def test_define(self) -> None:
        self.assertEqual(Lexicon(TEMPLATE).define("capital"), [
            {"lemma": "capital", "pos": "a", "synset": FIRST_RATE,
             "definition": ["first-rate"]},
            {"lemma": "capital", "pos": "n", "synset": ASSETS,
             "definition": [CAPITAL[1][1]]},
            {"lemma": "capital", "pos": "n", "synset": CAPITAL_CITY,
             "definition": ["a seat of government"]},
        ])

    def test_define_reads_its_word_page_and_its_senses_synset_pages(self):
        for word in ("capital", "The Hague", "A", "bass"):
            with self.subTest(word=word):
                lexicon = Lexicon(TEMPLATE)
                with counting_reads() as reads:
                    items = lexicon.define(word)
                pages = ({word_page(normalize_word(word))}
                         | {synset_page(item["synset"]) for item in items})
                self.assertEqual(sorted(reads), sorted(pages))
        lexicon = Lexicon(TEMPLATE)
        with counting_reads() as reads:
            self.assertEqual(lexicon.define("xyzzy"), [])
        self.assertEqual(reads, [word_page("xyzzy")])

    def test_pages_stay_cached_between_lookups(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        lexicon.define("capital")
        with counting_reads() as reads:
            lexicon.define("capital")
            lexicon.senses("Capital")
        self.assertEqual(reads, [])

    def test_the_cache_is_bounded(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        for number in range(300):
            lexicon.senses(f"word {number}")
        self.assertEqual(len(lexicon._pages), CACHE_PAGES)

    def test_a_lookup_never_writes(self) -> None:
        catalog = TEMPLATE / "lexicon" / "catalog.json"
        before = catalog.read_bytes()
        lexicon = Lexicon(TEMPLATE)
        with mock.patch.object(ShardVault, "_write_catalog",
                               side_effect=AssertionError("catalog written")), \
                mock.patch.object(ShardVault, "flush_stats",
                                  side_effect=AssertionError("stats flushed")):
            for word in ("capital", "The Hague", "A. A. Milne", "xyzzy"):
                lexicon.senses(word)
                lexicon.define(word)
                lexicon.synonyms(word)
                lexicon.kinds(word)
            lexicon.synset(CITY)
            lexicon.about()
        self.assertEqual(catalog.read_bytes(), before)

    def test_a_synset_is_a_copy(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        record = lexicon.synset(CITY)
        self.assertEqual(record["lexfile"], "noun.location")
        self.assertEqual(record["instance"], [AMSTERDAM, HAGUE])
        record["definition"].append("scribbled on")
        self.assertEqual(len(lexicon.synset(CITY)["definition"]), 2)
        self.assertIsNone(lexicon.synset("99999999-n"))

    def test_synonyms(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        self.assertEqual(lexicon.synonyms("capital"),
                         ["excellent", "working capital"])
        self.assertEqual(lexicon.synonyms("The Hague"),
                         ["'s Gravenhage", "Den Haag"])
        self.assertEqual(lexicon.synonyms("a"), ["angstrom", "angstrom unit"])
        # Two senses in one synset: "dash" once; both spellings are the word.
        self.assertEqual(lexicon.synonyms("elan"), ["dash"])
        self.assertEqual(lexicon.synonyms("xyzzy"), [])

    def test_kinds(self) -> None:
        lexicon = Lexicon(TEMPLATE)
        self.assertEqual(lexicon.kinds("capital"),
                         [[], [ENTITY], [CITY, ENTITY]])
        # An instance climbs through instance_hypernym to its class.
        self.assertEqual(lexicon.kinds("The Hague"), [[CITY, ENTITY]])
        self.assertEqual(lexicon.kinds("A. A. Milne"), [[WRITER, ENTITY]])
        self.assertEqual(lexicon.kinds("entity"), [[]])
        self.assertEqual(lexicon.kinds("xyzzy"), [])

    def test_about(self) -> None:
        about = Lexicon(TEMPLATE).about()
        self.assertEqual(about["attribution_line"], CREDIT)
        self.assertEqual((about["forms"], about["synsets"], about["senses"]),
                         (14, 20, 23))

    def test_a_long_lookup_reads_each_page_once_past_the_cache(self) -> None:
        chain = [f"{number:08d}-n" for number in range(150)]
        records = {}
        for number, synset_id in enumerate(chain):
            records[synset_id] = {"definition": [f"level {number}"],
                                  "members": [f"level {number}"],
                                  "partOfSpeech": "n"}
            if number:
                records[synset_id]["hypernym"] = [chain[number - 1]]
        release = self.dir / "chain.zip"
        sha = write_release(release, [
            ("entries-d.json", {"deep": {"n": {"sense": [
                {"id": "deep%1", "synset": chain[-1]},
                {"id": "deep%2", "synset": chain[-2]}]}}}),
            ("noun.Tops.json", records),
        ])
        build(self.dir / "home", release, expected_sha256=sha)
        pages = {synset_page(synset_id) for synset_id in chain}
        self.assertGreater(len(pages), CACHE_PAGES)
        lexicon = Lexicon(self.dir / "home")
        with counting_reads() as reads:
            kinds = lexicon.kinds("deep")
        self.assertEqual(kinds, [chain[-2::-1], chain[-3::-1]])
        self.assertEqual(sorted(reads), sorted(pages | {word_page("deep")}))


class RelocationTests(_Temp):
    """A copied home re-points its lexicon at its own library, once."""

    def test_a_copied_home_reads_its_own_library(self) -> None:
        first = self.copy_home("first")
        Lexicon(first)  # the copy of the template relocates onto first
        moved = self.dir / "moved"
        shutil.copytree(first, moved)
        shutil.rmtree(first)
        attached: list[Path] = []
        original = ShardVault.attach

        def counting(vault, path):
            attached.append(Path(path))
            return original(vault, path)

        with mock.patch.object(ShardVault, "attach", counting), \
                counting_reads() as reads:
            lexicon = Lexicon.open(moved)
        self.assertEqual(reads, [])
        self.assertEqual(attached, [moved / "lexicon" / "wordnet.uql"])
        self.assertEqual(
            [Path(path).resolve() for path in lexicon.vault.libraries()],
            [(moved / "lexicon" / "wordnet.uql").resolve()])
        self.assertEqual(lexicon.define("The Hague")[0]["definition"],
                         ["the seat of government of the Netherlands"])
        # Re-pointed on disk: the next open has nothing to fix.
        with mock.patch.object(ShardVault, "attach",
                               side_effect=AssertionError("attached twice")):
            self.assertIsNotNone(Lexicon.open(moved))


class ChatTests(_Temp):
    """Word questions answered from the dictionary; everything else as before."""

    def setUp(self) -> None:
        super().setUp()
        self.home = self.copy_home()
        self.session = build_session(self.home, seed=0)

    def ask(self, text: str, session=None) -> str:
        return run_pipeline(text, session or self.session)[0]

    def bare(self):
        return build_session(self.dir / "bare", seed=0)

    def test_the_three_questions_give_the_dictionary_text(self) -> None:
        self.assertEqual(self.ask("What does capital mean?"),
                         expected("capital", *CAPITAL))
        self.assertEqual(self.ask("Define Capital"),
                         expected("Capital", *CAPITAL))
        self.assertEqual(self.ask("what is the meaning of capital?"),
                         expected("capital", *CAPITAL))

    def test_quotes_phrases_and_several_definition_strings(self) -> None:
        self.assertEqual(
            self.ask('What does "The Hague" mean?'),
            expected("The Hague",
                     ("noun", "the seat of government of the Netherlands")))
        self.assertEqual(
            self.ask("define 'A. A. Milne'"),
            expected("A. A. Milne",
                     ("noun", "English writer of stories for children")))
        self.assertEqual(
            self.ask("What is the meaning of city?"),
            expected("city", ("noun", "a large and densely populated urban "
                                      "area; may include several independent "
                                      "administrative districts")))
        self.assertEqual(
            self.ask("define bass"),
            expected("bass", ("noun", "the lowest part of the musical range"),
                     ("noun", "the lean flesh of a saltwater fish")))

    def test_every_part_of_speech_is_named(self) -> None:
        self.assertEqual(self.ask("Define fast"), expected(
            "fast",
            ("noun", "abstaining from food"),
            ("verb", "abstain from certain foods"),
            ("adjective", "acting or moving or capable of acting or moving "
                          "quickly"),
            ("adjective", "securely fixed in place"),
            ("adverb", "quickly or rapidly")))

    def test_no_word_is_stripped_from_the_asked_word(self) -> None:
        # "word" is in the dictionary; "this word" is not, and is asked as
        # before rather than cut down to a word the dictionary holds.
        self.assertEqual(self.ask("What does word mean?"), expected(
            "word", ("noun", "a unit of language that native speakers can "
                             "identify")))
        bare = self.bare()
        self.assertEqual(self.ask("What does this word mean?"),
                         self.ask("What does this word mean?", bare))
        self.assertNotIn(CREDIT, self.ask("What does this word mean?"))

    def test_the_define_command_says_the_same(self) -> None:
        out = io.StringIO()
        cli = ChatCLI(self.session, out=out)
        cli.handle(":define capital")
        self.assertEqual(out.getvalue(), expected("capital", *CAPITAL) + "\n")
        out.seek(0)
        out.truncate()
        cli.handle(":define A. A. Milne")
        self.assertEqual(out.getvalue(), expected(
            "A. A. Milne",
            ("noun", "English writer of stories for children")) + "\n")

    def test_the_define_command_explains_itself(self) -> None:
        out = io.StringIO()
        cli = ChatCLI(self.session, out=out)
        cli.handle(":define")
        cli.handle(":define xyzzy")
        self.assertEqual(out.getvalue().splitlines(), [
            "Usage: :define WORD", "'xyzzy' is not in the dictionary."])
        out = io.StringIO()
        ChatCLI(self.bare(), out=out).handle(":define capital")
        self.assertEqual(out.getvalue(), (
            "No dictionary in this library; build one with: python -m "
            f"ultraquant.lexicon.build --home {self.dir / 'bare'}\n"))
        self.assertIn(":define WORD", HELP)
        HELP.encode("cp1252")

    def test_a_word_it_lacks_is_asked_as_before(self) -> None:
        bare = self.bare()
        for text in ("What does this word mean?", "What does xyzzy mean?",
                     "Define xyzzy", "what is the meaning of life?",
                     "define the tower material is iron",
                     "What does the tower material mean?"):
            with self.subTest(text=text):
                self.assertEqual(self.ask(text), self.ask(text, bare))

    def test_a_fact_question_is_unchanged(self) -> None:
        bare = self.bare()
        for session in (self.session, bare):
            self.ask("the capital material is iron", session)
        for text in ("What is the capital material?", "What is capital?",
                     "What is the capital of France?", "what is a city?"):
            with self.subTest(text=text):
                self.assertEqual(self.ask(text), self.ask(text, bare))

    def test_other_utterances_never_consult_the_dictionary(self) -> None:
        with mock.patch.object(Lexicon, "senses",
                               side_effect=AssertionError("looked up")), \
                mock.patch.object(Lexicon, "define",
                                  side_effect=AssertionError("defined")):
            for text in ("hello there", "What is capital?",
                         "the tower material is iron", "calc: 2 + 2",
                         "what is 3 + 4?"):
                with self.subTest(text=text):
                    self.ask(text)

    def test_a_home_without_a_lexicon_has_none(self) -> None:
        bare = self.bare()
        self.assertIsNone(bare.lexicon)
        self.assertFalse((self.dir / "bare" / "lexicon").exists())
        self.assertIsInstance(self.session.lexicon, Lexicon)

    def test_a_define_turn_stores_no_fact_and_records_its_episode(self):
        self.ask("the tower material is iron")
        keys = self.session.memory.fact_keys()
        reply = self.ask("Define capital")
        self.assertEqual(self.session.memory.fact_keys(), keys)
        episode = self.session.memory.recall_episodes(kind="interaction",
                                                      limit=1)[0]
        self.assertEqual(episode["content"]["intent"], "define")
        self.assertEqual(episode["content"]["response"], reply)
        # Saved as usual: a fresh session over the same home sees it.
        again = build_session(self.home, seed=0)
        self.assertEqual(again.memory.recall_episodes(
            kind="interaction", limit=1)[0]["content"]["response"], reply)

    def test_approvals_wait_for_the_next_turn(self) -> None:
        home = self.copy_home("approving")
        (home / "stash.json").write_text(json.dumps({"next_id": 2, "entries": [{
            "id": 1, "claim": "The capital of Kenya is Nairobi.",
            "classification": "unclassified", "status": "staged",
            "sources": ["a.example"], "netloc": "a.example",
            "url": "https://a.example/1", "title": "t",
            "fetched": "2026-09-28T00:00:00+00:00", "notes": ""}]}),
            encoding="utf-8")
        session = build_session(home, seed=0, auto_approve=True)
        self.assertEqual(self.ask("Define capital", session),
                         expected("capital", *CAPITAL))
        self.assertEqual(session.stash.get(1)["status"], "staged")
        self.assertNotIn("capital of kenya", session.memory.fact_keys())
        self.assertIn("Approved entry 1", self.ask("hello there", session))


class TUITests(_Temp):
    """The TUI's ':define' says what the chat CLI's says."""

    def setUp(self) -> None:
        super().setUp()
        self._old_config = os.environ.get("ULTRAQUANT_CONFIG")
        os.environ["ULTRAQUANT_CONFIG"] = str(self.dir / "settings.json")

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("ULTRAQUANT_CONFIG", None)
        else:
            os.environ["ULTRAQUANT_CONFIG"] = self._old_config
        super().tearDown()

    def test_define_in_the_tui(self) -> None:
        from ultraquant.tui import UltraQuantTUI

        tui = UltraQuantTUI(home=self.copy_home(), stream=io.StringIO(),
                            color=False)
        self.assertEqual(tui.handle(":define capital"),
                         expected("capital", *CAPITAL))
        self.assertEqual(tui.handle(":define   A. A.  Milne"), expected(
            "A. A. Milne", ("noun", "English writer of stories for children")))
        self.assertEqual(tui.handle(":define"), "usage: :define WORD")
        self.assertEqual(tui.handle(":define xyzzy"),
                         "'xyzzy' is not in the dictionary.")
        self.assertIn(":define WORD", tui.help_text())
        bare = UltraQuantTUI(home=self.dir / "bare", stream=io.StringIO(),
                             color=False)
        self.assertEqual(bare.handle(":define capital"), (
            "No dictionary in this library; build one with: python -m "
            f"ultraquant.lexicon.build --home {self.dir / 'bare'}"))


if __name__ == "__main__":
    unittest.main()
