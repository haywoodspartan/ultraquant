"""Dictionary targets (§11.170) over a fake lexicon and scratch stores only.

The lexicon here is a few dicts in the shapes Lexicon returns, never the
WordNet release: senses, and synset records carrying the inverse links
(hyponym, instance) that a build writes.
"""

from copy import deepcopy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_frontier import FakeTeacher
from ultraquant.distill import frontier, seeds, session, sources, targets
from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.lexicon import normalize_word
from ultraquant.memory.factshards import FactShards
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


def synset(members, *, kind=(), instance_of=(), pos="n"):
    """A release record, whose links are present only when non-empty."""
    record = {"members": list(members), "partOfSpeech": pos}
    if instance_of:
        record["instance_hypernym"] = list(instance_of)
    if kind:
        record["hypernym"] = list(kind)
    return record


#: A dictionary in miniature, with separate roots so coherence can fail. Dict
#: order is sense order: Quebec the city comes before the province, a title's
#: hero before its novel, a lake or river before its country, and the golf
#: club before the metal.
SYNSETS = {
    "location.n": synset(["location"]),
    "country.n": synset(["country"], kind=["location.n"]),
    "african.n": synset(["African country", "African nation"], kind=["country.n"]),
    "commonwealth.n": synset(["Commonwealth country"], kind=["country.n"]),
    "sahel.n": synset(["Sahel country"], kind=["country.n"]),
    "river.n": synset(["river"], kind=["location.n"]),
    "lake.n": synset(["lake"], kind=["location.n"]),
    "city.n": synset(["city"], kind=["location.n"]),
    "province.n": synset(["province"], kind=["location.n"]),
    "nigeria.n": synset(["Nigeria", "Federal Republic of Nigeria"],
                        instance_of=["african.n", "commonwealth.n"]),
    "kenya.n": synset(["Kenya", "Republic of Kenya"],
                      instance_of=["african.n", "commonwealth.n"]),
    "ghana.n": synset(["Ghana", "Republic of Ghana"],
                      instance_of=["african.n", "commonwealth.n"]),
    "ivory_coast.n": synset(["Cote d'Ivoire", "Ivory Coast"], instance_of=["african.n"]),
    "gambia.n": synset(["Gambia", "The Gambia"], instance_of=["african.n"]),
    "nameless.n": synset([], instance_of=["african.n"]),
    "jamaica.n": synset(["Jamaica"], instance_of=["commonwealth.n"]),
    "niger_river.n": synset(["Niger", "Niger River"], instance_of=["river.n"]),
    "niger.n": synset(["Niger", "Republic of Niger"], instance_of=["african.n", "sahel.n"]),
    "lake_chad.n": synset(["Chad", "Lake Chad"], instance_of=["lake.n"]),
    "chad.n": synset(["Chad", "Republic of Chad"], instance_of=["african.n", "sahel.n"]),
    "lagos.n": synset(["Lagos"], instance_of=["city.n"]),
    "quebec_city.n": synset(["Quebec", "Quebec City"], instance_of=["city.n"]),
    "quebec.n": synset(["Quebec", "Province of Quebec"], instance_of=["province.n"]),
    "ontario.n": synset(["Ontario"], instance_of=["province.n"]),
    "alberta.n": synset(["Alberta"], instance_of=["province.n"]),
    "manitoba.n": synset(["Manitoba"], instance_of=["province.n"]),
    "substance.n": synset(["substance"]),
    "element.n": synset(["chemical element", "element"], kind=["substance.n"]),
    "metal.n": synset(["metallic element", "metal"], kind=["element.n"]),
    "noble_metal.n": synset(["noble metal"], kind=["metal.n"]),
    "implement.n": synset(["implement"]),
    "golf_club.n": synset(["golf club"], kind=["implement.n"]),
    "iron_club.n": synset(["iron"], kind=["golf_club.n"]),
    "iron.n": synset(["iron", "Fe", "atomic number 26"], kind=["metal.n"]),
    "copper.n": synset(["copper", "Cu", "atomic number 29"], kind=["metal.n"]),
    "zinc.n": synset(["zinc", "Zn", "atomic number 30"], kind=["metal.n"]),
    "copernicium.n": synset(["copernicium", "Cn", "atomic number 112"], kind=["metal.n"]),
    "ununbium.n": synset(["ununbium", "Uub", "element 112", "atomic number 112"],
                         kind=["metal.n"]),
    "gold.n": synset(["gold", "Au", "atomic number 79"], kind=["noble_metal.n"]),
    "change.v": synset(["change"], pos="v"),
    "press.v": synset(["iron", "iron out", "press"], kind=["change.v"], pos="v"),
    "smooth.v": synset(["smooth", "flatten"], kind=["change.v"], pos="v"),
    "work.n": synset(["work", "piece of work"]),
    "novel.n": synset(["novel"], kind=["work.n"]),
    "character.n": synset(["fictional character", "character"]),
    "emma_woodhouse.n": synset(["Emma", "Emma Woodhouse"], instance_of=["character.n"]),
    "emma.n": synset(["Emma"], instance_of=["novel.n"]),
    "oliver.n": synset(["Oliver Twist"], instance_of=["character.n"]),
    "oliver_twist.n": synset(["Oliver Twist"], instance_of=["novel.n"]),
    "persuasion.n": synset(["Persuasion"], instance_of=["novel.n"]),
    "middlemarch.n": synset(["Middlemarch"], instance_of=["novel.n"]),
    "dracula.n": synset(["Dracula"], instance_of=["novel.n"]),
    "sherlock.n": synset(["Sherlock Holmes"], instance_of=["character.n"]),
}


class FakeLexicon:
    """Lexicon.senses and Lexicon.synset over dicts; every member is a word."""

    def __init__(self, synsets=SYNSETS):
        self.synsets = deepcopy(synsets)
        # The inverse links a build writes, each list sorted.
        for inverse, relation in (("hyponym", "hypernym"),
                                  ("instance", "instance_hypernym")):
            pointing = {}
            for synset_id, record in self.synsets.items():
                for target in record.get(relation, ()):
                    pointing.setdefault(target, []).append(synset_id)
            for target, ids in pointing.items():
                self.synsets[target][inverse] = sorted(ids)
        self.words = {}
        for synset_id, record in self.synsets.items():
            for member in record["members"]:
                self.words.setdefault(normalize_word(member), []).append({
                    "lemma": member, "pos": record["partOfSpeech"],
                    "synset": synset_id, "sense_key": f"{member}%{synset_id}"})

    def senses(self, word):
        return deepcopy(self.words.get(normalize_word(word), []))

    def synset(self, synset_id):
        record = self.synsets.get(synset_id)
        return deepcopy(record) if record is not None else None

    def kinds(self, word):
        raise AssertionError("Lexicon.kinds climbs every part of speech")


class SenseTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = FakeLexicon()

    def test_noun_senses_keep_the_dictionary_order_of_nouns_only(self):
        self.assertEqual(seeds.noun_senses(self.lexicon, "iron"), ["iron_club.n", "iron.n"])
        self.assertEqual(seeds.noun_senses(self.lexicon, " QUEBEC "),
                         ["quebec_city.n", "quebec.n"])
        self.assertEqual(seeds.noun_senses(self.lexicon, "Côte d'Ivoire"), ["ivory_coast.n"])
        self.assertEqual(seeds.noun_senses(self.lexicon, "smooth"), [])
        self.assertEqual(seeds.noun_senses(self.lexicon, "Atlantis"), [])

    def test_direct_kinds_prefer_instance_links_then_kind_links(self):
        self.assertEqual(seeds.direct_kinds(self.lexicon, "nigeria.n"),
                         ["african.n", "commonwealth.n"])
        self.assertEqual(seeds.direct_kinds(self.lexicon, "zinc.n"), ["metal.n"])
        self.assertEqual(seeds.direct_kinds(self.lexicon, "location.n"), [])
        self.assertEqual(seeds.direct_kinds(self.lexicon, "missing.n"), [])
        both = FakeLexicon({
            "both.n": {"members": ["both"], "partOfSpeech": "n",
                       "instance_hypernym": ["class.n"], "hypernym": ["kind.n"]},
            "empty.n": {"members": ["empty"], "partOfSpeech": "n",
                        "instance_hypernym": [], "hypernym": ["kind.n"]},
            "class.n": synset(["class"]), "kind.n": synset(["kind"])})
        self.assertEqual(seeds.direct_kinds(both, "both.n"), ["class.n"])
        self.assertEqual(seeds.direct_kinds(both, "empty.n"), ["kind.n"])

    def test_coherent_needs_half_the_subjects_under_one_ancestor(self):
        # Nigeria and Kenya share kinds; the novel, the metal and the hero none.
        self.assertTrue(seeds.coherent(self.lexicon, ["Nigeria", "Kenya", "Persuasion", "zinc"]))
        self.assertFalse(seeds.coherent(self.lexicon, [
            "Nigeria", "Kenya", "Persuasion", "zinc", "Sherlock Holmes"]))
        # A country and a city meet only at the root, two links up.
        self.assertTrue(seeds.coherent(self.lexicon, {
            "Nigeria", "Lagos", "Persuasion", "Sherlock Holmes"}))
        self.assertTrue(seeds.coherent(self.lexicon, ["zinc"]))
        self.assertFalse(seeds.coherent(self.lexicon, []))
        self.assertFalse(seeds.coherent(self.lexicon, ["Atlantis", "Lemuria"]))
        self.assertFalse(seeds.coherent(self.lexicon, ["location", "substance"]))

    def test_coherent_counts_subjects_not_senses_and_never_the_sense_itself(self):
        # "location" is a root: Lagos reaches it, but it reaches nothing.
        self.assertFalse(seeds.coherent(self.lexicon, ["location", "Lagos", "Persuasion"]))
        # Quebec reaches the root through both of its senses, and counts once.
        self.assertFalse(seeds.coherent(self.lexicon, ["Quebec", "Persuasion", "zinc"]))
        # As verbs, smooth and press share a kind; only nouns are walked, and
        # never through Lexicon.kinds, which raises here.
        self.assertFalse(seeds.coherent(self.lexicon, ["smooth", "press"]))
        self.assertTrue(seeds.coherent(self.lexicon, ["iron", "copper", "smooth"]))

    def test_core_kinds_of_unambiguous_subjects_need_two(self):
        self.assertEqual(seeds.core_kinds(self.lexicon, ["Nigeria", "Kenya", "Ivory Coast"]),
                         {"african.n": 3, "commonwealth.n": 2})
        self.assertEqual(seeds.core_kinds(self.lexicon, ["Nigeria", "Ivory Coast"]),
                         {"african.n": 2})
        self.assertEqual(seeds.core_kinds(self.lexicon, ["zinc", "copper", "smooth", "Atlantis"]),
                         {"metal.n": 2})
        self.assertEqual(seeds.core_kinds(self.lexicon, ["zinc", "Ontario"]), {})
        self.assertEqual(seeds.core_kinds(self.lexicon, []), {})

    def test_an_ambiguous_subject_counts_its_first_sense_of_a_core_kind(self):
        # Quebec counts as a province: the city is no kind the others name.
        self.assertEqual(seeds.core_kinds(self.lexicon, ["Ontario", "Alberta", "Quebec"]),
                         {"province.n": 3})
        # With cities named too, its first sense wins, and only that one.
        self.assertEqual(seeds.core_kinds(self.lexicon, [
            "Ontario", "Alberta", "Lagos", "Quebec City", "Quebec"]),
            {"province.n": 2, "city.n": 3})
        # Titles that name their heroes count as novels, never as characters.
        self.assertEqual(seeds.core_kinds(self.lexicon, [
            "Persuasion", "Middlemarch", "Emma", "Oliver Twist"]), {"novel.n": 4})
        # With no unambiguous subject there is no core at all.
        self.assertEqual(seeds.core_kinds(self.lexicon, ["Emma", "Oliver Twist"]), {})
        # The golf club is passed over for the metal; Quebec adds nothing.
        self.assertEqual(seeds.core_kinds(self.lexicon, ["iron", "copper", "zinc", "Quebec"]),
                         {"metal.n": 3})
        # Of the sense taken, only core kinds count: two Sahel countries do
        # not make the Sahel a kind.
        self.assertEqual(seeds.core_kinds(self.lexicon, [
            "Ivory Coast", "Gambia", "Niger", "Chad"]), {"african.n": 4})

    def test_is_alias_compares_normalized_values(self):
        held = {"112", "copernicium"}
        self.assertTrue(seeds.is_alias("112", held))
        self.assertTrue(seeds.is_alias(" COPERNICIUM. ", held))
        self.assertFalse(seeds.is_alias("113", held))
        self.assertFalse(seeds.is_alias("112", set()))


class DictionaryKindTests(unittest.TestCase):
    """§11.174: the nearest ancestor half the subjects reach names their kind."""

    def setUp(self):
        self.lexicon = FakeLexicon()

    def kind(self, subjects, lexicon=None):
        return seeds.dictionary_kind(lexicon or self.lexicon, subjects)

    def test_half_the_subjects_must_reach_the_kind(self):
        # Two of four is half: the Commonwealth, one link from Nigeria and Jamaica.
        self.assertEqual(self.kind(["Nigeria", "Jamaica", "zinc", "Persuasion"]),
                         "Commonwealth country")
        # Two of five is not, and nothing else is shared.
        self.assertIsNone(self.kind(["Nigeria", "Jamaica", "zinc", "Persuasion",
                                     "Sherlock Holmes"]))
        # Subjects are counted, not senses: Quebec reaches the root twice.
        self.assertIsNone(self.kind(["Quebec", "zinc", "Persuasion", "Sherlock Holmes"]))
        self.assertEqual(self.kind(["Quebec", "Lagos", "zinc", "Persuasion"]), "city")

    def test_the_smallest_mean_level_wins_over_more_subjects(self):
        # Provinces are one link from two subjects; the root is reached by all
        # four, at 2.25 links on average.
        self.assertEqual(self.kind(["Ontario", "Alberta", "Lagos", "Nigeria"]), "province")
        # Two subjects reach "country" at two links; all four reach the root, at 2.5.
        self.assertEqual(self.kind(["Jamaica", "Ivory Coast", "Lagos", "Ontario"]), "country")
        # Gold is two links below the metals and the rest one (iron by its
        # metal sense); the metals' first member names them.
        self.assertEqual(self.kind(["iron", "copper", "zinc", "gold"]), "metallic element")

    def test_ties_go_to_more_subjects_then_to_the_smaller_synset_id(self):
        # Both one link away: three Commonwealth countries beat two African ones.
        self.assertEqual(self.kind(["Nigeria", "Kenya", "Jamaica"]), "Commonwealth country")
        self.assertEqual(self.kind(["Nigeria", "Kenya", "Ivory Coast"]), "African country")
        # Same level, same subjects: the smaller id, whichever link comes first.
        self.assertEqual(self.kind(["Nigeria", "Kenya"]), "African country")
        tied = FakeLexicon({
            "b_kind.n": synset(["later kind"]), "a_kind.n": synset(["earlier kind"]),
            "one.n": synset(["one"], instance_of=["b_kind.n", "a_kind.n"]),
            "two.n": synset(["two"], instance_of=["b_kind.n", "a_kind.n"])})
        self.assertEqual(self.kind(["one", "two"], tied), "earlier kind")

    def test_each_subject_reaches_an_ancestor_at_its_nearest_sense(self):
        lexicon = FakeLexicon({
            "group.n": synset(["group"]),
            "middle.n": synset(["middle"], kind=["group.n"]),
            "far.n": synset(["dual"], instance_of=["middle.n"]),
            "near.n": synset(["dual"], instance_of=["group.n"]),
            "both.n": synset(["both"], instance_of=["middle.n", "group.n"]),
            "one.n": synset(["one"], instance_of=["group.n"])})
        # "dual" is two links below the group by its first sense and one by its
        # second; "both" is one link below it however else it climbs. At one
        # link from all three, the group ties the middle and has more subjects.
        self.assertEqual(self.kind(["dual", "both", "one"], lexicon), "group")
        self.assertEqual(self.kind(["dual", "both"], lexicon), "group")
        self.assertEqual(self.kind(["both"], lexicon), "group")

    def test_no_kind_without_subjects_or_an_ancestor_half_of_them_share(self):
        for subjects in ([], set(), ["Atlantis", "Lemuria"], ["smooth", "press"],
                         ["location", "substance"],
                         ["Nigeria", "Kenya", "Persuasion", "zinc", "Sherlock Holmes"]):
            with self.subTest(subjects=subjects):
                self.assertIsNone(self.kind(subjects))

    def test_senses_are_read_through_the_module(self):
        with mock.patch.object(seeds, "noun_senses", return_value=["zinc.n"]) as senses:
            self.assertEqual(self.kind(["Atlantis", "Lemuria"]), "metallic element")
        self.assertEqual(senses.call_args_list, [mock.call(self.lexicon, "Atlantis"),
                                                 mock.call(self.lexicon, "Lemuria")])


class LibraryCase(unittest.TestCase):
    """A scratch library beside the fake dictionary."""

    sharded = False

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="uq_seeds_test_")
        self.addCleanup(scratch.cleanup)
        self.home = Path(scratch.name)
        shards = (FactShards(ShardVault(self.home / "vault"))
                  if self.sharded else None)
        self.memory = SystematicMemory(self.home / "memory.json", shards=shards)
        self.stash = ContemporaryStash(self.home / "stash.json")
        self.lexicon = FakeLexicon()

    def hold(self, attribute, values):
        for subject, value in values.items():
            self.memory.remember_fact(f"{attribute} of {subject}".lower(), value,
                                      subject=subject, attribute=attribute)

    def form(self, attribute, category):
        """Promote a template question, which question_form learns from."""
        subject, value = "Template", "Label"
        qid = f"{category}:{subject}"
        entry_id = self.stash.add_claim(
            f"https://distill.invalid/seed/{qid}",
            f"What is the {attribute} of {subject}?",
            f"The {attribute} of {subject} is {value}.",
            provenance={"run_id": "seed", "question_id": qid},
            fields={"subject": subject, "attribute": attribute, "value": value,
                    "key": f"{attribute} of {subject.lower()}"})
        self.stash._entries[entry_id]["status"] = "promoted"
        self.stash.save()

    def expected(self, attribute, category, *names):
        return [targets.Target(category, name, f"What is the {attribute} of {name}?", attribute)
                for name in names]


class DictionaryTargetTests(LibraryCase):
    def targets(self):
        return seeds.dictionary_targets(self.memory, self.stash, self.lexicon)

    def test_instances_of_the_core_kinds_named_by_their_first_member(self):
        self.hold("capital", {"Nigeria": "Abuja", "Kenya": "Nairobi",
                              "Ivory Coast": "Yamoussoukro", "The Gambia": "Banjul"})
        self.form("capital", "capital")
        # Ghana is an instance of both core kinds and is asked once. Cote
        # d'Ivoire and Gambia are held under another member, the nameless
        # instance has no member, and the rivers and lakes are no countries.
        self.assertEqual(self.targets(), self.expected(
            "capital", "capital", "Chad", "Ghana", "Jamaica", "Niger"))

    def test_witnessed_subkinds_by_their_value_except_aliases(self):
        self.hold("atomic number", {"iron": "26", "copper": "29", "copernicium": "112"})
        self.form("atomic number", "number")
        # Ununbium's value is copernicium's: an old name, not a new element.
        # The noble metals give no value, and gold is not a direct subkind.
        self.assertEqual(self.targets(), self.expected("atomic number", "number", "zinc"))
        with mock.patch.object(seeds, "is_alias", return_value=False) as alias:
            self.assertEqual([t.subject for t in self.targets()], ["ununbium", "zinc"])
        self.assertIn(mock.call("112", {"26", "29", "112"}), alias.call_args_list)

    def test_subkinds_wait_for_half_the_held_values_to_be_witnessed(self):
        self.form("atomic number", "number")
        # Zinc's held value is not the dictionary's, but copper's is: half.
        self.hold("atomic number", {"copper": "29", "zinc": "31"})
        self.assertEqual([t.subject for t in self.targets()],
                         ["copernicium", "iron", "ununbium"])
        # One of three is not.
        self.hold("atomic number", {"iron": "27"})
        self.assertEqual(self.targets(), [])

    def test_quebec_counts_as_a_province_and_asks_about_no_city(self):
        self.hold("premier", {"Ontario": "Ford", "Alberta": "Smith", "Quebec": "Legault"})
        self.form("premier", "premier")
        self.assertEqual(self.targets(), self.expected("premier", "premier", "Manitoba"))

    def test_titles_that_name_their_heroes_ask_about_novels(self):
        self.hold("author", {"Persuasion": "Jane Austen", "Middlemarch": "George Eliot",
                             "Emma": "Jane Austen", "Oliver Twist": "Charles Dickens"})
        self.form("author", "author")
        self.assertEqual(self.targets(), self.expected("author", "author", "Dracula"))

    def test_nothing_without_a_question_form_or_coherence(self):
        self.hold("capital", {"Nigeria": "Abuja", "Kenya": "Nairobi"})
        self.hold("motto", {"smooth": "A", "press": "B", "Atlantis": "C"})
        self.form("motto", "motto")
        # The capital has no question form yet, and the mottos no nouns.
        with mock.patch.object(seeds, "core_kinds", side_effect=AssertionError("skipped")):
            self.assertEqual(self.targets(), [])
        self.form("capital", "capital")
        self.assertEqual(self.targets(), self.expected(
            "capital", "capital", "Chad", "Cote d'Ivoire", "Gambia", "Ghana", "Jamaica", "Niger"))

    def test_structured_facts_only_sorted_by_attribute_then_name(self):
        self.hold("capital", {"Nigeria": "Abuja", "Kenya": "Nairobi",
                              "Ivory Coast": "Yamoussoukro", "The Gambia": "Banjul"})
        self.hold("atomic number", {"iron": 26, "copper": 29, "copernicium": 112})
        self.memory.remember_fact("loose", "Accra")
        self.memory.remember_fact("no attribute", "Accra", subject="Ghana")
        self.memory.remember_fact("no subject", "Kingston", attribute="capital")
        self.form("capital", "capital")
        self.form("atomic number", "number")
        self.assertEqual([(t.attribute, t.subject) for t in self.targets()], [
            ("atomic number", "zinc"), ("capital", "Chad"), ("capital", "Ghana"),
            ("capital", "Jamaica"), ("capital", "Niger")])

    def test_each_step_is_called_through_the_module(self):
        self.hold("capital", {"Nigeria": "Abuja", "Kenya": "Nairobi"})
        self.form("capital", "capital")
        provinces = ["Alberta", "Manitoba", "Ontario", "Quebec"]
        with mock.patch.object(seeds, "coherent", return_value=False) as coherent:
            self.assertEqual(self.targets(), [])
        coherent.assert_called_once_with(self.lexicon, {"Nigeria", "Kenya"})
        with mock.patch.object(seeds, "core_kinds", return_value={"province.n": 2}) as kinds:
            self.assertEqual([t.subject for t in self.targets()], provinces)
        kinds.assert_called_once_with(self.lexicon, {"Nigeria", "Kenya"})
        with mock.patch.object(seeds, "direct_kinds", return_value=["province.n"]):
            self.assertEqual([t.subject for t in self.targets()], provinces)
        with mock.patch.object(seeds, "noun_senses", return_value=[]):
            self.assertEqual(self.targets(), [])
        with mock.patch.object(targets, "question_form", return_value=None) as form:
            self.assertEqual(self.targets(), [])
        form.assert_called_once_with(self.stash, "capital")


class FrontierLexiconTests(LibraryCase):
    def setUp(self):
        super().setUp()
        self.teacher = FakeTeacher(self.home)
        gguf = self.teacher.spec.gguf
        index = {**sources._teacher_index(),
                 self.teacher.spec.name: f"{gguf.name}:{gguf.stat().st_size}"}
        self.enterContext(mock.patch.object(sources, "_teacher_index", return_value=index))
        self.enterContext(mock.patch.object(
            sources.LMStudioTeacher, "ask", side_effect=AssertionError("Live teacher forbidden")))
        self.ledger = sources.SourceLedger(self.home / "ledger.json")

    def africa(self):
        """Two African capitals, the asking form, and the source's answers."""
        self.hold("capital", {"Nigeria": "Abuja", "Ivory Coast": "Yamoussoukro"})
        self.form("capital", "capital")
        # A learned kind with a property verdict: propose has nothing to ask.
        self.memory.learn_kind("capital", "country")
        self.memory.learn_property("capital", "population", "refused")
        answers = {"Ghana": "Accra", "Jamaica": "Kingston"}
        for name in ("Chad", "Gambia", "Ghana", "Kenya", "Niger", "Jamaica"):
            self.teacher.replies[f"What is the capital of {name}?"] = (
                [answers.get(name, "UNKNOWN")] * 5)

    def study(self, run_id, **kwargs):
        return frontier.study_round(
            self.memory, self.stash, self.teacher, self.ledger, "source", confidence=0.99,
            run_id=run_id, records_path=self.home / f"{run_id}.jsonl", **kwargs)

    def test_pending_adds_dictionary_targets_last_and_only_with_a_lexicon(self):
        completion = targets.Target("capital", "Ghana", "completion", "capital")
        reverse = frontier.ReverseTarget("number", "5", "reverse", "atomic number", "5")
        growth = targets.Target("mass", "Iron", "growth", "atomic mass")
        seeded = [targets.Target("capital", "Ghana", "duplicate", "capital"),
                  targets.Target("capital", "Kenya", "asked", "capital"),
                  targets.Target("capital", "Jamaica", "dictionary", "capital")]
        self.ledger.record("source", "capital:Kenya", False)
        with mock.patch.object(targets, "completion_targets", return_value=[completion]), \
                mock.patch.object(frontier, "reverse_targets", return_value=[reverse]), \
                mock.patch.object(frontier, "growth_targets", return_value=[growth]), \
                mock.patch.object(seeds, "dictionary_targets", return_value=seeded) as dictionary:
            self.assertEqual(frontier.pending(
                self.memory, self.stash, self.teacher, self.ledger, "source",
                lexicon=self.lexicon), [completion, reverse, growth, seeded[2]])
            dictionary.assert_called_once_with(self.memory, self.stash, self.lexicon)
            for extra in ({}, {"lexicon": None}):
                self.assertEqual(frontier.pending(
                    self.memory, self.stash, self.teacher, self.ledger, "source", **extra),
                    [completion, reverse, growth])
        dictionary.assert_called_once()

    def test_study_asks_the_dictionary_until_its_kinds_are_used_up(self):
        self.africa()
        outcomes = [self.study(f"seed-{n}", lexicon=self.lexicon) for n in range(3)]
        self.assertEqual([r["asked"] for r in outcomes], [5, 1, 0])
        self.assertEqual([r["filed"] for r in outcomes], [1, 1, 0])
        # Ghana made Commonwealth countries a core kind: Jamaica kept it going.
        self.assertEqual([r["used_up"] for r in outcomes], [False, True, True])
        self.assertEqual([questions for questions, _ in self.teacher.calls], [
            [f"What is the capital of {name}?"
             for name in ("Chad", "Gambia", "Ghana", "Kenya", "Niger")],
            ["What is the capital of Jamaica?"]])
        self.assertEqual([(row["question_id"], row["promoted"])
                          for row in self.ledger.history("source")], [
            ("capital:Chad", False), ("capital:Gambia", False), ("capital:Ghana", True),
            ("capital:Kenya", False), ("capital:Niger", False), ("capital:Jamaica", True)])
        self.assertEqual(self.memory.held_value("Ghana", "capital"), "Accra")
        self.assertEqual(self.memory.held_value("Jamaica", "capital"), "Kingston")

    def test_without_a_lexicon_the_same_library_is_used_up_at_once(self):
        self.africa()
        with mock.patch.object(seeds, "dictionary_targets",
                               side_effect=AssertionError("No lexicon was given")):
            result = self.study("plain")
        self.assertEqual(result, {"asked": 0, "checked": 0, "agreed": 0, "contested": 0,
                                  "revised": 0, "filed": 0, "queued": 0, "proposed": [],
                                  "adopted": [], "refused": [], "used_up": True})
        self.assertEqual(self.teacher.calls, [])

    def test_used_up_stays_false_while_a_dictionary_target_is_unasked(self):
        self.africa()
        with mock.patch.object(frontier, "pending", return_value=[]) as pending:
            self.assertTrue(self.study("judge-0")["used_up"])
            self.assertFalse(self.study("judge-1", lexicon=self.lexicon)["used_up"])
            for name in ("Chad", "Gambia", "Ghana", "Kenya"):
                self.ledger.record("source", f"capital:{name}", False)
            self.assertFalse(self.study("judge-2", lexicon=self.lexicon)["used_up"])
            self.ledger.record("source", "capital:Niger", False)
            self.assertTrue(self.study("judge-3", lexicon=self.lexicon)["used_up"])
        self.assertEqual([call.kwargs for call in pending.call_args_list],
                         [{}] + [{"lexicon": self.lexicon}] * 3)
        self.assertEqual(self.teacher.calls, [])

    def test_without_a_lexicon_a_round_is_todays(self):
        # test_frontier's abstention: one reverse question, answered UNKNOWN.
        for number in (1, 2, 3, 4, 6):
            self.hold("atomic number", {f"Member {number}": str(number)})
        self.form("atomic number", "number")
        self.memory.learn_kind("atomic number", "element")
        self.memory.learn_property("atomic number", "mass", "refused")
        question = frontier.reverse_targets(self.memory, self.stash, self.teacher)[0].question
        self.teacher.replies[question] = ["UNKNOWN"] * 5
        today = {"asked": 1, "checked": 0, "agreed": 0, "contested": 0, "revised": 0,
                 "filed": 0, "queued": 0, "proposed": [], "adopted": [], "refused": [],
                 "used_up": True}
        rows = [{"question_id": "number:5", "promoted": False}]
        real = frontier.pending

        def positional(*args):
            # The seam test_corroborate uses: today's call has no keyword.
            return real(*args)

        with mock.patch.object(frontier, "pending", side_effect=positional), \
                mock.patch.object(seeds, "dictionary_targets",
                                  side_effect=AssertionError("No lexicon was given")):
            for run_id, extra in (("omitted", {}), ("none", {"lexicon": None})):
                self.ledger = sources.SourceLedger(self.home / f"{run_id}.json")
                self.assertEqual(self.study(run_id, **extra), today)
                self.assertEqual(self.ledger.history("source"), rows)
        # A dictionary that knows none of the subjects changes nothing either.
        self.ledger = sources.SourceLedger(self.home / "lexicon.json")
        self.assertEqual(self.study("lexicon", lexicon=self.lexicon), today)
        self.assertEqual(self.ledger.history("source"), rows)
        self.assertEqual([qs for qs, _ in self.teacher.calls], [[question]] * 3)

    # §11.174: the dictionary names a kind the teachers could not.

    #: African countries all, and Commonwealth ones two of them: the nearest
    #: kind half of them share is the African country.
    AFRICA = {"Nigeria": "Abuja", "Kenya": "Nairobi", "Ivory Coast": "Yamoussoukro",
              "The Gambia": "Banjul"}

    def kind_replies(self, *rows):
        """The teacher's replies to the kind question for kind_of's three capitals."""
        members = random.Random(158).sample(sorted(frontier.cluster(self.memory, "capital")), 3)
        questions = [frontier._seed_questions()["kind"].format(a=member) for member in members]
        self.teacher.replies.update(zip(questions, rows))
        return questions

    def kind(self, **kwargs):
        return frontier.kind_of(self.memory, "capital", self.teacher, **kwargs)

    def stored(self):
        return self.memory._attribute_vocabulary()["capital"]

    def test_a_stored_kind_is_returned_without_asking_anyone(self):
        self.hold("capital", self.AFRICA)
        self.memory.learn_kind("capital", "nation")
        with mock.patch.object(seeds, "dictionary_kind",
                               side_effect=AssertionError("The dictionary was asked")), \
                mock.patch.object(frontier, "_seed_questions",
                                  side_effect=AssertionError("The teacher was asked")):
            self.assertEqual(self.kind(lexicon=self.lexicon), "nation")
            self.assertEqual(self.kind(), "nation")
        self.assertEqual(self.teacher.calls, [])

    def test_an_absent_kind_is_asked_of_the_teacher_first(self):
        self.hold("capital", self.AFRICA)
        questions = self.kind_replies(["country"] * 5, ["Country."] * 5, ["country"] * 5)
        with mock.patch.object(seeds, "dictionary_kind",
                               side_effect=AssertionError("The teacher named the kind")):
            self.assertEqual(self.kind(lexicon=self.lexicon), "country")
        self.assertEqual([qs for qs, _ in self.teacher.calls], [questions])
        self.assertEqual(self.stored()["kind"], "country")

    def test_the_teachers_none_is_stored_then_the_dictionary_names_the_kind(self):
        self.hold("capital", self.AFRICA)
        questions = self.kind_replies(["country"] * 5, ["nation"] * 5, ["state"] * 5)
        with mock.patch.object(seeds, "dictionary_kind", wraps=seeds.dictionary_kind) as named, \
                mock.patch.object(self.memory, "learn_kind", wraps=self.memory.learn_kind) as learn:
            self.assertEqual(self.kind(lexicon=self.lexicon), "African country")
        named.assert_called_once_with(self.lexicon, set(self.AFRICA))
        self.assertEqual(learn.call_args_list, [mock.call("capital", None),
                                                mock.call("capital", "African country")])
        self.assertEqual(self.stored()["kind"], "African country")
        # Learned, the kind is reused: nobody is asked again, lexicon or not.
        with mock.patch.object(seeds, "dictionary_kind",
                               side_effect=AssertionError("The dictionary was asked again")):
            self.assertEqual(self.kind(lexicon=self.lexicon), "African country")
            self.assertEqual(self.kind(), "African country")
        self.assertEqual([qs for qs, _ in self.teacher.calls], [questions])

    def test_a_stored_none_is_named_by_the_dictionary_without_a_teacher(self):
        self.hold("capital", self.AFRICA)
        self.memory.learn_kind("capital", None)
        with mock.patch.object(frontier, "_seed_questions",
                               side_effect=AssertionError("The teacher was asked")), \
                mock.patch.object(seeds, "dictionary_kind", wraps=seeds.dictionary_kind) as named:
            self.assertEqual(self.kind(lexicon=self.lexicon), "African country")
        named.assert_called_once_with(self.lexicon, set(self.AFRICA))
        self.assertEqual(self.stored()["kind"], "African country")
        self.assertEqual(self.teacher.calls, [])

    def test_without_a_lexicon_none_stays_none(self):
        self.hold("capital", self.AFRICA)
        questions = self.kind_replies(["country"] * 5, ["nation"] * 5, ["state"] * 5)
        with mock.patch.object(seeds, "dictionary_kind",
                               side_effect=AssertionError("No lexicon was given")):
            # The teacher's None, then the stored None.
            for extra in ({}, {"lexicon": None}):
                self.assertIsNone(self.kind(**extra))
                self.assertIsNone(self.stored()["kind"])
        self.assertEqual([qs for qs, _ in self.teacher.calls], [questions])

    def test_a_dictionary_naming_no_kind_leaves_the_stored_none(self):
        self.hold("capital", {"Atlantis": "Poseidonis", "Lemuria": "Unknown", "Mu": "Unknown"})
        questions = self.kind_replies(["island"] * 5, ["continent"] * 5, ["myth"] * 5)
        with mock.patch.object(self.memory, "learn_kind", wraps=self.memory.learn_kind) as learn:
            # The teacher's None, then the stored None: the dictionary names neither.
            self.assertIsNone(self.kind(lexicon=self.lexicon))
            self.assertIsNone(self.kind(lexicon=self.lexicon))
        learn.assert_called_once_with("capital", None)
        self.assertIsNone(self.stored()["kind"])
        self.assertEqual([qs for qs, _ in self.teacher.calls], [questions])

    def test_the_dictionary_waits_for_a_none_the_teacher_path_stored(self):
        # Fewer than three subjects, or no kind question: no teacher is asked,
        # nothing is stored, and so the dictionary is not asked either.
        self.hold("capital", {"Nigeria": "Abuja", "Kenya": "Nairobi"})
        with mock.patch.object(seeds, "dictionary_kind",
                               side_effect=AssertionError("The dictionary was asked")):
            self.assertIsNone(self.kind(lexicon=self.lexicon))
            self.hold("capital", {"Ivory Coast": "Yamoussoukro"})
            with mock.patch.object(frontier, "_seed_questions", return_value={}):
                self.assertIsNone(self.kind(lexicon=self.lexicon))
        self.assertNotIn("kind", self.stored())
        self.assertEqual(self.teacher.calls, [])

    def growth_world(self):
        """Five capitals the teachers named no kind for, and their populations."""
        self.hold("capital", {**self.AFRICA, "Ghana": "Accra"})
        self.memory.learn_kind("capital", None)
        question = frontier._seed_questions()["property"].format(kind="African country")
        self.teacher.replies[question] = ["population"] * 5
        for subject in (*self.AFRICA, "Ghana"):
            self.teacher.replies[f"What is the population of {subject}?"] = ["1000"] * 5
        return question

    def propose(self, **kwargs):
        return frontier.propose(
            self.memory, self.stash, self.teacher, self.ledger, "source", confidence=0.99,
            run_id="grow", records_path=self.home / "grow.jsonl", **kwargs)

    def test_propose_asks_the_property_of_the_kind_the_dictionary_names(self):
        question = self.growth_world()
        # Without the dictionary there is no kind, so nothing to ask.
        for extra in ({}, {"lexicon": None}):
            self.assertEqual(self.propose(**extra), {
                "proposed": [], "adopted": [], "refused": [], "asked": 0})
        self.assertEqual(self.teacher.calls, [])
        with mock.patch.object(frontier, "kind_of", wraps=frontier.kind_of) as kinds:
            result = self.propose(lexicon=self.lexicon)
        self.assertEqual([(call.args[1], call.kwargs) for call in kinds.call_args_list],
                         [("capital", {"lexicon": self.lexicon})])
        self.assertEqual(result, {"proposed": ["population"], "adopted": ["population"],
                                  "refused": [], "asked": 6})
        self.assertEqual(self.teacher.calls[0][0], [question])
        self.assertEqual(len(self.teacher.calls), 2)
        vocabulary = self.memory._attribute_vocabulary()
        self.assertEqual(vocabulary["capital"]["kind"], "African country")
        self.assertEqual(vocabulary["capital"]["properties"], {"population": "adopted"})
        self.assertEqual(vocabulary["population"]["kind"], "African country")

    def test_study_passes_its_lexicon_to_propose_and_omits_it_without_one(self):
        real = frontier.propose

        def todays(memory, stash, teacher, ledger, source, *, confidence, run_id,
                   records_path):
            # Today's signature: a seam without the keyword still fits.
            return real(memory, stash, teacher, ledger, source, confidence=confidence,
                        run_id=run_id, records_path=records_path)

        with mock.patch.object(frontier, "pending", return_value=[]):
            with mock.patch.object(frontier, "propose", side_effect=todays) as plain:
                self.study("omitted")
                self.study("none", lexicon=None)
            with mock.patch.object(frontier, "propose", wraps=real) as given:
                self.study("given", lexicon=self.lexicon)
        self.assertEqual([call.kwargs for call in plain.call_args_list], [
            {"confidence": 0.99, "run_id": run_id, "records_path": self.home / f"{run_id}.jsonl"}
            for run_id in ("omitted", "none")])
        given.assert_called_once()
        self.assertEqual(given.call_args.args[:3], (self.memory, self.stash, self.teacher))
        self.assertEqual(given.call_args.args[4], "source")
        self.assertEqual(given.call_args.kwargs, {
            "confidence": 0.99, "run_id": "given", "records_path": self.home / "given.jsonl",
            "lexicon": self.lexicon})

    def test_study_grows_a_property_of_a_kind_only_the_dictionary_names(self):
        question = self.growth_world()
        plain = self.study("plain")
        self.assertEqual((plain["asked"], plain["proposed"]), (0, []))
        self.assertEqual(self.teacher.calls, [])
        grown = self.study("grown", lexicon=self.lexicon)
        self.assertEqual((grown["asked"], grown["proposed"], grown["adopted"], grown["filed"]),
                         (6, ["population"], ["population"], 5))
        self.assertEqual(self.teacher.calls[0][0], [question])
        self.assertEqual(self.memory.held_value("Ghana", "population"), "1000")


class ShardedDictionaryTargetTests(DictionaryTargetTests):
    sharded = True


class ShardedFrontierLexiconTests(FrontierLexiconTests):
    sharded = True


class SessionLexiconTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.home / "library"
        self.root.mkdir()
        memory = SystematicMemory(self.root / "memory.json")
        memory.remember_fact("capital of somewhere", "Alpha", 0.99,
                             subject="Somewhere", attribute="capital")
        memory.save()
        question = "What is the capital of Somewhere?"
        self.pairs = [(targets.Target("capital", "Somewhere", question, "capital"), "Alpha")]
        self.teacher = FakeTeacher(self.home, {question: ["Alpha"] * 5})
        self.plan = [session.SourcePlan(name, str(self.teacher.spec.gguf), 4096)
                     for name in ("first", "second")]
        self.swapper = mock.Mock()
        self.swapper.snapshot.return_value = []
        # Unexpected external calls fail instead of reaching a process or network.
        self.enterContext(mock.patch("subprocess.run", side_effect=AssertionError("external process")))
        self.enterContext(mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")))

    def rounds(self):
        """Run two rounds a source; return each round's keyword arguments."""
        calls = []

        def study_round(memory, stash, teacher, ledger, source, **kwargs):
            calls.append(kwargs)
            return {"used_up": len(calls) % 2 == 0}

        with mock.patch.object(frontier, "study_round", side_effect=study_round):
            report = session.run_session(
                self.root, self.plan, swapper=self.swapper,
                teacher_factory=lambda source: self.teacher,
                backup_dir=self.home / "backups", report_path=self.home / "report.json",
                pairs=self.pairs)
        self.assertEqual(report["status"], "complete")
        return calls

    def test_every_round_gets_the_one_lexicon_the_session_opened(self):
        lexicon = FakeLexicon()
        with mock.patch.object(session.Lexicon, "open", return_value=lexicon) as opened:
            calls = self.rounds()
        opened.assert_called_once_with(self.root)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(kwargs["lexicon"] is lexicon for kwargs in calls))

    def test_a_home_without_a_dictionary_passes_none_and_creates_nothing(self):
        self.assertEqual([kwargs["lexicon"] for kwargs in self.rounds()], [None] * 4)
        self.assertFalse((self.root / "lexicon").exists())


if __name__ == "__main__":
    unittest.main()
