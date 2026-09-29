"""§11.141: catalogue answers and learned question words, without grammar.

Claude converted Astra's pytest functions to unittest, the project's
runner (``python -m unittest discover -s tests`` ran none of them), with
every assertion kept.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from ultraquant.interpreter.stash import ContemporaryStash
from ultraquant.interpreter.thoughts import build_session, run_pipeline
from ultraquant.memory import migrate
from ultraquant.memory.factshards import FactShards, question_words
from ultraquant.memory.systematic import SystematicMemory
from ultraquant.shards.vault import ShardVault


def hold(memory, subject, attribute="capital", value="v", key=None):
    key = key or f"{attribute} of {subject.lower()}"
    memory.remember_fact(key, value, confidence=.8, subject=subject,
                         attribute=attribute)
    return key


class _Scratch(unittest.TestCase):
    """Scratch directories, and both memory backings as subtests."""

    def scratch(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="uq_answers_test_"))
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def memories(self):
        # §11.141: all writes belong to scratch libraries; both backings
        # share the same pins.
        for label, sharded in (("ram", False), ("sharded", True)):
            root = self.scratch()
            shards = FactShards(ShardVault(root / "vault")) if sharded else None
            yield label, SystematicMemory(root / "memory.json", shards=shards), root


class ChooseSubjectTests(_Scratch):

    def test_choose_subject(self) -> None:
        cases = [
            (set(), None),
            ({"australia", "south australia"}, "south australia"),
            ({"mexico", "new mexico", "us state of new mexico"},
             "us state of new mexico"),
            ({"australia", "mexico"}, None),
            ({"art", "earth"}, None),
            ({"north tower", "north old tower"}, None),
            ({"tower", "north tower", "south tower"}, None),
        ]
        shards = FactShards(ShardVault(self.scratch() / "vault"))
        for subjects, expected in cases:
            with self.subTest(subjects=sorted(subjects)):
                self.assertEqual(shards._choose_subject(subjects), expected)


class CatalogueAnswerTests(_Scratch):

    def test_nested_subject_answers_and_unrelated_ambiguity(self) -> None:
        pairs = [("Australia", "Canberra"), ("South Australia", "Adelaide"),
                 ("Mexico", "Mexico City"), ("the US state of New Mexico", "Santa Fe")]
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                for subject, value in pairs:
                    hold(memory, subject, value=value)
                for subject, value in pairs:
                    answer = memory.catalogue_answer(f"What is {subject}'s capital?")
                    self.assertEqual(answer["record"]["value"], value)
                self.assertIsNone(
                    memory.catalogue_answer("capital of Australia and Mexico?"))

    def test_removes_subject_span_once_before_stopwords(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                subject = "the Australian Capital Territory"
                key = hold(memory, subject)
                self.assertEqual(
                    question_words("the capital of the Australian Capital Territory",
                                   "australian capital territory"), {"capital"})
                self.assertEqual(
                    memory.catalogue_answer(
                        "the capital of the Australian Capital Territory"),
                    {"form": "exact", "key": key, "record": memory.recall_fact(key)})
                memory.learn_asking(
                    "capital", subject,
                    "Which city is the capital of the Australian Capital Territory?")
                self.assertEqual(
                    memory._attribute_vocabulary()["capital"]["asked_by"],
                    {"city": ["australian capital territory"]})
                self.assertEqual(
                    question_words("north tower near north tower", "north tower"),
                    {"north", "tower", "near"})
                self.assertEqual(question_words("north old tower", "north tower"),
                                 {"north", "old", "tower"})

    def test_normalized_spans_are_not_normalized_twice(self) -> None:
        # §11.141: normalization strips one article, never another on a
        # second pass.
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                hold(memory, "the an echo", "author")
                memory.learn_asking("author", "the an echo", "Who wrote an echo?")
                self.assertEqual(
                    memory._attribute_vocabulary()["author"]["asked_by"],
                    {"wrote": ["an echo"]})
                self.assertEqual(
                    memory.catalogue_answer("author of an echo?")["form"], "exact")

    def test_asking_evidence_can_precede_fact_storage(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                memory.learn_asking("hue", "Echo", "Which chroma for Echo?")
                self.assertIsNone(memory.catalogue_answer("Which chroma for Zorbia?"))
                memory.learn_asking("hue", "Foxtrot", "Which chroma for Foxtrot?")
                hold(memory, "Echo", "hue", "amber")
                self.assertEqual(
                    memory.catalogue_answer("Which chroma for Echo?")["form"], "exact")

    def test_learning_is_idempotent_and_needs_two_normalized_subjects(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                key = hold(memory, "The Écho", "author", "Writer")
                hold(memory, "Foxtrot", "author")
                for subject in ("The Écho", "ECHO", "the echo"):
                    memory.learn_asking("The AUTHOR", subject,
                                        f"Who wrote the novel {subject}?")
                self.assertIsNone(memory.catalogue_answer("Who wrote Echo?"))
                self.assertEqual(
                    memory._attribute_vocabulary()["author"]["asked_by"],
                    {"novel": ["echo"], "wrote": ["echo"]})
                memory.learn_asking("author", "Foxtrot", "Who wrote the novel Foxtrot?")
                answer = memory.catalogue_answer("Who wrote Echo?")
                self.assertEqual((answer["form"], answer["key"]), ("exact", key))
                before = deepcopy(memory._attribute_vocabulary())
                memory.learn_asking("author", "Foxtrot", "Who wrote the novel Foxtrot?")
                self.assertEqual(memory._attribute_vocabulary(), before)

    def test_exact_reading_unique_best_tie_and_zero(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                symbol = hold(memory, "Zinc", "chemical symbol", "Zn")
                number = hold(memory, "Zinc", "atomic number", "30")
                self.assertEqual(
                    memory.catalogue_answer(
                        "Which chemical symbol does Zinc have?")["form"], "exact")
                answer = memory.catalogue_answer(
                    "Which shiny chemical symbol does Zinc have?")
                self.assertEqual((answer["form"], answer["key"]), ("reading", symbol))
                self.assertEqual(
                    memory.catalogue_answer("atomic number symbol of Zinc?")["key"],
                    number)
                self.assertIsNone(memory.catalogue_answer("number symbol of Zinc?"))
                self.assertIsNone(memory.catalogue_answer("Who wrote Zinc?"))
                hold(memory, "Zinc", "chemical symbol", "Zn", key="duplicate symbol")
                self.assertIsNone(memory.catalogue_answer("chemical symbol of Zinc?"))

    def test_unknown_subject_requires_attribute_and_library_unknown_words(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                hold(memory, "Kenya", value="Nairobi")
                hold(memory, "Ghana", value="Accra")
                self.assertEqual(
                    memory.catalogue_answer("What is the capital of Zorbia?"),
                    {"form": "unknown-subject"})
                self.assertIsNone(memory.catalogue_answer("Who wrote Zorbia?"))
                self.assertIsNone(memory.catalogue_answer("capital of Kenya and Ghana?"))
                for subject in ("Kenya", "Ghana"):
                    memory.learn_asking("capital", subject,
                                        f"Which city is the capital of {subject}?")
                self.assertEqual(memory.catalogue_answer("Which city for Zorbia?"),
                                 {"form": "unknown-subject"})
                memory.remember_fact("freedonia capital", "Sylvania")
                self.assertIsNone(
                    memory.catalogue_answer("What is the capital of Freedonia?"))
                # §11.141: every other word must be unknown, including shared
                # qualifiers.
                hold(memory, "the US state of Washington", value="Olympia")
                # §11.142: words shared with held subjects' names no longer make a subject known
                self.assertEqual(memory.catalogue_answer(
                    "What is the capital of the US state of Zorbia?"),
                    {"form": "unknown-subject"})

    def test_unstructured_facts_never_answer_through_catalogue(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                memory.remember_fact("freedonia capital", "Sylvania")
                self.assertIsNone(
                    memory.catalogue_answer("What is the capital of Freedonia?"))
                self.assertIsNone(
                    memory.catalogue_answer("What is the capital of Zorbia?"))


class PersistenceAndLearningTests(_Scratch):

    def test_question_words_persist_in_same_batch_and_retry(self) -> None:
        root = self.scratch()
        memory = SystematicMemory(root / "memory.json",
                                  shards=FactShards(ShardVault(root / "vault")))
        hold(memory, "Echo", "author")
        memory.learn_asking("author", "Echo", "Who wrote Echo?")
        memory.save()
        hold(memory, "Foxtrot", "author")
        memory.learn_asking("author", "Foxtrot", "Who wrote Foxtrot?")
        vault = memory.shards.vault
        real_add = vault.add_shard

        def fail_attribute(shard_id, *args, **kwargs):
            if shard_id == "index:attributes":
                raise OSError("injected asking index failure")
            return real_add(shard_id, *args, **kwargs)

        with mock.patch.object(vault, "add_shard", side_effect=fail_attribute):
            with self.assertRaisesRegex(OSError, "injected asking index failure"):
                memory.save()
        fresh = FactShards(ShardVault(root / "vault"))
        self.assertIsNone(fresh.get("author of foxtrot"))
        self.assertEqual(
            fresh._index_data("attributes")["author"]["asked_by"]["wrote"], ["echo"])
        memory.save()
        fresh = SystematicMemory(root / "memory.json",
                                 shards=FactShards(ShardVault(root / "vault")))
        self.assertEqual(fresh.catalogue_answer("Who wrote Echo?")["form"], "exact")
        with mock.patch.object(fresh.shards.vault, "add_shard",
                               wraps=fresh.shards.vault.add_shard) as add:
            fresh.learn_asking("author", "Foxtrot", "Who wrote Foxtrot?")
            fresh.save()
            add.assert_not_called()

    def test_promotion_and_migration_learn_only_eligible_titles(self) -> None:
        for label, memory, root in self.memories():
            with self.subTest(memory=label):
                stash = ContemporaryStash(root / "stash.json")
                for subject in ("Echo", "Foxtrot", "Staged", "Blank", "Unstructured"):
                    fields = {"key": f"author of {subject.lower()}", "value": "Writer"}
                    if subject != "Unstructured":
                        fields.update(subject=subject, attribute="author")
                    title = "  " if subject == "Blank" else f"Who wrote the novel {subject}?"
                    eid = stash.add_claim("https://example.test/record", title,
                                          f"author of {subject.lower()} is Writer",
                                          fields=fields)
                    if subject != "Staged":
                        with mock.patch.object(memory, "learn_asking",
                                               wraps=memory.learn_asking) as learn:
                            stash.promote(eid, memory)
                            self.assertEqual(learn.call_count,
                                             int(subject in ("Echo", "Foxtrot")))
                before = {key: memory.recall_fact(key) for key in memory.fact_keys()}
                vocabulary = memory._attribute_vocabulary()
                vocabulary["author"].pop("asked_by")
                self.assertIsNone(memory.catalogue_answer("Who wrote Echo?"))
                counts = migrate.learn_question_forms(memory, stash)
                self.assertEqual(counts, {"entries": 2, "subjects": 2, "attributes": 1})
                learned = deepcopy(memory._attribute_vocabulary())
                self.assertEqual(learned["author"]["asked_by"],
                                 {"novel": ["echo", "foxtrot"],
                                  "wrote": ["echo", "foxtrot"]})
                self.assertEqual(migrate.learn_question_forms(memory, stash), counts)
                self.assertEqual(memory._attribute_vocabulary(), learned)
                self.assertEqual(
                    {key: memory.recall_fact(key) for key in memory.fact_keys()}, before)


class ChatCatalogueTests(_Scratch):

    def test_chat_catalogue_formats_trace_curiosity_and_native_fallback(self) -> None:
        session = build_session(self.scratch() / "session", seed=0)
        for subject in ("Echo", "Foxtrot"):
            hold(session.memory, subject, "author", "Writer")
            session.memory.learn_asking("author", subject,
                                        f"Who wrote the novel {subject}?")
        questions = [
            ("Who wrote Echo?", "author of echo is Writer (confidence 0.80)."),
            ("Who wrote the strange novel Echo?",
             "Reading that as 'author of echo': author of echo is Writer "
             "(confidence 0.80)."),
            ("Who wrote Zorbia?",
             "I don't hold that: nothing it names is in my catalogue."),
        ]
        with mock.patch("ultraquant.reason.inference.infer",
                        side_effect=AssertionError("inference")), \
                mock.patch("ultraquant.reason.inference.missing_premise",
                           side_effect=AssertionError("curiosity")):
            for question, expected in questions:
                with self.subTest(question=question):
                    reply, trace = run_pipeline(question, session)
                    self.assertEqual(reply, expected)
                    self.assertIn("catalogue", str(trace))
                    self.assertEqual(session.curiosities, [])
        hold(session.memory, "Kenya", value="Nairobi")
        run_pipeline("freedonia capital is Sylvania", session)
        # §11.141: use the chat key's form to pin the unchanged recall path.
        reply, _ = run_pipeline("What is Freedonia capital?", session)
        self.assertEqual(reply, "freedonia capital is Sylvania (confidence 0.60).")

    def test_chat_exact_key_still_precedes_catalogue(self) -> None:
        session = build_session(self.scratch() / "session", seed=0)
        session.memory.remember_fact("tower height", "300", confidence=.8)
        with mock.patch.object(session.memory, "catalogue_answer",
                               side_effect=AssertionError("exact recall must win")):
            reply, _ = run_pipeline("What is the tower height?", session)
        self.assertEqual(reply, "tower height is 300 (confidence 0.80).")


# §11.142: requests and unheld subjects use the same catalogue in both stores.
class CatalogueRequestTests(_Scratch):

    def test_catalogue_request_returns_only_exact_answers(self) -> None:
        memory = SystematicMemory()
        for answer in (None, {"form": "unknown-subject"},
                       {"form": "reading", "key": "k", "record": {}},
                       {"form": "exact", "key": "k", "record": {}}):
            with self.subTest(answer=answer), mock.patch.object(
                    memory, "catalogue_answer", return_value=answer) as catalogue:
                expected = answer if answer and answer["form"] == "exact" else None
                self.assertIs(memory.catalogue_request("request"), expected)
                catalogue.assert_called_once_with("request")

    def test_catalogue_request_filters_real_readings_and_refusals(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                key = hold(memory, "Kenya", value="Nairobi")
                self.assertEqual(memory.catalogue_request("Tell me the capital of Kenya."),
                                 {"form": "exact", "key": key,
                                  "record": memory.recall_fact(key)})
                visited = "I visited the capital of Kenya last year."
                self.assertEqual(memory.catalogue_answer(visited)["form"], "reading")
                self.assertIsNone(memory.catalogue_request(visited))
                self.assertIsNone(memory.catalogue_request("Tell me the capital of Zorbia."))
                self.assertIsNone(memory.catalogue_request("Hello there."))

    def test_chat_intent_requests_use_catalogue_format_and_polarity(self) -> None:
        session = build_session(self.scratch(), seed=0)
        key = hold(session.memory, "Kenya", value="Nairobi")
        for negated in (False, True):
            with self.subTest(negated=negated):
                session.memory.remember_fact(key, "Nairobi", confidence=.8,
                                             negated=negated, subject="Kenya",
                                             attribute="capital")
                record = session.memory.recall_fact(key)
                with mock.patch.object(session.memory, "catalogue_request",
                                       wraps=session.memory.catalogue_request) as request:
                    reply, trace = run_pipeline("Tell me the capital of Kenya.", session)
                request.assert_called_once_with("Tell me the capital of Kenya.")
                shown = "not Nairobi" if negated else "Nairobi"
                # §11.145: chat mentions the catalogue's fact; it never asserts it.
                self.assertEqual(reply, f"That lands near '{key}', which I hold as: "
                                 f"{shown}.")
                self.assertNotIn(f"{record['confidence']:.2f}", reply)
                self.assertIn("intent=chat", str(trace))
                self.assertIn("catalogue mention", str(trace))
                self.assertEqual(session.curiosities, [])

    def test_nonexact_chat_keeps_reply_without_catalogue_trace(self) -> None:
        session = build_session(self.scratch(), seed=0)
        hold(session.memory, "Kenya", value="Nairobi")
        for message in ("I visited the capital of Kenya last year.",
                        "Tell me the capital of Zorbia.", "Hello there."):
            with self.subTest(message=message):
                with mock.patch.object(session.memory, "catalogue_request", return_value=None):
                    expected, _ = run_pipeline(message, session)
                reply, trace = run_pipeline(message, session)
                self.assertEqual(reply, expected)
                self.assertIn("intent=chat", str(trace))
                self.assertNotIn("catalogue", str(trace))
                self.assertEqual(session.curiosities, [])


# §11.142: shared qualifiers need one unstructured key covering all other words.
class UnheldSubjectTests(_Scratch):

    def test_empty_others_never_probes(self) -> None:
        memory = SystematicMemory()
        with mock.patch.object(memory, "find_facts") as find:
            self.assertFalse(memory._unheld_subject({"capital"}, {"capital"}))
            self.assertFalse(memory._unheld_subject(set(), {"capital"}))
        find.assert_not_called()

    def test_shared_qualifiers_refuse_but_chat_taught_subject_survives(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                hold(memory, "the US state of Washington", value="Olympia")
                query = "What is the capital of the US state of Zorbia?"
                self.assertTrue(memory._unheld_subject(question_words(query, ""),
                                                       {"capital"}))
                self.assertEqual(memory.catalogue_answer(query), {"form": "unknown-subject"})
                memory.remember_fact("freedonia capital", "Fredville")
                self.assertFalse(memory._unheld_subject({"freedonia", "capital"},
                                                        {"capital"}))
                self.assertIsNone(memory.catalogue_answer("What is the capital of Freedonia?"))

    def test_one_unstructured_key_must_cover_all_others(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                memory.remember_fact("north capital", "N")
                memory.remember_fact("freedonia capital", "F")
                words, known = {"north", "freedonia", "capital"}, {"capital"}
                self.assertTrue(memory._unheld_subject(words, known))
                hold(memory, "Elsewhere", key="north freedonia capital")
                self.assertTrue(memory._unheld_subject(words, known))
                memory.remember_fact("north-freedonia seat", "Fredville")
                self.assertFalse(memory._unheld_subject(words, known))

    # §11.145: absence comes from an exhaustive index lookup, never a ranked top 10.
    def test_exhaustive_lookup_skips_missing_records_without_ranking(self) -> None:
        for label, memory, _root in self.memories():
            with self.subTest(memory=label):
                memory.remember_fact("Fréedonia-Capital", "Fredville")
                words = {"capital", "fr", "edonia"}   # the index's own tokens
                real = memory.recall_fact
                with mock.patch.object(memory, "find_facts",
                                       side_effect=AssertionError("ranked")), \
                        mock.patch.object(memory, "recall_fact",
                                          side_effect=lambda key: None):
                    self.assertTrue(memory._unheld_subject(words, {"capital"}))
                with mock.patch.object(memory, "find_facts",
                                       side_effect=AssertionError("ranked")), \
                        mock.patch.object(memory, "recall_fact", side_effect=real):
                    self.assertFalse(memory._unheld_subject(words, {"capital"}))


# §11.142: covering candidates win in order before embeddings or nearest-held.
class FirstCoveringTests(_Scratch):

    def test_candidate_order_skips_partial_and_missing_keys(self) -> None:
        from ultraquant.interpreter.thoughts import _first_covering

        memory = SystematicMemory()
        for key in ("capital of ghana", "freedonia capital", "freedonia capital seat"):
            memory.remember_fact(key, "value")
        candidates = ["capital of ghana", "missing freedonia capital",
                      "freedonia capital seat", "freedonia capital"]
        asked = {"freedonia", "capital"}
        self.assertEqual(_first_covering(candidates, asked, memory), "freedonia capital seat")
        self.assertEqual(_first_covering(list(reversed(candidates)), asked, memory),
                         "freedonia capital")
        self.assertIsNone(_first_covering(candidates[:2], asked, memory))
        self.assertIsNone(_first_covering([], asked, memory))

    def test_covering_uses_existing_informative_plural_normalization(self) -> None:
        from ultraquant.interpreter.thoughts import _first_covering

        memory = SystematicMemory()
        memory.remember_fact("The TOWERS height", "300")
        self.assertEqual(_first_covering(["The TOWERS height"], {"tower", "height"}, memory),
                         "The TOWERS height")
        self.assertIsNone(_first_covering(["The TOWERS height"], {"the", "height"}, memory))

    def test_chat_taught_capital_answers_past_structured_partial_match(self) -> None:
        session = build_session(self.scratch(), seed=0)
        hold(session.memory, "Kenya", value="Nairobi")
        hold(session.memory, "Ghana", value="Accra")
        run_pipeline("freedonia capital is Fredville", session)
        session.semantic = mock.Mock()
        reply, trace = run_pipeline("What is the capital of Freedonia?", session)
        self.assertEqual(reply, "freedonia capital is Fredville (confidence 0.60).")
        self.assertIn("answered from keyword fact", str(trace))
        session.semantic.suggest.assert_not_called()

    def test_embedding_attempt_once_then_first_sharing_candidate(self) -> None:
        session = build_session(self.scratch(), seed=0)
        session.memory.remember_fact("copper weight", "10", .8)
        session.memory.remember_fact("iron weight", "20", .8)
        session.semantic = mock.Mock()
        session.semantic.suggest.return_value = None
        with mock.patch("ultraquant.reason.inference.infer", return_value=None), \
                mock.patch("ultraquant.reason.inference.missing_premise", return_value={
                    "premise_key": "alloy weight", "via_key": "copper weight"}), \
                mock.patch.object(session.memory, "find_facts",
                                  return_value=["copper weight", "iron weight"]):
            reply, _ = run_pipeline("What is the alloy weight?", session)
        session.semantic.suggest.assert_called_once_with("What is the alloy weight?",
                                                        session.memory)
        self.assertTrue(reply.startswith("I don't hold that exactly. Nearest I hold: "
                                         "copper weight is 10 (confidence 0.80)."))
        self.assertIn("If I knew the alloy weight", reply)


class EmptyQuestionTests(_Scratch):
    """Claude's review of §11.142: a question without informative tokens."""

    def test_an_empty_question_asserts_nothing(self) -> None:
        # "What is it?" folds to no informative token. The coverage pass
        # must not assert whatever key find_facts happens to rank first.
        session = build_session(self.scratch() / "session", seed=0)
        session.memory.remember_fact(
            "author of the moon is a harsh mistress", "Robert Heinlein", 0.9)
        reply, _ = run_pipeline("What is it?", session)
        self.assertNotIn("Robert Heinlein", reply)


if __name__ == "__main__":
    unittest.main()
