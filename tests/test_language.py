"""Grounded language: induced from examples, spoken about what is seen.

The gate's verdict is in ARCHITECTURE.md §11.10. Asserted here is the machinery
and the properties that make the result mean something — above all that there
is no template: the hidden languages' distinctive words must not appear in the
learner's source, and the same code must learn two structurally different
languages.
"""

from __future__ import annotations

import json
import pathlib
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.experiments import language_gate as gate
from ultraquant.reason.language import GroundedLanguage, store_language, verbalise


def _pairs(grammar_name: str, meanings: list[list[str]]) -> list[tuple[str, list[str]]]:
    grammar = gate.GRAMMARS[grammar_name]
    return [(gate.hidden_utterance(levels, grammar), levels) for levels in meanings]


class NoTemplateTests(unittest.TestCase):
    """The claim the whole stage rests on."""

    def test_the_hidden_languages_appear_nowhere_in_the_learner(self) -> None:
        """A template would need the words; induction never sees this file.

        A template has to live in a string literal, so the assertion covers
        every non-docstring string constant and every identifier in the
        module. Docstring prose is exempt — the first run of this test failed
        on the English word "inside" in a sentence about packed libraries,
        which is not a template.
        """
        import ast

        source = pathlib.Path("ultraquant/reason/language.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in docstrings
        ]
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for word in ("inside", "holding", "saltire", "speck", "grill", "nooks",
                     "a plus", "a box"):
            for literal in literals:
                self.assertNotIn(
                    word, literal,
                    f"{word!r} belongs to a hidden language, not the learner",
                )
            self.assertNotIn(word, names)

    def test_the_same_code_learns_two_different_languages(self) -> None:
        """One baked-in order could satisfy one grammar; never both."""
        train = gate.meanings(2) + gate.meanings(3)
        levels = ["rails", "box", "dot"]
        outputs = {}
        for name in ("nested", "holding"):
            model = GroundedLanguage()
            model.fit(_pairs(name, train))
            outputs[name] = model.generate(levels)
            self.assertEqual(outputs[name],
                             gate.hidden_utterance(levels, gate.GRAMMARS[name]))
        self.assertNotEqual(outputs["nested"], outputs["holding"])

    def test_order_is_learned_not_assumed(self) -> None:
        train = gate.meanings(2) + gate.meanings(3)
        nested = GroundedLanguage()
        nested.fit(_pairs("nested", train))
        holding = GroundedLanguage()
        holding.fit(_pairs("holding", train))
        self.assertTrue(nested.mark_first)
        self.assertTrue(nested.inner_first)
        self.assertFalse(holding.mark_first)
        self.assertFalse(holding.inner_first)


class InductionTests(unittest.TestCase):
    """Lexicon, frame, and both directions."""

    def setUp(self) -> None:
        self.train = gate.meanings(2) + gate.meanings(3)
        self.model = GroundedLanguage()
        self.model.fit(_pairs("nested", self.train))

    def test_alignment_finds_words_that_differ_from_atom_names(self) -> None:
        """String matching cannot substitute for learned alignment."""
        self.assertEqual(self.model.lexicon[("mark", "ex")], "saltire")
        self.assertEqual(self.model.lexicon[("container", "rails")], "grill")

    def test_function_words_are_frame_not_lexicon(self) -> None:
        content = set(self.model.reverse)
        self.assertNotIn("a", content)
        self.assertNotIn("inside", content)

    def test_generation_matches_the_hidden_grammar_exactly(self) -> None:
        for levels in (["box", "plus"], ["caps", "rails", "dot"]):
            with self.subTest(levels=levels):
                self.assertEqual(
                    self.model.generate(levels),
                    gate.hidden_utterance(levels, gate.GRAMMARS["nested"]),
                )

    def test_depth_never_seen_is_still_spoken(self) -> None:
        """Role-pair gaps are what make the recursion extrapolate."""
        levels = ["box", "rails", "caps", "ex"]  # depth 4; trained on 2-3
        self.assertEqual(
            self.model.generate(levels),
            gate.hidden_utterance(levels, gate.GRAMMARS["nested"]),
        )

    def test_a_bare_thing_is_a_depth_one_meaning(self) -> None:
        """Whole-pattern labels are meanings with no containers at all."""
        model = GroundedLanguage()
        model.fit([
            ("a quad", ["square"]), ("a lozenge", ["diamond"]),
            ("a speck within a box", ["box", "dot"]),
            ("a plus within a grill", ["rails", "plus"]),
        ])
        self.assertEqual(model.generate(["square"]), "a quad")
        self.assertEqual(model.parse("a lozenge"), ["diamond"])
        self.assertEqual(model.generate(["box", "dot"]), "a speck within a box")

    def test_parse_inverts_generation(self) -> None:
        for levels in (["box", "plus"], ["rails", "caps", "bar"]):
            with self.subTest(levels=levels):
                self.assertEqual(self.model.parse(self.model.generate(levels)),
                                 levels)

    def test_a_missing_word_is_a_refusal_not_a_hole(self) -> None:
        self.assertIsNone(self.model.generate(["box", "unknown_mark"]))

    def test_nonsense_parses_to_none(self) -> None:
        self.assertIsNone(self.model.parse("wibble wobble"))
        self.assertIsNone(self.model.parse("a box inside a grill"))  # no mark

    def test_state_round_trips_through_json(self) -> None:
        restored = GroundedLanguage.from_state_dict(
            json.loads(json.dumps(self.model.state_dict()))
        )
        levels = ["caps", "box", "dot"]
        self.assertEqual(restored.generate(levels), self.model.generate(levels))
        self.assertEqual(restored.parse(restored.generate(levels)), levels)


class GroundingTests(unittest.TestCase):
    """Speech about what the library actually perceives."""

    def setUp(self) -> None:
        from ultraquant.experiments.composition_gate import (
            SHAPES, MARKS, compound, split_combinations, _samples,
        )
        from ultraquant.experts.moe import ExpertPool
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_lang_"))
        self.session = build_session(self.dir / "home", seed=0)
        rng = random.Random(2)
        self.seen, self.unseen = split_combinations(rng, held_out=6)
        train_x, train_y = _samples(rng, self.seen, 24, 2)
        shapes, marks = sorted(SHAPES), sorted(MARKS)
        self.session.experts.train_expert(
            "border", train_x, [shapes.index(s) for s, _ in train_y], shapes,
            epochs=40, lr=0.05)
        self.session.experts.train_expert(
            "inner", train_x, [marks.index(m) for _, m in train_y], marks,
            epochs=40, lr=0.05)
        prototypes = [compound(s, m) for s, m in self.seen]
        for category, slot in (("border", "shape"), ("inner", "mark")):
            shard = ExpertPool.shard_id(category)
            self.session.vault.set_slot(shard, slot)
            self.session.vault.set_signature(shard, prototypes)

        model = GroundedLanguage()
        model.fit([
            (f"a {mark} inside a {shape}", [shape, mark])
            for shape, mark in self.seen
        ])
        store_language(self.session, model,
                       {"shape": "container", "mark": "mark"})

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_language_lives_in_the_library(self) -> None:
        """Learned data belongs in the shard store — the fact-shards lesson."""
        entry = self.session.vault.entry("language:model")
        self.assertEqual(entry["kind"], "language-model")

    def test_verbalise_speaks_the_composed_reading(self) -> None:
        sentence = verbalise(self.session, {"shape": "box", "mark": "plus"})
        self.assertEqual(sentence, "a plus inside a box")

    def test_verbalise_declines_without_a_full_reading(self) -> None:
        self.assertIsNone(verbalise(self.session, {"shape": "box"}))

    def test_chat_says_what_it_sees_for_an_unseen_scene(self) -> None:
        """The grounding, end to end: novel percept -> composed reading ->
        induced sentence. No template produced these words in this order."""
        from ultraquant.experiments.composition_gate import compound
        from ultraquant.interpreter.thoughts import run_pipeline

        shape, mark = next(
            (s, m) for s, m in self.unseen if (s, m) == ("box", "plus")
        ) if ("box", "plus") in self.unseen else self.unseen[-1]
        pixels = compound(shape, mark)
        rows = ["".join("#" if pixels[r * 5 + c] else "." for c in range(5))
                for r in range(5)]
        response, _trace = run_pipeline("\n".join(rows), self.session)
        said = [line for line in response.splitlines() if line.startswith("I see")]
        self.assertTrue(said, f"no verbalisation in: {response}")
        self.assertIn(mark, said[0])
        self.assertIn(shape, said[0])

    def test_a_session_without_a_language_stays_silent(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        bare = build_session(self.dir / "bare", seed=0)
        self.assertIsNone(verbalise(bare, {"shape": "box", "mark": "plus"}))


class MixedLibraryTests(unittest.TestCase):
    """Whole-pattern and factored interpretations sharing one library.

    The default slot is not an aspect — it is the competing *whole* reading.
    When factored experts joined a library of plain categories, a plain
    'square' pushed the shape expert past 0.95 (a square IS the box border)
    and 3 of 8 plain glyphs were mangled into three-slot compositions. The
    rule these tests hold: the interpretation with the single best route wins
    outright, and the whole slot never joins a composition.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from ultraquant.experiments.composition_gate import (
            MARKS, SHAPES, compound, _samples,
        )
        from ultraquant.experts.moe import ExpertPool
        from ultraquant.forge.corpus import (
            BUILTIN_KEYWORDS, build_corpora, builtin_taxonomy,
        )
        from ultraquant.forge.forge import ModelForge
        from ultraquant.interpreter.thoughts import build_session

        cls.dir = Path(tempfile.mkdtemp(prefix="uq_mixedlib_"))
        forge = ModelForge(cls.dir / "home", seed=0, hidden=16, tier="auto")
        forge.build(
            build_corpora(builtin_taxonomy(), n_per_class=24, seed=1,
                          keywords=BUILTIN_KEYWORDS),
            epochs=25,
        )
        cls.session = build_session(cls.dir / "home", seed=0)
        rng = random.Random(3)
        pairs = [(shape, mark) for shape in sorted(SHAPES)
                 for mark in sorted(MARKS)]
        train_x, train_y = _samples(rng, pairs, 24, 2)
        shapes, marks = sorted(SHAPES), sorted(MARKS)
        cls.session.experts.train_expert(
            "border", train_x, [shapes.index(s) for s, _ in train_y], shapes,
            epochs=40, lr=0.05)
        cls.session.experts.train_expert(
            "inner", train_x, [marks.index(m) for _, m in train_y], marks,
            epochs=40, lr=0.05)
        prototypes = [compound(s, m) for s, m in pairs]
        for category, slot in (("border", "shape"), ("inner", "mark")):
            shard = ExpertPool.shard_id(category)
            cls.session.vault.set_slot(shard, slot)
            cls.session.vault.set_signature(shard, prototypes)

        from ultraquant.forge.languages import deploy_languages

        deploy_languages(cls.session, shapes, marks)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _ask(self, rows: list[str]) -> str:
        from ultraquant.interpreter.thoughts import run_pipeline

        return run_pipeline("\n".join(rows), self.session)[0]

    def test_plain_glyphs_are_not_mangled_into_compositions(self) -> None:
        """The regression: a square must stay 'square', not become a scene.

        'cross' is exempt for a measured reason — its pixels are identical to
        the corners+ex compound, so both interpretations are true of it.
        """
        from ultraquant.pattern.recognition import PATTERNS

        for glyph in PATTERNS:
            if glyph == "cross":
                continue
            with self.subTest(glyph=glyph):
                response = self._ask(PATTERNS[glyph])
                self.assertIn(f"'{glyph}'", response)
                self.assertNotIn("shape:", response)

    def test_the_genuinely_dual_image_gets_a_correct_reading(self) -> None:
        """Plain 'cross' IS corners+ex, pixel for pixel. Either answer is
        right; the rule prefers the factored one, and it must be correct."""
        from ultraquant.experiments.composition_gate import compound
        from ultraquant.pattern.recognition import PATTERNS, render

        self.assertEqual(render(PATTERNS["cross"]), compound("corners", "ex"))
        response = self._ask(PATTERNS["cross"])
        self.assertIn("shape:corners", response)
        self.assertIn("mark:ex", response)

    def test_compound_scenes_never_include_the_whole_slot(self) -> None:
        """'pattern:cross' is an alternative reading, not a third aspect."""
        from ultraquant.experiments.composition_gate import compound

        for shape, mark in (("box", "plus"), ("rails", "dot"), ("caps", "bar")):
            with self.subTest(scene=(shape, mark)):
                pixels = compound(shape, mark)
                rows = ["".join("#" if pixels[r * 5 + c] else "."
                                for c in range(5)) for r in range(5)]
                response = self._ask(rows)
                self.assertIn(f"shape:{shape} + mark:{mark}", response)
                self.assertNotIn("pattern:", response)

    def test_whole_patterns_are_spoken_in_every_language(self) -> None:
        """'a square' in plain, 'a quad' in the coined languages."""
        from ultraquant.pattern.recognition import PATTERNS

        response = self._ask(PATTERNS["square"])
        self.assertIn("I see a square", response)
        self.assertIn("a quad", response)

    def test_synthetic_style_labels_stay_silent(self) -> None:
        """A language must not say what it cannot mean; uncovered labels are
        simply not verbalised rather than given an ambiguous word."""
        from ultraquant.reason.language import verbalise_all

        spoken = verbalise_all(self.session, {"pattern": "sym_0"})
        self.assertEqual(spoken, {})

    def test_compound_scenes_are_spoken(self) -> None:
        from ultraquant.experiments.composition_gate import compound

        pixels = compound("box", "plus")
        rows = ["".join("#" if pixels[r * 5 + c] else "." for c in range(5))
                for r in range(5)]
        response = self._ask(rows)
        self.assertIn("I see a plus inside a box", response)


class MultiLanguageTests(unittest.TestCase):
    """Several languages in one library, each its own shard."""

    def setUp(self) -> None:
        from ultraquant.interpreter.thoughts import build_session

        self.dir = Path(tempfile.mkdtemp(prefix="uq_multilang_"))
        self.session = build_session(self.dir, seed=0)
        train = gate.meanings(2) + gate.meanings(3)
        for name in ("nested", "holding"):
            model = GroundedLanguage()
            model.fit(_pairs(name, train))
            store_language(self.session, model,
                           {"shape": "container", "mark": "mark"}, name=name)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_each_language_is_its_own_shard(self) -> None:
        from ultraquant.reason.language import list_languages

        self.assertEqual(list_languages(self.session), ["holding", "nested"])
        for name in ("nested", "holding"):
            self.assertEqual(
                self.session.vault.entry(f"language:{name}")["kind"],
                "language-model",
            )

    def test_verbalise_all_speaks_every_stored_language(self) -> None:
        from ultraquant.reason.language import verbalise_all

        spoken = verbalise_all(self.session, {"shape": "box", "mark": "dot"})
        self.assertEqual(set(spoken), {"nested", "holding"})
        self.assertNotEqual(spoken["nested"], spoken["holding"])
        self.assertEqual(
            spoken["nested"],
            gate.hidden_utterance(["box", "dot"], gate.GRAMMARS["nested"]),
        )

    def test_a_named_language_can_be_asked_for(self) -> None:
        sentence = verbalise(self.session, {"shape": "rails", "mark": "ex"},
                             language="holding")
        self.assertEqual(
            sentence,
            gate.hidden_utterance(["rails", "ex"], gate.GRAMMARS["holding"]),
        )


class GateIntegrityTests(unittest.TestCase):
    """The experiment must stay able to give an honest answer."""

    def test_held_out_combinations_never_appear_in_training(self) -> None:
        train, held = gate._split(random.Random(0))
        train_set = {tuple(levels) for levels in train}
        for levels in held:
            self.assertNotIn(tuple(levels), train_set)

    def test_every_atom_of_a_held_out_meaning_was_seen(self) -> None:
        train, held = gate._split(random.Random(0))
        seen = {value for levels in train for value in levels}
        for levels in held:
            for value in levels:
                self.assertIn(value, seen)

    def test_the_two_grammars_produce_different_sentences(self) -> None:
        levels = ["rails", "box", "dot"]
        self.assertNotEqual(
            gate.hidden_utterance(levels, gate.GRAMMARS["nested"]),
            gate.hidden_utterance(levels, gate.GRAMMARS["holding"]),
        )

    def test_the_memoriser_scores_zero_on_held_out_by_construction(self) -> None:
        result = gate.one_trial(0)
        self.assertEqual(result["memoriser"], 0.0)


if __name__ == "__main__":
    unittest.main()
