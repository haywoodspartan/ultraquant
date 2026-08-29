"""The suggester with the network unplugged.

§11.112 wired the distilled encoder into the live question path. These
pins hold the pluggable-embedder contract, the fact that the LM Studio
path is untouched, and the caveat the run surfaced: the cosine floor
never binds, so the anchor rule is doing the filtering.
"""

from __future__ import annotations

import unittest

from ultraquant.model.distilled import DistilledEmbedder, DistilledEncoder
from ultraquant.reason.semantic import COSINE_FLOOR, SemanticSuggester


def _embedder(width: int = 6, dim: int = 8) -> DistilledEmbedder:
    return DistilledEmbedder(
        DistilledEncoder(width, dim=dim, hidden=4, seed=0),
        ["tower", "height", "bridge", "length", "steel", "hardness"])


class EmbedderTests(unittest.TestCase):
    """A drop-in for the client, indistinguishable to the suggester."""

    def test_it_returns_one_vector_per_input(self) -> None:
        got = _embedder().embed(["the tower height", "the bridge length"])
        self.assertEqual(len(got), 2)
        self.assertEqual(len(got[0]), 8)

    def test_a_bare_string_is_accepted(self) -> None:
        self.assertEqual(len(_embedder().embed("the tower height")), 1)

    def test_the_model_argument_is_accepted_and_ignored(self) -> None:
        """The client's signature, so the suggester needs no branch."""
        embedder = _embedder()
        self.assertEqual(embedder.embed(["a"], model="anything"),
                         embedder.embed(["a"]))

    def test_it_is_always_available(self) -> None:
        self.assertTrue(_embedder().available())

    def test_unknown_words_contribute_nothing(self) -> None:
        """§11.111's boundary, in the adapter."""
        embedder = _embedder()
        self.assertEqual(embedder.embed(["the tower height"]),
                         embedder.embed(["the tower height zzzqqq"]))

    def test_it_round_trips_through_a_state_dict(self) -> None:
        embedder = _embedder()
        back = DistilledEmbedder.from_state_dict(embedder.state_dict())
        self.assertEqual(back.embed(["the tower height"]),
                         embedder.embed(["the tower height"]))

    def test_it_touches_no_network(self) -> None:
        """The CODE, not the prose.

        The first version of this grepped the source and matched the
        docstring, which names `LMStudioClient.embed` to say what
        interface is being imitated. Docstrings are stripped here so
        the pin tests behaviour rather than vocabulary.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(DistilledEmbedder))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value,
                                                         ast.Constant):
                node.value = ast.Constant(value="")
        code = ast.unparse(tree)
        for forbidden in ("LMStudio", "urlopen", "requests", "socket",
                          "http"):
            self.assertNotIn(forbidden, code)


class SuggesterTests(unittest.TestCase):
    """Pluggable, without disturbing the path that shipped."""

    def test_an_embedder_bypasses_the_client_entirely(self) -> None:
        suggester = SemanticSuggester(embedder=_embedder())
        ready = suggester._ready()
        self.assertIsNotNone(ready)
        self.assertIs(ready[0], suggester._embedder)
        self.assertIsNone(ready[1], "no model id is needed offline")

    def test_the_default_floor_is_unchanged(self) -> None:
        """The live path must behave exactly as it did."""
        self.assertEqual(SemanticSuggester().floor, COSINE_FLOOR)

    def test_a_floor_can_be_supplied(self) -> None:
        self.assertEqual(SemanticSuggester(floor=0.5).floor, 0.5)

    def test_an_unavailable_embedder_yields_nothing(self) -> None:
        class _Down:
            def available(self):
                return False

            def embed(self, texts, model=None):
                raise AssertionError("must not be called")

        self.assertIsNone(SemanticSuggester(embedder=_Down())._ready())


class GateVerdictTests(unittest.TestCase):
    """The pass, and the criterion that fitted nothing."""

    def setUp(self) -> None:
        from ultraquant.experiments import offline_gate
        self.doc = " ".join(offline_gate.__doc__.split())

    def test_the_criteria_are_written_down(self) -> None:
        for phrase in ("The wall moves",
                       "Decoys stay dead",
                       "Everything else is untouched",
                       "The floor is fitted, not inherited",
                       "Nothing touches the network during evaluation"):
            self.assertIn(phrase, self.doc)

    def test_the_three_inherited_criteria_are_named_as_inherited(self):
        self.assertIn("§11.39's, verbatim", self.doc)

    def test_the_result_is_recorded(self) -> None:
        self.assertIn("+0.625 at seed sd 0.484", self.doc)
        self.assertIn("none of them during evaluation", self.doc)

    def test_the_floor_caveat_is_recorded(self) -> None:
        """It fitted something that turned out not to matter."""
        self.assertIn("The threshold never binds", self.doc)
        self.assertIn("a floor that never binds is not protection",
                      self.doc)

    def test_the_cosines_are_shown_not_to_have_collapsed(self) -> None:
        self.assertIn("0.41 to 0.97, spread 0.55", self.doc)


if __name__ == "__main__":
    unittest.main()
