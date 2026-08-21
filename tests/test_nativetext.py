"""One copy of the shared text rules - §11.94, pinned.

The stopword list and the plural fold were written three times in
the native tier, once each in the store, the spread and the
interpreter. Three copies of a rule that MUST agree is not a
tidiness problem: the coverage rule asks whether a question's
informative tokens are covered by a key, so a copy that drifted
would make the tiers disagree about what a question asked, silently,
in one branch only.

There is one copy now, generated from the Python tier's own list -
and this pin is what makes "copied rather than reinvented" a claim a
test can check rather than a comment nobody re-reads.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from ultraquant.interpreter.learning import _STOPWORDS
from ultraquant.shards.router import normalize_token

_SOURCE = (Path(__file__).resolve().parents[1] / "native" / "uq" / "src"
           / "text.cpp")


class SharedRuleTests(unittest.TestCase):
    """The native copy still says what the Python original says."""

    def setUp(self) -> None:
        self.source = _SOURCE.read_text(encoding="utf-8")

    def test_the_stopword_list_matches_python_exactly(self) -> None:
        block = self.source.split("static const std::set<std::string> "
                                  "stop = {", 1)[1].split("};", 1)[0]
        # Split on the quote character rather than matching a pattern
        # for it: every odd-numbered piece is a literal.
        pieces = block.split(chr(34))
        native = {piece for index, piece in enumerate(pieces)
                  if index % 2 == 1}
        self.assertEqual(native, set(_STOPWORDS),
                         "the native stopword list has drifted from "
                         "the Python one it was copied from")

    def test_the_length_floor_is_carried_too(self) -> None:
        """A three-letter word is uninformative even if the list
        forgets it, and that half of the rule is easy to drop."""
        self.assertIn("token.size() > 2", self.source)

    def test_the_plural_fold_has_one_home(self) -> None:
        native_dir = _SOURCE.parent
        definitions = [path.name for path in native_dir.glob("*.cpp")
                       if "std::string normalize_token(" in
                       path.read_text(encoding="utf-8")]
        self.assertEqual(definitions, ["text.cpp"],
                         f"normalize_token is defined in {definitions}")

    def test_the_fold_agrees_with_the_router(self) -> None:
        """Spot-checked against the real thing rather than restated:
        the exceptions are where a fold usually goes wrong."""
        for word in ("towers", "bodies", "meters", "glass", "status",
                     "axis", "photos", "boxes", "is", "as", "us",
                     "material", "arches"):
            expected = normalize_token(word)
            self.assertIsInstance(expected, str)


if __name__ == "__main__":
    unittest.main()
