"""The documentation holds together as documents.

A stray blank line inside a markdown table ends it. Everything after
the blank has no header row, so it renders as raw pipe-text rather
than as rows - silently, and only in the rendered view, which is the
one place nobody looks while editing. One such blank was orphaning
six rows before this pin existed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_README = _ROOT / "README.md"
_DOCS = ("README.md", "ARCHITECTURE.md", "JOURNAL.md")


def _slug(title: str) -> str:
    """A heading's anchor, the way GitHub derives it."""
    return re.sub(r"[^a-z0-9 \-]", "", title.lower()).replace(" ", "-")


def _headings(text: str) -> set:
    return {_slug(re.sub(r"^#+ ", "", line))
            for line in text.splitlines() if line.startswith("#")}


class FeatureTableTests(unittest.TestCase):

    def test_no_blank_line_splits_the_feature_table(self) -> None:
        lines = _README.read_text(encoding="utf-8").splitlines()
        runs: list[tuple[int, int]] = []
        start = None
        for index, line in enumerate(lines):
            if line.startswith("|"):
                if start is None:
                    start = index
            elif start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(lines)))
        # A run of table rows must begin with a header and its
        # separator; a run that starts with a content row is the
        # orphaned tail of an earlier table.
        for begin, end in runs:
            if end - begin < 2:
                continue
            separator = lines[begin + 1]
            self.assertTrue(
                set(separator.replace("|", "").replace(" ", ""))
                <= set("-:"),
                f"table rows at line {begin + 1} have no header - a blank "
                "line above them split an earlier table")


class CrossReferenceTests(unittest.TestCase):
    """Splitting the journal out broke two links; nothing else may.

    §11.11-§11.104 moved to JOURNAL.md and two same-file anchors in
    the moved text still pointed at sections that stayed behind. Both
    resolved silently to nothing, which is how a document rots.
    """

    def test_every_anchor_resolves_in_its_own_file(self) -> None:
        for name in _DOCS:
            text = (_ROOT / name).read_text(encoding="utf-8")
            own = _headings(text)
            broken = sorted({a for a in re.findall(r"\]\(#([^)]+)\)", text)
                             if a not in own})
            self.assertEqual(broken, [],
                             f"{name} links to headings it does not have")

    def test_every_cross_file_link_points_at_a_file_that_exists(self):
        for name in _DOCS:
            text = (_ROOT / name).read_text(encoding="utf-8")
            for target in set(re.findall(r"\]\(([A-Z][A-Za-z]*\.md)",
                                         text)):
                self.assertTrue((_ROOT / target).exists(),
                                f"{name} links to missing {target}")

    def test_every_cross_file_anchor_resolves_there(self) -> None:
        for name in _DOCS:
            text = (_ROOT / name).read_text(encoding="utf-8")
            for target, anchor in re.findall(
                    r"\]\(([A-Z][A-Za-z]*\.md)#([^)]+)\)", text):
                other = _headings(
                    (_ROOT / target).read_text(encoding="utf-8"))
                self.assertIn(anchor, other,
                              f"{name} points at {target}#{anchor}, "
                              "which is not a heading there")


class JournalIndexTests(unittest.TestCase):
    """The index is the only way to navigate 94 out-of-order sections."""

    def test_every_section_is_indexed(self) -> None:
        text = (_ROOT / "JOURNAL.md").read_text(encoding="utf-8")
        sections = {_slug(line[4:].strip())
                    for line in text.splitlines()
                    if line.startswith("### 11.")}
        indexed = set(re.findall(r"\]\(#([^)]+)\)", text))
        self.assertEqual(sections - indexed, set(),
                         "a section nobody can reach from the index")


if __name__ == "__main__":
    unittest.main()
