"""The README's feature table renders as a table.

A stray blank line inside a markdown table ends it. Everything after
the blank has no header row, so it renders as raw pipe-text rather
than as rows - silently, and only in the rendered view, which is the
one place nobody looks while editing. One such blank was orphaning
six rows before this pin existed.
"""

from __future__ import annotations

import unittest
from pathlib import Path

_README = Path(__file__).resolve().parents[1] / "README.md"


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


if __name__ == "__main__":
    unittest.main()
