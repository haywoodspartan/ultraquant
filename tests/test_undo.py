"""An undo that respects what came after.

11.135: a dispute restores exactly only when nothing has changed the
fact since; a later correction wins; two approvals of one key are
undone newest first; a stale conclusion stays retracted; approvals and
disputes are transactions that survive a crash. The whole exam runs in
about a second against temporary disk-backed stores.
"""

from __future__ import annotations

import unittest

from ultraquant.experiments import undo_gate as G


class UndoGateTests(unittest.TestCase):

    def test_the_whole_exam(self) -> None:
        report = G.run_gate()
        self.assertTrue(report.passes, report.reason)
        self.assertEqual(len(report.planted), 8)

    def test_each_case_on_its_own(self) -> None:
        for name, case in G.CASES.items():
            with self.subTest(case=name):
                self.assertTrue(case())

    def test_the_verdict_is_recorded(self) -> None:
        doc = " ".join(G.__doc__.split())
        for phrase in ("Version-aware disputes", "Transactions", "PASSED",
                       "legacy approval"):
            self.assertIn(phrase, doc)


if __name__ == "__main__":
    unittest.main()
