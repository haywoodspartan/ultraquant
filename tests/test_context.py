"""The byte-bounded context window and its reference index.

The design claim is a ratio: content bounded in bytes and resident, a 24-byte
reference per turn for everything ever written, and recall that follows those
references to disk instead of scanning. These tests pin the ratio, the bound,
and the two ways the signature was got wrong before it was got right.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.memory.context import (
    REFERENCE_BYTES,
    ContextWindow,
    _overlap,
    _sketch,
)


class _Windowed(unittest.TestCase):
    """A window on a temporary log."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="uq_ctx_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def window(self, budget: int = 512) -> ContextWindow:
        return ContextWindow(self.dir, budget_bytes=budget)


class SignatureTests(_Windowed):
    """Two earlier designs were wrong; both failures are pinned here."""

    def test_unrelated_texts_get_distinct_signatures(self):
        """The first design collapsed everything to one constant.

        It mixed a per-token CRC32 seed with the bit index, also via CRC32.
        CRC32 is linear over GF(2), so every bit reduced to the same functional
        of the seed up to a flip: three unrelated sentences all produced
        0xaaaa5555aaaa5555 and the signature carried about one bit.
        """
        texts = ["the tower height is 324 metres",
                 "the bridge length is 1280 metres",
                 "the vault capacity is 9400 shards",
                 "completely unrelated chatter here"]
        self.assertEqual(len({_sketch(t) for t in texts}), len(texts))

    def test_the_right_turn_outranks_an_unrelated_one(self):
        """The second design was a SimHash, and ranked chatter above the answer.

        SimHash is a random projection and needs many terms for its per-bit
        vote sums to be stable. A conversational turn has four to six content
        tokens, so each bit was the sign of a sum of five votes and flipped on
        noise: against "how tall is the tower", unrelated chatter scored
        Hamming 23 while the answer scored 25.
        """
        query = _sketch("how tall is the tower")
        answer = _overlap(query, _sketch("the tower height is 324 metres"))
        chatter = _overlap(query, _sketch("completely unrelated chatter here"))
        self.assertGreater(answer, chatter)

    def test_the_signature_is_stable_across_processes(self):
        """A rebuilt index must agree with the one that wrote it.

        Python's hash() for str is randomised per interpreter, so a signature
        built on it would make the index unreadable by the next run.
        """
        import subprocess
        import sys

        code = ("from ultraquant.memory.context import _sketch;"
                "print(_sketch('the tower height is 324 metres'))")
        first = _sketch("the tower height is 324 metres")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, cwd=str(Path(__file__).parent.parent))
        self.assertEqual(int(done.stdout.strip()), first)

    def test_stopwords_do_not_set_bits(self):
        """They appear in every turn, so they would be pure noise."""
        self.assertEqual(_sketch("the and of to is a"), 0)

    def test_an_empty_turn_has_an_empty_signature(self):
        self.assertEqual(_sketch(""), 0)


class BudgetTests(_Windowed):
    """The bound is in bytes, because bytes are what a card has."""

    def test_resident_content_stays_within_the_budget(self):
        window = self.window(budget=512)
        for index in range(200):
            window.add(f"turn number {index} carrying some words of content")
            self.assertLessEqual(window.stats()["resident_bytes"], 512)

    def test_eviction_keeps_the_reference(self):
        """The whole point: evicted content is still findable."""
        window = self.window(budget=256)
        first = window.add("the tower height is 324 metres")
        for index in range(60):
            window.add(f"filler turn {index} about the weather and coffee")
        self.assertNotIn(first, {t["id"] for t in window.resident()})
        self.assertEqual(len(window._ids), 61, "every turn keeps a reference")
        found = window.recall("how tall is the tower", top_k=3)
        self.assertTrue(any("324 metres" in t["text"] for t in found))

    def test_the_index_costs_the_documented_bytes_per_turn(self):
        window = self.window()
        for index in range(500):
            window.add(f"turn {index} with a handful of content words in it")
        per_turn = window.stats()["index_bytes"] / 500
        self.assertLessEqual(per_turn, REFERENCE_BYTES + 8,
                             "arrays over-allocate a little; not per-object")

    def test_the_index_is_far_smaller_than_the_content(self):
        """If it were not, there would be no reason to build any of this."""
        window = self.window()
        for index in range(500):
            window.add(f"turn {index} with a handful of content words in it")
        stats = window.stats()
        self.assertLess(stats["index_bytes"], stats["stored_bytes"] / 2)

    def test_one_turn_larger_than_the_budget_is_still_kept(self):
        """Evicting to nothing would lose the turn just added."""
        window = self.window(budget=32)
        window.add("a turn considerably longer than the whole byte budget is")
        self.assertEqual(len(window.resident()), 1)


class RecallTests(_Windowed):
    """Following a reference to disk."""

    def test_recall_reads_only_the_winners(self):
        window = self.window(budget=128)
        for index in range(80):
            window.add(f"filler turn {index} about assorted daily matters")
        window.add("the archive checksum is blake2b")
        for index in range(80):
            window.add(f"more filler {index} about assorted daily matters")
        before = window.stats()["recall_reads"]
        window.recall("which checksum does the archive use", top_k=3)
        self.assertLessEqual(window.stats()["recall_reads"] - before, 3,
                             "a screen, not a scan")

    def test_recall_skips_resident_turns_by_default(self):
        """A caller already has those; the question is what recall adds."""
        window = self.window()
        window.add("the tower height is 324 metres")
        self.assertEqual(window.recall("how tall is the tower"), [])
        self.assertTrue(window.recall("how tall is the tower",
                                      include_resident=True))

    def test_no_shared_vocabulary_returns_nothing(self):
        """Returning the least-bad match would be a confident wrong answer."""
        window = self.window(budget=64)
        for index in range(40):
            window.add(f"filler turn {index} about coffee and weather")
        self.assertEqual(
            window.recall("zygomatic parallax thaumaturgy", top_k=3), [])

    def test_recall_on_an_empty_window_is_empty(self):
        self.assertEqual(self.window().recall("anything"), [])


class DurabilityTests(_Windowed):
    """The log is the only thing that has to survive."""

    def test_the_index_is_rebuilt_from_the_log(self):
        first = self.window()
        first.add("the tower height is 324 metres")
        for index in range(40):
            first.add(f"filler {index} about the weather")
        reopened = self.window()
        self.assertEqual(len(reopened._ids), 41)
        found = reopened.recall("how tall is the tower", top_k=3)
        self.assertTrue(any("324 metres" in t["text"] for t in found))

    def test_a_torn_line_is_skipped_not_fatal(self):
        window = self.window()
        window.add("the tower height is 324 metres")
        with open(window.log, "ab") as handle:
            handle.write(b'{"id": 2, "text": "trunc\n')
        reopened = self.window()
        self.assertGreaterEqual(len(reopened._ids), 1)
        self.assertIsNotNone(reopened.read(1))

    def test_reading_a_missing_turn_returns_none(self):
        self.assertIsNone(self.window().read(999))


if __name__ == "__main__":
    unittest.main()
