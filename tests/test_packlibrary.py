"""Tensors packed into the shard library - §11.96, pinned.

The exactness pins run anywhere; the ones needing a published
checkpoint skip without one.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ultraquant.convert import library, pack
from ultraquant.experiments import convert_gate, packlibrary_gate
from ultraquant.shards.vault import ShardVault


class PackingTests(unittest.TestCase):
    """The codec, and the spelling it chooses."""

    def test_a_ragged_row_round_trips(self) -> None:
        """Widths that are not a multiple of five are the case a
        packer gets wrong."""
        for width in range(1, 17):
            rows = [[(index % 3) - 1 for index in range(width)]]
            blob = pack.encode_rows(rows)
            self.assertEqual(pack.decode_rows(blob, 1, width), rows,
                             f"width {width}")

    def test_rows_are_independently_addressable(self) -> None:
        """Each row starts on a byte boundary, so one can be read
        without decoding the row before it."""
        rows = [[1] * 7, [-1] * 7, [0] * 7]
        blob = pack.encode_rows(rows)
        per_row = len(blob) // 3
        middle = pack.decode_rows(blob[per_row:2 * per_row], 1, 7)
        self.assertEqual(middle, [rows[1]])

    def test_the_chosen_spelling_is_never_the_larger_one(self) -> None:
        """The rule, not a guessed outcome. An earlier version of
        this pin asserted WHICH spelling would win for a made-up
        row, and was wrong about it - the property that actually
        matters is that the choice is never worse than either
        fixed option."""
        rows = [[0] * 600], [[(i % 3) - 1 for i in range(600)]],             [[1, 0, -1] * 13]
        for case in rows:
            chosen = pack.to_payload(case, [1.0] * len(case), "n", "F32")
            sizes = [pack._stored_size(
                pack.to_payload(case, [1.0] * len(case), "n", "F32",
                                spelling=spelling))
                for spelling in ("base243", "ints")]
            self.assertLessEqual(pack._stored_size(chosen), min(sizes))
            self.assertIn(chosen["packing"], ("base243", "ints"))

    def test_both_spellings_decode_to_the_same_trits(self) -> None:
        rows = [[(index % 3) - 1 for index in range(37)]]
        for spelling in ("base243", "ints"):
            payload = pack.to_payload(rows, [0.125], "n", "F32",
                                      spelling=spelling)
            back, scales = pack.from_payload(payload)
            self.assertEqual(back, rows, spelling)
            self.assertEqual(scales, [0.125], spelling)

    def test_a_mismatched_scale_count_refuses(self) -> None:
        with self.assertRaises(ValueError):
            pack.to_payload([[1, 0], [0, 1]], [1.0], "n", "F32")

    def test_an_unknown_packing_refuses(self) -> None:
        with self.assertRaises(ValueError):
            pack.from_payload({"packing": "runes", "trits": "", "rows": 0,
                               "width": 0, "scales": []})


class AddressTests(unittest.TestCase):
    """Category and keywords - the two halves of a shard's address."""

    def test_the_category_is_the_layer(self) -> None:
        self.assertEqual(library.category_of("v.blk.7.attn_out.weight"),
                         "blk.7")
        self.assertEqual(library.category_of("model.layers.3.mlp.up"),
                         "blk.3")
        self.assertEqual(library.category_of("token_embd.weight"),
                         "trunk")

    def test_digits_are_not_keywords(self) -> None:
        """A layer number as a keyword would associate every
        layer-zero tensor with every other tensor holding a zero."""
        weights = library.associations_of("v.blk.0.attn_out.weight")
        self.assertNotIn("0", weights)
        self.assertIn("attn", weights)

    def test_a_packed_shard_is_reachable_both_ways(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="uq_addr_"))
        try:
            vault = ShardVault(root)
            name = "v.blk.2.ffn_up.weight"
            rows = [[(i % 3) - 1 for i in range(40)] for _ in range(4)]
            vault.add_shard(shard_id=f"tensor:{name}",
                            category=library.category_of(name),
                            payload=pack.to_payload(rows, [1.0] * 4, name,
                                                    "F16"),
                            kind="ternary-tensor",
                            associations=library.associations_of(name))
            self.assertIn(f"tensor:{name}", vault.shards_in("blk.2"))
            self.assertGreater(
                vault.association_scores({"ffn", "up"}).get("blk.2", 0.0),
                0.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(convert_gate.checkpoint_path().exists(),
                     "no local checkpoint to pack")
class RealCheckpointTests(unittest.TestCase):
    """Real tensors, into a real vault and back."""

    def test_a_real_pack_round_trips_exactly(self) -> None:
        report = packlibrary_gate.run_gate(tensors=6, rows_each=32)
        self.assertEqual(report.trit_differences, 0, report.reason)
        self.assertEqual(report.scale_differences, 0, report.reason)
        self.assertEqual(report.unreachable, 0, report.reason)
        self.assertLess(report.packed_bits, report.naive_bits)


class GateVerdictTests(unittest.TestCase):
    """The PASS, and the two mistakes behind it."""

    def setUp(self) -> None:
        self.doc = " ".join(packlibrary_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_pass(self) -> None:
        self.assertIn("| trit differences | **0** |", self.doc)
        self.assertIn("+0.477 at 0.300 tensor sd", self.doc)

    def test_the_retraction_is_recorded(self) -> None:
        """A finding that did not survive a fair comparison should
        read as retracted, not quietly vanish."""
        self.assertIn("The story was an artifact and is retracted "
                      "here", self.doc)
        self.assertIn("twelve of fourteen", self.doc)

    def test_the_harness_defect_is_recorded(self) -> None:
        self.assertIn("measuring the spelling AND the metadata at "
                      "once", self.doc)

    def test_the_exactness_line_is_argued(self) -> None:
        self.assertIn("no future measurement of the quantiser could be "
                      "trusted", self.doc)


if __name__ == "__main__":
    unittest.main()
