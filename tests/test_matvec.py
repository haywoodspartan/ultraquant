"""Command-R's matrices, multiplied natively.

11.125: the CPU kernel that decodes k-quant rows as ggml does and
multiplies them by a vector. These pins hold its parity with the
pure-Python oracle - on golden blocks everywhere, and on the real model
wherever it is on disk - plus the NaN lesson the exam learned on itself.
"""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.native import accel


def _golden():
    from tests.test_kquant import GOLDEN
    return GOLDEN


class KernelParityTests(unittest.TestCase):
    """The golden blocks from 11.123, through the native kernel."""

    def setUp(self) -> None:
        if accel.load_cpu() is None or not hasattr(accel.load_cpu(),
                                                    "uq_kq_matvec"):
            self.skipTest("the native CPU tier is not built here")

    def test_native_decode_is_bit_identical(self) -> None:
        import struct
        for kind, (_name, raw_hex, out_hex) in _golden().items():
            with self.subTest(kind=kind):
                raw = bytes.fromhex("".join(raw_hex))
                ours = accel.kq_dequant_cpu({"Q4_K": 12, "Q5_K": 13,
                                             "Q6_K": 14}[kind], raw)
                theirs = bytes.fromhex("".join(out_hex))
                self.assertEqual(struct.pack("<256f", *ours), theirs)

    def test_native_dot_equals_the_ordered_float64_dot(self) -> None:
        import struct
        rng = random.Random(0)
        for kind, (_name, raw_hex, out_hex) in _golden().items():
            with self.subTest(kind=kind):
                raw = bytes.fromhex("".join(raw_hex))
                weights = struct.unpack("<256f",
                                        bytes.fromhex("".join(out_hex)))
                x = [rng.uniform(-1, 1) for _ in range(256)]
                want = 0.0
                for w, v in zip(weights, x):
                    want += w * v
                got = accel.kq_matvec_cpu({"Q4_K": 12, "Q5_K": 13,
                                           "Q6_K": 14}[kind], raw, 1, 256,
                                          x, 1)[0]
                self.assertEqual(got, want)


class ExamTests(unittest.TestCase):

    def setUp(self) -> None:
        from ultraquant.experiments import matvec_gate
        self.gate = matvec_gate
        self.doc = " ".join(matvec_gate.__doc__.split())

    def test_nan_is_never_agreement(self) -> None:
        """The exam's own first defect: NaN compared False with 1e-9."""
        self.assertEqual(self.gate._relative(math.nan, 1.0), math.inf)
        self.assertEqual(self.gate._relative(1.0, math.inf), math.inf)
        self.assertFalse(self.gate._relative(math.nan, 1.0) <= 1e-9)

    def test_the_defect_and_the_result_are_recorded(self) -> None:
        self.assertIn("The exam's own defect came first", self.doc)
        self.assertIn("PASSED on all four measured criteria", self.doc)
        self.assertIn("6.2 GB/s", self.doc)

    def test_the_gate_passes_here(self) -> None:
        if not self.gate.COMMAND_R.exists():
            self.skipTest("Command-R is not on this machine")
        if accel.load_cpu() is None:
            self.skipTest("the native CPU tier is not built here")
        self.assertTrue(self.gate.run_gate().passes)


if __name__ == "__main__":
    unittest.main()
