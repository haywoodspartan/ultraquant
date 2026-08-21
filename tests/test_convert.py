"""The transformer conversion kit - §11.95, pinned.

The reader is pinned against a synthetic GGUF built here, so these
run on any machine; the measurements against a real checkpoint skip
when none is present, because a published model is not a test
fixture.
"""

from __future__ import annotations

import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from ultraquant.convert import gguf, ternary
from ultraquant.experiments import convert_gate


def _write_gguf(path: Path, tensors: list) -> None:
    """A minimal well-formed GGUF: (name, dims, type_number, floats)."""
    def string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    head = b"GGUF" + struct.pack("<I", 3)
    head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", 1)
    head += string("general.architecture") + struct.pack("<I", 8)
    head += string("test")

    infos = b""
    blobs = []
    offset = 0
    for name, dims, type_number, values in tensors:
        infos += string(name) + struct.pack("<I", len(dims))
        infos += struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<I", type_number) + struct.pack("<Q", offset)
        if type_number == 0:
            blob = struct.pack(f"<{len(values)}f", *values)
        elif type_number == 1:
            blob = struct.pack(f"<{len(values)}e", *values)
        else:
            packed = b""
            for value in values:
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                packed += struct.pack("<H", bits >> 16)
            blob = packed
        blobs.append(blob)
        offset += len(blob)
        offset += (32 - (offset % 32)) % 32
    body = head + infos
    body += bytes((32 - (len(body) % 32)) % 32)
    for blob in blobs:
        body += blob
        body += bytes((32 - (len(blob) % 32)) % 32)
    path.write_bytes(body)


class ReaderTests(unittest.TestCase):
    """Formats, shapes, and the refusal that is not a guess."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="uq_convert_"))
        self.path = self.root / "tiny.gguf"

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_f32_round_trips_exactly(self) -> None:
        values = [0.5, -0.25, 1.0, -2.0, 0.0, 3.25]
        _write_gguf(self.path, [("w", (3, 2), 0, values)])
        opened = gguf.read(self.path)
        rows = opened.rows_of(opened.by_name("w"))
        self.assertEqual(rows, [[0.5, -0.25, 1.0], [-2.0, 0.0, 3.25]])

    def test_bf16_widens_by_shifting(self) -> None:
        """bfloat16 is the top half of a float32, so values that fit
        come back exactly."""
        values = [0.5, -0.25, 2.0, -4.0]
        _write_gguf(self.path, [("w", (2, 2), 30, values)])
        opened = gguf.read(self.path)
        self.assertEqual(opened.rows_of(opened.by_name("w")),
                         [[0.5, -0.25], [2.0, -4.0]])

    def test_f16_is_read_as_half_precision(self) -> None:
        _write_gguf(self.path, [("w", (2, 1), 1, [1.5, -0.75])])
        opened = gguf.read(self.path)
        self.assertEqual(opened.rows_of(opened.by_name("w")),
                         [[1.5, -0.75]])

    def test_an_unsupported_type_refuses_by_name(self) -> None:
        """A reader that guessed at Q4_K blocks would produce numbers,
        and wrong numbers nobody checked are worse than a refusal."""
        _write_gguf(self.path, [("w", (2, 1), 12, [0.0, 0.0])])
        opened = gguf.read(self.path)
        info = opened.by_name("w")
        self.assertEqual(info.type_name, "Q4_K")
        self.assertFalse(info.readable)
        with self.assertRaises(gguf.UnsupportedTensorType) as caught:
            opened.rows_of(info)
        self.assertIn("Q4_K", str(caught.exception))

    def test_metadata_and_census(self) -> None:
        _write_gguf(self.path, [("a", (2, 1), 0, [1.0, 2.0]),
                                ("b", (2, 1), 12, [0.0, 0.0])])
        opened = gguf.read(self.path)
        self.assertEqual(opened.metadata["general.architecture"], "test")
        self.assertEqual(opened.type_census(), {"F32": 1, "Q4_K": 1})


class TernaryTests(unittest.TestCase):
    """The quantiser's own arithmetic."""

    def test_output_is_ternary_with_one_scale_per_row(self) -> None:
        rows = [[0.9, -0.8, 0.01, 0.7], [3.0, -3.1, 0.02, 2.9]]
        quantised, scales = ternary.ternarise_rows(rows, per_row=True)
        self.assertEqual(len(scales), 2)
        for row in quantised:
            for value in row:
                self.assertIn(value, (-1, 0, 1))
        self.assertGreater(scales[1], scales[0])

    def test_the_optimal_threshold_minimises_weight_error(self) -> None:
        """It is chosen to minimise exactly that, so this checks the
        closed form was implemented - not a claim about outputs,
        which is a different measure and the gate's."""
        rows = [[3.0, 2.9, 0.05, -0.04, -3.1, 0.02]]
        heuristic = ternary.ternarise_rows(rows, per_row=True)
        optimal = ternary.ternarise_rows(rows, per_row=True, optimal=True)
        self.assertLessEqual(
            ternary.weight_error(rows, *optimal),
            ternary.weight_error(rows, *heuristic) + 1e-12)

    def test_a_zero_row_does_not_divide_by_zero(self) -> None:
        quantised, scales = ternary.ternarise_rows([[0.0, 0.0]],
                                                   per_row=True)
        self.assertEqual(quantised, [[0, 0]])
        self.assertEqual(scales, [1.0])

    def test_choosing_reports_which_rule_it_chose(self) -> None:
        rows = [[0.9, -0.8, 0.01, 0.7]]
        _q, _s, picked = ternary.choose_rules(rows)
        self.assertIn(picked, ("per row, 0.75", "per row, optimal"))

    def test_error_measures_are_finite_and_bounded(self) -> None:
        rows = [[0.4, -0.3, 0.2], [0.1, 0.9, -0.5]]
        quantised, scales = ternary.ternarise_rows(rows, per_row=True)
        error, cosine = ternary.output_error(rows, quantised, scales,
                                             probes=3)
        self.assertTrue(math.isfinite(error) and error >= 0.0)
        self.assertLessEqual(cosine, 1.0 + 1e-9)


@unittest.skipUnless(convert_gate.checkpoint_path().exists(),
                     "no local checkpoint to convert")
class RealCheckpointTests(unittest.TestCase):
    """The measurement, on a real model, when one is present."""

    def test_a_real_conversion_is_measured_and_lossy(self) -> None:
        report = convert_gate.run_gate(matrices=4, rows_each=32,
                                       probes=2)
        self.assertGreater(report.tensors, 0)
        self.assertEqual(report.violations, 0)
        # Not an aspiration: this is what ternary costs, and the pin
        # exists so a future change claiming otherwise has to face a
        # number.
        self.assertGreater(report.errors["chosen per tensor"], 0.2)


class GateVerdictTests(unittest.TestCase):
    """The recorded FAIL and the two lessons behind it."""

    def setUp(self) -> None:
        self.doc = " ".join(convert_gate.__doc__.split())

    def test_the_recorded_verdict_is_a_failure(self) -> None:
        self.assertIn("FAILED criterion 1 - and the failure is the "
                      "finding", self.doc)

    def test_the_absolute_cost_is_stated(self) -> None:
        """Criterion 2 exists so the kit cannot report only a
        relative improvement."""
        self.assertIn("0.4667 relative output error, cosine 0.8837",
                      self.doc)
        self.assertIn("not a substitute for it", self.doc)

    def test_weight_error_is_not_output_error(self) -> None:
        self.assertIn("Minimising weight error is not minimising "
                      "output error", self.doc)

    def test_the_harness_artifact_is_recorded(self) -> None:
        self.assertIn("The conclusion that looked so clean was an "
                      "artifact of the wrong question", self.doc)


if __name__ == "__main__":
    unittest.main()
