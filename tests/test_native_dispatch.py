"""Tests for SPEC-NATIVE.md sections 4-7: native backends, tier dispatch,
the heterogeneous CPU+GPU feature extractor, and the recognizer batch hook.

Every native-tier assertion is guarded so the module stays green (skips, never
fails) on machines with no compiler and no GPU; ``best_backend`` and the
extractor always have the pure-Python tier to land on, so their tests run
everywhere.

Run from the project root:  python -m unittest tests.test_native_dispatch -v
"""

from __future__ import annotations

import math
import random
import unittest

from ultraquant.native import accel
from ultraquant.native.backends import CudaSimulatorBackend, NativeSimulatorBackend
from ultraquant.native.dispatch import best_backend, tier_report
from ultraquant.native.hetero import HeterogeneousFeatureExtractor
from ultraquant.pattern.recognition import LABELS, PATTERNS, PatternRecognizer
from ultraquant.quantum.backend import (
    Backend,
    BackendUnavailable,
    SimulatorBackend,
)
from ultraquant.quantum.circuit import Circuit
from ultraquant.quantum.qlayer import QuantumFeatureMap

TOL = 1e-9

_CPU_OK = accel.load_cpu() is not None
_GPU_OK = accel.load_gpu() is not None


def _bell() -> Circuit:
    """H then CNOT: exact <Z> is 0.0 on both qubits."""
    return Circuit(2).h(0).cnot(0, 1)


def _random_circuit(rng: random.Random, num_qubits: int, depth: int) -> Circuit:
    c = Circuit(num_qubits)
    one_q = ("h", "x", "y", "z", "s", "t")
    rot = ("rx", "ry", "rz")
    two_q = ("cnot", "cz", "swap")
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.4:
            getattr(c, rng.choice(one_q))(rng.randrange(num_qubits))
        elif roll < 0.7:
            getattr(c, rng.choice(rot))(
                rng.randrange(num_qubits), rng.uniform(-2 * math.pi, 2 * math.pi)
            )
        else:
            a, b = rng.sample(range(num_qubits), 2)
            getattr(c, rng.choice(two_q))(a, b)
    return c


def _python_feature_rows(
    samples: list[list[float]], num_qubits: int
) -> list[list[float]]:
    """Pure-Python oracle: QuantumFeatureMap.extract per sample."""
    qmap = QuantumFeatureMap(num_qubits, SimulatorBackend(seed=0), shots=None)
    return [qmap.extract(row) for row in samples]


class _Close(unittest.TestCase):
    def assertRowsClose(self, got, want, msg=""):
        self.assertEqual(len(got), len(want), msg)
        for r, (grow, wrow) in enumerate(zip(got, want)):
            self.assertEqual(len(grow), len(wrow), f"{msg} [row {r}]")
            for i, (g, w) in enumerate(zip(grow, wrow)):
                self.assertLess(
                    abs(g - w), TOL, f"{msg} [row {r}][{i}]: {g!r} vs {w!r}"
                )


class TestBestBackend(unittest.TestCase):
    """best_backend(): SPEC-NATIVE.md section 6 tier order, never raises."""

    def test_returns_backend_with_bell_expectations(self):
        backend = best_backend()
        self.assertIsInstance(backend, Backend)
        result = backend.run(_bell(), shots=None)
        self.assertIsNone(result.counts)
        self.assertEqual(len(result.expectations_z), 2)
        for q, value in enumerate(result.expectations_z):
            self.assertLess(abs(value), TOL, f"<Z_{q}> of Bell state must be 0")

    def test_tier_order_without_qpu(self):
        backend = best_backend(prefer_qpu=False)
        self.assertNotEqual(backend.name, "ibm-qpu")
        if _GPU_OK:
            self.assertIsInstance(backend, CudaSimulatorBackend)
            self.assertEqual(backend.name, "ultraquant-cuda")
        elif _CPU_OK:
            self.assertIsInstance(backend, NativeSimulatorBackend)
            self.assertEqual(backend.name, "ultraquant-native-cpu")
        else:
            self.assertIsInstance(backend, SimulatorBackend)
            self.assertEqual(backend.name, "ultraquant-statevector")

    def test_seeded_shots_deterministic(self):
        r1 = best_backend(seed=123).run(_bell(), shots=256)
        r2 = best_backend(seed=123).run(_bell(), shots=256)
        self.assertEqual(r1.counts, r2.counts)
        self.assertEqual(r1.expectations_z, r2.expectations_z)
        self.assertEqual(sum(r1.counts.values()), 256)
        self.assertTrue(set(r1.counts) <= {"00", "11"})


class TestTierReport(unittest.TestCase):
    """tier_report(): SPEC-NATIVE.md section 6 key/shape contract."""

    def test_keys_and_types(self):
        report = tier_report()
        self.assertEqual(set(report), {"qpu", "cuda", "native_cpu", "python"})
        self.assertIs(report["python"], True)
        self.assertIsInstance(report["qpu"], bool)
        self.assertIsInstance(report["native_cpu"], bool)
        self.assertIsInstance(report["cuda"], (bool, str))

    def test_matches_accel_probes(self):
        report = tier_report()
        self.assertEqual(report["native_cpu"], _CPU_OK)
        self.assertEqual(bool(report["cuda"]), _GPU_OK)
        if isinstance(report["cuda"], str):
            self.assertGreater(len(report["cuda"]), 0)


class TestNativeSimulatorBackend(_Close):
    """SPEC-NATIVE.md section 4: the C++ CPU statevector backend."""

    def test_constructor_contract(self):
        if _CPU_OK:
            backend = NativeSimulatorBackend()
            self.assertEqual(backend.name, "ultraquant-native-cpu")
        else:
            with self.assertRaises(BackendUnavailable):
                NativeSimulatorBackend()

    @unittest.skipUnless(_CPU_OK, "native CPU tier not available")
    def test_bell_exact(self):
        result = NativeSimulatorBackend().run(_bell())
        self.assertIsNone(result.counts)
        for value in result.expectations_z:
            self.assertLess(abs(value), TOL)

    @unittest.skipUnless(_CPU_OK, "native CPU tier not available")
    def test_exact_parity_vs_pure_python(self):
        rng = random.Random(2026)
        for k in range(4):
            num_qubits = 2 + k
            circuit = _random_circuit(rng, num_qubits, 18)
            with self.subTest(k=k, num_qubits=num_qubits):
                want = SimulatorBackend(seed=0).run(circuit).expectations_z
                got = NativeSimulatorBackend(seed=0).run(circuit).expectations_z
                self.assertRowsClose([got], [want], "exact <Z> parity")

    @unittest.skipUnless(_CPU_OK, "native CPU tier not available")
    def test_shots_sampling_seeded_and_sane(self):
        r1 = NativeSimulatorBackend(seed=5).run(_bell(), shots=4096)
        r2 = NativeSimulatorBackend(seed=5).run(_bell(), shots=4096)
        self.assertEqual(r1.counts, r2.counts)
        self.assertTrue(set(r1.counts) <= {"00", "11"})
        self.assertEqual(sum(r1.counts.values()), 4096)
        for est in r1.expectations_z:
            self.assertLess(abs(est), 0.1)


class TestCudaSimulatorBackend(_Close):
    """SPEC-NATIVE.md section 4: the CUDA statevector backend."""

    def test_constructor_contract(self):
        if _GPU_OK:
            backend = CudaSimulatorBackend()
            self.assertEqual(backend.name, "ultraquant-cuda")
        else:
            with self.assertRaises(BackendUnavailable):
                CudaSimulatorBackend()

    @unittest.skipUnless(_GPU_OK, "CUDA tier not available")
    def test_bell_exact(self):
        result = CudaSimulatorBackend().run(_bell())
        self.assertIsNone(result.counts)
        for value in result.expectations_z:
            self.assertLess(abs(value), TOL)

    @unittest.skipUnless(_GPU_OK, "CUDA tier not available")
    def test_exact_parity_vs_pure_python(self):
        rng = random.Random(1926)
        for k in range(4):
            num_qubits = 2 + k
            circuit = _random_circuit(rng, num_qubits, 18)
            with self.subTest(k=k, num_qubits=num_qubits):
                want = SimulatorBackend(seed=0).run(circuit).expectations_z
                got = CudaSimulatorBackend(seed=0).run(circuit).expectations_z
                self.assertRowsClose([got], [want], "exact <Z> parity")

    @unittest.skipUnless(_GPU_OK, "CUDA tier not available")
    def test_shots_sampling_seeded_and_sane(self):
        r1 = CudaSimulatorBackend(seed=5).run(_bell(), shots=4096)
        r2 = CudaSimulatorBackend(seed=5).run(_bell(), shots=4096)
        self.assertEqual(r1.counts, r2.counts)
        self.assertTrue(set(r1.counts) <= {"00", "11"})
        self.assertEqual(sum(r1.counts.values()), 4096)
        for est in r1.expectations_z:
            self.assertLess(abs(est), 0.1)


class TestHeterogeneousFeatureExtractor(_Close):
    """SPEC-NATIVE.md section 5: parity for every force mode and the auto split."""

    def _samples(self, n_samples: int, n_features: int, seed: int = 42):
        rng = random.Random(seed)
        return [
            [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
            for _ in range(n_samples)
        ]

    def _assert_parity(self, force: str | None, expect_tier: str | None = None):
        # 7 features / 3 qubits: the data re-uploading case, in batch form.
        samples = self._samples(24, 7)
        extractor = HeterogeneousFeatureExtractor(3, force=force)
        got = extractor.extract_batch(samples)
        want = _python_feature_rows(samples, 3)
        self.assertRowsClose(got, want, f"force={force!r}")
        self.assertEqual(set(extractor.last_split), {"gpu", "cpu", "python"})
        self.assertEqual(sum(extractor.last_split.values()), len(samples))
        if expect_tier is not None:
            self.assertEqual(extractor.last_split[expect_tier], len(samples))

    def test_force_python_parity(self):
        self._assert_parity("python", expect_tier="python")

    @unittest.skipUnless(_CPU_OK, "native CPU tier not available")
    def test_force_cpu_parity(self):
        self._assert_parity("cpu", expect_tier="cpu")

    @unittest.skipUnless(_GPU_OK, "CUDA tier not available")
    def test_force_gpu_parity(self):
        self._assert_parity("gpu", expect_tier="gpu")

    def test_every_force_mode_degrades_gracefully(self):
        """All force modes give correct results whatever tiers exist."""
        for force in (None, "cpu", "gpu", "python"):
            with self.subTest(force=force):
                self._assert_parity(force)

    def test_auto_split_parity_and_introspection(self):
        samples = self._samples(30, 5, seed=9)
        extractor = HeterogeneousFeatureExtractor(5, force=None)
        want = _python_feature_rows(samples, 5)

        got = extractor.extract_batch(samples)
        self.assertRowsClose(got, want, "auto split (warmup batch)")
        self.assertEqual(sum(extractor.last_split.values()), len(samples))
        if _CPU_OK and _GPU_OK:
            # The split is decided by measured throughput, so which tier gets
            # the work is hardware-dependent: narrow feature maps are decisively
            # CPU work (the statevector fits in L1), wide ones favour the GPU.
            # What must hold is that the batch is partitioned across the native
            # tiers with nothing falling back to pure Python.
            self.assertEqual(extractor.last_split["python"], 0)
            self.assertEqual(
                extractor.last_split["gpu"] + extractor.last_split["cpu"],
                len(samples),
            )

        # Second batch reuses the cached throughput ratio; order must hold.
        got2 = extractor.extract_batch(samples)
        self.assertRowsClose(got2, want, "auto split (cached ratio)")
        self.assertEqual(sum(extractor.last_split.values()), len(samples))

    def test_empty_batch(self):
        extractor = HeterogeneousFeatureExtractor(3)
        self.assertEqual(extractor.extract_batch([]), [])
        self.assertEqual(
            extractor.last_split, {"gpu": 0, "cpu": 0, "python": 0}
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            HeterogeneousFeatureExtractor(0)
        with self.assertRaises(ValueError):
            HeterogeneousFeatureExtractor(3, force="quantum")


class TestRecognizerExtractorHook(unittest.TestCase):
    """SPEC-NATIVE.md section 7: PatternRecognizer(extractor=...) batch hook."""

    @classmethod
    def setUpClass(cls):
        # The small classical-check training config from tests/test_pattern.
        train_kwargs = dict(epochs=30, n_per_class=15, seed=1)
        eval_kwargs = dict(n_per_class=15, flips=3, seed=99)

        cls.baseline = PatternRecognizer(
            mode="hybrid", backend=SimulatorBackend(seed=7), shots=None, seed=0
        )
        cls.baseline.train(**train_kwargs)
        cls.baseline_acc = cls.baseline.evaluate(**eval_kwargs)

        cls.extractor = HeterogeneousFeatureExtractor(5)
        cls.hooked = PatternRecognizer(
            mode="hybrid",
            backend=SimulatorBackend(seed=7),
            shots=None,
            seed=0,
            extractor=cls.extractor,
        )
        cls.hooked.train(**train_kwargs)
        cls.hooked_acc = cls.hooked.evaluate(**eval_kwargs)
        cls.eval_size = 15 * len(LABELS)

    def test_same_accuracy_within_tolerance(self):
        self.assertLessEqual(
            abs(self.baseline_acc - self.hooked_acc),
            0.05,
            f"baseline {self.baseline_acc} vs extractor {self.hooked_acc}",
        )
        # The same config must stay in test_pattern's passing regime.
        self.assertGreaterEqual(self.baseline_acc, 0.7)
        self.assertGreaterEqual(self.hooked_acc, 0.7)

    def test_extractor_actually_used_for_evaluate(self):
        # evaluate() was the most recent batch through the extractor.
        self.assertEqual(
            sum(self.extractor.last_split.values()), self.eval_size
        )

    def test_per_sample_paths_unchanged(self):
        feats = self.hooked.features(
            [1.0 if ch == "#" else 0.0 for row in PATTERNS["plus"] for ch in row]
        )
        self.assertEqual(len(feats), 30)
        label, confidence = self.hooked.recognize(PATTERNS["plus"])
        self.assertIn(label, LABELS)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestRowsZeroCopy(unittest.TestCase):
    """The zero-copy output view that replaced per-row list re-boxing.

    Measured problem: at 131k samples the split spent its wrapper time
    re-boxing a flat C buffer into n Python lists (49%) and flattening inputs
    (33%), leaving the kernels 16%. ``Rows`` returns one object whose rows are
    memoryview slices of the buffer the DLL wrote - the join costs nothing per
    row, and the old list price is paid only by an explicit ``tolist()``.
    """

    def _rows(self):
        from array import array
        from ultraquant.native.accel import Rows

        return Rows(array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3)

    def test_rows_present_the_batch_shape(self):
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0]), [1.0, 2.0, 3.0])
        self.assertEqual(list(rows[1]), [4.0, 5.0, 6.0])

    def test_rows_iterate_in_order(self):
        self.assertEqual([list(r) for r in self._rows()],
                         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_negative_and_out_of_range_indexing(self):
        rows = self._rows()
        self.assertEqual(list(rows[-1]), [4.0, 5.0, 6.0])
        with self.assertRaises(IndexError):
            _ = rows[2]

    def test_rows_are_read_only(self):
        """Every row is a window on one shared buffer; mutation would corrupt
        the batch for every other holder, so it is refused."""
        rows = self._rows()
        with self.assertRaises(TypeError):
            rows[0][0] = 99.0

    def test_tolist_reproduces_the_old_representation(self):
        self.assertEqual(self._rows().tolist(),
                         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_empty_width_is_survivable(self):
        from array import array
        from ultraquant.native.accel import Rows

        self.assertEqual(len(Rows(array("d", []), 0)), 0)


class TestSplitReturnsAView(_Close):
    """The heterogeneous split hands back a Rows view, values unchanged."""

    def _extractor(self, force=None):
        from ultraquant.native.hetero import HeterogeneousFeatureExtractor

        return HeterogeneousFeatureExtractor(num_qubits=4, force=force)

    def test_split_output_matches_the_cpu_tier_exactly(self):
        import random

        rng = random.Random(0)
        batch = [[rng.random() for _ in range(4)] for _ in range(200)]
        split = self._extractor().extract_batch(batch)
        cpu = self._extractor("cpu").extract_batch(batch)
        self.assertEqual(len(split), len(cpu))
        for a_row, b_row in zip(split, cpu):
            for a, b in zip(a_row, b_row):
                self.assertAlmostEqual(a, b, places=9)

    def test_a_view_row_equals_the_python_tier(self):
        import random

        rng = random.Random(1)
        batch = [[rng.random() for _ in range(4)] for _ in range(50)]
        got = self._extractor().extract_batch(batch)
        want = self._extractor("python").extract_batch(batch)
        for a_row, b_row in zip(got, want):
            for a, b in zip(list(a_row), b_row):
                self.assertAlmostEqual(a, b, places=9)


class TestFlatBatchInput(_Close):
    """The zero-flatten input path - the counterpart of the Rows output view.

    Measured at 131k x 8 with the tier held constant so split luck cannot
    flatter the result: 70% saved forcing cpu, 80% forcing gpu, 66% on a
    pinned split. The saving is the flattening pass, which costs more than
    either kernel and cannot be viewed away because a list-of-lists must
    genuinely be copied once.

    The honest limit is in the class docstring and repeated here: building a
    FlatBatch *from* lists still costs ~29.7 ms of the ~39.8 ms flatten, so
    this only truly pays for callers that generate flat and never materialise
    the lists at all.
    """

    def _flat(self, n=64, width=4):
        import random
        from array import array
        from ultraquant.native.hetero import FlatBatch

        rng = random.Random(0)
        return FlatBatch(array("d", [rng.random() for _ in range(n * width)]),
                         width)

    def _extractor(self, force=None, qubits=4):
        from ultraquant.native.hetero import HeterogeneousFeatureExtractor

        return HeterogeneousFeatureExtractor(num_qubits=qubits, force=force)

    def test_flat_input_matches_the_list_path_exactly(self):
        """The two input paths must be indistinguishable in their output."""
        from ultraquant.native.hetero import FlatBatch

        batch = self._flat()
        rows = batch.to_rows()
        from_flat = self._extractor().extract_batch(batch)
        from_lists = self._extractor().extract_batch(rows)
        self.assertEqual(len(from_flat), len(from_lists))
        for a_row, b_row in zip(from_flat, from_lists):
            for a, b in zip(list(a_row), list(b_row)):
                self.assertAlmostEqual(a, b, places=12)

    def test_flat_input_matches_the_python_tier(self):
        batch = self._flat()
        got = self._extractor().extract_batch(batch)
        want = self._extractor("python").extract_batch(batch.to_rows())
        for a_row, b_row in zip(got, want):
            for a, b in zip(list(a_row), b_row):
                self.assertAlmostEqual(a, b, places=9)

    def test_flat_input_returns_a_zero_copy_view(self):
        from ultraquant.native.accel import Rows

        self.assertIsInstance(self._extractor().extract_batch(self._flat()),
                              Rows)

    def test_the_split_is_recorded_for_flat_input_too(self):
        extractor = self._extractor()
        extractor.extract_batch(self._flat(n=256))
        total = sum(extractor.last_split.values())
        self.assertEqual(total, 256)

    def test_round_trip_through_rows_and_back(self):
        from ultraquant.native.hetero import FlatBatch

        original = self._flat()
        rebuilt = FlatBatch.from_rows(original.to_rows())
        self.assertEqual(len(rebuilt), len(original))
        for a, b in zip(rebuilt.values, original.values):
            self.assertAlmostEqual(a, b, places=12)

    def test_malformed_flat_batches_are_refused(self):
        from array import array
        from ultraquant.native.hetero import FlatBatch

        with self.assertRaises(TypeError):
            FlatBatch([1.0, 2.0], 2)                      # not an array
        with self.assertRaises(TypeError):
            FlatBatch(array("i", [1, 2]), 2)              # wrong typecode
        with self.assertRaises(ValueError):
            FlatBatch(array("d", [1.0, 2.0, 3.0]), 2)     # ragged
        with self.assertRaises(ValueError):
            FlatBatch(array("d", [1.0]), 0)               # zero width

    def test_an_empty_flat_batch_is_survivable(self):
        from array import array
        from ultraquant.native.hetero import FlatBatch

        empty = FlatBatch(array("d"), 4)
        self.assertEqual(len(self._extractor().extract_batch(empty)), 0)

    def test_forced_python_tier_accepts_flat_input(self):
        """Every tier must handle the flat path, including the fallback."""
        batch = self._flat()
        got = self._extractor("python").extract_batch(batch)
        self.assertEqual(len(got), len(batch))


if __name__ == "__main__":
    unittest.main()
