"""The GPU says when it fails (§11.176).

The five CUDA compute exports returned void. On any CUDA error the host
buffer was left as the caller passed it, zeroed, so a launch that never ran
read as ``<Z> = +1`` on every qubit or an all-zero feature row, and a DLL
built for a newer architecture than the device was selected and wrong on
every call. At ABI 2 every export returns a status, the loader refuses the
old ABI, and a one-qubit probe at load reports a DLL this GPU cannot run as
absent, with the reason.

The fake-DLL classes run everywhere; the last class runs against the
rebuilt DLL on a machine with a CUDA device and skips otherwise. The exam
was frozen before the code (sha256 d46f0020..., Amendment A corrected one
keyword after run 1) and copied here verbatim, except that the DLL path is
resolved from this file rather than written out.
"""

from __future__ import annotations

import ctypes
import os
import unittest
from array import array
from pathlib import Path
from unittest import mock


class _Export:
    """A callable that accepts argtypes/restype assignment like a ctypes export."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


def _h_probe_result(amp, num_qubits, *_rest):
    """Fill ``amp`` with H|0> on one qubit and report success."""
    r = 2 ** -0.5
    amp[0], amp[1], amp[2], amp[3] = r, 0.0, r, 0.0
    return 0


class _FakeGpuDll:
    """Every export of ultraquant_cuda.dll at ABI 2, returning ``status``."""

    def __init__(self, status: int = 0, abi: int | None = 2, probe=None) -> None:
        self.calls: list[str] = []

        def status_export(name):
            def call(*_args):
                self.calls.append(name)
                return status
            return _Export(call)

        self.uqg_available = _Export(lambda: 1)
        self.uqg_device_name = _Export(lambda buf, n: 0)
        if abi is not None:
            self.uqg_abi_version = _Export(lambda: abi)
            self.uqg_status_string = _Export(
                lambda code: b"ok" if code == 0 else b"simulated failure")
        self.uqg_run_circuit = (_Export(probe) if probe is not None
                                else status_export("uqg_run_circuit"))
        self.uqg_expectations_z_from = status_export("uqg_expectations_z_from")
        self.uqg_feature_map_batch = status_export("uqg_feature_map_batch")
        self.uqg_ternary_forward_batch = status_export("uqg_ternary_forward_batch")
        self.uqg_layer_forward = status_export("uqg_layer_forward")
        self.uqg_layer_upload = _Export(lambda *a: 0)
        self.uqg_layer_free = _Export(lambda slot: None)
        self.uqg_layer_resident_bytes = _Export(lambda: 0)


class GpuStatusTests(unittest.TestCase):
    """Criterion 5: a nonzero status is an exception, never a number."""

    def _with(self, dll):
        from ultraquant.native import accel
        return mock.patch.object(accel, "_gpu_dll", dll)

    def test_every_wrapper_raises_on_a_nonzero_status(self) -> None:
        from ultraquant.native import accel
        fake = _FakeGpuDll(status=7)
        calls = {
            "uqg_run_circuit": lambda: accel.run_ops_gpu(1, [0], [0, -1], [0.0]),
            "uqg_expectations_z_from": lambda: accel.expectations_z_gpu(
                array("d", [1.0, 0.0, 0.0, 0.0]), 1),
            "uqg_feature_map_batch": lambda: accel.feature_map_batch_gpu([[0.1]], 1),
            "uqg_ternary_forward_batch": lambda: accel.ternary_forward_batch_gpu(
                [[1]], 1.0, None, [[0.5]]),
            "uqg_layer_forward": lambda: accel.layer_forward(0, [[0.5]], 1, 1),
        }
        with self._with(fake):
            for export, call in calls.items():
                with self.subTest(export=export):
                    with self.assertRaises(accel.GPUError) as caught:
                        call()
                    self.assertIn(export, str(caught.exception))
                    self.assertIn("simulated failure", str(caught.exception))

    def test_status_zero_returns(self) -> None:
        from ultraquant.native import accel
        with self._with(_FakeGpuDll(status=0)):
            amp = accel.run_ops_gpu(1, [0], [0, -1], [0.0])
            self.assertEqual(len(amp), 4)
            self.assertEqual(accel.feature_map_batch_gpu([[0.1]], 1), [[0.0]])

    def test_a_dll_without_the_abi_export_is_refused_with_a_rebuild_hint(self) -> None:
        from ultraquant.native import accel
        legacy = _FakeGpuDll(abi=None)
        with mock.patch.object(accel.ctypes, "CDLL", return_value=legacy), \
                mock.patch.object(accel.os.path, "isfile", return_value=True), \
                mock.patch.object(accel, "_gpu_reason", None):
            self.assertIsNone(accel._try_load_gpu())
            self.assertIn("rebuild", accel._gpu_reason)

    def test_a_dll_whose_probe_fails_is_refused_with_the_reason(self) -> None:
        from ultraquant.native import accel
        broken = _FakeGpuDll(status=209)
        with mock.patch.object(accel.ctypes, "CDLL", return_value=broken), \
                mock.patch.object(accel.os.path, "isfile", return_value=True), \
                mock.patch.object(accel, "_gpu_reason", None):
            self.assertIsNone(accel._try_load_gpu())
            self.assertIn("probe", accel._gpu_reason)
            self.assertIn("simulated failure", accel._gpu_reason)

    def test_a_dll_whose_probe_passes_is_loaded(self) -> None:
        from ultraquant.native import accel
        healthy = _FakeGpuDll(probe=_h_probe_result)
        with mock.patch.object(accel.ctypes, "CDLL", return_value=healthy), \
                mock.patch.object(accel.os.path, "isfile", return_value=True), \
                mock.patch.object(accel, "_gpu_reason", None):
            self.assertIs(accel._try_load_gpu(), healthy)
            self.assertIsNone(accel._gpu_reason)


class CudaBackendFallbackTests(unittest.TestCase):
    """Criterion 6: a failing GPU answers through the tier below, and says so."""

    def test_run_falls_back_and_records_why(self) -> None:
        from ultraquant.native import accel
        from ultraquant.native.backends import CudaSimulatorBackend
        from ultraquant.quantum.circuit import Circuit
        bell = Circuit(2).h(0).cnot(0, 1)
        with mock.patch.object(accel, "_gpu_dll", _FakeGpuDll(status=7)):
            backend = CudaSimulatorBackend(seed=0)
            self.assertIsNone(backend.fell_back)
            result = backend.run(bell)
            for value in result.expectations_z:
                self.assertLess(abs(value), 1e-9)
            self.assertIn("uqg_run_circuit", backend.fell_back)
            again = backend.run(bell, shots=64)
            self.assertEqual(sum(again.counts.values()), 64)
            self.assertTrue(set(again.counts) <= {"00", "11"})


class VramForwardFailureTests(unittest.TestCase):
    """Criterion 7: a forward that fails releases its slot and yields to the tier below."""

    def test_forward_returns_none_and_frees_the_slot(self) -> None:
        from ultraquant.native.accel import GPUError
        from ultraquant.native.vram import VramLayerCache

        class _Accel:
            def __init__(self) -> None:
                self.used: set[int] = set()

            def gpu_available(self) -> bool:
                return True

            def layer_upload(self, qw, alpha, bias, in_dim, out_dim) -> int:
                self.used.add(0)
                return 0

            def layer_forward(self, slot, xs, in_dim, out_dim):
                raise GPUError("uqg_layer_forward failed: simulated")

            def layer_free(self, slot) -> None:
                self.used.discard(slot)

            def layer_resident_bytes(self) -> int:
                return 0

        fake = _Accel()
        cache = VramLayerCache(budget_bytes=1 << 20, accel=fake)
        qw = [[1] * 6 for _ in range(4)]
        self.assertIsNone(cache.forward("k", qw, 1.0, [0.0] * 4, [[0.5] * 6]))
        self.assertEqual(fake.used, set())
        self.assertIn("simulated", cache.last_error)


# --------------------------------------------------------------------------- #
# Criterion 4: the rebuilt DLL on a machine with a CUDA device.
# --------------------------------------------------------------------------- #

_REAL_DLL = Path(__file__).resolve().parents[1] / "ultraquant" / "native" / "_bin" / (
    "ultraquant_cuda.dll" if os.name == "nt" else "libultraquant_cuda.so")


def _load_raw():
    try:
        if not _REAL_DLL.exists():
            return None
        dll = ctypes.CDLL(str(_REAL_DLL))
        dll.uqg_available.restype = ctypes.c_int
        dll.uqg_available.argtypes = []
        if dll.uqg_available() < 1:
            return None
        PD = ctypes.POINTER(ctypes.c_double)
        PI = ctypes.POINTER(ctypes.c_int)
        dll.uqg_abi_version.restype = ctypes.c_int
        dll.uqg_abi_version.argtypes = []
        dll.uqg_status_string.restype = ctypes.c_char_p
        dll.uqg_status_string.argtypes = [ctypes.c_int]
        dll.uqg_run_circuit.restype = ctypes.c_int
        dll.uqg_run_circuit.argtypes = [PD, ctypes.c_int, PI, PI, PD, ctypes.c_int]
        return dll
    except Exception:
        return None


_RAW = _load_raw()


@unittest.skipUnless(_RAW is not None, "no CUDA device or the DLL predates ABI 2")
class RebuiltDllStatusTests(unittest.TestCase):
    def _dbuf(self, a):
        return (ctypes.c_double * len(a)).from_buffer(a)

    def _ibuf(self, a):
        return (ctypes.c_int * len(a)).from_buffer(a)

    def test_abi_version_is_two(self) -> None:
        self.assertEqual(_RAW.uqg_abi_version(), 2)
        self.assertEqual(_RAW.uqg_status_string(0), b"ok")

    def test_a_rejected_argument_reports_and_leaves_the_buffer(self) -> None:
        amp = array("d", [1.0, 0.0, 0.0, 0.0])
        opc, qub, par = array("i", [0]), array("i", [0, -1]), array("d", [0.0])
        status = _RAW.uqg_run_circuit(self._dbuf(amp), 31, self._ibuf(opc),
                                      self._ibuf(qub), self._dbuf(par), 1)
        self.assertNotEqual(status, 0)
        self.assertEqual(list(amp), [1.0, 0.0, 0.0, 0.0])

    def test_a_gate_outside_the_register_reports_and_leaves_the_buffer(self) -> None:
        amp = array("d", [1.0, 0.0] + [0.0] * 6)
        opc, qub, par = array("i", [1]), array("i", [5, -1]), array("d", [0.0])
        status = _RAW.uqg_run_circuit(self._dbuf(amp), 2, self._ibuf(opc),
                                      self._ibuf(qub), self._dbuf(par), 1)
        self.assertEqual(status, -2)
        self.assertEqual(list(amp), [1.0, 0.0] + [0.0] * 6)

    def test_a_bell_circuit_reports_success_with_the_right_amplitudes(self) -> None:
        amp = array("d", [1.0, 0.0] + [0.0] * 6)
        opc, qub, par = array("i", [0, 9]), array("i", [0, -1, 0, 1]), array("d", [0.0, 0.0])
        status = _RAW.uqg_run_circuit(self._dbuf(amp), 2, self._ibuf(opc),
                                      self._ibuf(qub), self._dbuf(par), 2)
        self.assertEqual(status, 0)
        r = 2 ** -0.5
        want = [r, 0.0, 0.0, 0.0, 0.0, 0.0, r, 0.0]
        for got, expected in zip(amp, want):
            self.assertAlmostEqual(got, expected, places=12)


if __name__ == "__main__":
    unittest.main()
