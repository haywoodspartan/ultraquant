"""Parity tests: native CPU tier (ultraquant_native.dll) vs the pure-Python tier.

Per SPEC-NATIVE.md §9: skips (never fails) when the CPU DLL is absent. The DLL
is loaded directly with ctypes at module level inside a try/except so test
collection never crashes; the ultraquant package is imported lazily inside
setUp for the same reason (the pure-Python tier may be built concurrently).

Run from the project root:  python -m unittest tests.test_native_cpu -v
"""

import ctypes
import math
import os
import random
import unittest
from array import array

TOL = 1e-9

_DLL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "ultraquant", "native", "_bin", "ultraquant_native.dll",
    )
)

# Wire format per SPEC-NATIVE.md: 0=h 1=x 2=y 3=z 4=s 5=t 6=rx 7=ry 8=rz
# 9=cnot 10=cz 11=swap.
_OPCODES = {
    "h": 0, "x": 1, "y": 2, "z": 3, "s": 4, "t": 5,
    "rx": 6, "ry": 7, "rz": 8, "cnot": 9, "cz": 10, "swap": 11,
}


def _load_dll():
    """Load and type the CPU DLL; return None on any failure (never raise)."""
    try:
        dll = ctypes.CDLL(_DLL_PATH)
        c_int = ctypes.c_int
        c_dbl = ctypes.c_double
        p_dbl = ctypes.POINTER(ctypes.c_double)
        p_int = ctypes.POINTER(ctypes.c_int)
        p_i8 = ctypes.POINTER(ctypes.c_int8)
        dll.uq_version.argtypes = []
        dll.uq_version.restype = c_int
        dll.uq_init_state.argtypes = [p_dbl, c_int]
        dll.uq_init_state.restype = None
        dll.uq_run_circuit.argtypes = [p_dbl, c_int, p_int, p_int, p_dbl, c_int]
        dll.uq_run_circuit.restype = None
        dll.uq_probabilities.argtypes = [p_dbl, c_int, p_dbl]
        dll.uq_probabilities.restype = None
        dll.uq_expectations_z.argtypes = [p_dbl, c_int, p_dbl]
        dll.uq_expectations_z.restype = None
        dll.uq_feature_map.argtypes = [p_dbl, c_int, c_int, p_dbl]
        dll.uq_feature_map.restype = None
        dll.uq_feature_map_batch.argtypes = [p_dbl, c_int, c_int, c_int, p_dbl, c_int]
        dll.uq_feature_map_batch.restype = None
        dll.uq_ternary_forward.argtypes = [p_i8, c_dbl, p_dbl, p_dbl, c_int, c_int, p_dbl]
        dll.uq_ternary_forward.restype = None
        dll.uq_ternary_forward_batch.argtypes = [
            p_i8, c_dbl, p_dbl, p_dbl, c_int, c_int, c_int, p_dbl, c_int
        ]
        dll.uq_ternary_forward_batch.restype = None
        if dll.uq_version() != 1:
            return None
        return dll
    except Exception:
        return None


_DLL = _load_dll()


def _dbuf(values):
    arr = array("d", values)
    return (ctypes.c_double * len(arr)).from_buffer(arr), arr


def _ibuf(values):
    arr = array("i", values)
    return (ctypes.c_int * len(arr)).from_buffer(arr), arr


def _i8buf(values):
    arr = array("b", values)
    return (ctypes.c_int8 * len(arr)).from_buffer(arr), arr


def _encode_ops(ops):
    """Circuit.ops -> (opcodes, qubits, params) arrays per the wire format."""
    opcodes = array("i")
    qubits = array("i")
    params = array("d")
    for op in ops:
        opcodes.append(_OPCODES[op.name])
        qs = list(op.qubits)
        qubits.append(qs[0] if len(qs) > 0 else -1)
        qubits.append(qs[1] if len(qs) > 1 else -1)
        params.append(op.params[0] if op.params else 0.0)
    return opcodes, qubits, params


def _native_run(dll, num_qubits, ops):
    """Replay ops natively; return (probabilities, expectations_z)."""
    dim = 1 << num_qubits
    amp_c, _amp = _dbuf([0.0] * (2 * dim))
    dll.uq_init_state(amp_c, num_qubits)
    opcodes, qubits, params = _encode_ops(ops)
    n = len(opcodes)
    if n:
        opc_c = (ctypes.c_int * n).from_buffer(opcodes)
        qub_c = (ctypes.c_int * (2 * n)).from_buffer(qubits)
        par_c = (ctypes.c_double * n).from_buffer(params)
        dll.uq_run_circuit(amp_c, num_qubits, opc_c, qub_c, par_c, n)
    probs_c, probs = _dbuf([0.0] * dim)
    dll.uq_probabilities(amp_c, num_qubits, probs_c)
    exp_c, expz = _dbuf([0.0] * num_qubits)
    dll.uq_expectations_z(amp_c, num_qubits, exp_c)
    return list(probs), list(expz)


_GATES_1Q = ("h", "x", "y", "z", "s", "t")
_GATES_ROT = ("rx", "ry", "rz")
_GATES_2Q = ("cnot", "cz", "swap")


def _random_circuit(circuit_cls, rng, num_qubits, depth):
    c = circuit_cls(num_qubits)
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.4:
            getattr(c, rng.choice(_GATES_1Q))(rng.randrange(num_qubits))
        elif roll < 0.7:
            getattr(c, rng.choice(_GATES_ROT))(
                rng.randrange(num_qubits), rng.uniform(-2 * math.pi, 2 * math.pi)
            )
        else:
            a, b = rng.sample(range(num_qubits), 2)
            getattr(c, rng.choice(_GATES_2Q))(a, b)
    return c


@unittest.skipUnless(_DLL is not None, "native CPU DLL not available (build with native\\build.ps1 -Target cpu)")
class TestNativeCpuParity(unittest.TestCase):
    """Native CPU tier must match the pure-Python oracle to 1e-9."""

    def setUp(self):
        # Lazy import: the pure-Python tier is written by a separate workflow,
        # so collection must never crash and absence must skip, not fail.
        try:
            from ultraquant.quantum.circuit import Circuit
            from ultraquant.quantum.state import QuantumState
            from ultraquant.quantum.backend import SimulatorBackend
            from ultraquant.quantum.qlayer import QuantumFeatureMap
            from ultraquant.model.quantize import TernaryLinear, quantize_ternary
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"pure-Python ultraquant tier not importable: {exc!r}")
        self.Circuit = Circuit
        self.QuantumState = QuantumState
        self.SimulatorBackend = SimulatorBackend
        self.QuantumFeatureMap = QuantumFeatureMap
        self.TernaryLinear = TernaryLinear
        self.quantize_ternary = quantize_ternary

    # -- helpers ----------------------------------------------------------

    def _python_run(self, num_qubits, circuit):
        state = self.QuantumState(num_qubits)
        circuit.apply_to(state)
        probs = state.probabilities()
        expz = [state.expectation_z(q) for q in range(num_qubits)]
        return probs, expz

    def _python_feature_map(self, features, num_qubits):
        qmap = self.QuantumFeatureMap(
            num_qubits, self.SimulatorBackend(seed=0), shots=None
        )
        return qmap.extract(features)

    def _native_feature_map(self, features, num_qubits):
        f_c, _f = _dbuf(features)
        out_c, out = _dbuf([0.0] * num_qubits)
        _DLL.uq_feature_map(f_c, len(features), num_qubits, out_c)
        return list(out)

    def assertListsClose(self, got, want, msg=""):
        self.assertEqual(len(got), len(want), msg)
        for i, (g, w) in enumerate(zip(got, want)):
            self.assertAlmostEqual(g, w, delta=TOL, msg=f"{msg} [index {i}]")

    # -- random circuit parity -------------------------------------------

    def test_random_circuits_parity(self):
        """20 random seeded circuits, 2-8 qubits: probs + <Z> within 1e-9."""
        rng = random.Random(20260728)
        for k in range(20):
            num_qubits = 2 + k % 7  # cycles 2..8
            depth = rng.randrange(10, 30)
            circuit = _random_circuit(self.Circuit, rng, num_qubits, depth)
            with self.subTest(circuit=k, qubits=num_qubits, depth=depth):
                py_probs, py_expz = self._python_run(num_qubits, circuit)
                nat_probs, nat_expz = _native_run(_DLL, num_qubits, circuit.ops)
                self.assertListsClose(nat_probs, py_probs, "probabilities")
                self.assertListsClose(nat_expz, py_expz, "expectations_z")

    # -- feature map parity ----------------------------------------------

    def test_feature_map_single_parity(self):
        cases = [
            ([0.4, -1.1, 0.2, 0.9, -0.6], 5),                # exact chunk
            ([0.3, -0.7, 1.2, 0.5, -0.2, 0.9, 0.1], 3),      # re-uploading 7f/3q
            ([0.8, -0.3], 4),                                # fewer features than qubits
            ([1.5], 1),                                      # 1 qubit, ring skipped
        ]
        for features, nq in cases:
            with self.subTest(n_features=len(features), num_qubits=nq):
                want = self._python_feature_map(features, nq)
                got = self._native_feature_map(features, nq)
                self.assertListsClose(got, want, "feature map")

    def test_feature_map_batch_parity(self):
        rng = random.Random(7)
        n_samples, n_features, nq = 12, 7, 3  # re-uploading case in batch form
        samples = [
            [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
            for _ in range(n_samples)
        ]
        flat = [f for row in samples for f in row]
        flat_c, _flat = _dbuf(flat)
        out_c, out = _dbuf([0.0] * (n_samples * nq))
        _DLL.uq_feature_map_batch(flat_c, n_samples, n_features, nq, out_c, 1)
        want = []
        for row in samples:
            want.extend(self._python_feature_map(row, nq))
        self.assertListsClose(list(out), want, "feature map batch vs python")

    def test_feature_map_batch_threads_equal(self):
        """n_threads=4 must equal n_threads=1 and single-sample calls."""
        rng = random.Random(11)
        n_samples, n_features, nq = 10, 7, 3
        flat = [rng.uniform(-2.0, 2.0) for _ in range(n_samples * n_features)]
        flat_c, _flat = _dbuf(flat)
        out1_c, out1 = _dbuf([0.0] * (n_samples * nq))
        _DLL.uq_feature_map_batch(flat_c, n_samples, n_features, nq, out1_c, 1)
        out4_c, out4 = _dbuf([0.0] * (n_samples * nq))
        _DLL.uq_feature_map_batch(flat_c, n_samples, n_features, nq, out4_c, 4)
        self.assertEqual(list(out1), list(out4))
        single = []
        for s in range(n_samples):
            single.extend(
                self._native_feature_map(
                    flat[s * n_features:(s + 1) * n_features], nq
                )
            )
        self.assertEqual(list(out1), single)

    # -- ternary forward parity ------------------------------------------

    def _make_layer(self, in_dim, out_dim, seed):
        layer = self.TernaryLinear(in_dim, out_dim, random.Random(seed), quantized=True)
        # Exercise the bias path too (TernaryLinear inits bias to zeros).
        rng = random.Random(seed + 1)
        layer.b = [rng.uniform(-0.5, 0.5) for _ in range(out_dim)]
        q, alpha = self.quantize_ternary(layer.w)
        qw_flat = [int(q[o][i]) for o in range(out_dim) for i in range(in_dim)]
        return layer, qw_flat, alpha

    def test_ternary_forward_parity(self):
        in_dim, out_dim = 6, 4
        layer, qw_flat, alpha = self._make_layer(in_dim, out_dim, seed=3)
        rng = random.Random(5)
        qw_c, _qw = _i8buf(qw_flat)
        bias_c, _b = _dbuf(layer.b)
        for trial in range(5):
            x = [rng.uniform(-3.0, 3.0) for _ in range(in_dim)]
            with self.subTest(trial=trial):
                want = layer.forward(x)
                x_c, _x = _dbuf(x)
                out_c, out = _dbuf([0.0] * out_dim)
                _DLL.uq_ternary_forward(qw_c, alpha, bias_c, x_c, in_dim, out_dim, out_c)
                self.assertListsClose(list(out), want, "ternary forward")

    def test_ternary_forward_batch_parity_and_threads(self):
        in_dim, out_dim, n_samples = 5, 3, 16
        layer, qw_flat, alpha = self._make_layer(in_dim, out_dim, seed=9)
        rng = random.Random(13)
        xs = [[rng.uniform(-3.0, 3.0) for _ in range(in_dim)] for _ in range(n_samples)]
        flat = [v for row in xs for v in row]
        qw_c, _qw = _i8buf(qw_flat)
        bias_c, _b = _dbuf(layer.b)
        flat_c, _flat = _dbuf(flat)
        out1_c, out1 = _dbuf([0.0] * (n_samples * out_dim))
        _DLL.uq_ternary_forward_batch(
            qw_c, alpha, bias_c, flat_c, n_samples, in_dim, out_dim, out1_c, 1
        )
        out4_c, out4 = _dbuf([0.0] * (n_samples * out_dim))
        _DLL.uq_ternary_forward_batch(
            qw_c, alpha, bias_c, flat_c, n_samples, in_dim, out_dim, out4_c, 4
        )
        # n_threads=4 == n_threads=1, bit-for-bit.
        self.assertEqual(list(out1), list(out4))
        # And both match TernaryLinear.forward per sample within 1e-9.
        want = []
        for row in xs:
            want.extend(layer.forward(row))
        self.assertListsClose(list(out1), want, "ternary batch vs python")


if __name__ == "__main__":
    unittest.main()
