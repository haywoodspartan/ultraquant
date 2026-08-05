"""Tests for the UltraQuant CUDA acceleration tier (SPEC-NATIVE.md sections 2 and 9).

The GPU DLL is exercised directly through its documented C ABI (ctypes), with
the pure-Python tier as the correctness oracle.  The GPU availability probe
runs at import time via ctypes only; all ultraquant imports are lazy (inside
setUp) so this module imports cleanly while the pure-Python tier is absent or
in flux.  Every test skips gracefully when the CUDA tier is unavailable.

Run from the project root:  python -m unittest tests.test_native_cuda -v
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
        os.pardir,
        "ultraquant",
        "native",
        "_bin",
        "ultraquant_cuda.dll",
    )
)

_PD = ctypes.POINTER(ctypes.c_double)
_PI = ctypes.POINTER(ctypes.c_int)
_PB = ctypes.POINTER(ctypes.c_byte)

# SPEC-NATIVE.md circuit wire format opcode table.
OPCODES = {
    "h": 0, "x": 1, "y": 2, "z": 3, "s": 4, "t": 5,
    "rx": 6, "ry": 7, "rz": 8, "cnot": 9, "cz": 10, "swap": 11,
}


def _load_gpu():
    """Load ultraquant_cuda.dll and verify a usable device; None on failure."""
    try:
        if not os.path.exists(_DLL_PATH):
            return None
        dll = ctypes.CDLL(_DLL_PATH)
        dll.uqg_available.restype = ctypes.c_int
        dll.uqg_available.argtypes = []
        if dll.uqg_available() < 1:
            return None
        dll.uqg_device_name.restype = ctypes.c_int
        dll.uqg_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.uqg_run_circuit.restype = None
        dll.uqg_run_circuit.argtypes = [
            _PD, ctypes.c_int, _PI, _PI, _PD, ctypes.c_int,
        ]
        dll.uqg_expectations_z_from.restype = None
        dll.uqg_expectations_z_from.argtypes = [_PD, ctypes.c_int, _PD]
        dll.uqg_feature_map_batch.restype = None
        dll.uqg_feature_map_batch.argtypes = [
            _PD, ctypes.c_int, ctypes.c_int, ctypes.c_int, _PD,
        ]
        dll.uqg_ternary_forward_batch.restype = None
        dll.uqg_ternary_forward_batch.argtypes = [
            _PB, ctypes.c_double, _PD, _PD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, _PD,
        ]
        return dll
    except Exception:
        return None


_GPU = _load_gpu()


def _dbuf(a):
    return (ctypes.c_double * len(a)).from_buffer(a)


def _ibuf(a):
    return (ctypes.c_int * len(a)).from_buffer(a)


def _bbuf(a):
    return (ctypes.c_byte * len(a)).from_buffer(a)


def _encode_ops(ops):
    """Encode Circuit.ops into the SPEC-NATIVE wire format arrays.

    Works for whichever op names the Circuit builder emits (e.g. swap either
    as a dedicated 'swap' op or lowered to three 'cnot' ops).
    """
    opcodes = array("i", [])
    qubits = array("i", [])
    params = array("d", [])
    for op in ops:
        if op.name not in OPCODES:
            raise ValueError("unknown op name: %r" % (op.name,))
        opcodes.append(OPCODES[op.name])
        q = list(op.qubits)
        qubits.append(q[0])
        qubits.append(q[1] if len(q) > 1 else -1)
        params.append(op.params[0] if op.params else 0.0)
    return opcodes, qubits, params


def _run_circuit_gpu(circuit):
    """Replay a Circuit on the GPU; returns interleaved re/im amplitudes."""
    opcodes, qubits, params = _encode_ops(circuit.ops)
    amp = array("d", [0.0] * (2 * (1 << circuit.num_qubits)))
    amp[0] = 1.0
    _GPU.uqg_run_circuit(
        _dbuf(amp), circuit.num_qubits,
        _ibuf(opcodes), _ibuf(qubits), _dbuf(params), len(opcodes),
    )
    return amp


def _probs_from_amp(amp, num_qubits):
    return [
        amp[2 * i] * amp[2 * i] + amp[2 * i + 1] * amp[2 * i + 1]
        for i in range(1 << num_qubits)
    ]


def _expectations_z_gpu(amp, num_qubits):
    out = array("d", [0.0] * num_qubits)
    _GPU.uqg_expectations_z_from(_dbuf(amp), num_qubits, _dbuf(out))
    return list(out)


def _feature_map_gpu(samples, num_qubits):
    n_samples = len(samples)
    n_features = len(samples[0]) if n_samples else 0
    flat = array("d", [f for row in samples for f in row])
    out = array("d", [0.0] * (n_samples * num_qubits))
    _GPU.uqg_feature_map_batch(
        _dbuf(flat) if len(flat) else None,
        n_samples, n_features, num_qubits, _dbuf(out),
    )
    return [
        list(out[s * num_qubits:(s + 1) * num_qubits]) for s in range(n_samples)
    ]


def _ternary_gpu(qw_rows, alpha, bias, xs):
    out_dim = len(qw_rows)
    in_dim = len(qw_rows[0])
    n_samples = len(xs)
    qw = array("b", [int(v) for row in qw_rows for v in row])
    bias_a = array("d", bias)
    xs_a = array("d", [v for row in xs for v in row])
    out = array("d", [0.0] * (n_samples * out_dim))
    _GPU.uqg_ternary_forward_batch(
        _bbuf(qw), alpha, _dbuf(bias_a), _dbuf(xs_a),
        n_samples, in_dim, out_dim, _dbuf(out),
    )
    return [
        list(out[s * out_dim:(s + 1) * out_dim]) for s in range(n_samples)
    ]


@unittest.skipUnless(
    _GPU is not None,
    "CUDA tier unavailable (ultraquant_cuda.dll missing, unloadable, or "
    "uqg_available() < 1)",
)
class TestNativeCuda(unittest.TestCase):
    """Parity battery: CUDA tier vs the pure-Python correctness oracle."""

    def setUp(self):
        # Lazy imports: the pure-Python tier is only needed when tests run.
        from ultraquant.quantum.circuit import Circuit
        from ultraquant.quantum.state import QuantumState
        from ultraquant.quantum.backend import SimulatorBackend
        from ultraquant.quantum.qlayer import QuantumFeatureMap
        from ultraquant.model.quantize import TernaryLinear, quantize_ternary

        self.Circuit = Circuit
        self.QuantumState = QuantumState
        self.SimulatorBackend = SimulatorBackend
        self.QuantumFeatureMap = QuantumFeatureMap
        self.TernaryLinear = TernaryLinear
        self.quantize_ternary = quantize_ternary

    # -- helpers ------------------------------------------------------------

    def _random_circuit(self, rng, num_qubits, depth):
        c = self.Circuit(num_qubits)
        one_q = ["h", "x", "y", "z", "s", "t"]
        rot = ["rx", "ry", "rz"]
        two_q = ["cnot", "cz", "swap"]
        for _ in range(depth):
            kind = rng.choice(one_q + rot + two_q)
            if kind in two_q:
                a, b = rng.sample(range(num_qubits), 2)
                getattr(c, kind)(a, b)
            elif kind in rot:
                getattr(c, kind)(
                    rng.randrange(num_qubits),
                    rng.uniform(-2.0 * math.pi, 2.0 * math.pi),
                )
            else:
                getattr(c, kind)(rng.randrange(num_qubits))
        return c

    def _assert_circuit_parity(self, circuit):
        state = self.QuantumState(circuit.num_qubits)
        circuit.apply_to(state)
        want_probs = state.probabilities()
        want_ez = [
            state.expectation_z(q) for q in range(circuit.num_qubits)
        ]

        amp = _run_circuit_gpu(circuit)
        got_probs = _probs_from_amp(amp, circuit.num_qubits)
        got_ez = _expectations_z_gpu(amp, circuit.num_qubits)

        for i, (g, w) in enumerate(zip(got_probs, want_probs)):
            self.assertLess(
                abs(g - w), TOL,
                "probability mismatch at basis index %d: gpu %r vs python %r"
                % (i, g, w),
            )
        for q, (g, w) in enumerate(zip(got_ez, want_ez)):
            self.assertLess(
                abs(g - w), TOL,
                "<Z_%d> mismatch: gpu %r vs python %r" % (q, g, w),
            )

    # -- tests --------------------------------------------------------------

    def test_available_and_device_name(self):
        self.assertGreaterEqual(_GPU.uqg_available(), 1)
        buf = ctypes.create_string_buffer(256)
        self.assertEqual(_GPU.uqg_device_name(buf, 256), 0)
        self.assertTrue(len(buf.value) > 0)

    def test_bell_circuit(self):
        c = self.Circuit(2)
        c.h(0).cnot(0, 1)
        amp = _run_circuit_gpu(c)
        probs = _probs_from_amp(amp, 2)
        for got, want in zip(probs, [0.5, 0.0, 0.0, 0.5]):
            self.assertLess(abs(got - want), TOL)
        for ez in _expectations_z_gpu(amp, 2):
            self.assertLess(abs(ez), TOL)

    def test_random_circuits_parity(self):
        """20 seeded random circuits (mixed gates, 2-8 qubits): probabilities
        and Z expectations agree with the pure-Python simulator to 1e-9."""
        for seed in range(20):
            rng = random.Random(1000 + seed)
            num_qubits = rng.randint(2, 8)
            depth = rng.randint(8, 24)
            circuit = self._random_circuit(rng, num_qubits, depth)
            with self.subTest(seed=seed, num_qubits=num_qubits, depth=depth):
                self._assert_circuit_parity(circuit)

    def test_run_circuit_12_qubits(self):
        """uqg_run_circuit at 12 qubits vs pure Python (1e-9)."""
        rng = random.Random(4242)
        circuit = self._random_circuit(rng, 12, 40)
        self._assert_circuit_parity(circuit)
        amp = _run_circuit_gpu(circuit)
        norm = math.sqrt(math.fsum(a * a for a in amp))
        self.assertLess(abs(norm - 1.0), TOL)

    def test_feature_map_single_parity(self):
        """Single-sample feature map parity, incl. the plain (no re-upload)
        case, against pure-Python QuantumFeatureMap.extract."""
        cases = [
            (2, [0.3, -0.7]),              # exactly num_qubits features
            (5, [0.1, -0.2, 0.9, -1.5, 0.4]),
            (4, [0.25, -0.5]),             # fewer features than qubits
        ]
        for num_qubits, features in cases:
            with self.subTest(num_qubits=num_qubits, n_features=len(features)):
                qmap = self.QuantumFeatureMap(
                    num_qubits, self.SimulatorBackend(seed=0), shots=None
                )
                want = qmap.extract(features)
                got = _feature_map_gpu([features], num_qubits)[0]
                self.assertEqual(len(got), num_qubits)
                for q in range(num_qubits):
                    self.assertLess(
                        abs(got[q] - want[q]), TOL,
                        "feature map qubit %d: gpu %r vs python %r"
                        % (q, got[q], want[q]),
                    )

    def test_feature_map_batch_parity_reuploading(self):
        """Batch feature map with data re-uploading (7 features / 3 qubits)
        matches per-sample pure Python and the GPU's own single-sample calls."""
        num_qubits, n_features = 3, 7
        rng = random.Random(7)
        samples = [
            [rng.uniform(-2.0, 2.0) for _ in range(n_features)]
            for _ in range(16)
        ]
        qmap = self.QuantumFeatureMap(
            num_qubits, self.SimulatorBackend(seed=0), shots=None
        )
        got_batch = _feature_map_gpu(samples, num_qubits)
        self.assertEqual(len(got_batch), len(samples))
        for s, features in enumerate(samples):
            want = qmap.extract(features)
            for q in range(num_qubits):
                self.assertLess(
                    abs(got_batch[s][q] - want[q]), TOL,
                    "batch feature map sample %d qubit %d: gpu %r vs python %r"
                    % (s, q, got_batch[s][q], want[q]),
                )
            single = _feature_map_gpu([features], num_qubits)[0]
            for q in range(num_qubits):
                self.assertLess(
                    abs(single[q] - got_batch[s][q]), TOL,
                    "batch/single inconsistency at sample %d qubit %d" % (s, q),
                )

    def test_ternary_forward_parity(self):
        """uqg_ternary_forward_batch matches TernaryLinear.forward."""
        rng = random.Random(3)
        in_dim, out_dim = 6, 4
        layer = self.TernaryLinear(in_dim, out_dim, rng, quantized=True)
        for o in range(out_dim):
            layer.b[o] = 0.125 * (o + 1) * (-1.0 if o % 2 else 1.0)
        qw, alpha = self.quantize_ternary(layer.w)
        xs = [
            [rng.uniform(-2.0, 2.0) for _ in range(in_dim)] for _ in range(8)
        ]
        got = _ternary_gpu(qw, alpha, list(layer.b), xs)
        for s, x in enumerate(xs):
            want = layer.forward(x)
            for o in range(out_dim):
                self.assertLess(
                    abs(got[s][o] - want[o]), TOL,
                    "ternary forward [%d][%d]: gpu %r vs python %r"
                    % (s, o, got[s][o], want[o]),
                )


if __name__ == "__main__":
    unittest.main()
