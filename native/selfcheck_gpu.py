"""Standalone GPU self-check for the UltraQuant CUDA tier (ultraquant_cuda.dll).

Deliberately self-contained: uses only ctypes / math / array (plus os.path and
sys for locating the DLL and exiting) and NO ultraquant imports, so it can
validate the native tier even while the pure-Python package is in flux.

Checks (SPEC-NATIVE.md section 2 semantics, SPEC.md little-endian convention):
  1. uqg_available() >= 1 and uqg_device_name() succeeds (device name printed).
  2. Bell circuit (H on q0, CNOT(0,1)) via uqg_run_circuit:
     <Z> = [0, 0] and probabilities [0.5, 0, 0, 0.5].
  3. RY(1.0) on one qubit: <Z> = cos(1.0).
  4. uqg_feature_map_batch on 8 samples matches an in-script pure-Python
     reimplementation of the SPEC.md section 5 feature map to 1e-9, and the
     batch result matches per-sample (n_samples=1) calls.
  5. A 12-qubit random-ish mixed-gate circuit conserves the state norm.
  6. uqg_ternary_forward_batch matches a hand-rolled matmul.

Run from anywhere:  python native\\selfcheck_gpu.py
Exit code 0 on success, 1 on any failure.
"""

import ctypes
import math
import os
import sys
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


def fail(msg):
    print("SELFCHECK FAIL: " + msg)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


def dbuf(a):
    """array('d') -> ctypes double pointer (shares the buffer)."""
    return (ctypes.c_double * len(a)).from_buffer(a)


def ibuf(a):
    return (ctypes.c_int * len(a)).from_buffer(a)


def bbuf(a):
    return (ctypes.c_byte * len(a)).from_buffer(a)


def load_dll():
    check(os.path.exists(_DLL_PATH), "DLL not found at " + _DLL_PATH)
    dll = ctypes.CDLL(_DLL_PATH)
    dll.uqg_available.restype = ctypes.c_int
    dll.uqg_available.argtypes = []
    dll.uqg_device_name.restype = ctypes.c_int
    dll.uqg_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.uqg_run_circuit.restype = None
    dll.uqg_run_circuit.argtypes = [_PD, ctypes.c_int, _PI, _PI, _PD, ctypes.c_int]
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


# ---------------------------------------------------------------------------
# GPU call helpers
# ---------------------------------------------------------------------------

def run_circuit(dll, num_qubits, ops):
    """ops: list of (opcode, q0, q1, param). Returns interleaved amplitudes."""
    opcodes = array("i", [op[0] for op in ops])
    qubits = array("i", [])
    for op in ops:
        qubits.append(op[1])
        qubits.append(op[2])
    params = array("d", [op[3] for op in ops])
    amp = array("d", [0.0] * (2 * (1 << num_qubits)))
    amp[0] = 1.0
    dll.uqg_run_circuit(
        dbuf(amp), num_qubits, ibuf(opcodes), ibuf(qubits), dbuf(params), len(ops)
    )
    return amp


def expectations_z(dll, amp, num_qubits):
    out = array("d", [0.0] * num_qubits)
    dll.uqg_expectations_z_from(dbuf(amp), num_qubits, dbuf(out))
    return list(out)


def probabilities(amp, num_qubits):
    return [
        amp[2 * i] * amp[2 * i] + amp[2 * i + 1] * amp[2 * i + 1]
        for i in range(1 << num_qubits)
    ]


def feature_map_gpu(dll, samples, num_qubits):
    """samples: list of equal-length feature lists. Returns list of rows."""
    n_samples = len(samples)
    n_features = len(samples[0]) if n_samples else 0
    flat = array("d", [f for row in samples for f in row])
    out = array("d", [0.0] * (n_samples * num_qubits))
    dll.uqg_feature_map_batch(
        dbuf(flat) if len(flat) else None,
        n_samples,
        n_features,
        num_qubits,
        dbuf(out),
    )
    return [list(out[s * num_qubits:(s + 1) * num_qubits]) for s in range(n_samples)]


# ---------------------------------------------------------------------------
# Pure-Python oracle: SPEC.md section 5 feature map, little-endian indexing
# ---------------------------------------------------------------------------

def _py_ry(amp, num_qubits, qubit, theta):
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    mask = 1 << qubit
    for i0 in range(1 << num_qubits):
        if i0 & mask:
            continue
        i1 = i0 | mask
        a0 = amp[i0]
        a1 = amp[i1]
        amp[i0] = c * a0 - s * a1
        amp[i1] = s * a0 + c * a1


def _py_cnot(amp, num_qubits, control, target):
    cmask = 1 << control
    tmask = 1 << target
    for i0 in range(1 << num_qubits):
        if (i0 & tmask) or not (i0 & cmask):
            continue
        i1 = i0 | tmask
        amp[i0], amp[i1] = amp[i1], amp[i0]


def py_feature_map(features, num_qubits):
    """Exact reimplementation of SPEC.md section 5 QuantumFeatureMap.encode +
    per-qubit <Z>: theta_i = pi*tanh(f_i); RY layer per chunk of num_qubits
    features (missing -> angle 0); ring CNOT entangler (skip for 1 qubit);
    data re-uploading until features are exhausted."""
    dim = 1 << num_qubits
    amp = [0j] * dim
    amp[0] = 1.0 + 0j
    n_features = len(features)
    layers = max(1, -(-n_features // num_qubits))
    for layer in range(layers):
        for q in range(num_qubits):
            fi = layer * num_qubits + q
            theta = math.pi * math.tanh(features[fi]) if fi < n_features else 0.0
            _py_ry(amp, num_qubits, q, theta)
        if num_qubits > 1:
            for q in range(num_qubits):
                _py_cnot(amp, num_qubits, q, (q + 1) % num_qubits)
    probs = [abs(a) ** 2 for a in amp]
    return [
        sum(p if not (i >> q) & 1 else -p for i, p in enumerate(probs))
        for q in range(num_qubits)
    ]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_device(dll):
    n = dll.uqg_available()
    check(n >= 1, "uqg_available() returned %d (expected >= 1)" % n)
    buf = ctypes.create_string_buffer(256)
    rc = dll.uqg_device_name(buf, 256)
    check(rc == 0, "uqg_device_name returned %d" % rc)
    name = buf.value.decode("utf-8", "replace")
    check(len(name) > 0, "empty device name")
    print("[1] device count %d, device 0: %s" % (n, name))


def check_bell(dll):
    # H on q0 then CNOT(control=0, target=1). Opcodes: 0=h, 9=cnot.
    ops = [(0, 0, -1, 0.0), (9, 0, 1, 0.0)]
    amp = run_circuit(dll, 2, ops)
    ez = expectations_z(dll, amp, 2)
    check(abs(ez[0]) < TOL and abs(ez[1]) < TOL,
          "Bell <Z> expected [0, 0], got %r" % (ez,))
    probs = probabilities(amp, 2)
    expected = [0.5, 0.0, 0.0, 0.5]
    for i in range(4):
        check(abs(probs[i] - expected[i]) < TOL,
              "Bell probabilities expected %r, got %r" % (expected, probs))
    print("[2] Bell circuit: <Z> = [0, 0], probs = [0.5, 0, 0, 0.5]")


def check_ry(dll):
    ops = [(7, 0, -1, 1.0)]  # 7 = ry
    amp = run_circuit(dll, 1, ops)
    ez = expectations_z(dll, amp, 1)
    want = math.cos(1.0)
    check(abs(ez[0] - want) < TOL,
          "RY(1.0) <Z> expected cos(1.0)=%.15f, got %.15f" % (want, ez[0]))
    print("[3] RY(1.0): <Z> = cos(1.0) (delta %.3e)" % abs(ez[0] - want))


def check_feature_map(dll):
    num_qubits = 3
    n_features = 7  # forces the data re-uploading path (3 layers on 3 qubits)
    samples = [
        [math.sin(0.7 * s + 1.3 * j + 0.5) for j in range(n_features)]
        for s in range(8)
    ]
    got = feature_map_gpu(dll, samples, num_qubits)
    worst = 0.0
    for s in range(8):
        want = py_feature_map(samples[s], num_qubits)
        for q in range(num_qubits):
            d = abs(got[s][q] - want[q])
            worst = max(worst, d)
            check(d < TOL,
                  "feature map sample %d qubit %d: gpu %.15f vs python %.15f"
                  % (s, q, got[s][q], want[q]))
    # Per-sample consistency: n_samples=1 calls must match the batch rows.
    for s in range(8):
        single = feature_map_gpu(dll, [samples[s]], num_qubits)[0]
        for q in range(num_qubits):
            check(abs(single[q] - got[s][q]) < 1e-12,
                  "feature map batch/single mismatch at sample %d qubit %d" % (s, q))
    print("[4] feature map batch (8 samples, 7 features / 3 qubits, "
          "re-uploading): matches pure Python (worst delta %.3e) "
          "and per-sample calls" % worst)


def check_norm_12q(dll):
    num_qubits = 12
    ops = []
    for q in range(num_qubits):
        ops.append((0, q, -1, 0.0))  # h
    for layer in range(4):
        for q in range(num_qubits):
            ops.append((7, q, -1, math.sin(0.3 * layer + 0.17 * q + 0.05)))  # ry
        for q in range(num_qubits):
            ops.append((9, q, (q + 1) % num_qubits, 0.0))  # cnot ring
        ops.append((6, layer % num_qubits, -1, 0.4 + 0.1 * layer))  # rx
        ops.append((8, (layer + 3) % num_qubits, -1, -0.7 + 0.2 * layer))  # rz
    ops.append((4, 2, -1, 0.0))    # s
    ops.append((5, 7, -1, 0.0))    # t
    ops.append((1, 0, -1, 0.0))    # x
    ops.append((2, 5, -1, 0.0))    # y
    ops.append((3, 9, -1, 0.0))    # z
    ops.append((10, 1, 6, 0.0))    # cz
    ops.append((11, 3, 8, 0.0))    # swap
    amp = run_circuit(dll, num_qubits, ops)
    norm = math.sqrt(math.fsum(a * a for a in amp))
    check(abs(norm - 1.0) < TOL,
          "12-qubit circuit norm expected 1.0, got %.15f" % norm)
    ez = expectations_z(dll, amp, num_qubits)
    for q, v in enumerate(ez):
        check(-1.0 - TOL <= v <= 1.0 + TOL,
              "12-qubit <Z_%d> out of [-1, 1]: %r" % (q, v))
    print("[5] 12-qubit circuit (%d ops, all opcodes): norm conserved "
          "(delta %.3e)" % (len(ops), abs(norm - 1.0)))


def check_ternary(dll):
    in_dim, out_dim, n_samples = 4, 3, 5
    qw_rows = [[1, -1, 0, 1], [0, 1, 1, -1], [-1, 0, 1, 0]]
    alpha = 0.75
    bias = [0.5, -0.25, 0.125]
    xs = [
        [math.cos(0.3 * s + 0.9 * i) for i in range(in_dim)]
        for s in range(n_samples)
    ]
    qw = array("b", [v for row in qw_rows for v in row])
    bias_a = array("d", bias)
    xs_a = array("d", [v for row in xs for v in row])
    out = array("d", [0.0] * (n_samples * out_dim))
    dll.uqg_ternary_forward_batch(
        bbuf(qw), alpha, dbuf(bias_a), dbuf(xs_a),
        n_samples, in_dim, out_dim, dbuf(out),
    )
    worst = 0.0
    for s in range(n_samples):
        for o in range(out_dim):
            want = alpha * sum(
                qw_rows[o][i] * xs[s][i] for i in range(in_dim)
            ) + bias[o]
            got = out[s * out_dim + o]
            d = abs(got - want)
            worst = max(worst, d)
            check(d < TOL,
                  "ternary forward [%d][%d]: gpu %.15f vs python %.15f"
                  % (s, o, got, want))
    print("[6] ternary forward batch: matches hand matmul "
          "(worst delta %.3e)" % worst)


def main():
    print("UltraQuant CUDA tier self-check")
    print("DLL: " + _DLL_PATH)
    dll = load_dll()
    check_device(dll)
    check_bell(dll)
    check_ry(dll)
    check_feature_map(dll)
    check_norm_12q(dll)
    check_ternary(dll)
    print("SELFCHECK PASS")


if __name__ == "__main__":
    main()
