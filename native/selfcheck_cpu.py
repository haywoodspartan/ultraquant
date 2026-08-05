"""Standalone ground-truth selfcheck for ultraquant_native.dll (CPU tier).

Deliberately does NOT import the ultraquant package: only ctypes, math, array
(plus os/sys for locating the DLL and exiting). Every expected value is either
hand-computed or produced by a tiny pure-Python statevector reimplementation
written inside this file.

Run from anywhere:  python native\\selfcheck_cpu.py
Exits 0 and prints "ALL SELFCHECKS PASSED" on success; raises AssertionError
otherwise. Tolerance: 1e-9.
"""

import ctypes
import math
import os
import sys
from array import array

TOL = 1e-9

# Opcodes per SPEC-NATIVE.md wire format.
OP_H, OP_X, OP_Y, OP_Z, OP_S, OP_T = 0, 1, 2, 3, 4, 5
OP_RX, OP_RY, OP_RZ, OP_CNOT, OP_CZ, OP_SWAP = 6, 7, 8, 9, 10, 11

DLL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "ultraquant", "native", "_bin", "ultraquant_native.dll",
    )
)


def load_dll() -> ctypes.CDLL:
    dll = ctypes.CDLL(DLL_PATH)
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
    return dll


def dbuf(values):
    """array('d') -> ctypes double array sharing the same memory."""
    arr = array("d", values)
    return (ctypes.c_double * len(arr)).from_buffer(arr), arr


def ibuf(values):
    arr = array("i", values)
    return (ctypes.c_int * len(arr)).from_buffer(arr), arr


def i8buf(values):
    arr = array("b", values)
    return (ctypes.c_int8 * len(arr)).from_buffer(arr), arr


def run_circuit(dll, num_qubits, ops):
    """ops: list of (opcode, q0, q1, param). Returns (amps, probs, expz)."""
    dim = 1 << num_qubits
    amp_c, amp = dbuf([0.0] * (2 * dim))
    dll.uq_init_state(amp_c, num_qubits)
    if ops:
        opc_c, _opc = ibuf([o[0] for o in ops])
        qub_c, _qub = ibuf([q for o in ops for q in (o[1], o[2])])
        par_c, _par = dbuf([o[3] for o in ops])
        dll.uq_run_circuit(amp_c, num_qubits, opc_c, qub_c, par_c, len(ops))
    probs_c, probs = dbuf([0.0] * dim)
    dll.uq_probabilities(amp_c, num_qubits, probs_c)
    exp_c, expz = dbuf([0.0] * num_qubits)
    dll.uq_expectations_z(amp_c, num_qubits, exp_c)
    return list(amp), list(probs), list(expz)


def assert_close(got, want, label):
    if isinstance(want, (int, float)):
        got, want = [got], [want]
    assert len(got) == len(want), f"{label}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        assert abs(g - w) <= TOL, f"{label}[{i}]: got {g!r}, want {w!r}"
    print(f"  ok: {label}")


# ---------------------------------------------------------------------------
# Tiny pure-Python reimplementation of the SPEC.md §5 feature map (for check 4).
# ---------------------------------------------------------------------------

def py_feature_map(features, num_qubits):
    dim = 1 << num_qubits
    amp = [0j] * dim
    amp[0] = 1 + 0j

    def apply_1q(m00, m01, m10, m11, q):
        bit = 1 << q
        for i0 in range(dim):
            if i0 & bit:
                continue
            i1 = i0 | bit
            a0, a1 = amp[i0], amp[i1]
            amp[i0] = m00 * a0 + m01 * a1
            amp[i1] = m10 * a0 + m11 * a1

    def apply_cnot(c, t):
        cbit, tbit = 1 << c, 1 << t
        for i0 in range(dim):
            if (i0 & tbit) or not (i0 & cbit):
                continue
            i1 = i0 | tbit
            amp[i0], amp[i1] = amp[i1], amp[i0]

    n = len(features)
    start = 0
    while True:  # do-while: first layer always runs (missing features -> 0)
        for q in range(num_qubits):
            fi = start + q
            theta = math.pi * math.tanh(features[fi]) if fi < n else 0.0
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            apply_1q(c, -s, s, c, q)  # RY(theta)
        if num_qubits > 1:
            for q in range(num_qubits):
                apply_cnot(q, (q + 1) % num_qubits)  # ring entangler
        start += num_qubits
        if start >= n:
            break
    out = []
    for q in range(num_qubits):
        e = 0.0
        for i in range(dim):
            p = abs(amp[i]) ** 2
            e += -p if (i >> q) & 1 else p
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_version_and_init(dll):
    print("check: version + init_state")
    assert dll.uq_version() == 1, "uq_version() != 1"
    print("  ok: uq_version == 1")
    # init_state must overwrite junk with |00>.
    amp_c, amp = dbuf([7.5] * 8)  # 2 qubits, garbage-filled
    dll.uq_init_state(amp_c, 2)
    assert_close(list(amp), [1, 0, 0, 0, 0, 0, 0, 0], "init_state |00>")


def check_bell(dll):
    print("check: Bell circuit (H q0, CNOT 0->1)")
    _amp, probs, expz = run_circuit(
        dll, 2, [(OP_H, 0, -1, 0.0), (OP_CNOT, 0, 1, 0.0)]
    )
    assert_close(probs, [0.5, 0.0, 0.0, 0.5], "Bell probabilities")
    assert_close(expz, [0.0, 0.0], "Bell <Z>")


def check_little_endian_x(dll):
    print("check: X on qubit 0 of 2 qubits -> amplitude at index 1")
    amp, probs, expz = run_circuit(dll, 2, [(OP_X, 0, -1, 0.0)])
    # Little-endian: qubit 0 is the LSB, so |q1 q0> = |01> is basis index 1.
    assert_close(amp, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "X amplitudes")
    assert_close(probs, [0.0, 1.0, 0.0, 0.0], "X probabilities")
    assert_close(expz, [-1.0, 1.0], "X <Z>")


def check_ry(dll):
    print("check: RY(1.0) on |0> gives <Z> = cos(1.0)")
    _amp, _probs, expz = run_circuit(dll, 1, [(OP_RY, 0, -1, 1.0)])
    assert_close(expz, [math.cos(1.0)], "RY(1.0) <Z>")


def check_feature_map(dll):
    print("check: 3-qubit ring feature map vs in-file pure-Python model")
    cases = [
        ([0.3, -0.7, 1.2], 3),          # single layer, exact chunk
        ([0.3, -0.7, 1.2, 0.5, -0.2, 0.9, 0.1], 3),  # re-uploading, ragged
        ([0.25], 1),                     # 1 qubit: ring skipped
    ]
    for features, nq in cases:
        want = py_feature_map(features, nq)
        f_c, _f = dbuf(features)
        out_c, out = dbuf([0.0] * nq)
        dll.uq_feature_map(f_c, len(features), nq, out_c)
        assert_close(list(out), want, f"feature_map({len(features)}f/{nq}q)")


def check_ternary_forward(dll):
    print("check: uq_ternary_forward vs hand matmul")
    # qw (out=2, in=3) row-major, alpha=0.5, bias, x — hand computed:
    # out0 = 0.5*(1*1 + (-1)*2 + 0*3) + 0.1 = -0.4
    # out1 = 0.5*(0*1 + 1*2 + 1*3) - 0.2 = 2.3
    qw_c, _qw = i8buf([1, -1, 0, 0, 1, 1])
    bias_c, _b = dbuf([0.1, -0.2])
    x_c, _x = dbuf([1.0, 2.0, 3.0])
    out_c, out = dbuf([0.0, 0.0])
    dll.uq_ternary_forward(qw_c, 0.5, bias_c, x_c, 3, 2, out_c)
    assert_close(list(out), [-0.4, 2.3], "ternary_forward hand matmul")


def check_batches(dll):
    print("check: batch paths (n_threads=4 == n_threads=1 == single-sample)")
    # Feature map batch: 9 samples x 7 features on 3 qubits (re-uploading).
    n_samples, n_features, nq = 9, 7, 3
    feats = [
        math.sin(0.7 * s + 1.3 * f) * 1.5
        for s in range(n_samples)
        for f in range(n_features)
    ]
    feats_c, _feats = dbuf(feats)

    out1_c, out1 = dbuf([0.0] * (n_samples * nq))
    dll.uq_feature_map_batch(feats_c, n_samples, n_features, nq, out1_c, 1)
    out4_c, out4 = dbuf([0.0] * (n_samples * nq))
    dll.uq_feature_map_batch(feats_c, n_samples, n_features, nq, out4_c, 4)
    single = []
    for s in range(n_samples):
        f_c, _f = dbuf(feats[s * n_features:(s + 1) * n_features])
        o_c, o = dbuf([0.0] * nq)
        dll.uq_feature_map(f_c, n_features, nq, o_c)
        single.extend(o)
    assert_close(list(out4), list(out1), "feature_map_batch t4 == t1")
    assert_close(list(out1), single, "feature_map_batch t1 == single calls")

    # Ternary batch: 10 samples, in=4, out=3.
    n_s, in_dim, out_dim = 10, 4, 3
    qw = [(((o * in_dim + i) * 7) % 3) - 1 for o in range(out_dim) for i in range(in_dim)]
    qw_c, _qw = i8buf(qw)
    bias_c, _b = dbuf([0.05 * o - 0.1 for o in range(out_dim)])
    xs = [math.cos(0.9 * s + 0.4 * i) * 2.0 for s in range(n_s) for i in range(in_dim)]
    xs_c, _xs = dbuf(xs)
    alpha = 0.75

    t1_c, t1 = dbuf([0.0] * (n_s * out_dim))
    dll.uq_ternary_forward_batch(qw_c, alpha, bias_c, xs_c, n_s, in_dim, out_dim, t1_c, 1)
    t4_c, t4 = dbuf([0.0] * (n_s * out_dim))
    dll.uq_ternary_forward_batch(qw_c, alpha, bias_c, xs_c, n_s, in_dim, out_dim, t4_c, 4)
    tsingle = []
    for s in range(n_s):
        x_c, _x = dbuf(xs[s * in_dim:(s + 1) * in_dim])
        o_c, o = dbuf([0.0] * out_dim)
        dll.uq_ternary_forward(qw_c, alpha, bias_c, x_c, in_dim, out_dim, o_c)
        tsingle.extend(o)
    assert_close(list(t4), list(t1), "ternary_batch t4 == t1")
    assert_close(list(t1), tsingle, "ternary_batch t1 == single calls")

    # Hand ground truth for the ternary batch too.
    want = []
    for s in range(n_s):
        for o in range(out_dim):
            acc = sum(qw[o * in_dim + i] * xs[s * in_dim + i] for i in range(in_dim))
            want.append(alpha * acc + (0.05 * o - 0.1))
    assert_close(list(t1), want, "ternary_batch vs pure-Python matmul")


def main() -> int:
    assert os.path.isfile(DLL_PATH), f"DLL not found: {DLL_PATH}"
    dll = load_dll()
    check_version_and_init(dll)
    check_bell(dll)
    check_little_endian_x(dll)
    check_ry(dll)
    check_feature_map(dll)
    check_ternary_forward(dll)
    check_batches(dll)
    print("ALL SELFCHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
