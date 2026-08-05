// native/uq_core.cpp — UltraQuant native CPU acceleration tier.
//
// Implements SPEC-NATIVE.md §1: C++17, stdlib only, std::thread for batch paths.
// Builds to ultraquant/native/_bin/ultraquant_native.dll via native/build.ps1.
//
// Conventions (binding, from SPEC.md):
//   * Little-endian qubit indexing: qubit q is bit q of the basis-state index
//     (LSB = qubit 0), identical to SPEC.md §2.
//   * Statevectors cross the FFI as interleaved re/im doubles:
//     double[2 * 2^n], amplitude i lives at (amp[2i], amp[2i+1]).
//   * Circuit wire format: int32 opcodes[N], int32 qubits[2N] (q0, q1; unused
//     slot = -1), double params[N] (unused slot = 0.0). Opcodes:
//     0=h 1=x 2=y 3=z 4=s 5=t 6=rx 7=ry 8=rz 9=cnot 10=cz 11=swap.
//   * Feature map per SPEC.md §5: theta_i = pi * tanh(f_i); RY layer over a
//     chunk of num_qubits features (missing -> angle 0); ring CNOT entangler
//     cnot(q, (q+1) % num_qubits) (skipped for 1 qubit); data re-uploading
//     until features are exhausted; then per-qubit <Z>.

#include <cmath>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#define UQ_EXPORT extern "C" __declspec(dllexport)

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Apply a 2x2 gate [[m00, m01], [m10, m11]] (complex entries as re/im pairs)
// to `qubit` of an interleaved statevector.
void apply_1q(double* amp, int num_qubits, int qubit,
              double m00r, double m00i, double m01r, double m01i,
              double m10r, double m10i, double m11r, double m11i) {
    const long long dim = 1LL << num_qubits;
    const long long bit = 1LL << qubit;
    for (long long i0 = 0; i0 < dim; ++i0) {
        if (i0 & bit) continue;
        const long long i1 = i0 | bit;
        const double a0r = amp[2 * i0], a0i = amp[2 * i0 + 1];
        const double a1r = amp[2 * i1], a1i = amp[2 * i1 + 1];
        amp[2 * i0]     = m00r * a0r - m00i * a0i + m01r * a1r - m01i * a1i;
        amp[2 * i0 + 1] = m00r * a0i + m00i * a0r + m01r * a1i + m01i * a1r;
        amp[2 * i1]     = m10r * a0r - m10i * a0i + m11r * a1r - m11i * a1i;
        amp[2 * i1 + 1] = m10r * a0i + m10i * a0r + m11r * a1i + m11i * a1r;
    }
}

// Apply a 2x2 gate to `target`, only on basis states where the `control` bit
// is 1 (controlled-U; CNOT = controlled X, CZ = controlled Z).
void apply_ctrl(double* amp, int num_qubits, int control, int target,
                double m00r, double m00i, double m01r, double m01i,
                double m10r, double m10i, double m11r, double m11i) {
    const long long dim = 1LL << num_qubits;
    const long long cbit = 1LL << control;
    const long long tbit = 1LL << target;
    for (long long i0 = 0; i0 < dim; ++i0) {
        if ((i0 & tbit) || !(i0 & cbit)) continue;
        const long long i1 = i0 | tbit;
        const double a0r = amp[2 * i0], a0i = amp[2 * i0 + 1];
        const double a1r = amp[2 * i1], a1i = amp[2 * i1 + 1];
        amp[2 * i0]     = m00r * a0r - m00i * a0i + m01r * a1r - m01i * a1i;
        amp[2 * i0 + 1] = m00r * a0i + m00i * a0r + m01r * a1i + m01i * a1r;
        amp[2 * i1]     = m10r * a0r - m10i * a0i + m11r * a1r - m11i * a1i;
        amp[2 * i1 + 1] = m10r * a0i + m10i * a0r + m11r * a1i + m11i * a1r;
    }
}

// SWAP(a, b): exchange amplitudes of basis states that differ exactly in
// bits a and b.
void apply_swap(double* amp, int num_qubits, int a, int b) {
    const long long dim = 1LL << num_qubits;
    const long long abit = 1LL << a;
    const long long bbit = 1LL << b;
    for (long long i = 0; i < dim; ++i) {
        if ((i & abit) && !(i & bbit)) {
            const long long j = (i ^ abit) | bbit;
            std::swap(amp[2 * i], amp[2 * j]);
            std::swap(amp[2 * i + 1], amp[2 * j + 1]);
        }
    }
}

void apply_op(double* amp, int num_qubits, int opcode, int q0, int q1,
              double param) {
    switch (opcode) {
        case 0: {  // h
            const double s = 1.0 / std::sqrt(2.0);
            apply_1q(amp, num_qubits, q0, s, 0, s, 0, s, 0, -s, 0);
            break;
        }
        case 1:  // x
            apply_1q(amp, num_qubits, q0, 0, 0, 1, 0, 1, 0, 0, 0);
            break;
        case 2:  // y = [[0, -i], [i, 0]]
            apply_1q(amp, num_qubits, q0, 0, 0, 0, -1, 0, 1, 0, 0);
            break;
        case 3:  // z
            apply_1q(amp, num_qubits, q0, 1, 0, 0, 0, 0, 0, -1, 0);
            break;
        case 4:  // s = diag(1, i)
            apply_1q(amp, num_qubits, q0, 1, 0, 0, 0, 0, 0, 0, 1);
            break;
        case 5: {  // t = diag(1, e^{i pi/4})
            const double c = std::cos(kPi / 4.0), sn = std::sin(kPi / 4.0);
            apply_1q(amp, num_qubits, q0, 1, 0, 0, 0, 0, 0, c, sn);
            break;
        }
        case 6: {  // rx(t) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
            const double c = std::cos(param * 0.5), sn = std::sin(param * 0.5);
            apply_1q(amp, num_qubits, q0, c, 0, 0, -sn, 0, -sn, c, 0);
            break;
        }
        case 7: {  // ry(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            const double c = std::cos(param * 0.5), sn = std::sin(param * 0.5);
            apply_1q(amp, num_qubits, q0, c, 0, -sn, 0, sn, 0, c, 0);
            break;
        }
        case 8: {  // rz(t) = diag(e^{-i t/2}, e^{i t/2})
            const double c = std::cos(param * 0.5), sn = std::sin(param * 0.5);
            apply_1q(amp, num_qubits, q0, c, -sn, 0, 0, 0, 0, c, sn);
            break;
        }
        case 9:  // cnot(control=q0, target=q1)
            apply_ctrl(amp, num_qubits, q0, q1, 0, 0, 1, 0, 1, 0, 0, 0);
            break;
        case 10:  // cz(control=q0, target=q1)
            apply_ctrl(amp, num_qubits, q0, q1, 1, 0, 0, 0, 0, 0, -1, 0);
            break;
        case 11:  // swap(q0, q1)
            apply_swap(amp, num_qubits, q0, q1);
            break;
        default:  // unknown opcode: no-op (Python side validates)
            break;
    }
}

int resolve_threads(int n_threads, int n_samples) {
    if (n_threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        n_threads = hc > 0 ? static_cast<int>(hc) : 1;
    }
    if (n_samples < 1) return 1;
    if (n_threads > n_samples) n_threads = n_samples;
    return n_threads;
}

}  // namespace

UQ_EXPORT int uq_version(void) { return 1; }

UQ_EXPORT void uq_init_state(double* amp, int num_qubits) {
    const long long dim = 1LL << num_qubits;
    for (long long i = 0; i < 2 * dim; ++i) amp[i] = 0.0;
    amp[0] = 1.0;  // |0...0>
}

UQ_EXPORT void uq_run_circuit(double* amp, int num_qubits, const int* opcodes,
                              const int* qubits, const double* params,
                              int n_ops) {
    for (int k = 0; k < n_ops; ++k)
        apply_op(amp, num_qubits, opcodes[k], qubits[2 * k], qubits[2 * k + 1],
                 params[k]);
}

UQ_EXPORT void uq_probabilities(const double* amp, int num_qubits,
                                double* out /* 2**n */) {
    const long long dim = 1LL << num_qubits;
    for (long long i = 0; i < dim; ++i)
        out[i] = amp[2 * i] * amp[2 * i] + amp[2 * i + 1] * amp[2 * i + 1];
}

UQ_EXPORT void uq_expectations_z(const double* amp, int num_qubits,
                                 double* out /* n */) {
    const long long dim = 1LL << num_qubits;
    for (int q = 0; q < num_qubits; ++q) out[q] = 0.0;
    for (long long i = 0; i < dim; ++i) {
        const double p =
            amp[2 * i] * amp[2 * i] + amp[2 * i + 1] * amp[2 * i + 1];
        for (int q = 0; q < num_qubits; ++q)
            out[q] += ((i >> q) & 1) ? -p : p;
    }
}

UQ_EXPORT void uq_feature_map(const double* features, int n_features,
                              int num_qubits, double* out /* num_qubits */) {
    if (num_qubits <= 0) return;
    const size_t dim = static_cast<size_t>(1) << num_qubits;
    std::vector<double> amp(2 * dim, 0.0);
    amp[0] = 1.0;
    // Data re-uploading loop: the first RY layer + entangler always runs
    // (missing features encode angle 0), then repeats while features remain.
    int start = 0;
    do {
        for (int q = 0; q < num_qubits; ++q) {
            const int fi = start + q;
            const double theta =
                (fi < n_features) ? kPi * std::tanh(features[fi]) : 0.0;
            const double c = std::cos(theta * 0.5), sn = std::sin(theta * 0.5);
            apply_1q(amp.data(), num_qubits, q, c, 0, -sn, 0, sn, 0, c, 0);
        }
        if (num_qubits > 1) {
            for (int q = 0; q < num_qubits; ++q)
                apply_ctrl(amp.data(), num_qubits, q, (q + 1) % num_qubits,
                           0, 0, 1, 0, 1, 0, 0, 0);  // ring CNOT
        }
        start += num_qubits;
    } while (start < n_features);
    uq_expectations_z(amp.data(), num_qubits, out);
}

UQ_EXPORT void uq_feature_map_batch(
    const double* features /* n_samples*n_features */, int n_samples,
    int n_features, int num_qubits, double* out /* n_samples*num_qubits */,
    int n_threads) {
    if (n_samples <= 0) return;
    const int nt = resolve_threads(n_threads, n_samples);
    auto worker = [features, n_features, num_qubits, out](int begin, int end) {
        for (int s = begin; s < end; ++s)
            uq_feature_map(features + static_cast<long long>(s) * n_features,
                           n_features, num_qubits,
                           out + static_cast<long long>(s) * num_qubits);
    };
    if (nt <= 1) {
        worker(0, n_samples);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int begin =
            static_cast<int>(static_cast<long long>(n_samples) * t / nt);
        const int end =
            static_cast<int>(static_cast<long long>(n_samples) * (t + 1) / nt);
        pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
}

UQ_EXPORT void uq_ternary_forward(const signed char* qw /* out_dim*in_dim */,
                                  double alpha, const double* bias,
                                  const double* x, int in_dim, int out_dim,
                                  double* out) {
    for (int o = 0; o < out_dim; ++o) {
        const signed char* row = qw + static_cast<long long>(o) * in_dim;
        double acc = 0.0;
        for (int i = 0; i < in_dim; ++i)
            acc += static_cast<double>(row[i]) * x[i];
        out[o] = alpha * acc + (bias ? bias[o] : 0.0);
    }
}

UQ_EXPORT void uq_ternary_forward_batch(const signed char* qw, double alpha,
                                        const double* bias, const double* xs,
                                        int n_samples, int in_dim, int out_dim,
                                        double* out, int n_threads) {
    if (n_samples <= 0) return;
    const int nt = resolve_threads(n_threads, n_samples);
    auto worker = [qw, alpha, bias, xs, in_dim, out_dim, out](int begin,
                                                             int end) {
        for (int s = begin; s < end; ++s)
            uq_ternary_forward(qw, alpha, bias,
                               xs + static_cast<long long>(s) * in_dim, in_dim,
                               out_dim,
                               out + static_cast<long long>(s) * out_dim);
    };
    if (nt <= 1) {
        worker(0, n_samples);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) {
        const int begin =
            static_cast<int>(static_cast<long long>(n_samples) * t / nt);
        const int end =
            static_cast<int>(static_cast<long long>(n_samples) * (t + 1) / nt);
        pool.emplace_back(worker, begin, end);
    }
    for (auto& th : pool) th.join();
}
