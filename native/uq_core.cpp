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
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#define UQ_EXPORT extern "C" __declspec(dllexport)

// Keep ggml's separate float32 operations, including the multiply/subtract.
#pragma float_control(precise, on, push)
#pragma fp_contract(off)

namespace {

int kq_block_bytes(int type) {
    return type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : 0;
}

float kq_half(const uint8_t* p) {
    const uint32_t h = p[0] | (uint32_t(p[1]) << 8);
    const uint32_t exponent = (h >> 10) & 31, mantissa = h & 1023;
    if (exponent == 0)
        return std::ldexp(float(mantissa), -24) * ((h & 32768) ? -1.0f : 1.0f);
    const uint32_t bits = ((h & 32768) << 16) | (mantissa << 13)
        | (exponent == 31 ? 0x7f800000u : (exponent + 112) << 23);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void kq_decode(int type, const uint8_t* w, float* out) {
    if (type == 14) {
        const float d = kq_half(w + 208);
        for (int half = 0; half < 2; ++half) {
            for (int group = 0; group < 4; ++group) {
                const uint8_t* low = w + half * 64 + (group % 2) * 32;
                const uint8_t* high = w + 128 + half * 32;
                for (int lane = 0; lane < 32; ++lane) {
                    const int q = ((low[lane] >> (4 * (group / 2))) & 15)
                        | (((high[lane] >> (2 * group)) & 3) << 4);
                    const int s = w[192 + half * 8 + group * 2 + lane / 16];
                    const float ds = d * float(s < 128 ? s : s - 256);
                    out[half * 128 + group * 32 + lane] = ds * float(q - 32);
                }
            }
        }
        return;
    }
    const float d = kq_half(w), dmin = kq_half(w + 2);
    const uint8_t* scales = w + 4;
    const uint8_t* low = w + (type == 13 ? 48 : 16);
    for (int group = 0; group < 8; ++group) {
        const int scale = group < 4 ? scales[group] & 63
            : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        const int minimum = group < 4 ? scales[group + 4] & 63
            : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        const float ds = d * float(scale), dm = dmin * float(minimum);
        for (int lane = 0; lane < 32; ++lane) {
            int q = (low[(group / 2) * 32 + lane] >> (4 * (group % 2))) & 15;
            if (type == 13) q |= ((w[16 + lane] >> group) & 1) << 4;
            const float product = ds * float(q);
            out[group * 32 + lane] = product - dm;
        }
    }
}

}  // namespace

UQ_EXPORT int uq_kq_dequant(int type, const uint8_t* w, int64_t n, float* out) {
    const int bytes = kq_block_bytes(type);
    if (!bytes || n < 0 || n % 256 || n > INT64_MAX / int64_t(sizeof(float))
        || (n && (!w || !out))) return 1;
    for (int64_t block = 0; block < n / 256; ++block)
        kq_decode(type, w + block * bytes, out + block * 256);
    return 0;
}

UQ_EXPORT int uq_kq_matvec(int type, const uint8_t* w, int64_t rows,
                          int64_t cols, const double* x, double* y, int threads) {
    const int bytes = kq_block_bytes(type);
    if (!bytes || rows < 0 || cols <= 0 || cols % 256
        || cols > INT64_MAX / int64_t(sizeof(double))
        || rows > INT64_MAX / int64_t(sizeof(double))) return 1;
    const int64_t row_bytes = (cols / 256) * bytes;
    if (rows > INT64_MAX / row_bytes || (rows && (!w || !x || !y))) return 1;
    if (!rows) return 0;
    if (threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        threads = hc ? int(hc) : 1;
    }
    if (threads > rows) threads = int(rows);
    auto worker = [=](int64_t begin, int64_t end) {
        float decoded[256];
        for (int64_t row = begin; row < end; ++row) {
            double sum = 0.0;
            const uint8_t* packed = w + row * row_bytes;
            for (int64_t block = 0; block < cols / 256; ++block) {
                kq_decode(type, packed + block * bytes, decoded);
                for (int lane = 0; lane < 256; ++lane)
                    sum += double(decoded[lane]) * x[block * 256 + lane];
            }
            y[row] = sum;
        }
    };
    if (threads == 1) {
        worker(0, rows);
        return 0;
    }
    std::vector<std::thread> pool;
    try {
        pool.reserve(threads);
        int64_t begin = 0;
        for (int t = 0; t < threads; ++t) {
            const int64_t end = begin + rows / threads + (t < rows % threads);
            pool.emplace_back(worker, begin, end);
            begin = end;
        }
    } catch (...) {
        for (auto& th : pool) th.join();
        return 2;  // Never unwind a C++ exception across ctypes.
    }
    for (auto& th : pool) th.join();
    return 0;
}

// Prefill: decode each packed row once, then use it for every input vector.
// Each dot still adds columns in exactly uq_kq_matvec's order. Parallelism
// is over output rows, never over the reduction, so batching cannot change
// a result through reassociation or a different partial-sum tree.
UQ_EXPORT int uq_kq_matvec_batch(int type, const uint8_t* w, int64_t rows,
                                int64_t cols, const double* x, int64_t batch,
                                double* y, int threads) {
    const int bytes = kq_block_bytes(type);
    const int64_t max_doubles = INT64_MAX / int64_t(sizeof(double));
    if (!bytes || rows < 0 || cols <= 0 || cols % 256 || batch < 0
        || cols > max_doubles || rows > max_doubles
        || batch > max_doubles / cols
        || (rows && batch > max_doubles / rows)) return 1;
    const int64_t row_bytes = (cols / 256) * bytes;
    if (rows > INT64_MAX / row_bytes) return 1;
    if (!rows || !batch) return 0;
    if (!w || !x || !y) return 1;
    if (threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        threads = hc ? int(hc) : 1;
    }
    if (threads > rows) threads = int(rows);
    std::vector<std::thread> pool;
    std::vector<std::vector<float>> scratch;
    try {
        // Allocate before starting workers: allocation failure must return
        // through the C ABI, not escape a worker and terminate the process.
        scratch.resize(threads);
        for (auto& row : scratch) row.resize(size_t(cols));
        auto worker = [&](int t, int64_t begin, int64_t end) {
            float* decoded = scratch[t].data();
            for (int64_t row = begin; row < end; ++row) {
                const uint8_t* packed = w + row * row_bytes;
                for (int64_t block = 0; block < cols / 256; ++block)
                    kq_decode(type, packed + block * bytes, decoded + block * 256);
                for (int64_t item = 0; item < batch; ++item) {
                    const double* vector = x + item * cols;
                    double sum = 0.0;
                    for (int64_t col = 0; col < cols; ++col)
                        sum += double(decoded[col]) * vector[col];
                    y[item * rows + row] = sum;
                }
            }
        };
        if (threads == 1) {
            worker(0, 0, rows);
        } else {
            pool.reserve(threads);
            int64_t begin = 0;
            for (int t = 0; t < threads; ++t) {
                const int64_t end = begin + rows / threads + (t < rows % threads);
                pool.emplace_back(worker, t, begin, end);
                begin = end;
            }
        }
    } catch (...) {
        for (auto& th : pool) th.join();
        return 2;
    }
    for (auto& th : pool) th.join();
    return 0;
}

#pragma float_control(pop)

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
