// UltraQuant CUDA acceleration tier -- native/uq_cuda.cu -> ultraquant_cuda.dll
//
// Implements SPEC-NATIVE.md section 2.  C ABI (extern "C" + dllexport), double
// precision everywhere.  Qubit indexing is little-endian exactly as SPEC.md
// section 2: qubit q is bit q (LSB = qubit 0) of the basis-state index.
// Statevectors cross the FFI as interleaved re/im: double[2 * 2**n].
//
// Circuit wire format (SPEC-NATIVE.md): three parallel arrays
//   int32 opcodes[N], int32 qubits[2*N] (q0, q1; unused slot = -1),
//   double params[N] (unused slot = 0.0)
// Opcodes: 0=h 1=x 2=y 3=z 4=s 5=t 6=rx 7=ry 8=rz 9=cnot 10=cz 11=swap
//
// Build: powershell -ExecutionPolicy Bypass -File native\build.ps1 -Target cuda

#include <cuda_runtime.h>
#include <math.h>

#define UQ_API extern "C" __declspec(dllexport)

#define UQ_PI 3.14159265358979323846
static const int UQ_THREADS = 256; // power of two: block reduction relies on it
static const unsigned long long UQ_MAX_BLOCKS = 8192ULL;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Apply a 2x2 complex matrix to `target`, optionally conditioned on `control`
// (control < 0 means no control).  Grid-stride over the 2**(n-1) amplitude
// pairs that differ only in bit `target`.
__global__ void k_apply_gate(double2* amp, unsigned long long half, int target,
                             int control,
                             double m00r, double m00i, double m01r, double m01i,
                             double m10r, double m10i, double m11r, double m11i)
{
    const unsigned long long tmask = 1ULL << target;
    const unsigned long long low = tmask - 1ULL;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long p =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < half; p += stride) {
        // Insert a 0 bit at position `target` into p -> index with target bit 0.
        const unsigned long long i0 = ((p & ~low) << 1) | (p & low);
        if (control >= 0 && (((i0 >> control) & 1ULL) == 0ULL)) continue;
        const unsigned long long i1 = i0 | tmask;
        const double2 a0 = amp[i0];
        const double2 a1 = amp[i1];
        amp[i0] = make_double2(m00r * a0.x - m00i * a0.y + m01r * a1.x - m01i * a1.y,
                               m00r * a0.y + m00i * a0.x + m01r * a1.y + m01i * a1.x);
        amp[i1] = make_double2(m10r * a0.x - m10i * a0.y + m11r * a1.x - m11i * a1.y,
                               m10r * a0.y + m10i * a0.x + m11r * a1.y + m11i * a1.x);
    }
}

// SWAP(qa, qb): exchange amplitudes of indices with (bit qa, bit qb) = (1,0)
// and (0,1).  Host guarantees qa < qb.  Grid-stride over 2**(n-2) index stubs.
__global__ void k_apply_swap(double2* amp, unsigned long long quarter,
                             int qa, int qb)
{
    const unsigned long long lowa = (1ULL << qa) - 1ULL;
    const unsigned long long lowb = (1ULL << qb) - 1ULL;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * (unsigned long long)blockDim.x;
    for (unsigned long long p =
             (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < quarter; p += stride) {
        const unsigned long long t = ((p & ~lowa) << 1) | (p & lowa); // 0 at qa
        const unsigned long long base = ((t & ~lowb) << 1) | (t & lowb); // 0 at qb
        const unsigned long long i01 = base | (1ULL << qa);
        const unsigned long long i10 = base | (1ULL << qb);
        const double2 tmp = amp[i01];
        amp[i01] = amp[i10];
        amp[i10] = tmp;
    }
}

// One thread block per sample.  The per-sample statevector (2**num_qubits
// complex doubles, num_qubits <= 10) lives in dynamic shared memory; the
// threads of the block cooperate on every gate.  Implements the SPEC.md
// section 5 feature map exactly: angles theta_i = pi * tanh(f_i); RY layer
// over a chunk of num_qubits features (missing features -> angle 0); ring of
// CNOTs cnot(q, (q+1) % num_qubits) (skipped for 1 qubit); data re-uploading
// chunk by chunk until the features are exhausted; then per-qubit <Z>.
__global__ void k_feature_map(const double* features, int n_features,
                              int num_qubits, double* out)
{
    const int dim = 1 << num_qubits;
    const int half = dim >> 1;
    extern __shared__ double smem[];
    double2* sv = reinterpret_cast<double2*>(smem); // 2*dim doubles
    double* red = smem + 2 * dim;                   // blockDim.x doubles
    const double* f = features + (size_t)blockIdx.x * (size_t)n_features;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sv[i] = make_double2(0.0, 0.0);
    __syncthreads();
    if (threadIdx.x == 0) sv[0] = make_double2(1.0, 0.0);
    __syncthreads();

    int layers = (n_features + num_qubits - 1) / num_qubits;
    if (layers < 1) layers = 1;

    for (int l = 0; l < layers; ++l) {
        // RY layer over the l-th chunk of features.
        for (int q = 0; q < num_qubits; ++q) {
            const int fi = l * num_qubits + q;
            const double theta = (fi < n_features) ? UQ_PI * tanh(f[fi]) : 0.0;
            const double c = cos(0.5 * theta);
            const double s = sin(0.5 * theta);
            const int tmask = 1 << q;
            const int low = tmask - 1;
            for (int p = threadIdx.x; p < half; p += blockDim.x) {
                const int i0 = ((p & ~low) << 1) | (p & low);
                const int i1 = i0 | tmask;
                const double2 a0 = sv[i0];
                const double2 a1 = sv[i1];
                sv[i0] = make_double2(c * a0.x - s * a1.x, c * a0.y - s * a1.y);
                sv[i1] = make_double2(s * a0.x + c * a1.x, s * a0.y + c * a1.y);
            }
            __syncthreads();
        }
        // Ring entangler (skip for a single qubit).
        if (num_qubits > 1) {
            for (int q = 0; q < num_qubits; ++q) {
                const int ctrl = q;
                const int tgt = (q + 1) % num_qubits;
                const int tmask = 1 << tgt;
                const int low = tmask - 1;
                for (int p = threadIdx.x; p < half; p += blockDim.x) {
                    const int i0 = ((p & ~low) << 1) | (p & low);
                    if (((i0 >> ctrl) & 1) == 0) continue;
                    const int i1 = i0 | tmask;
                    const double2 tmp = sv[i0];
                    sv[i0] = sv[i1];
                    sv[i1] = tmp;
                }
                __syncthreads();
            }
        }
    }

    // Per-qubit <Z> = sum_i p(i) * (+1 if bit q of i is 0 else -1).
    for (int q = 0; q < num_qubits; ++q) {
        double part = 0.0;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            const double pr = sv[i].x * sv[i].x + sv[i].y * sv[i].y;
            part += ((i >> q) & 1) ? -pr : pr;
        }
        red[threadIdx.x] = part;
        __syncthreads();
        for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
            if ((int)threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            out[(size_t)blockIdx.x * (size_t)num_qubits + q] = red[0];
        __syncthreads();
    }
}

// out[s][o] = alpha * sum_i qw[o][i] * xs[s][i] + bias[o]
// (bias optional: has_bias == 0 means no bias term).
__global__ void k_ternary_forward(const signed char* qw, double alpha,
                                  const double* bias, int has_bias,
                                  const double* xs, long long n_samples,
                                  int in_dim, int out_dim, double* out)
{
    const long long total = n_samples * (long long)out_dim;
    const long long stride =
        (long long)gridDim.x * (long long)blockDim.x;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += stride) {
        const long long s = t / out_dim;
        const int o = (int)(t - s * out_dim);
        const double* x = xs + s * (long long)in_dim;
        const signed char* w = qw + (long long)o * (long long)in_dim;
        double acc = 0.0;
        for (int i = 0; i < in_dim; ++i)
            acc += (double)w[i] * x[i];
        out[t] = alpha * acc + (has_bias ? bias[o] : 0.0);
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static bool uq_ok(cudaError_t e)
{
    if (e != cudaSuccess) {
        cudaGetLastError(); // clear sticky error state
        return false;
    }
    return true;
}

// SS11.176: every compute export reports what happened. 0 is success, a
// positive value is the cudaError_t (sticky state cleared), and the
// negatives below are argument rejections decided before the device is
// touched. On any nonzero status the caller's output buffer is untouched,
// so a launch that never ran cannot be read as a result.
#define UQ_ABI_VERSION   2
#define UQ_ERR_ARGUMENT  (-1)
#define UQ_ERR_OPERAND   (-2)
#define UQ_ERR_SLOT      (-3)

static int uq_status(cudaError_t e)
{
    if (e == cudaSuccess) return 0;
    cudaGetLastError(); // clear sticky error state
    return (int)e;
}

static int uq_blocks_for(unsigned long long work)
{
    unsigned long long b = (work + UQ_THREADS - 1ULL) / (unsigned long long)UQ_THREADS;
    if (b < 1ULL) b = 1ULL;
    if (b > UQ_MAX_BLOCKS) b = UQ_MAX_BLOCKS;
    return (int)b;
}

// Fill m = [m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i] for a single-qubit
// opcode (9=cnot uses X, 10=cz uses Z; the control is handled by the kernel).
static void uq_fill_matrix(int opcode, double theta, double m[8])
{
    for (int i = 0; i < 8; ++i) m[i] = 0.0;
    switch (opcode) {
    case 0: { // h
        const double r = 1.0 / sqrt(2.0);
        m[0] = r; m[2] = r; m[4] = r; m[6] = -r;
    } break;
    case 1: case 9: // x / cnot
        m[2] = 1.0; m[4] = 1.0;
        break;
    case 2: // y = [[0, -i], [i, 0]]
        m[3] = -1.0; m[5] = 1.0;
        break;
    case 3: case 10: // z / cz
        m[0] = 1.0; m[6] = -1.0;
        break;
    case 4: // s = diag(1, i)
        m[0] = 1.0; m[7] = 1.0;
        break;
    case 5: // t = diag(1, exp(i*pi/4))
        m[0] = 1.0; m[6] = cos(UQ_PI / 4.0); m[7] = sin(UQ_PI / 4.0);
        break;
    case 6: { // rx
        const double c = cos(0.5 * theta), s = sin(0.5 * theta);
        m[0] = c; m[3] = -s; m[5] = -s; m[6] = c;
    } break;
    case 7: { // ry
        const double c = cos(0.5 * theta), s = sin(0.5 * theta);
        m[0] = c; m[2] = -s; m[4] = s; m[6] = c;
    } break;
    case 8: { // rz = diag(exp(-i*t/2), exp(i*t/2))
        const double c = cos(0.5 * theta), s = sin(0.5 * theta);
        m[0] = c; m[1] = -s; m[6] = c; m[7] = s;
    } break;
    default: // unknown -> identity (no-op)
        m[0] = 1.0; m[6] = 1.0;
        break;
    }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

UQ_API int uqg_available(void)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (n > 0) ? n : 0;
}

UQ_API int uqg_device_name(char* buf, int buflen)
{
    if (buf == nullptr || buflen <= 0) return 1;
    cudaDeviceProp prop;
    if (!uq_ok(cudaGetDeviceProperties(&prop, 0))) return 2;
    int i = 0;
    for (; i < buflen - 1 && prop.name[i] != '\0'; ++i) buf[i] = prop.name[i];
    buf[i] = '\0';
    return 0;
}

UQ_API int uqg_abi_version(void)
{
    return UQ_ABI_VERSION;
}

UQ_API const char* uqg_status_string(int status)
{
    if (status == 0) return "ok";
    if (status == UQ_ERR_ARGUMENT) return "argument rejected";
    if (status == UQ_ERR_OPERAND) return "gate operand outside the register";
    if (status == UQ_ERR_SLOT) return "resident slot not in use";
    if (status < 0) return "unknown status";
    return cudaGetErrorString((cudaError_t)status);
}

UQ_API int uqg_run_circuit(double* amp_host, int num_qubits,
                           const int* opcodes, const int* qubits,
                           const double* params, int n_ops)
{
    if (amp_host == nullptr || num_qubits < 1 || num_qubits > 30) return UQ_ERR_ARGUMENT;
    if (n_ops < 0) return UQ_ERR_ARGUMENT;
    if (n_ops > 0 && (opcodes == nullptr || qubits == nullptr)) return UQ_ERR_ARGUMENT;

    // Every operand is checked before the device is touched, so a rejected
    // circuit leaves the host buffer exactly as it was. The old code skipped
    // a bad gate and carried on, which is a wrong state rather than an error.
    for (int k = 0; k < n_ops; ++k) {
        const int oc = opcodes[k];
        const int q0 = qubits[2 * k];
        const int q1 = qubits[2 * k + 1];
        if (oc < 0 || oc > 11) return UQ_ERR_OPERAND;
        if (oc == 11) { // swap
            if (q0 < 0 || q1 < 0 || q0 >= num_qubits || q1 >= num_qubits ||
                q0 == q1 || num_qubits < 2)
                return UQ_ERR_OPERAND;
        } else if (oc == 9 || oc == 10) { // cnot / cz: (control, target)
            if (q0 < 0 || q0 >= num_qubits || q1 < 0 || q1 >= num_qubits || q0 == q1)
                return UQ_ERR_OPERAND;
        } else if (q0 < 0 || q0 >= num_qubits) {
            return UQ_ERR_OPERAND;
        }
    }

    const unsigned long long dim = 1ULL << num_qubits;
    const size_t bytes = (size_t)dim * sizeof(double2);
    double2* d_amp = nullptr;
    int st = uq_status(cudaMalloc(&d_amp, bytes));
    if (st != 0) return st;
    st = uq_status(cudaMemcpy(d_amp, amp_host, bytes, cudaMemcpyHostToDevice));

    const unsigned long long half = dim >> 1;
    const int blocks = uq_blocks_for(half);

    for (int k = 0; st == 0 && k < n_ops; ++k) {
        const int oc = opcodes[k];
        int q0 = qubits[2 * k];
        int q1 = qubits[2 * k + 1];
        const double theta = (params != nullptr) ? params[k] : 0.0;

        if (oc == 11) { // swap
            if (q0 > q1) { const int t = q0; q0 = q1; q1 = t; }
            const unsigned long long quarter = dim >> 2;
            k_apply_swap<<<uq_blocks_for(quarter), UQ_THREADS>>>(
                d_amp, quarter, q0, q1);
            continue;
        }

        int control = -1;
        int target = q0;
        if (oc == 9 || oc == 10) { // cnot / cz: (control, target) = (q0, q1)
            control = q0;
            target = q1;
        }

        double m[8];
        uq_fill_matrix(oc, theta, m);
        k_apply_gate<<<blocks, UQ_THREADS>>>(d_amp, half, target, control,
                                             m[0], m[1], m[2], m[3],
                                             m[4], m[5], m[6], m[7]);
    }

    if (st == 0) st = uq_status(cudaDeviceSynchronize());
    if (st == 0) st = uq_status(cudaGetLastError());
    if (st == 0) st = uq_status(cudaMemcpy(amp_host, d_amp, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_amp);
    return st;
}

UQ_API int uqg_expectations_z_from(const double* amp_host, int num_qubits,
                                   double* out)
{
    // Host-side on purpose: exactness matters here, speed does not.
    if (amp_host == nullptr || out == nullptr) return UQ_ERR_ARGUMENT;
    if (num_qubits < 1 || num_qubits > 30) return UQ_ERR_ARGUMENT;
    const unsigned long long dim = 1ULL << num_qubits;
    for (int q = 0; q < num_qubits; ++q) out[q] = 0.0;
    for (unsigned long long i = 0; i < dim; ++i) {
        const double re = amp_host[2 * i];
        const double im = amp_host[2 * i + 1];
        const double p = re * re + im * im;
        for (int q = 0; q < num_qubits; ++q)
            out[q] += ((i >> q) & 1ULL) ? -p : p;
    }
    return 0;
}

UQ_API int uqg_feature_map_batch(const double* features, int n_samples,
                                 int n_features, int num_qubits, double* out)
{
    if (out == nullptr || n_samples <= 0 || n_features < 0) return UQ_ERR_ARGUMENT;
    if (num_qubits < 1 || num_qubits > 10) return UQ_ERR_ARGUMENT;
    if (n_features > 0 && features == nullptr) return UQ_ERR_ARGUMENT;

    const size_t feat_count = (size_t)n_samples * (size_t)n_features;
    const size_t out_count = (size_t)n_samples * (size_t)num_qubits;
    double* d_feat = nullptr;
    double* d_out = nullptr;
    int st = 0;

    if (feat_count > 0) {
        st = uq_status(cudaMalloc(&d_feat, feat_count * sizeof(double)));
        if (st == 0)
            st = uq_status(cudaMemcpy(d_feat, features, feat_count * sizeof(double),
                                      cudaMemcpyHostToDevice));
    }
    if (st == 0) st = uq_status(cudaMalloc(&d_out, out_count * sizeof(double)));

    if (st == 0) {
        const int dim = 1 << num_qubits;
        const size_t shmem =
            (size_t)dim * sizeof(double2) + (size_t)UQ_THREADS * sizeof(double);
        k_feature_map<<<(unsigned)n_samples, UQ_THREADS, shmem>>>(
            d_feat, n_features, num_qubits, d_out);
        st = uq_status(cudaDeviceSynchronize());
        if (st == 0) st = uq_status(cudaGetLastError());
    }
    if (st == 0)
        st = uq_status(cudaMemcpy(out, d_out, out_count * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    if (d_feat != nullptr) cudaFree(d_feat);
    if (d_out != nullptr) cudaFree(d_out);
    return st;
}

// ---------------------------------------------------------------------------
// VRAM-resident expert layers (SS11.45): upload once, forward many.
//
// uqg_ternary_forward_batch pays four mallocs and three host<->device
// copies per call; for a HOT expert the weights never change between
// queries, so residency moves everything but the activations off the
// per-call path. A fixed slot table, no hidden allocation policy: the
// Python cache decides what stays resident and evicts by calling free.
// ---------------------------------------------------------------------------

#define UQ_MAX_RESIDENT 64

typedef struct {
    signed char* d_qw;
    double* d_bias;
    int in_dim;
    int out_dim;
    int has_bias;
    int in_use;
    double alpha;
} uq_resident_layer;

static uq_resident_layer uq_residents[UQ_MAX_RESIDENT];

UQ_API int uqg_layer_upload(const signed char* qw, double alpha,
                            const double* bias, int in_dim, int out_dim)
{
    if (qw == nullptr || in_dim <= 0 || out_dim <= 0) return -1;
    int slot = -1;
    for (int i = 0; i < UQ_MAX_RESIDENT; ++i)
        if (!uq_residents[i].in_use) { slot = i; break; }
    if (slot < 0) return -1;

    uq_resident_layer* r = &uq_residents[slot];
    const size_t w_count = (size_t)out_dim * (size_t)in_dim;
    bool ok = uq_ok(cudaMalloc(&r->d_qw, w_count));
    ok = ok && uq_ok(cudaMemcpy(r->d_qw, qw, w_count,
                                cudaMemcpyHostToDevice));
    r->has_bias = (bias != nullptr) ? 1 : 0;
    r->d_bias = nullptr;
    if (ok && r->has_bias) {
        ok = uq_ok(cudaMalloc(&r->d_bias,
                              (size_t)out_dim * sizeof(double)));
        ok = ok && uq_ok(cudaMemcpy(r->d_bias, bias,
                                    (size_t)out_dim * sizeof(double),
                                    cudaMemcpyHostToDevice));
    }
    if (!ok) {
        if (r->d_qw != nullptr) cudaFree(r->d_qw);
        if (r->d_bias != nullptr) cudaFree(r->d_bias);
        r->d_qw = nullptr; r->d_bias = nullptr;
        return -1;
    }
    r->in_dim = in_dim;
    r->out_dim = out_dim;
    r->alpha = alpha;
    r->in_use = 1;
    return slot;
}

UQ_API int uqg_layer_forward(int slot, const double* xs, int n_samples,
                             double* out)
{
    if (slot < 0 || slot >= UQ_MAX_RESIDENT) return UQ_ERR_SLOT;
    if (xs == nullptr || out == nullptr || n_samples <= 0) return UQ_ERR_ARGUMENT;
    uq_resident_layer* r = &uq_residents[slot];
    if (!r->in_use) return UQ_ERR_SLOT;

    const size_t x_count = (size_t)n_samples * (size_t)r->in_dim;
    const size_t o_count = (size_t)n_samples * (size_t)r->out_dim;
    double* d_xs = nullptr;
    double* d_out = nullptr;
    int st = uq_status(cudaMalloc(&d_xs, x_count * sizeof(double)));
    if (st == 0) st = uq_status(cudaMalloc(&d_out, o_count * sizeof(double)));
    if (st == 0) st = uq_status(cudaMemcpy(d_xs, xs, x_count * sizeof(double),
                                           cudaMemcpyHostToDevice));
    if (st == 0) {
        k_ternary_forward<<<uq_blocks_for(o_count), UQ_THREADS>>>(
            r->d_qw, r->alpha, r->d_bias, r->has_bias, d_xs,
            (long long)n_samples, r->in_dim, r->out_dim, d_out);
        st = uq_status(cudaDeviceSynchronize());
        if (st == 0) st = uq_status(cudaGetLastError());
    }
    if (st == 0)
        st = uq_status(cudaMemcpy(out, d_out, o_count * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    if (d_xs != nullptr) cudaFree(d_xs);
    if (d_out != nullptr) cudaFree(d_out);
    return st;
}

UQ_API void uqg_layer_free(int slot)
{
    if (slot < 0 || slot >= UQ_MAX_RESIDENT) return;
    uq_resident_layer* r = &uq_residents[slot];
    if (!r->in_use) return;
    if (r->d_qw != nullptr) cudaFree(r->d_qw);
    if (r->d_bias != nullptr) cudaFree(r->d_bias);
    r->d_qw = nullptr;
    r->d_bias = nullptr;
    r->in_use = 0;
}

UQ_API long long uqg_layer_resident_bytes(void)
{
    long long total = 0;
    for (int i = 0; i < UQ_MAX_RESIDENT; ++i) {
        if (!uq_residents[i].in_use) continue;
        total += (long long)uq_residents[i].out_dim
                 * (long long)uq_residents[i].in_dim;
        if (uq_residents[i].has_bias)
            total += (long long)uq_residents[i].out_dim * 8LL;
    }
    return total;
}

UQ_API int uqg_ternary_forward_batch(const signed char* qw, double alpha,
                                     const double* bias, const double* xs,
                                     int n_samples, int in_dim, int out_dim,
                                     double* out)
{
    if (qw == nullptr || xs == nullptr || out == nullptr) return UQ_ERR_ARGUMENT;
    if (n_samples <= 0 || in_dim <= 0 || out_dim <= 0) return UQ_ERR_ARGUMENT;

    const size_t w_count = (size_t)out_dim * (size_t)in_dim;
    const size_t x_count = (size_t)n_samples * (size_t)in_dim;
    const size_t o_count = (size_t)n_samples * (size_t)out_dim;
    signed char* d_qw = nullptr;
    double* d_bias = nullptr;
    double* d_xs = nullptr;
    double* d_out = nullptr;
    const int has_bias = (bias != nullptr) ? 1 : 0;
    int st = 0;

    st = uq_status(cudaMalloc(&d_qw, w_count));
    if (st == 0 && has_bias)
        st = uq_status(cudaMalloc(&d_bias, (size_t)out_dim * sizeof(double)));
    if (st == 0) st = uq_status(cudaMalloc(&d_xs, x_count * sizeof(double)));
    if (st == 0) st = uq_status(cudaMalloc(&d_out, o_count * sizeof(double)));

    if (st == 0) st = uq_status(cudaMemcpy(d_qw, qw, w_count, cudaMemcpyHostToDevice));
    if (st == 0 && has_bias)
        st = uq_status(cudaMemcpy(d_bias, bias, (size_t)out_dim * sizeof(double),
                                  cudaMemcpyHostToDevice));
    if (st == 0)
        st = uq_status(cudaMemcpy(d_xs, xs, x_count * sizeof(double),
                                  cudaMemcpyHostToDevice));

    if (st == 0) {
        k_ternary_forward<<<uq_blocks_for(o_count), UQ_THREADS>>>(
            d_qw, alpha, d_bias, has_bias, d_xs, (long long)n_samples,
            in_dim, out_dim, d_out);
        st = uq_status(cudaDeviceSynchronize());
        if (st == 0) st = uq_status(cudaGetLastError());
    }
    if (st == 0)
        st = uq_status(cudaMemcpy(out, d_out, o_count * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    if (d_qw != nullptr) cudaFree(d_qw);
    if (d_bias != nullptr) cudaFree(d_bias);
    if (d_xs != nullptr) cudaFree(d_xs);
    if (d_out != nullptr) cudaFree(d_out);
    return st;
}
