// UltraQuant forge: multi-expert ternary training on the GPU.
//
// One CUDA block trains one expert start to finish.  A library of many small
// pattern experts is hundreds of independent training problems, which is the
// shape a GPU suits: hundreds of blocks run concurrently while a CPU works
// through them a few cores at a time.
//
// The inner loop is *batch-parallel*: a whole minibatch is forwarded at once
// (threads spread over sample x unit), and each weight gradient is computed as a
// single reduction over the batch, so a thread owns specific weight elements for
// the whole minibatch.  That gives two things at once -- far more parallelism,
// and one set of barriers per minibatch instead of per sample.  It also lets the
// weight update be fused into the gradient computation, so no gradient
// accumulators are stored at all.  Because each element's batch sum is
// accumulated by a single thread in a fixed order, results are deterministic
// (no atomics), and they agree with the CPU/Python tiers to floating-point noise.
//
// Layout: every array is expert-major and tightly packed.
//   xs   [n_experts][n_samples][in_dim]      ys [n_experts][n_samples]
//   w1   [n_experts][hidden][in_dim]         b1 [n_experts][hidden]
//   w2   [n_experts][n_classes][hidden]      b2 [n_experts][n_classes]
//   order[n_experts][epochs][n_samples]      precomputed on the host so every
//                                            tier sees identical minibatches

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#if defined(_WIN32)
#define UQ_API extern "C" __declspec(dllexport)
#else
#define UQ_API extern "C"
#endif

#define UQ_EPS 1e-12
#define UQ_THREADS 256
#define UQ_MAX_BATCH 256

namespace {

// Block-wide sum of `val`, broadcast to every thread.  Warp shuffles handle the
// first five levels without barriers, so this costs 3 __syncthreads instead of
// the log2(blockDim)+2 a pure shared-memory tree would need -- and with a
// barrier every few hundred flops, that difference dominates the kernel.
__device__ double block_sum(double val, double* scratch) {
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) scratch[warp] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nwarps = blockDim.x >> 5;
        double v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) scratch[0] = v;
    }
    __syncthreads();
    const double total = scratch[0];
    __syncthreads();
    return total;
}

// Two independent block-wide sums for the price of one barrier sequence.
__device__ void block_sum2(double a, double b, double* scratch,
                           double* out_a, double* out_b) {
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, off);
        b += __shfl_down_sync(0xffffffffu, b, off);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (lane == 0) { scratch[warp] = a; scratch[warp + nwarps] = b; }
    __syncthreads();
    if (threadIdx.x < 32) {
        double va = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0;
        double vb = (threadIdx.x < nwarps) ? scratch[threadIdx.x + nwarps] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            va += __shfl_down_sync(0xffffffffu, va, off);
            vb += __shfl_down_sync(0xffffffffu, vb, off);
        }
        if (threadIdx.x == 0) { scratch[0] = va; scratch[1] = vb; }
    }
    __syncthreads();
    *out_a = scratch[0];
    *out_b = scratch[1];
    __syncthreads();
}

// Ternarize w[0..n) into eff (= alpha * q), matching quantize_ternary():
// delta = 0.75 * mean|w|, alpha = mean|w| over the entries that survive.
__device__ void ternarize(const double* __restrict__ w, int n, double* eff,
                          double* scratch) {
    double local = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += fabs(w[i]);
    const double delta = 0.75 * (block_sum(local, scratch) / (double)n);

    if (delta == 0.0) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) eff[i] = 0.0;
        __syncthreads();
        return;
    }

    double surv_sum = 0.0;
    double surv_n = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double a = fabs(w[i]);
        if (a > delta) { surv_sum += a; surv_n += 1.0; }
    }
    double total = 0.0, count = 0.0;
    block_sum2(surv_sum, surv_n, scratch, &total, &count);
    const double alpha = (count > 0.0) ? (total / count) : 1.0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = w[i];
        eff[i] = (fabs(v) > delta) ? (v > 0.0 ? alpha : -alpha) : 0.0;
    }
    __syncthreads();
}

// fake_quantize_activation; nearbyint is round-half-to-even, like Python's round().
__device__ inline double fake_quant(double v, double step, double clip) {
    const double clamped = fmin(fmax(v, -clip), clip);
    return nearbyint((clamped + clip) / step) * step - clip;
}

__global__ void train_experts_kernel(
    int in_dim, int hidden, int n_classes, int n_samples,
    const double* __restrict__ xs, const int32_t* __restrict__ ys,
    double* w1, double* b1, double* w2, double* b2,
    int epochs, double lr, int batch_size, int act_bits, int quantize_head,
    double clip, const int32_t* __restrict__ order,
    double* out_loss, double* out_acc) {

    const int e = blockIdx.x;
    const int t = threadIdx.x;
    const int nthreads = blockDim.x;
    const int n1 = hidden * in_dim;
    const int n2 = n_classes * hidden;

    const double* my_xs = xs + (size_t)e * n_samples * in_dim;
    const int32_t* my_ys = ys + (size_t)e * n_samples;
    double* my_w1 = w1 + (size_t)e * n1;
    double* my_b1 = b1 + (size_t)e * hidden;
    double* my_w2 = w2 + (size_t)e * n2;
    double* my_b2 = b2 + (size_t)e * n_classes;
    const int32_t* my_order = order + (size_t)e * epochs * n_samples;

    const int bsz = (batch_size > 0) ? batch_size : n_samples;

    extern __shared__ double smem[];
    double* eff1 = smem;                    // n1
    double* eff2 = eff1 + n1;               // n2
    double* act = eff2 + n2;                // bsz * hidden
    double* gz = act + bsz * hidden;        // bsz * hidden  (ReLU mask, then grad)
    double* pr = gz + bsz * hidden;         // bsz * n_classes (logits, then probs/grad)
    double* scratch = pr + bsz * n_classes; // nthreads

    const int levels = (1 << act_bits) - 1;
    const double step = (levels >= 2) ? (2.0 * clip) / (levels - 1) : 0.0;

    double epoch_loss = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const int32_t* ep = my_order + (size_t)epoch * n_samples;
        double loss_sum = 0.0;
        int batches = 0;

        for (int start = 0; start < n_samples; start += bsz) {
            const int bs = min(bsz, n_samples - start);

            // Master weights are constant across a minibatch, so ternarize once
            // per batch rather than once per forward as the reference does.
            ternarize(my_w1, n1, eff1, scratch);
            if (quantize_head) {
                ternarize(my_w2, n2, eff2, scratch);
            } else {
                for (int i = t; i < n2; i += nthreads) eff2[i] = my_w2[i];
                __syncthreads();
            }

            // -- forward: hidden, one thread per (sample, unit)
            for (int idx = t; idx < bs * hidden; idx += nthreads) {
                const int k = idx / hidden, h = idx - k * hidden;
                const double* x = my_xs + (size_t)ep[start + k] * in_dim;
                const double* row = eff1 + (size_t)h * in_dim;
                double a = my_b1[h];
                for (int i = 0; i < in_dim; ++i) a += row[i] * x[i];
                gz[idx] = (a > 0.0) ? 1.0 : 0.0;          // ReLU mask for now
                act[idx] = fake_quant((a > 0.0) ? a : 0.0, step, clip);
            }
            __syncthreads();

            // -- forward: head, one thread per (sample, class)
            for (int idx = t; idx < bs * n_classes; idx += nthreads) {
                const int k = idx / n_classes, o = idx - k * n_classes;
                const double* row = eff2 + (size_t)o * hidden;
                const double* a = act + (size_t)k * hidden;
                double v = my_b2[o];
                for (int h = 0; h < hidden; ++h) v += row[h] * a[h];
                pr[idx] = v;
            }
            __syncthreads();

            // -- softmax, loss and logit gradient, one thread per sample
            double local_loss = 0.0;
            for (int k = t; k < bs; k += nthreads) {
                double* p = pr + (size_t)k * n_classes;
                double m = -1e308;
                for (int o = 0; o < n_classes; ++o) m = fmax(m, p[o]);
                double zs = 0.0;
                for (int o = 0; o < n_classes; ++o) { p[o] = exp(p[o] - m); zs += p[o]; }
                for (int o = 0; o < n_classes; ++o) p[o] /= zs;
                const int y = my_ys[ep[start + k]];
                local_loss += -log(fmax(p[y], UQ_EPS));
                for (int o = 0; o < n_classes; ++o) p[o] -= (o == y) ? 1.0 : 0.0;
            }
            // Only the final epoch's loss is reported, so skip the reduction
            // (and its barriers) on every earlier epoch -- the pr[] writes above
            // still need publishing, hence the plain barrier on that path.
            if (epoch == epochs - 1) {
                loss_sum += block_sum(local_loss, scratch) / (double)bs;
                ++batches;
            } else {
                __syncthreads();
            }

            // -- grad wrt hidden activations, folded with the ReLU mask
            for (int idx = t; idx < bs * hidden; idx += nthreads) {
                const int k = idx / hidden, h = idx - k * hidden;
                const double* g = pr + (size_t)k * n_classes;
                double ga = 0.0;
                for (int o = 0; o < n_classes; ++o) ga += g[o] * eff2[(size_t)o * hidden + h];
                gz[idx] *= ga;
            }
            __syncthreads();

            // -- gradients as batch reductions, fused straight into the update.
            // Each thread owns whole weight elements, so the batch sum happens in
            // a fixed order: no atomics, no accumulator arrays, deterministic.
            const double slr = lr / (double)bs;

            for (int idx = t; idx < n1; idx += nthreads) {
                const int h = idx / in_dim, i = idx - h * in_dim;
                double g = 0.0;
                for (int k = 0; k < bs; ++k)
                    g += gz[(size_t)k * hidden + h] * my_xs[(size_t)ep[start + k] * in_dim + i];
                my_w1[idx] = fmin(fmax(my_w1[idx] - slr * g, -1.0), 1.0);
            }
            for (int idx = t; idx < n2; idx += nthreads) {
                const int o = idx / hidden, h = idx - o * hidden;
                double g = 0.0;
                for (int k = 0; k < bs; ++k)
                    g += pr[(size_t)k * n_classes + o] * act[(size_t)k * hidden + h];
                my_w2[idx] = fmin(fmax(my_w2[idx] - slr * g, -1.0), 1.0);
            }
            for (int h = t; h < hidden; h += nthreads) {
                double g = 0.0;
                for (int k = 0; k < bs; ++k) g += gz[(size_t)k * hidden + h];
                my_b1[h] -= slr * g;
            }
            for (int o = t; o < n_classes; o += nthreads) {
                double g = 0.0;
                for (int k = 0; k < bs; ++k) g += pr[(size_t)k * n_classes + o];
                my_b2[o] -= slr * g;
            }
            __syncthreads();
        }
        epoch_loss = batches ? (loss_sum / (double)batches) : 0.0;
    }

    // -- final training accuracy, evaluated in batches the same way
    ternarize(my_w1, n1, eff1, scratch);
    if (quantize_head) {
        ternarize(my_w2, n2, eff2, scratch);
    } else {
        for (int i = t; i < n2; i += nthreads) eff2[i] = my_w2[i];
        __syncthreads();
    }

    double correct = 0.0;
    for (int start = 0; start < n_samples; start += bsz) {
        const int bs = min(bsz, n_samples - start);
        for (int idx = t; idx < bs * hidden; idx += nthreads) {
            const int k = idx / hidden, h = idx - k * hidden;
            const double* x = my_xs + (size_t)(start + k) * in_dim;
            const double* row = eff1 + (size_t)h * in_dim;
            double a = my_b1[h];
            for (int i = 0; i < in_dim; ++i) a += row[i] * x[i];
            act[idx] = fake_quant((a > 0.0) ? a : 0.0, step, clip);
        }
        __syncthreads();
        double local_ok = 0.0;
        for (int k = t; k < bs; k += nthreads) {
            const double* a = act + (size_t)k * hidden;
            int best = 0;
            double bestv = -1e308;
            for (int o = 0; o < n_classes; ++o) {
                const double* row = eff2 + (size_t)o * hidden;
                double v = my_b2[o];
                for (int h = 0; h < hidden; ++h) v += row[h] * a[h];
                if (v > bestv) { bestv = v; best = o; }
            }
            if (best == my_ys[start + k]) local_ok += 1.0;
        }
        correct += block_sum(local_ok, scratch);
    }

    if (t == 0) {
        out_loss[e] = epoch_loss;
        out_acc[e] = correct / (double)n_samples;
    }
}

}  // namespace

UQ_API int uqfg_available(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

// Shared doubles required per block for a given shape and minibatch.
UQ_API int uqfg_shared_doubles(int in_dim, int hidden, int n_classes, int batch) {
    return hidden * in_dim + n_classes * hidden
         + 2 * batch * hidden + batch * n_classes + UQ_THREADS;
}

// Returns 0 on success, negative on failure (caller falls back to another tier).
UQ_API int uqfg_train_experts(
    int n_experts, int in_dim, int hidden, int n_classes, int n_samples,
    const double* xs, const int32_t* ys,
    double* w1, double* b1, double* w2, double* b2,
    int epochs, double lr, int batch_size, int act_bits, int quantize_head,
    double clip, const int32_t* order, double* out_loss, double* out_acc) {

    if (n_experts <= 0 || n_samples <= 0) return 0;

    const int bsz = (batch_size > 0) ? batch_size : n_samples;
    if (bsz > UQ_MAX_BATCH) return -6;

    const size_t n1 = (size_t)hidden * in_dim;
    const size_t n2 = (size_t)n_classes * hidden;
    const size_t shared =
        (size_t)uqfg_shared_doubles(in_dim, hidden, n_classes, bsz) * sizeof(double);

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return -1;
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shared > (size_t)max_shared) return -2;

    const size_t bx = (size_t)n_experts * n_samples * in_dim * sizeof(double);
    const size_t by = (size_t)n_experts * n_samples * sizeof(int32_t);
    const size_t bw1 = (size_t)n_experts * n1 * sizeof(double);
    const size_t bb1 = (size_t)n_experts * hidden * sizeof(double);
    const size_t bw2 = (size_t)n_experts * n2 * sizeof(double);
    const size_t bb2 = (size_t)n_experts * n_classes * sizeof(double);
    const size_t bo = (size_t)n_experts * epochs * n_samples * sizeof(int32_t);
    const size_t br = (size_t)n_experts * sizeof(double);

    double *d_xs = nullptr, *d_w1 = nullptr, *d_b1 = nullptr, *d_w2 = nullptr,
           *d_b2 = nullptr, *d_loss = nullptr, *d_acc = nullptr;
    int32_t *d_ys = nullptr, *d_order = nullptr;
    int rc = 0;

#define UQ_TRY(call) do { if ((call) != cudaSuccess) { rc = -3; goto cleanup; } } while (0)
    UQ_TRY(cudaMalloc(&d_xs, bx));
    UQ_TRY(cudaMalloc(&d_ys, by));
    UQ_TRY(cudaMalloc(&d_w1, bw1));
    UQ_TRY(cudaMalloc(&d_b1, bb1));
    UQ_TRY(cudaMalloc(&d_w2, bw2));
    UQ_TRY(cudaMalloc(&d_b2, bb2));
    UQ_TRY(cudaMalloc(&d_order, bo));
    UQ_TRY(cudaMalloc(&d_loss, br));
    UQ_TRY(cudaMalloc(&d_acc, br));

    UQ_TRY(cudaMemcpy(d_xs, xs, bx, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_ys, ys, by, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_w1, w1, bw1, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_b1, b1, bb1, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_w2, w2, bw2, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_b2, b2, bb2, cudaMemcpyHostToDevice));
    UQ_TRY(cudaMemcpy(d_order, order, bo, cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(train_experts_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);

    train_experts_kernel<<<n_experts, UQ_THREADS, shared>>>(
        in_dim, hidden, n_classes, n_samples, d_xs, d_ys,
        d_w1, d_b1, d_w2, d_b2, epochs, lr, bsz, act_bits,
        quantize_head, clip, d_order, d_loss, d_acc);

    if (cudaGetLastError() != cudaSuccess) { rc = -4; goto cleanup; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = -5; goto cleanup; }

    UQ_TRY(cudaMemcpy(w1, d_w1, bw1, cudaMemcpyDeviceToHost));
    UQ_TRY(cudaMemcpy(b1, d_b1, bb1, cudaMemcpyDeviceToHost));
    UQ_TRY(cudaMemcpy(w2, d_w2, bw2, cudaMemcpyDeviceToHost));
    UQ_TRY(cudaMemcpy(b2, d_b2, bb2, cudaMemcpyDeviceToHost));
    UQ_TRY(cudaMemcpy(out_loss, d_loss, br, cudaMemcpyDeviceToHost));
    UQ_TRY(cudaMemcpy(out_acc, d_acc, br, cudaMemcpyDeviceToHost));
#undef UQ_TRY

cleanup:
    cudaFree(d_xs); cudaFree(d_ys); cudaFree(d_w1); cudaFree(d_b1);
    cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_order);
    cudaFree(d_loss); cudaFree(d_acc);
    return rc;
}
