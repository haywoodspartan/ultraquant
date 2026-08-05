// UltraQuant forge: multi-expert ternary training on the CPU.
//
// Trains N independent single-hidden-layer ternary networks concurrently, one
// worker thread per chunk of experts.  The arithmetic mirrors the pure-Python
// reference (ultraquant/model/network.py + quantize.py) operation for operation,
// including summation order and Python's round-half-to-even, so results agree to
// floating-point noise rather than merely "closely".
//
// Layout: every array is expert-major and tightly packed.
//   xs   [n_experts][n_samples][in_dim]
//   ys   [n_experts][n_samples]
//   w1   [n_experts][hidden][in_dim]        master weights, updated in place
//   b1   [n_experts][hidden]
//   w2   [n_experts][n_classes][hidden]
//   b2   [n_experts][n_classes]
//   order[n_experts][epochs][n_samples]     shuffle order, precomputed on the
//                                           host so every tier sees identical
//                                           minibatches (determinism)
//   loss [n_experts]                        mean loss of the final epoch
//   acc  [n_experts]                        training accuracy after the final epoch

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define UQ_API extern "C" __declspec(dllexport)
#else
#define UQ_API extern "C"
#endif

namespace {

constexpr double kEps = 1e-12;

// Ternarize a weight matrix in place into `eff` (= alpha * q), mirroring
// quantize_ternary(): delta = 0.75 * mean|w|, alpha = mean|w| over survivors.
void ternarize(const double* w, int n, double* eff) {
    double sum_abs = 0.0;
    for (int i = 0; i < n; ++i) sum_abs += std::fabs(w[i]);
    const double delta = 0.75 * (sum_abs / static_cast<double>(n));
    if (delta == 0.0) {
        for (int i = 0; i < n; ++i) eff[i] = 0.0;
        return;
    }
    double surv_sum = 0.0;
    int surv_n = 0;
    for (int i = 0; i < n; ++i) {
        const double a = std::fabs(w[i]);
        if (a > delta) { surv_sum += a; ++surv_n; }
    }
    const double alpha = surv_n ? (surv_sum / static_cast<double>(surv_n)) : 1.0;
    for (int i = 0; i < n; ++i) {
        const double v = w[i];
        eff[i] = (std::fabs(v) > delta) ? (v > 0.0 ? alpha : -alpha) : 0.0;
    }
}

// fake_quantize_activation with bits/clip; nearbyint matches Python's round()
// (both are round-half-to-even under the default FP rounding mode).
inline double fake_quant(double v, double step, double clip) {
    const double clamped = std::min(std::max(v, -clip), clip);
    const double idx = std::nearbyint((clamped + clip) / step);
    return idx * step - clip;
}

struct ExpertWork {
    int in_dim, hidden, n_classes, n_samples, epochs, batch_size, act_bits;
    int quantize_head;
    double lr, clip;
};

void train_one(const ExpertWork& cfg, const double* xs, const int32_t* ys,
               double* w1, double* b1, double* w2, double* b2,
               const int32_t* order, double* out_loss, double* out_acc) {
    const int in_dim = cfg.in_dim, hidden = cfg.hidden, ncls = cfg.n_classes;
    const int n1 = hidden * in_dim, n2 = ncls * hidden;
    const double clip = cfg.clip;
    const int levels = (1 << cfg.act_bits) - 1;
    const double step = (levels >= 2) ? (2.0 * clip) / (levels - 1) : 0.0;

    std::vector<double> eff1(n1), eff2(n2), gw1(n1, 0.0), gw2(n2, 0.0);
    std::vector<double> gb1(hidden, 0.0), gb2(ncls, 0.0);
    std::vector<double> z(hidden), act(hidden), mask(hidden);
    std::vector<double> logits(ncls), probs(ncls), grad_a(hidden);

    double epoch_loss = 0.0;
    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        const int32_t* ep_order = order + static_cast<size_t>(epoch) * cfg.n_samples;
        epoch_loss = 0.0;
        int batches = 0;

        for (int start = 0; start < cfg.n_samples; start += cfg.batch_size) {
            const int bs = std::min(cfg.batch_size, cfg.n_samples - start);
            // Master weights are constant across a minibatch, so ternarize once
            // per batch instead of once per sample -- identical result, far less
            // work than the reference's per-forward recomputation.
            ternarize(w1, n1, eff1.data());
            if (cfg.quantize_head) ternarize(w2, n2, eff2.data());
            else std::copy(w2, w2 + n2, eff2.begin());

            std::fill(gw1.begin(), gw1.end(), 0.0);
            std::fill(gw2.begin(), gw2.end(), 0.0);
            std::fill(gb1.begin(), gb1.end(), 0.0);
            std::fill(gb2.begin(), gb2.end(), 0.0);

            double batch_loss = 0.0;
            for (int k = 0; k < bs; ++k) {
                const int s = ep_order[start + k];
                const double* x = xs + static_cast<size_t>(s) * in_dim;
                const int y = ys[s];

                // -- forward: hidden
                for (int h = 0; h < hidden; ++h) {
                    double acc = b1[h];
                    const double* row = eff1.data() + static_cast<size_t>(h) * in_dim;
                    for (int i = 0; i < in_dim; ++i) acc += row[i] * x[i];
                    z[h] = acc;
                    mask[h] = (acc > 0.0) ? 1.0 : 0.0;
                    act[h] = fake_quant((acc > 0.0) ? acc : 0.0, step, clip);
                }
                // -- forward: head
                double maxlog = -1e308;
                for (int o = 0; o < ncls; ++o) {
                    double acc = b2[o];
                    const double* row = eff2.data() + static_cast<size_t>(o) * hidden;
                    for (int h = 0; h < hidden; ++h) acc += row[h] * act[h];
                    logits[o] = acc;
                    if (acc > maxlog) maxlog = acc;
                }
                double zsum = 0.0;
                for (int o = 0; o < ncls; ++o) { probs[o] = std::exp(logits[o] - maxlog); zsum += probs[o]; }
                for (int o = 0; o < ncls; ++o) probs[o] /= zsum;
                batch_loss += -std::log(std::max(probs[y], kEps));

                // -- backward: head
                for (int h = 0; h < hidden; ++h) grad_a[h] = 0.0;
                for (int o = 0; o < ncls; ++o) {
                    const double g = probs[o] - ((o == y) ? 1.0 : 0.0);
                    gb2[o] += g;
                    const double* wrow = eff2.data() + static_cast<size_t>(o) * hidden;
                    double* grow = gw2.data() + static_cast<size_t>(o) * hidden;
                    for (int h = 0; h < hidden; ++h) {
                        grad_a[h] += g * wrow[h];
                        grow[h] += g * act[h];
                    }
                }
                // -- backward: hidden (straight-through the fake-quant, then ReLU)
                for (int h = 0; h < hidden; ++h) {
                    const double g = grad_a[h] * mask[h];
                    gb1[h] += g;
                    double* grow = gw1.data() + static_cast<size_t>(h) * in_dim;
                    for (int i = 0; i < in_dim; ++i) grow[i] += g * x[i];
                }
            }

            // -- step (mean-gradient equivalent), masters clamped to [-1, 1]
            const double slr = cfg.lr / static_cast<double>(bs);
            for (int i = 0; i < n1; ++i)
                w1[i] = std::min(std::max(w1[i] - slr * gw1[i], -1.0), 1.0);
            for (int h = 0; h < hidden; ++h) b1[h] -= slr * gb1[h];
            for (int i = 0; i < n2; ++i)
                w2[i] = std::min(std::max(w2[i] - slr * gw2[i], -1.0), 1.0);
            for (int o = 0; o < ncls; ++o) b2[o] -= slr * gb2[o];

            epoch_loss += batch_loss / static_cast<double>(bs);
            ++batches;
        }
        if (batches) epoch_loss /= static_cast<double>(batches);
    }

    // -- final training accuracy with the trained weights
    ternarize(w1, n1, eff1.data());
    if (cfg.quantize_head) ternarize(w2, n2, eff2.data());
    else std::copy(w2, w2 + n2, eff2.begin());
    int correct = 0;
    for (int s = 0; s < cfg.n_samples; ++s) {
        const double* x = xs + static_cast<size_t>(s) * in_dim;
        for (int h = 0; h < hidden; ++h) {
            double acc = b1[h];
            const double* row = eff1.data() + static_cast<size_t>(h) * in_dim;
            for (int i = 0; i < in_dim; ++i) acc += row[i] * x[i];
            act[h] = fake_quant((acc > 0.0) ? acc : 0.0, step, clip);
        }
        int best = 0;
        double bestv = -1e308;
        for (int o = 0; o < ncls; ++o) {
            double acc = b2[o];
            const double* row = eff2.data() + static_cast<size_t>(o) * hidden;
            for (int h = 0; h < hidden; ++h) acc += row[h] * act[h];
            if (acc > bestv) { bestv = acc; best = o; }
        }
        if (best == ys[s]) ++correct;
    }
    *out_loss = epoch_loss;
    *out_acc = static_cast<double>(correct) / static_cast<double>(cfg.n_samples);
}

}  // namespace

UQ_API int uqf_version(void) { return 1; }

UQ_API void uqf_train_experts(
    int n_experts, int in_dim, int hidden, int n_classes, int n_samples,
    const double* xs, const int32_t* ys,
    double* w1, double* b1, double* w2, double* b2,
    int epochs, double lr, int batch_size, int act_bits, int quantize_head,
    double clip, const int32_t* order, double* out_loss, double* out_acc,
    int n_threads) {
    if (n_experts <= 0 || n_samples <= 0) return;

    ExpertWork cfg{in_dim, hidden, n_classes, n_samples, epochs,
                   batch_size > 0 ? batch_size : n_samples, act_bits,
                   quantize_head, lr, clip};

    const size_t sx = static_cast<size_t>(n_samples) * in_dim;
    const size_t s1 = static_cast<size_t>(hidden) * in_dim;
    const size_t s2 = static_cast<size_t>(n_classes) * hidden;
    const size_t so = static_cast<size_t>(epochs) * n_samples;

    int workers = n_threads;
    if (workers <= 0) {
        workers = static_cast<int>(std::thread::hardware_concurrency());
        if (workers <= 0) workers = 1;
    }
    workers = std::min(workers, n_experts);

    auto run_range = [&](int lo, int hi) {
        for (int e = lo; e < hi; ++e) {
            train_one(cfg,
                      xs + static_cast<size_t>(e) * sx,
                      ys + static_cast<size_t>(e) * n_samples,
                      w1 + static_cast<size_t>(e) * s1,
                      b1 + static_cast<size_t>(e) * hidden,
                      w2 + static_cast<size_t>(e) * s2,
                      b2 + static_cast<size_t>(e) * n_classes,
                      order + static_cast<size_t>(e) * so,
                      out_loss + e, out_acc + e);
        }
    };

    if (workers <= 1) { run_range(0, n_experts); return; }

    std::vector<std::thread> pool;
    pool.reserve(workers);
    const int chunk = (n_experts + workers - 1) / workers;
    for (int w = 0; w < workers; ++w) {
        const int lo = w * chunk;
        const int hi = std::min(lo + chunk, n_experts);
        if (lo >= hi) break;
        pool.emplace_back(run_range, lo, hi);
    }
    for (auto& t : pool) t.join();
}
