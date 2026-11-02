#include "detector/cuda_trainer.hpp"
#include "detector/trainer.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef DETECTOR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Trains two freshly-initialized (same seed) copies of `cfg` for `steps` applied-update
// iterations, one with the CPU Trainer and one with CudaTrainer, and asserts every learned
// parameter (weights, biases, and for batch_normalize layers scales/means/variances) matches
// within a tight tolerance. This is the correctness evidence for the new CUDA training kernels:
// it cross-checks against the CPU path already validated by the finite-difference gradient
// check in training_smoke.cpp, rather than re-deriving the batch-norm math a second time. Each
// iteration's minibatch is `batch_size` identical copies of `input`/`targets`: true batch-norm
// statistics over N identical copies equal that one sample's own statistics, so this keeps
// exercising the real batched code path (not a batch_size=1 shortcut) while staying comparable
// to a single, known input/target pair.
void check_parity(const std::string& cfg, const std::vector<float>& input,
                   const std::vector<detector::Target>& targets, int steps, int batch_size,
                   const char* label) {
    auto cpu_net = detector::Network::from_cfg(cfg);
    cpu_net.initialize_random(17);
    auto gpu_net = detector::Network::from_cfg(cfg);
    gpu_net.initialize_random(17);

    detector::Trainer cpu_trainer(cpu_net, detector::LearningRateSchedule{0.05f}, batch_size, 0.0005f);
    detector::CudaTrainer gpu_trainer(gpu_net, detector::LearningRateSchedule{0.05f}, batch_size, 0.0005f);
    std::vector<std::vector<float>> batch_inputs(static_cast<std::size_t>(batch_size), input);
    std::vector<std::vector<detector::Target>> batch_targets(static_cast<std::size_t>(batch_size), targets);

    float cpu_loss = 0, gpu_loss = 0;
    for (int i = 0; i < steps; ++i) {
        cpu_loss = cpu_trainer.step(batch_inputs, batch_targets).loss;
        gpu_loss = gpu_trainer.step(batch_inputs, batch_targets).loss;
    }
    gpu_trainer.sync_to_network();

    std::cout << "cuda_training_parity " << label << " cpu_loss=" << cpu_loss << " gpu_loss=" << gpu_loss << '\n';
    expect(std::abs(cpu_loss - gpu_loss) < 1e-2f * std::max(1.0f, std::abs(cpu_loss)), "CPU/GPU loss mismatch");

    const auto& cpu_layers = cpu_net.layers();
    const auto& gpu_layers = gpu_net.layers();
    expect(cpu_layers.size() == gpu_layers.size(), "layer count mismatch");
    for (std::size_t i = 0; i < cpu_layers.size(); ++i) {
        const auto& a = cpu_layers[i];
        const auto& b = gpu_layers[i];
        auto close = [](float x, float y) { return std::abs(x - y) < 1e-2f * std::max(1.0f, std::abs(x)); };
        expect(a.weights.size() == b.weights.size(), "weight count mismatch");
        float max_diff = 0; std::size_t max_p = 0;
        for (std::size_t p = 0; p < a.weights.size(); ++p) {
            float d = std::abs(a.weights[p] - b.weights[p]);
            if (d > max_diff) { max_diff = d; max_p = p; }
            expect(close(a.weights[p], b.weights[p]), "weight mismatch");
        }
        if (!a.weights.empty())
            std::cout << "  layer " << i << " bn=" << a.batch_normalize << " weights max_diff=" << max_diff
                      << " at " << max_p << " cpu=" << a.weights[max_p] << " gpu=" << b.weights[max_p] << '\n';
        expect(a.biases.size() == b.biases.size(), "bias count mismatch");
        float bias_diff = 0;
        for (std::size_t p = 0; p < a.biases.size(); ++p) {
            bias_diff = std::max(bias_diff, std::abs(a.biases[p] - b.biases[p]));
            expect(close(a.biases[p], b.biases[p]), "bias mismatch");
        }
        if (!a.biases.empty()) std::cout << "    bias max_diff=" << bias_diff << '\n';
        if (a.batch_normalize) {
            float scale_diff = 0, mean_diff = 0, var_diff = 0;
            for (std::size_t p = 0; p < a.scales.size(); ++p) {
                scale_diff = std::max(scale_diff, std::abs(a.scales[p] - b.scales[p]));
                expect(close(a.scales[p], b.scales[p]), "scale mismatch");
            }
            for (std::size_t p = 0; p < a.means.size(); ++p) {
                mean_diff = std::max(mean_diff, std::abs(a.means[p] - b.means[p]));
                expect(close(a.means[p], b.means[p]), "mean mismatch");
            }
            for (std::size_t p = 0; p < a.variances.size(); ++p) {
                var_diff = std::max(var_diff, std::abs(a.variances[p] - b.variances[p]));
                expect(close(a.variances[p], b.variances[p]), "variance mismatch");
            }
            std::cout << "    scale_diff=" << scale_diff << " mean_diff=" << mean_diff << " var_diff=" << var_diff << '\n';
        }
    }
}

// Like check_parity, but the minibatch holds `samples.size()` genuinely *different* inputs
// (not `batch_size` identical copies), so the batch mean/variance differ from any single
// sample's own statistics. This is the decisive CPU/GPU cross-check for the new batched
// batch-norm reduction kernels: a bug in the cross-sample sums (or the total_n normalizer)
// would show up here even if it happened to cancel out in the identical-copies case above.
void check_parity_batch(const std::string& cfg,
                        const std::vector<std::pair<std::vector<float>, std::vector<detector::Target>>>& samples,
                        int steps, const char* label) {
    auto cpu_net = detector::Network::from_cfg(cfg);
    cpu_net.initialize_random(17);
    auto gpu_net = detector::Network::from_cfg(cfg);
    gpu_net.initialize_random(17);
    int batch_size = static_cast<int>(samples.size());
    // Lower LR than check_parity's 0.05: SGD momentum's default 0.9 (~10x effective step size)
    // diverges this tiny two-distinct-sample batch-norm graph at 0.05 within 5 steps (CPU and
    // GPU still matched each other exactly while diverging, so this is a test-tuning issue, not
    // a parity bug); 0.01 keeps the same real momentum code path numerically stable.
    detector::Trainer cpu_trainer(cpu_net, detector::LearningRateSchedule{0.01f}, batch_size, 0.0005f);
    detector::CudaTrainer gpu_trainer(gpu_net, detector::LearningRateSchedule{0.01f}, batch_size, 0.0005f);
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<detector::Target>> targets;
    for (const auto& sample : samples) { inputs.push_back(sample.first); targets.push_back(sample.second); }

    float cpu_loss = 0, gpu_loss = 0;
    for (int i = 0; i < steps; ++i) {
        cpu_loss = cpu_trainer.step(inputs, targets).loss;
        gpu_loss = gpu_trainer.step(inputs, targets).loss;
    }
    gpu_trainer.sync_to_network();
    std::cout << "cuda_training_parity " << label << " cpu_loss=" << cpu_loss << " gpu_loss=" << gpu_loss << '\n';
    expect(std::abs(cpu_loss - gpu_loss) < 1e-2f * std::max(1.0f, std::abs(cpu_loss)), "CPU/GPU loss mismatch (distinct batch)");
    const auto& cpu_layers = cpu_net.layers();
    const auto& gpu_layers = gpu_net.layers();
    for (std::size_t i = 0; i < cpu_layers.size(); ++i) {
        const auto& a = cpu_layers[i];
        const auto& b = gpu_layers[i];
        auto close = [](float x, float y) { return std::abs(x - y) < 1e-2f * std::max(1.0f, std::abs(x)); };
        for (std::size_t p = 0; p < a.weights.size(); ++p) expect(close(a.weights[p], b.weights[p]), "weight mismatch (distinct batch)");
        if (a.batch_normalize)
            for (std::size_t p = 0; p < a.scales.size(); ++p) expect(close(a.scales[p], b.scales[p]), "scale mismatch (distinct batch)");
    }
}

// CPU/GPU parity through a Darknet-style multiscale resize mid-training: mini.cfg's one
// maxpool (stride 2) and one upsample (stride 2) exactly cancel, so doubling its 4x4 input to
// 8x8 keeps the route layer's two branches matched, just like network.cpp's from_cfg validates
// at the original size. Trains a few steps at 4x4, resizes both trainers, trains a few more at
// 8x8, and asserts every learned parameter still matches - the decisive cross-check for
// CudaTrainer::resize's device-buffer reallocation (the invasive, highest-risk part of
// multiscale support) against the already-proven CPU path.
void check_parity_resize(const std::string& cfg, int steps, const char* label) {
    auto cpu_net = detector::Network::from_cfg(cfg);
    cpu_net.initialize_random(17);
    auto gpu_net = detector::Network::from_cfg(cfg);
    gpu_net.initialize_random(17);
    detector::Trainer cpu_trainer(cpu_net, detector::LearningRateSchedule{0.01f}, 1, 0.0005f);
    detector::CudaTrainer gpu_trainer(gpu_net, detector::LearningRateSchedule{0.01f}, 1, 0.0005f);
    std::vector<detector::Target> targets{{0, 0.5f, 0.5f, 0.25f, 0.25f}};

    std::vector<float> small_input(cpu_net.input_shape().elements());
    for (std::size_t i = 0; i < small_input.size(); ++i)
        small_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
    float cpu_loss = 0, gpu_loss = 0;
    for (int i = 0; i < steps; ++i) {
        cpu_loss = cpu_trainer.step({small_input}, {targets}).loss;
        gpu_loss = gpu_trainer.step({small_input}, {targets}).loss;
    }

    cpu_trainer.resize(8, 8);
    gpu_trainer.resize(8, 8);
    expect(cpu_net.input_shape().h == 8 && gpu_net.input_shape().h == 8, "resize did not take effect");
    std::vector<float> large_input(cpu_net.input_shape().elements());
    for (std::size_t i = 0; i < large_input.size(); ++i)
        large_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
    for (int i = 0; i < steps; ++i) {
        cpu_loss = cpu_trainer.step({large_input}, {targets}).loss;
        gpu_loss = gpu_trainer.step({large_input}, {targets}).loss;
    }
    gpu_trainer.sync_to_network();

    std::cout << "cuda_training_parity " << label << " cpu_loss=" << cpu_loss << " gpu_loss=" << gpu_loss << '\n';
    expect(std::abs(cpu_loss - gpu_loss) < 1e-2f * std::max(1.0f, std::abs(cpu_loss)), "CPU/GPU loss mismatch after resize");
    const auto& cpu_layers = cpu_net.layers();
    const auto& gpu_layers = gpu_net.layers();
    auto close = [](float x, float y) { return std::abs(x - y) < 1e-2f * std::max(1.0f, std::abs(x)); };
    for (std::size_t i = 0; i < cpu_layers.size(); ++i) {
        const auto& a = cpu_layers[i];
        const auto& b = gpu_layers[i];
        for (std::size_t p = 0; p < a.weights.size(); ++p) expect(close(a.weights[p], b.weights[p]), "weight mismatch after resize");
        if (a.batch_normalize)
            for (std::size_t p = 0; p < a.scales.size(); ++p) expect(close(a.scales[p], b.scales[p]), "scale mismatch after resize");
    }
}

// Dropout draws its mask from unrelated RNGs on CPU (std::mt19937) and GPU (a device xorshift
// hash), so unlike every other layer here, CPU and GPU losses/weights are NOT expected to match
// bit-for-bit - this only checks that CudaTrainer's dropout kernels are wired correctly (finite
// loss, runs to completion, weights actually change) independently of the CPU trainer.
void check_dropout_runs(const std::string& cfg, const std::vector<float>& input,
                        const std::vector<detector::Target>& targets, int steps) {
    auto gpu_net = detector::Network::from_cfg(cfg);
    gpu_net.initialize_random(17);
    auto initial_weights = gpu_net.layers()[0].weights;
    detector::CudaTrainer gpu_trainer(gpu_net, detector::LearningRateSchedule{0.05f}, 1, 0.0005f);
    float loss = 0;
    for (int i = 0; i < steps; ++i) {
        loss = gpu_trainer.step({input}, {targets}).loss;
        expect(std::isfinite(loss), "dropout CUDA training step produced a non-finite loss");
    }
    gpu_trainer.sync_to_network();
    std::cout << "cuda_dropout_smoke loss=" << loss << '\n';
    bool changed = false;
    for (std::size_t p = 0; p < initial_weights.size(); ++p)
        if (std::abs(gpu_net.layers()[0].weights[p] - initial_weights[p]) > 1e-8f) changed = true;
    expect(changed, "dropout CUDA training did not update weights");
}

/** Checks weighted model/velocity averaging against an unsplit batch without batch normalization. */
void check_data_parallel_average(const std::string& cfg) {
    auto left_net = detector::Network::from_cfg(cfg);
    left_net.initialize_random(71);
    auto right_net = left_net;
    auto reference_net = left_net;
    detector::LearningRateSchedule schedule{0.002f};
    detector::CudaTrainer left(left_net, schedule, 2, 0.0005f, 0.9f, 0.99f, 0, 0);
    detector::CudaTrainer right(right_net, schedule, 2, 0.0005f, 0.9f, 0.99f, 0, 0);
    detector::CudaTrainer reference(reference_net, schedule, 3, 0.0005f, 0.9f, 0.99f, 0, 0);
    std::vector<std::vector<float>> inputs(3, std::vector<float>(16));
    for (std::size_t s = 0; s < inputs.size(); ++s)
        for (std::size_t j = 0; j < inputs[s].size(); ++j)
            inputs[s][j] = 0.1f + 0.02f * static_cast<float>((j + 3 * s) % 9);
    std::vector<std::vector<detector::Target>> targets{
        {{0, 0.5f, 0.5f, 0.25f, 0.25f}},
        {{1, 0.4f, 0.6f, 0.3f, 0.3f}},
        {{0, 0.7f, 0.3f, 0.2f, 0.2f}}};
    auto compare = [&] {
        left.sync_to_network();
        right.sync_to_network();
        reference.sync_to_network();
        expect(left.iteration() == right.iteration() && left.iteration() == reference.iteration(),
               "replica step counts differ");
        for (std::size_t i = 0; i < reference_net.layers().size(); ++i) {
            const auto& expected = reference_net.layers()[i];
            const auto& a = left_net.layers()[i];
            const auto& b = right_net.layers()[i];
            for (std::size_t j = 0; j < expected.weights.size(); ++j) {
                expect(std::abs(a.weights[j] - expected.weights[j]) < 2e-5f,
                       "weighted multi-GPU weight differs from full batch");
                expect(std::abs(b.weights[j] - expected.weights[j]) < 2e-5f,
                       "replica weights differ after synchronization");
            }
            for (std::size_t j = 0; j < expected.biases.size(); ++j) {
                expect(std::abs(a.biases[j] - expected.biases[j]) < 2e-5f,
                       "weighted multi-GPU bias differs from full batch");
                expect(std::abs(b.biases[j] - expected.biases[j]) < 2e-5f,
                       "replica biases differ after synchronization");
            }
        }
    };
    for (int step = 0; step < 2; ++step) {
        left.step({inputs[0]}, {targets[0]});
        right.step({inputs[1], inputs[2]}, {targets[1], targets[2]});
        reference.step(inputs, targets);
        detector::CudaTrainer::average_state({&left, &right}, {1, 2});
        compare();
    }
    // A final partial batch can leave one replica idle; it still receives parameters,
    // momentum and the global iteration before the next minibatch.
    right.step({inputs[2]}, {targets[2]});
    reference.step({inputs[2]}, {targets[2]});
    detector::CudaTrainer::average_state({&left, &right}, {0, 1});
    compare();
    std::cout << "cuda_data_parallel_average passed\n";
}

/** Confirms the same averaging operation broadcasts trainable BN and inference statistics. */
void check_data_parallel_batch_norm(const std::string& cfg) {
    auto a_net = detector::Network::from_cfg(cfg);
    a_net.initialize_random(72);
    auto b_net = a_net;
    detector::CudaTrainer a(a_net, detector::LearningRateSchedule{0.001f}, 1);
    detector::CudaTrainer b(b_net, detector::LearningRateSchedule{0.001f}, 1);
    std::vector<float> first(a_net.input_shape().elements(), 0.2f);
    std::vector<float> second(first.size(), 0.7f);
    a.step({first}, {{{0, 0.5f, 0.5f, 0.25f, 0.25f}}});
    b.step({second}, {{{0, 0.5f, 0.5f, 0.25f, 0.25f}}});
    a.sync_to_network();
    b.sync_to_network();
    const auto before_a = a_net.layers()[0];
    const auto before_b = b_net.layers()[0];
    detector::CudaTrainer::average_state({&a, &b}, {1, 1});
    a.sync_to_network();
    b.sync_to_network();
    for (std::size_t i = 0; i < before_a.scales.size(); ++i) {
        const float mean = (before_a.means[i] + before_b.means[i]) * 0.5f;
        const float variance = (before_a.variances[i] + before_b.variances[i]) * 0.5f;
        const float scale = (before_a.scales[i] + before_b.scales[i]) * 0.5f;
        expect(std::abs(a_net.layers()[0].means[i] - mean) < 1e-6f, "BN mean average mismatch");
        expect(std::abs(a_net.layers()[0].variances[i] - variance) < 1e-6f, "BN variance average mismatch");
        expect(std::abs(a_net.layers()[0].scales[i] - scale) < 1e-6f, "BN scale average mismatch");
        expect(std::abs(b_net.layers()[0].means[i] - mean) < 1e-6f, "BN replica mean mismatch");
        expect(std::abs(b_net.layers()[0].variances[i] - variance) < 1e-6f, "BN replica variance mismatch");
        expect(std::abs(b_net.layers()[0].scales[i] - scale) < 1e-6f, "BN replica scale mismatch");
    }
    std::cout << "cuda_data_parallel_bn_state passed\n";
}

}  // namespace

/** Checks CUDA detector training against the CPU reference across supported layer kinds. */
int main() {
    try {
#ifdef DETECTOR_HAS_CUDA
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
            std::cout << "no CUDA device available; skipping CUDA training parity test\n";
            return 0;
        }
        auto directory = std::filesystem::path(__FILE__).parent_path();
        std::vector<float> graph_input(detector::Network::from_cfg((directory / "mini.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < graph_input.size(); ++i)
            graph_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "mini.cfg").string(), graph_input, {{0, 0.5f, 0.5f, 0.25f, 0.25f}}, 5, 1, "bn_5step");

        std::vector<float> shortcut_input{0.3f};
        check_parity((directory / "shortcut-train.cfg").string(), shortcut_input,
                     {{0, 0.5f, 0.5f, 0.5f, 0.5f}}, 5, 1, "shortcut");

        // True minibatch: batch_size 3 (of identical copies, see check_parity) should still match.
        check_parity((directory / "shortcut-train.cfg").string(), shortcut_input,
                     {{0, 0.5f, 0.5f, 0.5f, 0.5f}}, 3, 3, "shortcut+minibatch");

        // Decisive true-batch batch-norm check: two genuinely different samples in one minibatch.
        std::vector<float> graph_input_b(graph_input.size());
        for (std::size_t i = 0; i < graph_input_b.size(); ++i) graph_input_b[i] = 0.1f + 0.05f * static_cast<float>(i % 5);
        check_parity_batch((directory / "mini.cfg").string(),
                           {{graph_input, {{0, 0.5f, 0.5f, 0.25f, 0.25f}}}, {graph_input_b, {{0, 0.3f, 0.7f, 0.2f, 0.3f}}}},
                           5, "bn_distinct_batch");

        check_parity_resize((directory / "mini.cfg").string(), 3, "resize_mid_training");

        // The CUDA trainer launches maxpool/upsample/reorg/route/shortcut/BN kernels once per
        // minibatch over sample-major blocks: three distinct samples through every such layer
        // kind (plus im2col 3x3, strided and pointwise convolutions) must still match the CPU.
        const auto graph_cfg = (directory / "batch-graph-train.cfg").string();
        const std::size_t graph_elements = detector::Network::from_cfg(graph_cfg).input_shape().elements();
        std::vector<std::pair<std::vector<float>, std::vector<detector::Target>>> graph_batch;
        for (int sample = 0; sample < 3; ++sample) {
            std::vector<float> values(graph_elements);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = 0.05f + 0.9f * static_cast<float>((i * (3 + 2 * sample) + 5 * sample) % 17) / 17.0f;
            graph_batch.push_back({values, {{0, 0.3f + 0.2f * sample, 0.6f - 0.1f * sample, 0.3f, 0.4f}}});
        }
        check_parity_batch(graph_cfg, graph_batch, 5, "mixed_graph_distinct_batch");

        // Reorg has no learnable parameters and no randomness, so unlike dropout it fits the
        // exact CPU/GPU parity harness directly.
        std::vector<float> reorg_input(detector::Network::from_cfg((directory / "reorg-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < reorg_input.size(); ++i) reorg_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "reorg-train.cfg").string(), reorg_input, {{0, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 1, "reorg");

        std::vector<float> norm_input(detector::Network::from_cfg((directory / "l2norm-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < norm_input.size(); ++i) norm_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "l2norm-train.cfg").string(), norm_input,
                     {{0, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 2, "l2norm");

        std::vector<float> average_input(detector::Network::from_cfg((directory / "avgpool-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < average_input.size(); ++i) average_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "avgpool-train.cfg").string(), average_input,
                     {{0, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 2, "avgpool");

        check_parity((directory / "normalization-train.cfg").string(),
                     {0.2f, 0.4f, 0.6f}, {{0, 0.5f, 0.5f, 0.3f, 0.3f}},
                     5, 2, "normalization");

        std::vector<float> local_input(detector::Network::from_cfg((directory / "local-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < local_input.size(); ++i) local_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "local-train.cfg").string(), local_input,
                     {{0, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 2, "local");

        std::vector<float> deconv_input(detector::Network::from_cfg((directory / "deconv-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < deconv_input.size(); ++i) deconv_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "deconv-train.cfg").string(), deconv_input,
                     {{0, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 2, "deconvolutional");

        check_dropout_runs((directory / "dropout-train.cfg").string(), {0.4f}, {{0, 0.5f, 0.5f, 0.5f, 0.5f}}, 20);

        // Region has no learnable parameters of its own and no randomness (rescore just decodes
        // the deterministic forward output), so like reorg it fits the exact CPU/GPU parity
        // harness directly - this is the decisive cross-check that CudaTrainer's region dispatch
        // (host-side yolo_loss_and_gradient call, same as yolo) produces identical gradients to
        // the CPU trainer's softmax/rescore/scale-knob code paths.
        std::vector<float> region_input(detector::Network::from_cfg((directory / "region-train.cfg").string()).input_shape().elements());
        for (std::size_t i = 0; i < region_input.size(); ++i) region_input[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
        check_parity((directory / "region-train.cfg").string(), region_input, {{1, 0.5f, 0.5f, 0.3f, 0.3f}}, 5, 1, "region");
        check_data_parallel_average((directory / "region-train.cfg").string());
        check_data_parallel_batch_norm((directory / "mini.cfg").string());

        std::cout << "cuda training smoke passed\n";
        return 0;
#else
        std::cout << "DETECTOR_HAS_CUDA not defined; nothing to test\n";
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "cuda training smoke failed: " << error.what() << '\n';
        return 1;
    }
}
