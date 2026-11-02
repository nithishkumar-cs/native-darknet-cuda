#include "detector/classifier_trainer.hpp"
#include "detector/cuda_classifier_trainer.hpp"
#include "detector/network.hpp"
#include "detector/vector_dataset.hpp"
#ifdef DETECTOR_HAS_TENSORRT
#include "detector/trt_session.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Training, dataset-limit, and checkpoint options for the semantic CLI. */
struct Options {
    int epochs = 1;
    int limit = 0;
    float learning_rate = 0.01f;
    float decay = 0.0f;
    unsigned seed = 2026;
    std::string initial_weights;
    std::string backend = "cpu";
};

/** Parses the shared train/eval options after the four positional arguments. */
Options parse(int argc, char** argv) {
    Options options;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--epochs") options.epochs = std::stoi(value);
        else if (key == "--limit") options.limit = std::stoi(value);
        else if (key == "--lr") options.learning_rate = std::stof(value);
        else if (key == "--decay") options.decay = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<unsigned>(std::stoul(value));
        else if (key == "--init") options.initial_weights = value;
        else if (key == "--backend") options.backend = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.backend != "cpu" && options.backend != "cuda" && options.backend != "tensorrt")
        throw std::runtime_error("backend must be cpu, cuda or tensorrt");
    if (options.epochs < 1 || options.limit < 0) throw std::runtime_error("invalid segmenter options");
    return options;
}

/** Prints the semantic segmentation dataset and command contract. */
void usage() {
    std::cerr << "Usage:\n"
              << "  cuda_segmenter train <cfg> <output.weights> <data.vecbin>"
                 " [--backend cpu|cuda] [--epochs N] [--lr 0.01] [--decay 0] [--seed 2026] [--limit N] [--init weights]\n"
              << "  cuda_segmenter eval <cfg> <weights> <data.vecbin> [--limit N] [--backend tensorrt]\n"
              << "    (with --backend tensorrt, <weights> is a Jetson-built TensorRT engine of the exported ONNX)\n"
              << "Each record: input CHW float32 pixels followed by H*W float32 integer class IDs.\n";
}

/** Validates and converts one labeled target value to its class index. */
std::size_t target_class(float value, int classes) {
    if (!std::isfinite(value) || value < 0.0f || value >= classes || std::floor(value) != value)
        throw std::runtime_error("segmentation targets must be integer class IDs or -1 for ignore");
    return static_cast<std::size_t>(value);
}

}  // namespace

/** Trains a pixelwise softmax head or evaluates pixel accuracy and class mIoU. */
int main(int argc, char** argv) {
    try {
        if (argc < 5) { usage(); return 2; }
        const std::string command = argv[1];
        if (command != "train" && command != "eval") { usage(); return 2; }
        auto options = parse(argc, argv);
        auto network = detector::Network::from_cfg(argv[2]);
        const auto shape = network.layers().back().output;
        if (shape.c < 2) throw std::runtime_error("segmentation head needs at least two class channels");
        const std::size_t pixels = static_cast<std::size_t>(shape.h) * shape.w;
        detector::VectorDataset data(argv[4], network.input_shape().elements(), pixels);
        const std::size_t count = options.limit > 0 ?
            std::min(data.size(), static_cast<std::size_t>(options.limit)) : data.size();
        const std::string weights = argv[3];
        if (options.backend == "tensorrt" && command == "train")
            throw std::runtime_error("tensorrt backend is eval-only");
        if (command == "train") {
            if (std::filesystem::exists(weights))
                throw std::runtime_error("refusing to overwrite existing output weights");
            if (options.initial_weights.empty()) network.initialize_random(options.seed);
            else network.load_weights(options.initial_weights);
            std::unique_ptr<detector::VectorTrainer> cpu_trainer;
#ifdef DETECTOR_HAS_CUDA
            std::unique_ptr<detector::CudaClassificationTrainer> cuda_trainer;
            if (options.backend == "cuda")
                cuda_trainer = std::make_unique<detector::CudaClassificationTrainer>(
                    network, options.learning_rate, options.decay, detector::VectorLoss::pixel_softmax_ce);
#else
            if (options.backend == "cuda") throw std::runtime_error("CUDA backend not built; use -DDETECTOR_CUDA=ON");
#endif
            if (options.backend == "cpu")
                cpu_trainer = std::make_unique<detector::VectorTrainer>(
                    network, detector::VectorLoss::pixel_softmax_ce, options.learning_rate, options.decay);
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 rng(options.seed);
            for (int epoch = 1; epoch <= options.epochs; ++epoch) {
                std::shuffle(order.begin(), order.end(), rng);
                double loss_sum = 0.0;
                for (std::size_t i : order) {
                    detector::VectorResult result;
#ifdef DETECTOR_HAS_CUDA
                    if (cuda_trainer) result = cuda_trainer->step_vector(data.input(i), data.target(i));
                    else
#endif
                        result = cpu_trainer->step(data.input(i), data.target(i));
                    loss_sum += result.loss;
                }
#ifdef DETECTOR_HAS_CUDA
                if (cuda_trainer) cuda_trainer->sync_to_network();
#endif
                network.save_weights(weights);
                std::cout << "epoch=" << epoch << " samples=" << count
                          << " backend=" << options.backend << " mean_loss=" << std::fixed << std::setprecision(6) << loss_sum / count
                          << " checkpoint=" << weights << '\n';
            }
        } else {
            if (!options.initial_weights.empty()) throw std::runtime_error("--init is train-only");
#ifdef DETECTOR_HAS_TENSORRT
            std::unique_ptr<detector::TrtSession> trt;
            if (options.backend == "tensorrt") trt = std::make_unique<detector::TrtSession>(network, weights);
#else
            if (options.backend == "tensorrt")
                throw std::runtime_error("TensorRT backend not built; use -DDETECTOR_TENSORRT=ON");
#endif
            if (options.backend != "tensorrt") network.load_weights(weights);
            std::vector<std::size_t> intersection(static_cast<std::size_t>(shape.c), 0);
            std::vector<std::size_t> union_count(static_cast<std::size_t>(shape.c), 0);
            std::size_t correct = 0;
            std::size_t labeled_pixels = 0;
            for (std::size_t i = 0; i < count; ++i) {
#ifdef DETECTOR_HAS_TENSORRT
                auto heads = trt ? trt->forward(data.input(i))
                                 : network.forward(data.input(i), detector::Backend::cpu);
#else
                auto heads = network.forward(data.input(i), detector::Backend::cpu);
#endif
                if (heads.empty() || heads.back().values.size() != shape.elements())
                    throw std::runtime_error("segmentation head output missing");
                const auto& logits = heads.back().values;
                auto targets = data.target(i);
                for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                    if (targets[pixel] == -1.0f) continue;
                    ++labeled_pixels;
                    std::size_t prediction = 0;
                    for (int c = 1; c < shape.c; ++c)
                        if (logits[static_cast<std::size_t>(c) * pixels + pixel] >
                            logits[prediction * pixels + pixel]) prediction = static_cast<std::size_t>(c);
                    const auto truth = target_class(targets[pixel], shape.c);
                    if (prediction == truth) { ++correct; ++intersection[truth]; ++union_count[truth]; }
                    else { ++union_count[truth]; ++union_count[prediction]; }
                }
            }
            double sum_iou = 0.0;
            std::size_t present_classes = 0;
            for (int c = 0; c < shape.c; ++c) {
                if (!union_count[c]) continue;
                sum_iou += static_cast<double>(intersection[c]) / union_count[c];
                ++present_classes;
            }
            if (!labeled_pixels || !present_classes)
                throw std::runtime_error("segmentation evaluation has no labeled pixels");
            std::cout << "samples=" << count << " pixel_accuracy=" << std::fixed << std::setprecision(6)
                      << static_cast<double>(correct) / labeled_pixels
                      << " mean_iou=" << sum_iou / present_classes
                      << " classes_present=" << present_classes << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
