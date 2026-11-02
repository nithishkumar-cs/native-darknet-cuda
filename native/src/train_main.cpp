#include "detector/cuda_trainer.hpp"
#include "detector/image.hpp"
#include "detector/trainer.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

namespace {

struct Options {
    int epochs = 1;
    int limit = 0;
    float learning_rate = 0.0001f;
    float decay = 0.0005f;
    float momentum = 0.9f;  // Darknet cfg's SGD momentum default
    std::uint32_t seed = 2026;
    std::string initial_weights;
    int init_layers = -1;  // -1: full load of initial_weights; >=0: partial/pretrained-backbone load
                            // (Darknet `partial N`) - random-init first, then overlay layers [0, N).
    // Resumes a crashed/interrupted run: full-loads these weights (like --init with no
    // --init-layers) AND seeds the LearningRateSchedule/epoch-loop step counter from the sidecar
    // "<path>.step" file written alongside every checkpoint save, so burn-in/step-decay continues
    // exactly instead of restarting at iteration 0 against already-trained weights. Mutually
    // exclusive with --init/--init-layers.
    std::string resume_from;
    int batch_size = 1;
    int burn_in = 0;
    std::vector<int> steps;
    std::vector<float> scales;
    float flip_probability = 0.0f;
    float exposure_jitter = 0.0f;    // max relative brightness (HSV value) deviation, e.g. 0.3 => factor in [0.7, 1.3]
    float saturation_jitter = 0.0f;  // max relative saturation deviation, same convention as exposure
    float hue_jitter = 0.0f;         // max hue shift as a fraction of the full circle, e.g. 0.1 => +-36 degrees
    float crop_jitter = 0.0f;        // max fraction of width/height cropped from each edge before letterboxing
    std::string backend = "cpu";
    std::vector<int> gpus{0};
    bool gpus_specified = false;
    // Darknet-style multiscale/resize-jitter training: every `multiscale_interval` completed
    // minibatches, resize the network to a new square dimension, a random multiple of 32 in
    // [multiscale_min, multiscale_max]. Disabled (default) when multiscale_max is 0.
    int multiscale_min = 0;
    int multiscale_max = 0;
    int multiscale_interval = 10;  // Darknet's default resize period, in batches
    // Threads decoding/augmenting the next minibatch while the current one trains.
    int loader_threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 2u, 17u)) - 1;
};

std::vector<int> int_list(const std::string& value) {
    std::vector<int> result;
    std::istringstream stream(value);
    std::string field;
    while (std::getline(stream, field, ',')) if (!field.empty()) result.push_back(std::stoi(field));
    return result;
}

std::vector<float> float_list(const std::string& value) {
    std::vector<float> result;
    std::istringstream stream(value);
    std::string field;
    while (std::getline(stream, field, ',')) if (!field.empty()) result.push_back(std::stof(field));
    return result;
}

// Parses and validates the `--key value` options after the four positional arguments; throws
// std::runtime_error on an unknown key or an out-of-range value.
Options options_from(int argc, char** argv) {
    Options options;
    for (int i = 5; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("option requires a value");
        std::string key = argv[i];
        std::string value = argv[i + 1];
        if (key == "--epochs") options.epochs = std::stoi(value);
        else if (key == "--limit") options.limit = std::stoi(value);
        else if (key == "--lr") options.learning_rate = std::stof(value);
        else if (key == "--decay") options.decay = std::stof(value);
        else if (key == "--momentum") options.momentum = std::stof(value);
        else if (key == "--seed") options.seed = static_cast<std::uint32_t>(std::stoul(value));
        else if (key == "--init") options.initial_weights = value;
        else if (key == "--init-layers") options.init_layers = std::stoi(value);
        else if (key == "--resume-from") options.resume_from = value;
        else if (key == "--batch") options.batch_size = std::stoi(value);
        else if (key == "--burn-in") options.burn_in = std::stoi(value);
        else if (key == "--steps") options.steps = int_list(value);
        else if (key == "--scales") options.scales = float_list(value);
        else if (key == "--flip-probability") options.flip_probability = std::stof(value);
        else if (key == "--exposure-jitter") options.exposure_jitter = std::stof(value);
        else if (key == "--saturation-jitter") options.saturation_jitter = std::stof(value);
        else if (key == "--hue-jitter") options.hue_jitter = std::stof(value);
        else if (key == "--crop-jitter") options.crop_jitter = std::stof(value);
        else if (key == "--backend") options.backend = value;
        else if (key == "--gpus") { options.gpus = int_list(value); options.gpus_specified = true; }
        else if (key == "--multiscale-min") options.multiscale_min = std::stoi(value);
        else if (key == "--multiscale-max") options.multiscale_max = std::stoi(value);
        else if (key == "--multiscale-interval") options.multiscale_interval = std::stoi(value);
        else if (key == "--loader-threads") options.loader_threads = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.epochs < 1 || options.epochs > 100000 || options.limit < 0)
        throw std::runtime_error("invalid epochs or limit");
    if (options.init_layers >= 0 && options.initial_weights.empty())
        throw std::runtime_error("--init-layers requires --init");
    if (!options.resume_from.empty() && (!options.initial_weights.empty() || options.init_layers >= 0))
        throw std::runtime_error("--resume-from cannot be combined with --init or --init-layers");
    if (options.batch_size < 1) throw std::runtime_error("invalid batch size");
    if (options.steps.size() != options.scales.size())
        throw std::runtime_error("--steps and --scales must have the same length");
    if (!(options.flip_probability >= 0.0f && options.flip_probability <= 1.0f))
        throw std::runtime_error("invalid flip probability");
    if (!(options.exposure_jitter >= 0.0f && options.exposure_jitter < 1.0f))
        throw std::runtime_error("invalid exposure jitter");
    if (!(options.saturation_jitter >= 0.0f && options.saturation_jitter < 1.0f))
        throw std::runtime_error("invalid saturation jitter");
    if (!(options.hue_jitter >= 0.0f && options.hue_jitter <= 0.5f))
        throw std::runtime_error("invalid hue jitter");
    if (!(options.crop_jitter >= 0.0f && options.crop_jitter < 0.5f))
        throw std::runtime_error("invalid crop jitter");
    if (options.backend != "cpu" && options.backend != "cuda")
        throw std::runtime_error("--backend must be cpu or cuda");
    if (options.gpus_specified && options.backend != "cuda")
        throw std::runtime_error("--gpus requires --backend cuda");
    if (options.gpus.empty() || std::any_of(options.gpus.begin(), options.gpus.end(), [](int id) { return id < 0; }))
        throw std::runtime_error("--gpus requires non-negative CUDA device indices");
    for (std::size_t i = 0; i < options.gpus.size(); ++i)
        if (std::find(options.gpus.begin(), options.gpus.begin() + i, options.gpus[i]) != options.gpus.begin() + i)
            throw std::runtime_error("--gpus requires distinct CUDA device indices");
    if (options.gpus.size() > 1 && options.batch_size < static_cast<int>(options.gpus.size()))
        throw std::runtime_error("--batch must be at least the number of selected GPUs");
    if (options.multiscale_min != 0 || options.multiscale_max != 0) {
        if (options.multiscale_min <= 0 || options.multiscale_max <= 0 || options.multiscale_min > options.multiscale_max ||
            options.multiscale_min % 32 != 0 || options.multiscale_max % 32 != 0)
            throw std::runtime_error("--multiscale-min/--multiscale-max must be positive multiples of 32 with min <= max");
    }
    if (options.multiscale_interval < 1) throw std::runtime_error("invalid multiscale interval");
    if (options.loader_threads < 1) throw std::runtime_error("--loader-threads must be at least 1");
    return options;
}

// Converts one [0,1] RGB pixel to HSV (h,s,v each in [0,1]); h is undefined (set to 0) for gray pixels.
void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
    float max_c = std::max({r, g, b}), min_c = std::min({r, g, b}), delta = max_c - min_c;
    v = max_c;
    s = max_c > 0.0f ? delta / max_c : 0.0f;
    if (delta < 1e-8f) { h = 0.0f; return; }
    if (max_c == r) h = std::fmod((g - b) / delta, 6.0f);
    else if (max_c == g) h = (b - r) / delta + 2.0f;
    else h = (r - g) / delta + 4.0f;
    h /= 6.0f;
    if (h < 0.0f) h += 1.0f;
}

// Inverse of rgb_to_hsv: converts HSV (h,s,v in [0,1], h wrapped) back to [0,1] RGB.
void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    float hh = h * 6.0f;
    int i = ((static_cast<int>(std::floor(hh)) % 6) + 6) % 6;
    float f = hh - std::floor(hh);
    float p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

// Jitters hue (circular shift), saturation and value (brightness) in HSV space, approximating
// Darknet's hue/saturation/exposure augmentation triple; identity when all deltas are zero.
void jitter_hsv(std::vector<float>& chw, const detector::Shape& shape, float hue_delta,
                float saturation_factor, float value_factor) {
    int area = shape.h * shape.w;
    for (int p = 0; p < area; ++p) {
        float h, s, v;
        rgb_to_hsv(chw[p], chw[area + p], chw[2 * area + p], h, s, v);
        h += hue_delta;
        h -= std::floor(h);
        s = std::clamp(s * saturation_factor, 0.0f, 1.0f);
        v = std::clamp(v * value_factor, 0.0f, 1.0f);
        float r, g, b;
        hsv_to_rgb(h, s, v, r, g, b);
        chw[p] = std::clamp(r, 0.0f, 1.0f);
        chw[area + p] = std::clamp(g, 0.0f, 1.0f);
        chw[2 * area + p] = std::clamp(b, 0.0f, 1.0f);
    }
}

// Turns pre-drawn per-edge crop fractions (left, right, top, bottom; each in [0, jitter)) into a
// crop rectangle within [0,width)x[0,height) (Darknet's "jitter" aspect/scale augmentation,
// crop-only variant).
struct CropRect { int x0, y0, w, h; };
CropRect crop_rect_from(int width, int height, const std::array<float, 4>& fractions) {
    int left = static_cast<int>(fractions[0] * width), right = static_cast<int>(fractions[1] * width);
    int top = static_cast<int>(fractions[2] * height), bottom = static_cast<int>(fractions[3] * height);
    int w = std::max(1, width - left - right), h = std::max(1, height - top - bottom);
    left = std::min(left, width - w);
    top = std::min(top, height - h);
    return {left, top, w, h};
}

// Remaps a target's box (normalized to the pre-crop image) into crop-relative normalization,
// clipping to the crop's bounds; returns false (leaving `target` unspecified) if the box's
// center falls outside the crop, i.e. the object was cropped out.
bool crop_target(detector::Target& target, const CropRect& crop, int orig_w, int orig_h) {
    float x1 = (target.cx - target.width / 2) * orig_w, x2 = (target.cx + target.width / 2) * orig_w;
    float y1 = (target.cy - target.height / 2) * orig_h, y2 = (target.cy + target.height / 2) * orig_h;
    x1 = std::clamp(x1, static_cast<float>(crop.x0), static_cast<float>(crop.x0 + crop.w));
    x2 = std::clamp(x2, static_cast<float>(crop.x0), static_cast<float>(crop.x0 + crop.w));
    y1 = std::clamp(y1, static_cast<float>(crop.y0), static_cast<float>(crop.y0 + crop.h));
    y2 = std::clamp(y2, static_cast<float>(crop.y0), static_cast<float>(crop.y0 + crop.h));
    if (x2 <= x1 || y2 <= y1) return false;
    float cx = (x1 + x2) / 2 - crop.x0, cy = (y1 + y2) / 2 - crop.y0;
    if (cx < 0 || cx > crop.w || cy < 0 || cy > crop.h) return false;
    target.cx = cx / crop.w;
    target.cy = cy / crop.h;
    target.width = (x2 - x1) / crop.w;
    target.height = (y2 - y1) / crop.h;
    return true;
}

// Mirrors the letterboxed CHW buffer left-right in place (one row-reverse per channel/row).
void flip_horizontal(std::vector<float>& chw, const detector::Shape& shape) {
    for (int c = 0; c < shape.c; ++c)
        for (int y = 0; y < shape.h; ++y) {
            auto* row = chw.data() + (static_cast<std::size_t>(c) * shape.h + y) * shape.w;
            std::reverse(row, row + shape.w);
        }
}

// Parses YOLO-format labels, normalized to whatever image they describe (before any crop/letterbox).
std::vector<detector::Target> raw_labels_from(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("missing label file: " + path.string());
    std::vector<detector::Target> targets;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream stream(line);
        detector::Target target;
        std::string extra;
        if (!(stream >> target.class_id >> target.cx >> target.cy >> target.width >> target.height) || (stream >> extra))
            throw std::runtime_error("malformed YOLO label in " + path.string());
        targets.push_back(target);
    }
    return targets;
}

// Converts a box normalized to the pre-letterbox image into network letterbox normalization.
void letterbox_target(detector::Target& target, const detector::Letterbox& box, detector::Shape network_shape) {
    target.cx = (target.cx * box.resized_width + box.left) / network_shape.w;
    target.cy = (target.cy * box.resized_height + box.top) / network_shape.h;
    target.width *= static_cast<float>(box.resized_width) / network_shape.w;
    target.height *= static_cast<float>(box.resized_height) / network_shape.h;
}

// One image's augmentation draws, taken on the planning thread in a fixed order so the number
// of loader threads never changes which minibatches a given seed produces.
struct SamplePlan {
    fs::path image;
    bool crop = false;
    std::array<float, 4> crop_fractions{};  // left, right, top, bottom
    bool flip = false;
    bool color = false;
    float hue = 0.0f, saturation = 1.0f, exposure = 1.0f;
};

// A minibatch's sample plans plus the network input shape its images are letterboxed to (the
// shape changes between minibatches only under multiscale training).
struct BatchPlan {
    std::vector<SamplePlan> samples;
    detector::Shape shape;
};

// A prepared minibatch, ready for Trainer/CudaTrainer::step().
struct Batch {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<detector::Target>> targets;
    detector::Shape shape;
};

// Loads one image and its labels, then applies the planned crop, letterbox, flip and HSV jitter.
// Touches no shared state, so samples can be prepared concurrently.
void prepare_sample(const SamplePlan& plan, const fs::path& labels_dir, detector::Shape shape,
                    std::vector<float>& input, std::vector<detector::Target>& labels) {
    auto image = detector::read_image(plan.image.string());
    labels = raw_labels_from(labels_dir / (plan.image.stem().string() + ".txt"));
    if (plan.crop) {
        int orig_w = image.width, orig_h = image.height;
        auto rect = crop_rect_from(orig_w, orig_h, plan.crop_fractions);
        image = detector::crop(image, rect.x0, rect.y0, rect.w, rect.h);
        std::vector<detector::Target> kept;
        for (auto label : labels) if (crop_target(label, rect, orig_w, orig_h)) kept.push_back(label);
        labels = std::move(kept);
    }
    auto box = detector::letterbox(image, shape);
    for (auto& label : labels) letterbox_target(label, box, shape);
    if (plan.flip) {
        flip_horizontal(box.chw, shape);
        for (auto& label : labels) label.cx = 1.0f - label.cx;
    }
    if (plan.color) jitter_hsv(box.chw, shape, plan.hue, plan.saturation, plan.exposure);
    input = std::move(box.chw);
}

// Prepares every sample of `plan` on up to `threads` threads (each fills a disjoint strided
// subset of slots, so no locking); rethrows the first worker error after all threads join.
Batch prepare_batch(const BatchPlan& plan, const fs::path& labels_dir, int threads) {
    Batch batch;
    const std::size_t n = plan.samples.size();
    batch.inputs.resize(n);
    batch.targets.resize(n);
    batch.shape = plan.shape;
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(static_cast<std::size_t>(threads), n));
    std::vector<std::exception_ptr> errors(workers);
    auto work = [&](std::size_t t) {
        try {
            for (std::size_t i = t; i < n; i += workers)
                prepare_sample(plan.samples[i], labels_dir, plan.shape, batch.inputs[i], batch.targets[i]);
        } catch (...) {
            errors[t] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& thread : pool) thread.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    return batch;
}

// Shared epoch loop for both Trainer (CPU) and CudaTrainer (GPU): same data pipeline (random
// crop, letterbox, flip/HSV augmentation, labels), differing only in which trainer type steps
// and whether its device-resident weights need syncing back to `network` before each checkpoint.
// Samples are assembled into minibatches of `options.batch_size` (a smaller final batch per
// epoch is fine) so step() computes true cross-sample batch-norm statistics, not per-sample ones.
// The next minibatch is decoded/augmented on `options.loader_threads` threads while the current
// one trains (Darknet's background load_data pattern); every random draw stays on this thread,
// one minibatch ahead, so results are identical for any loader thread count.
// `start_epoch` is the number of epochs already completed by a prior (crashed/interrupted) run
// being resumed via --resume-from (0 for a fresh run); the loop continues toward the same
// --epochs target total rather than restarting the epoch count from scratch. Per-epoch image
// order is freshly shuffled regardless, so no image-in-epoch position needs to be restored -
// only the completed-epoch count and (via the trainer's seeded iteration) the LR schedule step.
template <typename TrainerT>
void train_loop(TrainerT& trainer, detector::Network& network, std::vector<fs::path>& images,
                const fs::path& labels_dir, const fs::path& output_path, const Options& options,
                std::mt19937& rng, bool sync_before_checkpoint, int start_epoch = 0) {
    std::bernoulli_distribution flip_coin(options.flip_probability);
    std::uniform_real_distribution<float> exposure_dist(1.0f - options.exposure_jitter, 1.0f + options.exposure_jitter);
    std::uniform_real_distribution<float> saturation_dist(1.0f - options.saturation_jitter, 1.0f + options.saturation_jitter);
    std::uniform_real_distribution<float> hue_dist(-options.hue_jitter, options.hue_jitter);
    std::uniform_real_distribution<float> crop_fraction(0.0f, options.crop_jitter);
    const bool crop = options.crop_jitter > 0.0f;
    const bool color = options.exposure_jitter > 0.0f || options.saturation_jitter > 0.0f || options.hue_jitter > 0.0f;
    // Multiscale resize-jitter: a single random multiple-of-32 dimension applied to both width
    // and height (Darknet always resizes to a square), redrawn every `multiscale_interval`
    // minibatches across the whole run (the planned-batch counter is never reset per epoch).
    const bool multiscale = options.multiscale_max > 0;
    std::uniform_int_distribution<int> multiscale_dist(options.multiscale_min / 32, options.multiscale_max / 32);
    detector::Shape planned_shape = network.input_shape();
    long long planned_batches = 0;
    const std::size_t batch_size = static_cast<std::size_t>(options.batch_size);
    auto plan_batch = [&](std::size_t begin) {
        if (multiscale && planned_batches > 0 && planned_batches % options.multiscale_interval == 0)
            planned_shape.w = planned_shape.h = multiscale_dist(rng) * 32;
        ++planned_batches;
        BatchPlan plan;
        plan.shape = planned_shape;
        for (std::size_t i = begin; i < std::min(images.size(), begin + batch_size); ++i) {
            SamplePlan sample;
            sample.image = images[i];
            sample.crop = crop;
            if (crop) for (auto& fraction : sample.crop_fractions) fraction = crop_fraction(rng);
            sample.flip = flip_coin(rng);
            sample.color = color;
            if (color) {
                sample.hue = hue_dist(rng);
                sample.saturation = saturation_dist(rng);
                sample.exposure = exposure_dist(rng);
            }
            plan.samples.push_back(std::move(sample));
        }
        return plan;
    };
    auto launch = [&](std::size_t begin) {
        return std::async(std::launch::async, prepare_batch, plan_batch(begin), std::cref(labels_dir),
                          options.loader_threads);
    };
    for (int epoch = start_epoch + 1; epoch <= options.epochs; ++epoch) {
        std::shuffle(images.begin(), images.end(), rng);
        double loss_sum = 0;
        std::size_t positives = 0;
        auto pending = launch(0);
        for (std::size_t begin = 0; begin < images.size(); begin += batch_size) {
            Batch batch = pending.get();
            if (begin + batch_size < images.size()) pending = launch(begin + batch_size);
            if (batch.shape.w != network.input_shape().w || batch.shape.h != network.input_shape().h)
                trainer.resize(batch.shape.w, batch.shape.h);
            auto result = trainer.step(batch.inputs, batch.targets);
            loss_sum += result.loss;
            positives += result.positives;
        }
        if (sync_before_checkpoint) trainer.sync_to_network();
        network.save_weights(output_path.string());
        // Sidecar "<iteration> <completed epochs>", read back by a later --resume-from run so the
        // LearningRateSchedule and epoch loop continue exactly instead of restarting at iteration/
        // epoch 0. Not part of the Darknet weights format itself, so it never affects
        // compatibility with external Darknet.
        std::ofstream(output_path.string() + ".step") << trainer.iteration() << ' ' << epoch << '\n';
        std::cout << "epoch=" << epoch << " mean_loss=" << loss_sum / images.size()
                  << " positives=" << positives << " lr=" << trainer.current_learning_rate()
                  << " checkpoint=" << output_path.string() << '\n';
    }
}

std::vector<fs::path> images_from(const fs::path& directory) {
    if (!fs::is_directory(directory)) throw std::runtime_error("not an image directory: " + directory.string());
    std::vector<fs::path> images;
    for (const auto& item : fs::directory_iterator(directory)) {
        if (!item.is_regular_file()) continue;
        std::string extension = item.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".ppm" || extension == ".jpg" || extension == ".jpeg" || extension == ".png")
            images.push_back(item.path());
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) throw std::runtime_error("no images found in " + directory.string());
    return images;
}

}  // namespace

// CLI entry point: builds the network, initializes or resumes weights, and trains on the CPU,
// one GPU or several GPUs; returns 0 on success, 1 on error, 2 on a usage error.
int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            std::cerr << "Usage: cuda_detector_train <cfg> <images_dir> <labels_dir> <output.weights> "
                         "[--init weights] [--init-layers N] [--resume-from weights] [--epochs 1] [--lr 0.0001] "
                         "[--decay 0.0005] [--momentum 0.9] [--seed 2026] [--limit N] "
                         "[--batch 1] [--burn-in 0] [--steps s1,s2,...] [--scales f1,f2,...] "
                         "[--flip-probability 0.0] [--exposure-jitter 0.0] [--saturation-jitter 0.0] "
                         "[--hue-jitter 0.0] [--crop-jitter 0.0] [--backend cpu|cuda] [--gpus 0,1] "
                         "[--multiscale-min N] [--multiscale-max N] [--multiscale-interval 10] "
                         "[--loader-threads N]\n"
                         "  --resume-from: full-loads <weights> and its sidecar \"<weights>.step\" (written "
                         "alongside every checkpoint save) to continue the LR schedule/epoch loop exactly, "
                         "instead of restarting it at iteration 0 like --init does; mutually exclusive with "
                         "--init/--init-layers.\n";
            return 2;
        }
        auto options = options_from(argc, argv);
        fs::path output_path = argv[4];
        if (fs::exists(output_path)) throw std::runtime_error("refusing to overwrite existing output: " + output_path.string());
        auto network = detector::Network::from_cfg(argv[1]);
        int initial_iteration = 0;
        int start_epoch = 0;
        if (!options.resume_from.empty()) {
            network.load_weights(options.resume_from);
            fs::path step_path = options.resume_from + ".step";
            std::ifstream step_file(step_path);
            long long saved_step = -1, saved_epoch = -1;
            if (!step_file || !(step_file >> saved_step >> saved_epoch) || saved_step < 0 || saved_epoch < 0)
                throw std::runtime_error("--resume-from requires a valid sidecar step file: " + step_path.string());
            initial_iteration = static_cast<int>(saved_step);
            start_epoch = static_cast<int>(saved_epoch);
            if (start_epoch >= options.epochs)
                throw std::runtime_error("nothing to resume: checkpoint already completed " +
                                          std::to_string(start_epoch) + " of --epochs " + std::to_string(options.epochs));
        } else if (options.initial_weights.empty()) {
            network.initialize_random(options.seed);
        } else if (options.init_layers < 0) {
            network.load_weights(options.initial_weights);
        } else {
            // Pretrained-backbone transfer: random-init every layer first (so shape-diverging
            // head layers get a valid init), then overlay the pretrained file's first N layers.
            network.initialize_random(options.seed);
            network.load_weights(options.initial_weights, options.init_layers);
        }
        detector::LearningRateSchedule schedule{options.learning_rate, options.burn_in, 4.0f,
                                                 options.steps, options.scales};
        auto images = images_from(argv[2]);
        if (options.limit > 0 && images.size() > static_cast<std::size_t>(options.limit)) images.resize(options.limit);
        fs::path labels_dir = argv[3];
        std::mt19937 rng(options.seed);
        std::cout << "training_backend=" << options.backend << " images=" << images.size()
                  << " epochs=" << options.epochs << " batch=" << options.batch_size << " frozen_batchnorm=0"
                  << " flip_probability=" << options.flip_probability
                  << " exposure_jitter=" << options.exposure_jitter
                  << " saturation_jitter=" << options.saturation_jitter
                  << " hue_jitter=" << options.hue_jitter
                  << " crop_jitter=" << options.crop_jitter << " momentum=" << options.momentum
                  << " multiscale_min=" << options.multiscale_min << " multiscale_max=" << options.multiscale_max
                   << " multiscale_interval=" << options.multiscale_interval
                   << " loader_threads=" << options.loader_threads
                   << " gpu_count=" << (options.backend == "cuda" ? options.gpus.size() : 0) << "\n";
        if (options.backend == "cuda") {
#ifdef DETECTOR_HAS_CUDA
            if (options.gpus.size() == 1) {
                detector::CudaTrainer trainer(network, schedule, options.batch_size, options.decay, options.momentum,
                                               0.99f, initial_iteration, options.gpus[0]);
                train_loop(trainer, network, images, labels_dir, output_path, options, rng,
                           /*sync_before_checkpoint=*/true, start_epoch);
            } else {
                detector::MultiCudaTrainer trainer(network, schedule, options.batch_size, options.gpus,
                                                    options.decay, options.momentum, 0.99f, initial_iteration);
                train_loop(trainer, network, images, labels_dir, output_path, options, rng,
                           /*sync_before_checkpoint=*/true, start_epoch);
            }
#else
            throw std::runtime_error("--backend cuda requires a build with -DDETECTOR_CUDA=ON");
#endif
        } else {
            detector::Trainer trainer(network, schedule, options.batch_size, options.decay, options.momentum,
                                       0.99f, initial_iteration);
            train_loop(trainer, network, images, labels_dir, output_path, options, rng, /*sync_before_checkpoint=*/false,
                       start_epoch);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
