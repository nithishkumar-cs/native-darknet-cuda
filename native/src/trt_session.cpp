#include "detector/trt_session.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

struct Logger final : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

struct DeviceMemory {
    std::vector<void*> allocations;
    ~DeviceMemory() { for (void* pointer : allocations) cudaFree(pointer); }
    float* floats(std::size_t count) {
        void* pointer = nullptr;
        check(cudaMalloc(&pointer, count * sizeof(float)), "TensorRT cudaMalloc");
        allocations.push_back(pointer);
        return static_cast<float*>(pointer);
    }
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreate(&value), "TensorRT cudaStreamCreate"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};

void expect_tensor(const nvinfer1::ICudaEngine& engine, const char* name,
                   nvinfer1::TensorIOMode mode, Shape shape) {
    if (engine.getTensorIOMode(name) != mode ||
        engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(std::string("TensorRT tensor mode/type mismatch: ") + name);
    auto dims = engine.getTensorShape(name);
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != shape.c ||
        dims.d[2] != shape.h || dims.d[3] != shape.w)
        throw std::runtime_error(std::string("TensorRT tensor shape mismatch: ") + name);
}

std::vector<char> read_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open TensorRT engine: " + path);
    auto length = file.tellg();
    if (length <= 0 || length > 1'000'000'000)
        throw std::runtime_error("invalid TensorRT engine size");
    std::vector<char> data(static_cast<std::size_t>(length));
    file.seekg(0);
    if (!file.read(data.data(), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("cannot read TensorRT engine: " + path);
    return data;
}

}  // namespace

struct TrtSession::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    DeviceMemory memory;
    Stream stream;
    float* input_device = nullptr;
    std::vector<float*> head_devices;
    std::vector<std::pair<std::size_t, Shape>> heads;
    std::vector<std::string> head_names;

    Impl(const Network& network, const std::string& path) {
        for (std::size_t index = 0; index < network.layers().size(); ++index)
            if (network.layers()[index].kind == Kind::yolo || network.layers()[index].kind == Kind::region)
                heads.emplace_back(index, network.layers()[index].output);
        if (!heads.empty()) {
            for (std::size_t index = 0; index < heads.size(); ++index)
                head_names.push_back("head_" + std::to_string(index));
        } else {
            // Dense-prediction cfg (segmentation): single "logits" output from the last layer.
            if (network.layers().empty() ||
                (network.layers().back().kind != Kind::convolutional &&
                 network.layers().back().kind != Kind::deconvolutional))
                throw std::runtime_error("TensorRT backend expects a [yolo]/[region] head or a convolutional/deconvolutional final layer");
            heads.emplace_back(network.layers().size() - 1, network.layers().back().output);
            head_names.push_back("logits");
        }
        auto data = read_engine(path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("TensorRT runtime creation failed");
        engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
        if (!engine) throw std::runtime_error("TensorRT engine deserialization failed; rebuild on this Jetson");
        if (engine->getNbIOTensors() != static_cast<int>(heads.size()) + 1)
            throw std::runtime_error("TensorRT engine must have one input and one raw head output per [yolo]/[region] layer");
        expect_tensor(*engine, "images", nvinfer1::TensorIOMode::kINPUT, network.input_shape());
        for (std::size_t index = 0; index < heads.size(); ++index) {
            expect_tensor(*engine, head_names[index].c_str(), nvinfer1::TensorIOMode::kOUTPUT, heads[index].second);
        }
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("TensorRT execution context creation failed");
        input_device = memory.floats(network.input_shape().elements());
        if (!context->setTensorAddress("images", input_device))
            throw std::runtime_error("TensorRT input binding failed");
        for (std::size_t index = 0; index < heads.size(); ++index) {
            const auto& name = head_names[index];
            auto* pointer = memory.floats(heads[index].second.elements());
            head_devices.push_back(pointer);
            if (!context->setTensorAddress(name.c_str(), pointer))
                throw std::runtime_error("TensorRT output binding failed: " + name);
        }
    }

    std::vector<Head> forward(const std::vector<float>& host_input) {
        if (host_input.size() * sizeof(float) !=
            static_cast<std::size_t>(engine->getTensorShape("images").d[1] *
                                     engine->getTensorShape("images").d[2] *
                                     engine->getTensorShape("images").d[3]) * sizeof(float))
            throw std::runtime_error("TensorRT input does not match engine shape");
        check(cudaMemcpyAsync(input_device, host_input.data(), host_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream.value), "TensorRT input copy");
        if (!context->enqueueV3(stream.value))
            throw std::runtime_error("TensorRT enqueueV3 failed");
        std::vector<Head> result;
        result.reserve(heads.size());
        for (std::size_t index = 0; index < heads.size(); ++index) {
            Head head;
            head.layer = heads[index].first;
            head.values.resize(heads[index].second.elements());
            check(cudaMemcpyAsync(head.values.data(), head_devices[index],
                                  head.values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream.value), "TensorRT output copy");
            result.push_back(std::move(head));
        }
        check(cudaStreamSynchronize(stream.value), "TensorRT stream sync");
        return result;
    }
};

TrtSession::TrtSession(const Network& network, const std::string& engine_path)
    : impl_(std::make_unique<Impl>(network, engine_path)) {}
TrtSession::~TrtSession() = default;
std::vector<Head> TrtSession::forward(const std::vector<float>& input) {
    return impl_->forward(input);
}

}  // namespace detector
