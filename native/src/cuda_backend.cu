#include "detector/cuda_session.hpp"
#include "detector/cuda_recurrent.hpp"
#include "l2norm_cuda.cuh"
#include "normalization_cuda.cuh"
#include "local_cuda.cuh"
#include "deconv_cuda.cuh"
#include "avgpool_cuda.cuh"
#ifdef DETECTOR_HAS_CUDNN
#include "detector/cudnn_backend.hpp"
#else
#include "im2col_cuda.cuh"
#include <cublas_v2.h>
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace detector {
namespace {

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

#ifndef DETECTOR_HAS_CUDNN
/** Raises an exception when a cuBLAS operation fails. */
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}
#endif

struct DeviceMemory {
    std::vector<void*> allocations;
    ~DeviceMemory() { for (void* item : allocations) cudaFree(item); }
    float* floats(std::size_t count) {
        float* pointer = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(float)), "cudaMalloc");
        allocations.push_back(pointer);
        return pointer;
    }
    float* upload(const std::vector<float>& host) {
        float* pointer = floats(host.size());
        check(cudaMemcpy(pointer, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        return pointer;
    }
};

/** Applies the folded BN affine transform (or bias) and activation after a raw convolution,
 *  in one pass instead of separate normalize/scale/bias/activate kernels. */
__global__ void convolution_epilogue(float* values, const float* multiplier, const float* offset,
                                    int count, int plane, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    int channel = pos / plane;
    float value = values[pos] * multiplier[channel] + offset[channel];
    values[pos] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void max_pool(const float* input, float* output, int channels, int ih, int iw,
                         int oh, int ow, int kernel, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * oh * ow;
    if (pos >= total) return;
    int x = pos % ow;
    int y = pos / ow % oh;
    int c = pos / (ow * oh);
    int pad = (kernel - 1) / 2;
    float best = -3.402823466e+38f;
    for (int ky = 0; ky < kernel; ++ky)
        for (int kx = 0; kx < kernel; ++kx) {
            int iy = y * stride + ky - pad;
            int ix = x * stride + kx - pad;
            if (iy < 0 || ix < 0 || iy >= ih || ix >= iw) continue;
            best = fmaxf(best, input[(static_cast<std::size_t>(c) * ih + iy) * iw + ix]);
        }
    output[pos] = best;
}

__global__ void upsample(const float* input, float* output, int channels, int ih, int iw,
                         int oh, int ow, int stride) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= channels * oh * ow) return;
    int x = pos % ow;
    int y = pos / ow % oh;
    int c = pos / (ow * oh);
    output[pos] = input[(static_cast<std::size_t>(c) * ih + y / stride) * iw + x / stride];
}

__global__ void shortcut_add(const float* input, const float* branch, float* output, int count, bool leaky) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= count) return;
    float value = input[pos] + branch[pos];
    output[pos] = leaky && value < 0 ? value * 0.1f : value;
}

// Space-to-depth rearrangement (see network.cpp's reorg_forward for the shared convention this
// mirrors); one thread per OUTPUT element, each reading its one corresponding input position.
__global__ void reorg(const float* input, float* output, int channels, int ih, int iw, int stride) {
    int out_h = ih / stride, out_w = iw / stride, out_c = channels * stride * stride;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= out_c * out_h * out_w) return;
    int ox = pos % out_w, oy = pos / out_w % out_h, oc = pos / (out_w * out_h);
    int ic = oc / (stride * stride), offset = oc % (stride * stride);
    int ix = ox * stride + offset % stride, iy = oy * stride + offset / stride;
    output[pos] = input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix];
}

__global__ void connected(const float* input, const float* weights, const float* bias,
                          float* output, int inputs, int outputs, bool leaky) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= outputs) return;
    float value = bias[oc];
    for (int ic = 0; ic < inputs; ++ic)
        value += input[ic] * weights[static_cast<std::size_t>(oc) * inputs + ic];
    output[oc] = leaky && value < 0 ? value * 0.1f : value;
}

__global__ void softmax_vector(const float* input, float* output, int count) {
    if (blockIdx.x || threadIdx.x) return;
    float maximum = input[0];
    for (int i = 1; i < count; ++i) maximum = fmaxf(maximum, input[i]);
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        output[i] = expf(input[i] - maximum);
        total += output[i];
    }
    for (int i = 0; i < count; ++i) output[i] /= total;
}

void launch_check() { check(cudaGetLastError(), "CUDA kernel launch"); }

}  // namespace

/** Owns the uploaded graph and cached cuDNN convolution contexts for one inference session. */
struct CudaSession::Impl {
    struct ConvolutionParameters { float* weights = nullptr; float* scales = nullptr; float* offsets = nullptr; };
    const Network& network;
    DeviceMemory memory;
    float* input = nullptr;
    std::vector<float*> output;
    std::vector<ConvolutionParameters> parameters;
#ifdef DETECTOR_HAS_CUDNN
    std::vector<std::unique_ptr<CudnnConvolution>> cudnn_convolutions;
#else
    cublasHandle_t blas = nullptr;
    float* columns = nullptr;  // im2col scratch shared by every non-pointwise convolution
#endif

    /** Uploads graph buffers and prepares the cuDNN contexts or the cuBLAS handle and im2col scratch. */
    explicit Impl(const Network& net) : network(net) {
        int devices = 0;
        check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 1) throw std::runtime_error("no CUDA device available");
        input = memory.floats(network.input_shape().elements());
        const auto& layers = network.layers();
        output.resize(layers.size());
        parameters.resize(layers.size());
#ifdef DETECTOR_HAS_CUDNN
        cudnn_convolutions.resize(layers.size());
#endif
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            output[i] = memory.floats(layer.output.elements());
            if (layer.kind != Kind::convolutional && layer.kind != Kind::connected &&
                layer.kind != Kind::local && layer.kind != Kind::deconvolutional) continue;
            if (layer.biases.size() != (layer.kind == Kind::local ? layer.output.elements() :
                static_cast<std::size_t>(layer.filters)) || layer.weights.empty())
                throw std::runtime_error("load weights before constructing CudaSession");
            if (layer.kind == Kind::deconvolutional) {
                parameters[i] = {memory.upload(layer.weights), nullptr, memory.upload(layer.biases)};
                continue;
            }
            if (layer.kind == Kind::local) {
                parameters[i] = {memory.upload(layer.weights), nullptr, memory.upload(layer.biases)};
                continue;
            }
            std::vector<float> multiplier(static_cast<std::size_t>(layer.filters), 1.0f);
            std::vector<float> offsets = layer.biases;
            if (layer.kind == Kind::convolutional && layer.batch_normalize)
                for (int c = 0; c < layer.filters; ++c) {
                    multiplier[c] = layer.scales[c] / std::sqrt(layer.variances[c] + 1e-5f);
                    offsets[c] -= layer.means[c] * multiplier[c];
                }
            parameters[i] = {memory.upload(layer.weights), memory.upload(multiplier), memory.upload(offsets)};
#ifdef DETECTOR_HAS_CUDNN
            if (layer.kind == Kind::convolutional)
                cudnn_convolutions[i] = std::make_unique<CudnnConvolution>(layer.input.c, layer.input.h,
                    layer.input.w, layer.output.c, layer.size, layer.stride, layer.padding);
#endif
        }
#ifndef DETECTOR_HAS_CUDNN
        std::size_t column_capacity = 1;
        for (const auto& layer : layers)
            if (layer.kind == Kind::convolutional && !pointwise(layer))
                column_capacity = std::max(column_capacity, static_cast<std::size_t>(layer.input.c) * layer.size *
                                                            layer.size * layer.output.h * layer.output.w);
        columns = memory.floats(column_capacity);
        check(cublasCreate(&blas), "cublasCreate");
        check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
#endif
    }

#ifndef DETECTOR_HAS_CUDNN
    /** Releases the cuBLAS handle; device buffers are freed by `memory`. */
    ~Impl() { if (blas) cublasDestroy(blas); }

    /** True when a convolution's CHW input already is its im2col matrix (1x1, stride 1, no pad). */
    static bool pointwise(const Layer& layer) { return layer.size == 1 && layer.stride == 1 && layer.padding == 0; }
#endif

    /** Runs one input through the graph and downloads detection, classification, or plain terminal heads. */
    std::vector<Head> forward(const std::vector<float>& host_input) {
        if (host_input.size() != network.input_shape().elements()) throw std::runtime_error("input does not match cfg shape");
        check(cudaMemcpy(input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy input");
        const auto& layers = network.layers();
        std::vector<Head> heads;
        constexpr int threads = 256;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& layer = layers[index];
            const float* previous = index ? output[index - 1] : input;
            float* result = output[index];
            int elements = static_cast<int>(layer.output.elements());
            int blocks = (elements + threads - 1) / threads;
            if (layer.kind == Kind::deconvolutional) {
                const auto& param = parameters[index];
                deconv_forward_kernel<<<blocks, threads>>>(previous, param.weights, param.offsets,
                    result, elements, layer.input.c, layer.input.h, layer.input.w, layer.filters,
                    layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding,
                    layer.leaky, layer.logistic);
                launch_check();
            } else if (layer.kind == Kind::local) {
                const auto& param = parameters[index];
                local_forward_kernel<<<blocks, threads>>>(previous, param.weights, param.offsets,
                    result, elements, layer.input.c, layer.input.h, layer.input.w, layer.filters,
                    layer.output.h, layer.output.w, layer.size, layer.stride, layer.padding,
                    layer.leaky, layer.logistic);
                launch_check();
            } else if (layer.kind == Kind::convolutional) {
                const auto& param = parameters[index];
                const int plane = layer.output.h * layer.output.w;
#ifdef DETECTOR_HAS_CUDNN
                cudnn_convolutions[index]->forward(previous, param.weights, result);
#else
                // Darknet's GPU path: im2col + SGEMM (pointwise layers multiply the input directly).
                const int key_count = layer.input.c * layer.size * layer.size;
                const float* matrix = previous;
                if (!pointwise(layer)) {
                    im2col_kernel<<<(key_count * plane + threads - 1) / threads, threads>>>(previous, columns,
                        layer.input.c, layer.input.h, layer.input.w, layer.output.h, layer.output.w,
                        layer.size, layer.stride, layer.padding);
                    launch_check();
                    matrix = columns;
                }
                const float one = 1.0f, zero = 0.0f;
                check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, plane, layer.output.c, key_count,
                                  &one, matrix, plane, param.weights, key_count, &zero, result, plane),
                      "cublasSgemm convolution");
#endif
                convolution_epilogue<<<blocks, threads>>>(result, param.scales, param.offsets,
                    elements, plane, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::maxpool) {
                max_pool<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h,
                    layer.input.w, layer.output.h, layer.output.w, layer.size, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::avgpool) {
                avgpool_forward_kernel<<<layer.input.c, threads>>>(previous, result,
                    layer.input.c, layer.input.h * layer.input.w);
                launch_check();
            } else if (layer.kind == Kind::route) {
                std::size_t offset = 0;
                for (int branch : layer.sources) {
                    std::size_t count = layers[static_cast<std::size_t>(branch)].output.elements();
                    check(cudaMemcpy(result + offset, output[static_cast<std::size_t>(branch)],
                                     count * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy route");
                    offset += count;
                }
            } else if (layer.kind == Kind::upsample) {
                upsample<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h,
                    layer.input.w, layer.output.h, layer.output.w, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::shortcut) {
                shortcut_add<<<blocks, threads>>>(previous, output[static_cast<std::size_t>(layer.sources.front())],
                    result, elements, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::connected) {
                const auto& param = parameters[index];
                connected<<<blocks, threads>>>(previous, param.weights, param.offsets, result,
                    static_cast<int>(layer.input.elements()), layer.filters, layer.leaky);
                launch_check();
            } else if (layer.kind == Kind::dropout) {
                check(cudaMemcpy(result, previous, layer.output.elements() * sizeof(float), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy dropout");  // no-op at inference, matching Darknet
            } else if (layer.kind == Kind::l2norm) {
                l2norm_forward_kernel<<<1, threads>>>(previous, result, elements);
                launch_check();
            } else if (layer.kind == Kind::normalization) {
                normalization_forward_kernel<<<blocks, threads>>>(previous, result, nullptr, elements,
                    layer.input.c, layer.input.h * layer.input.w, layer.size,
                    layer.norm_alpha, layer.norm_beta, layer.norm_kappa);
                launch_check();
            } else if (layer.kind == Kind::reorg) {
                reorg<<<blocks, threads>>>(previous, result, layer.input.c, layer.input.h, layer.input.w, layer.stride);
                launch_check();
            } else if (layer.kind == Kind::softmax) {
                softmax_vector<<<1, 1>>>(previous, result, elements);
                launch_check();
                Head head;
                head.layer = index;
                head.values.resize(layer.output.elements());
                check(cudaMemcpy(head.values.data(), result, head.values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy softmax output");
                heads.push_back(std::move(head));
            } else if (layer.kind == Kind::rnn || layer.kind == Kind::lstm || layer.kind == Kind::gru) {
                auto* cell = network.cell_at(index);
                if (!cell) throw std::runtime_error("initialize or load recurrent weights before CUDA inference");
                std::vector<float> timestep(layer.input.elements());
                check(cudaMemcpy(timestep.data(), previous, timestep.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy recurrent input");
                auto state = recurrent_forward_cuda(*cell, {timestep}, cell->hidden_state(), cell->cell_state());
                cell->set_state(state.final_hidden, state.final_cell);
                check(cudaMemcpy(result, state.outputs.front().data(), elements * sizeof(float), cudaMemcpyHostToDevice),
                      "cudaMemcpy recurrent output");
            } else {
                check(cudaMemcpy(result, previous, layer.output.elements() * sizeof(float), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy YOLO head");
                Head head;
                head.layer = index;
                head.values.resize(layer.output.elements());
                check(cudaMemcpy(head.values.data(), result, head.values.size() * sizeof(float), cudaMemcpyDeviceToHost),
                      "cudaMemcpy YOLO output");
                heads.push_back(std::move(head));
            }
        }
        if (!layers.empty() && (heads.empty() || heads.back().layer != layers.size() - 1)) {
            const Kind kind = layers.back().kind;
            if (kind == Kind::connected || kind == Kind::convolutional || kind == Kind::local ||
                kind == Kind::deconvolutional || kind == Kind::avgpool || kind == Kind::l2norm ||
                kind == Kind::normalization) {
                Head head;
                head.layer = layers.size() - 1;
                head.values.resize(layers.back().output.elements());
                check(cudaMemcpy(head.values.data(), output.back(), head.values.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy plain terminal output");
                heads.push_back(std::move(head));
            }
        }
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        return heads;
    }
};

CudaSession::CudaSession(const Network& network) : impl_(std::make_unique<Impl>(network)) {}
CudaSession::~CudaSession() = default;
std::vector<Head> CudaSession::forward(const std::vector<float>& input) { return impl_->forward(input); }

std::vector<Head> forward_cuda(const Network& network, const std::vector<float>& input) {
    CudaSession session(network);
    return session.forward(input);
}

}  // namespace detector
