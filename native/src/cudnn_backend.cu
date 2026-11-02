#include "detector/cudnn_backend.hpp"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace detector {
namespace {

void check(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
    }
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// Owns one cuDNN resource and frees it (if non-null) on scope exit - avoids leaking
// descriptors/memory on an exception thrown mid-setup by any `check()` above it.
template <typename Handle, cudnnStatus_t (*Destroy)(Handle)>
struct Scoped {
    Handle handle = nullptr;
    ~Scoped() { if (handle) Destroy(handle); }
};

/** Largest workspace a benchmarked algorithm may need: 256 MiB or a quarter of free memory. */
std::size_t workspace_budget() {
    std::size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return std::min<std::size_t>(std::size_t{256} << 20, free_bytes / 4);
}

/** Device-wide scratch shared by every convolution (layers run one at a time on one stream),
 *  grown on demand; cudaFree synchronizes the device, so growth never frees a buffer an
 *  in-flight convolution still reads. Kept for the process lifetime. */
void* shared_workspace(std::size_t bytes) {
    static std::mutex mutex;
    static std::map<int, std::pair<void*, std::size_t>> by_device;
    if (bytes == 0) return nullptr;
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = by_device[device];
    if (entry.second < bytes) {
        if (entry.first) check(cudaFree(entry.first), "cudaFree(cuDNN workspace)");
        entry = {nullptr, 0};
        check(cudaMalloc(&entry.first, bytes), "cudaMalloc(cuDNN workspace)");
        entry.second = bytes;
    }
    return entry.first;
}

/** Fastest successful full-FP32 algorithm (cuDNN returns results sorted by measured time)
 *  whose workspace fits `budget`; tensor-op (TF32/FP16) variants are excluded. */
template <typename Perf>
Perf fastest(const Perf* results, int count, std::size_t budget, const char* direction) {
    for (int i = 0; i < count; ++i)
        if (results[i].status == CUDNN_STATUS_SUCCESS && results[i].memory <= budget &&
            results[i].mathType != CUDNN_TENSOR_OP_MATH && results[i].mathType != CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)
            return results[i];
    throw std::runtime_error(std::string("no cuDNN ") + direction + " convolution algorithm fits the workspace budget");
}

}  // namespace

int cudnn_runtime_version() {
    cudnnHandle_t handle = nullptr;
    check(cudnnCreate(&handle), "cudnnCreate");
    int version = static_cast<int>(cudnnGetVersion());
    check(cudnnDestroy(handle), "cudnnDestroy");
    return version;
}

void cudnn_convolution_forward(const float* input, const float* weights, float* output,
                                int in_channels, int in_h, int in_w, int out_channels,
                                int kernel, int stride, int pad) {
    const int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    const int out_w = (in_w + 2 * pad - kernel) / stride + 1;

    Scoped<cudnnHandle_t, cudnnDestroy> handle;
    check(cudnnCreate(&handle.handle), "cudnnCreate");

    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> input_desc;
    check(cudnnCreateTensorDescriptor(&input_desc.handle), "cudnnCreateTensorDescriptor(input)");
    check(cudnnSetTensor4dDescriptor(input_desc.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                      1, in_channels, in_h, in_w), "cudnnSetTensor4dDescriptor(input)");

    Scoped<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor> filter_desc;
    check(cudnnCreateFilterDescriptor(&filter_desc.handle), "cudnnCreateFilterDescriptor");
    check(cudnnSetFilter4dDescriptor(filter_desc.handle, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                       out_channels, in_channels, kernel, kernel), "cudnnSetFilter4dDescriptor");

    Scoped<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor> conv_desc;
    check(cudnnCreateConvolutionDescriptor(&conv_desc.handle), "cudnnCreateConvolutionDescriptor");
    check(cudnnSetConvolution2dDescriptor(conv_desc.handle, pad, pad, stride, stride, 1, 1,
                                            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");

    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> output_desc;
    check(cudnnCreateTensorDescriptor(&output_desc.handle), "cudnnCreateTensorDescriptor(output)");
    check(cudnnSetTensor4dDescriptor(output_desc.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                       1, out_channels, out_h, out_w), "cudnnSetTensor4dDescriptor(output)");

    const cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    std::size_t workspace_bytes = 0;
    check(cudnnGetConvolutionForwardWorkspaceSize(handle.handle, input_desc.handle, filter_desc.handle,
                                                    conv_desc.handle, output_desc.handle, algo, &workspace_bytes),
          "cudnnGetConvolutionForwardWorkspaceSize");
    void* workspace = nullptr;
    if (workspace_bytes > 0) check(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)");

    const float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t status = cudnnConvolutionForward(handle.handle, &alpha, input_desc.handle, input,
                                                     filter_desc.handle, weights, conv_desc.handle, algo,
                                                     workspace, workspace_bytes, &beta, output_desc.handle, output);
    if (workspace) cudaFree(workspace);
    check(status, "cudnnConvolutionForward");
}

/** Owns the descriptors and benchmarked algorithms reused by convolution forward and backward. */
struct CudnnConvolution::Impl {
    /** Algorithms measured fastest (cudnnFind*, like PyTorch's benchmark mode) for one batch size;
     *  backward ones are found on first use so inference-only sessions never pay for them. */
    struct Algorithms {
        cudnnConvolutionFwdAlgo_t forward{};
        std::size_t forward_bytes = 0;
        bool backward_ready = false;
        cudnnConvolutionBwdFilterAlgo_t filter{};
        cudnnConvolutionBwdDataAlgo_t data{};
        std::size_t filter_bytes = 0, data_bytes = 0;
    };
    Scoped<cudnnHandle_t, cudnnDestroy> handle;
    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> input;
    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> output;
    Scoped<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor> filter;
    Scoped<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor> convolution;
    std::map<int, Algorithms> algorithms;  // keyed by minibatch size
    Algorithms* active = nullptr;
    int in_channels, in_height, in_width, out_channels, out_height, out_width;
    int batch_size = 0;

    /** Creates NCHW descriptors; the active minibatch size is set on demand. */
    Impl(int in_c, int in_h, int in_w, int out_c, int kernel, int stride, int pad)
        : in_channels(in_c), in_height(in_h), in_width(in_w), out_channels(out_c),
          out_height(0), out_width(0) {
        if (in_c < 1 || in_h < 1 || in_w < 1 || out_c < 1 || kernel < 1 || stride < 1 || pad < 0)
            throw std::runtime_error("invalid cuDNN convolution shape");
        out_height = (in_h + 2 * pad - kernel) / stride + 1;
        out_width = (in_w + 2 * pad - kernel) / stride + 1;
        if (out_height < 1 || out_width < 1) throw std::runtime_error("invalid cuDNN convolution output shape");
        check(cudnnCreate(&handle.handle), "cudnnCreate");
        check(cudnnCreateTensorDescriptor(&input.handle), "cudnnCreateTensorDescriptor(input)");
        check(cudnnCreateTensorDescriptor(&output.handle), "cudnnCreateTensorDescriptor(output)");
        check(cudnnCreateFilterDescriptor(&filter.handle), "cudnnCreateFilterDescriptor");
        check(cudnnCreateConvolutionDescriptor(&convolution.handle), "cudnnCreateConvolutionDescriptor");
        check(cudnnSetFilter4dDescriptor(filter.handle, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          out_c, in_c, kernel, kernel), "cudnnSetFilter4dDescriptor");
        check(cudnnSetConvolution2dDescriptor(convolution.handle, pad, pad, stride, stride, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
              "cudnnSetConvolution2dDescriptor");
        check(cudnnSetConvolutionMathType(convolution.handle, CUDNN_FMA_MATH),
              "cudnnSetConvolutionMathType");
    }

    /** Points the tensor descriptors at a minibatch of `n` and selects that size's algorithms,
     *  benchmarking the forward algorithm the first time `n` is seen. */
    void set_batch(int n) {
        if (n < 1) throw std::runtime_error("invalid cuDNN convolution batch size");
        if (n == batch_size) return;
        check(cudnnSetTensor4dDescriptor(input.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              n, in_channels, in_height, in_width), "cudnnSetTensor4dDescriptor(input)");
        check(cudnnSetTensor4dDescriptor(output.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              n, out_channels, out_height, out_width), "cudnnSetTensor4dDescriptor(output)");
        batch_size = n;
        auto found = algorithms.find(n);
        if (found == algorithms.end()) {
            cudnnConvolutionFwdAlgoPerf_t results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            int count = 0;
            check(cudnnFindConvolutionForwardAlgorithm(handle.handle, input.handle, filter.handle,
                  convolution.handle, output.handle, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &count, results),
                  "cudnnFindConvolutionForwardAlgorithm");
            const auto best = fastest(results, count, workspace_budget(), "forward");
            Algorithms chosen;
            chosen.forward = best.algo;
            chosen.forward_bytes = best.memory;
            found = algorithms.emplace(n, chosen).first;
        }
        active = &found->second;
    }

    /** Benchmarks the backward filter/data algorithms for the active batch size on first use. */
    void prepare_backward() {
        if (active->backward_ready) return;
        const std::size_t budget = workspace_budget();
        cudnnConvolutionBwdFilterAlgoPerf_t filters[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
        int count = 0;
        check(cudnnFindConvolutionBackwardFilterAlgorithm(handle.handle, input.handle, output.handle,
              convolution.handle, filter.handle, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &count, filters),
              "cudnnFindConvolutionBackwardFilterAlgorithm");
        const auto best_filter = fastest(filters, count, budget, "backward-filter");
        cudnnConvolutionBwdDataAlgoPerf_t datas[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
        check(cudnnFindConvolutionBackwardDataAlgorithm(handle.handle, filter.handle, output.handle,
              convolution.handle, input.handle, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &count, datas),
              "cudnnFindConvolutionBackwardDataAlgorithm");
        const auto best_data = fastest(datas, count, budget, "backward-data");
        active->filter = best_filter.algo;
        active->filter_bytes = best_filter.memory;
        active->data = best_data.algo;
        active->data_bytes = best_data.memory;
        active->backward_ready = true;
    }
};

CudnnConvolution::CudnnConvolution(int in_c, int in_h, int in_w, int out_c,
                                   int kernel, int stride, int pad)
    : impl_(std::make_unique<Impl>(in_c, in_h, in_w, out_c, kernel, stride, pad)) {}
CudnnConvolution::~CudnnConvolution() = default;

void CudnnConvolution::forward(const float* input, const float* weights, float* output, int n) {
    impl_->set_batch(n);
    const auto& algorithms = *impl_->active;
    const float one = 1.0f, zero = 0.0f;
    check(cudnnConvolutionForward(impl_->handle.handle, &one, impl_->input.handle, input,
          impl_->filter.handle, weights, impl_->convolution.handle, algorithms.forward,
          shared_workspace(algorithms.forward_bytes), algorithms.forward_bytes,
          &zero, impl_->output.handle, output), "cudnnConvolutionForward");
}

void CudnnConvolution::backward_filter(const float* input, const float* dy, float* dw, int n) {
    impl_->set_batch(n);
    impl_->prepare_backward();
    const auto& algorithms = *impl_->active;
    const float one = 1.0f;
    check(cudnnConvolutionBackwardFilter(impl_->handle.handle, &one, impl_->input.handle, input,
          impl_->output.handle, dy, impl_->convolution.handle, algorithms.filter,
          shared_workspace(algorithms.filter_bytes), algorithms.filter_bytes, &one, impl_->filter.handle, dw),
          "cudnnConvolutionBackwardFilter");
}

void CudnnConvolution::backward_data(const float* weights, const float* dy, float* dx, int n) {
    impl_->set_batch(n);
    impl_->prepare_backward();
    const auto& algorithms = *impl_->active;
    const float one = 1.0f;
    check(cudnnConvolutionBackwardData(impl_->handle.handle, &one, impl_->filter.handle, weights,
          impl_->output.handle, dy, impl_->convolution.handle, algorithms.data,
          shared_workspace(algorithms.data_bytes), algorithms.data_bytes, &one, impl_->input.handle, dx),
          "cudnnConvolutionBackwardData");
}

/** Owns spatial batch-normalization descriptors for a fixed channel/spatial shape. */
struct CudnnBatchNorm::Impl {
    Scoped<cudnnHandle_t, cudnnDestroy> handle;
    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> tensor;
    Scoped<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor> parameters;
    int channels, height, width;

    /** Creates reusable descriptors; the actual minibatch size is set per call. */
    Impl(int c, int h, int w) : channels(c), height(h), width(w) {
        if (c < 1 || h < 1 || w < 1) throw std::runtime_error("invalid cuDNN batch-norm shape");
        check(cudnnCreate(&handle.handle), "cudnnCreate(batch norm)");
        check(cudnnCreateTensorDescriptor(&tensor.handle), "cudnnCreateTensorDescriptor(batch norm)");
        check(cudnnCreateTensorDescriptor(&parameters.handle), "cudnnCreateTensorDescriptor(BN parameters)");
        set_batch(1);
        check(cudnnDeriveBNTensorDescriptor(parameters.handle, tensor.handle, CUDNN_BATCHNORM_SPATIAL),
              "cudnnDeriveBNTensorDescriptor");
    }

    /** Updates only the sample dimension of the NCHW tensor descriptor. */
    void set_batch(int n) {
        if (n < 1) throw std::runtime_error("invalid cuDNN batch-norm batch size");
        check(cudnnSetTensor4dDescriptor(tensor.handle, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
              n, channels, height, width), "cudnnSetTensor4dDescriptor(batch norm)");
    }
};

CudnnBatchNorm::CudnnBatchNorm(int c, int h, int w) : impl_(std::make_unique<Impl>(c, h, w)) {}
CudnnBatchNorm::~CudnnBatchNorm() = default;

void CudnnBatchNorm::forward(const float* raw, const float* scale, const float* bias, float* output,
                             float* running_mean, float* running_variance, float* saved_mean,
                             float* saved_inv_variance, int n) const {
    impl_->set_batch(n);
    const float one = 1.0f, zero = 0.0f;
    check(cudnnBatchNormalizationForwardTraining(impl_->handle.handle, CUDNN_BATCHNORM_SPATIAL,
          &one, &zero, impl_->tensor.handle, raw, impl_->tensor.handle, output,
          impl_->parameters.handle, scale, bias, 0.0, running_mean, running_variance,
          1e-5, saved_mean, saved_inv_variance), "cudnnBatchNormalizationForwardTraining");
}

void CudnnBatchNorm::backward(const float* raw, const float* dy, const float* scale,
                              const float* saved_mean, const float* saved_inv_variance,
                              float* dx, float* dscale, float* dbias, int n) const {
    impl_->set_batch(n);
    const float one = 1.0f, zero = 0.0f;
    check(cudnnBatchNormalizationBackward(impl_->handle.handle, CUDNN_BATCHNORM_SPATIAL,
          &one, &zero, &one, &one, impl_->tensor.handle, raw, impl_->tensor.handle, dy,
          impl_->tensor.handle, dx, impl_->parameters.handle, scale, dscale, dbias,
          1e-5, saved_mean, saved_inv_variance), "cudnnBatchNormalizationBackward");
}

}  // namespace detector
