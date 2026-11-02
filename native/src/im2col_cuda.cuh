#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace detector {

/** Expands one CHW image into the (kernel position x output pixel) column matrix that a
 *  convolution multiplies by its weights, zero-filling padded taps. Shared by the CUDA inference
 *  session and the cuBLAS training path; `columns` is row-major [channels*kernel*kernel][oh*ow]. */
static __global__ void im2col_kernel(const float* input, float* columns, int channels, int ih, int iw,
                                     int oh, int ow, int kernel, int stride, int pad) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = oh * ow;
    int count = channels * kernel * kernel * plane;
    if (pos >= count) return;
    int pixel = pos % plane;
    int key = pos / plane;
    int kx = key % kernel, ky = key / kernel % kernel, ic = key / (kernel * kernel);
    int iy = pixel / ow * stride + ky - pad, ix = pixel % ow * stride + kx - pad;
    columns[pos] = iy >= 0 && ix >= 0 && iy < ih && ix < iw
        ? input[(static_cast<std::size_t>(ic) * ih + iy) * iw + ix] : 0.0f;
}

}  // namespace detector
