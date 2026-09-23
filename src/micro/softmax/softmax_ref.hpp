#pragma once

#include <algorithm>
#include <cmath>

#include "micro/softmax.hpp"

namespace llm_bricks::detail {

inline Status softmax_ref(SoftmaxParams& params) {
    const auto& input = params.input;
    auto& output = params.output;
    if (!input.ptr() || !output.ptr() || input.dtype() != DType::f32 || output.dtype() != DType::f32 ||
        input.ndims() == 0 || input.shape() != output.shape()) {
        return Status::invalid_argument;
    }
    const size_t width = input.dim(input.ndims() - 1);
    const auto* src = static_cast<const float*>(input.ptr());
    auto* dst = static_cast<float*>(output.ptr());
    for (size_t row = 0; row < input.numel() / width; ++row) {
        const float* x = src + row * width;
        float* y = dst + row * width;
        float maximum = x[0];
        for (size_t i = 1; i < width; ++i) {
            maximum = std::max(maximum, x[i]);
        }
        float sum = 0.0F;
        for (size_t i = 0; i < width; ++i) {
            y[i] = std::exp(x[i] - maximum);
            sum += y[i];
        }
        for (size_t i = 0; i < width; ++i) {
            y[i] /= sum;
        }
    }
    return Status::ok;
}

}  // namespace llm_bricks::detail