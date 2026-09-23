#pragma once

#include <cmath>

#include "micro/layer_norm.hpp"

namespace llm_bricks::detail {

inline Status layer_norm_ref(LayerNormParams& params) {
    const auto& input = params.input;
    const auto& gamma = params.gamma;
    const auto& beta = params.beta;
    auto& output = params.output;
    if (!input.ptr() || !gamma.ptr() || !beta.ptr() || !output.ptr() || input.dtype() != DType::f32 ||
        gamma.dtype() != DType::f32 || beta.dtype() != DType::f32 || output.dtype() != DType::f32 ||
        input.ndims() == 0 || input.shape() != output.shape() || params.epsilon <= 0.0F) {
        return Status::invalid_argument;
    }
    const size_t width = input.dim(input.ndims() - 1);
    if (gamma.numel() != width || beta.numel() != width) {
        return Status::invalid_argument;
    }
    const auto* src = static_cast<const float*>(input.ptr());
    const auto* g = static_cast<const float*>(gamma.ptr());
    const auto* b = static_cast<const float*>(beta.ptr());
    auto* dst = static_cast<float*>(output.ptr());
    for (size_t row = 0; row < input.numel() / width; ++row) {
        const float* x = src + row * width;
        float* y = dst + row * width;
        float mean = 0.0F;
        float variance = 0.0F;
        for (size_t i = 0; i < width; ++i) {
            mean += x[i];
        }
        mean /= static_cast<float>(width);
        for (size_t i = 0; i < width; ++i) {
            const float delta = x[i] - mean;
            variance += delta * delta;
        }
        const float inv_std = 1.0F / std::sqrt(variance / static_cast<float>(width) + params.epsilon);
        for (size_t i = 0; i < width; ++i) {
            y[i] = (x[i] - mean) * inv_std * g[i] + b[i];
        }
    }
    return Status::ok;
}

}  // namespace llm_bricks::detail