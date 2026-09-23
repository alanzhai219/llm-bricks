#pragma once

#include <cmath>

#include "micro/gelu.hpp"

namespace llm_bricks::detail {

inline Status gelu_ref(GeluParams& params) {
    auto& input = params.input;
    if (!input.ptr() || input.dtype() != DType::f32) {
        return Status::invalid_argument;
    }
    constexpr float sqrt_2_over_pi = 0.7978845608028654F;
    constexpr float coefficient = 0.044715F;
    auto* data = static_cast<float*>(input.ptr());
    for (size_t i = 0; i < input.numel(); ++i) {
        const float x = data[i];
        data[i] = 0.5F * x * (1.0F + std::tanh(sqrt_2_over_pi * (x + coefficient * x * x * x)));
    }
    return Status::ok;
}

}  // namespace llm_bricks::detail