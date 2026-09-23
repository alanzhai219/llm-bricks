#pragma once

#include "micro/scale.hpp"

namespace llm_bricks::detail {

inline Status scale_ref(ScaleParams& params) {
    const auto& input = params.input;
    auto& output = params.output;
    if (!input.ptr() || !output.ptr() || input.dtype() != DType::f32 || output.dtype() != DType::f32 ||
        input.shape() != output.shape()) {
        return Status::invalid_argument;
    }
    const auto* source = static_cast<const float*>(input.ptr());
    auto* target = static_cast<float*>(output.ptr());
    for (size_t i = 0; i < input.numel(); ++i) {
        target[i] = source[i] * params.factor;
    }
    return Status::ok;
}

}  // namespace llm_bricks::detail