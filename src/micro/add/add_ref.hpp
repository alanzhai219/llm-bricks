#pragma once

#include "micro/add.hpp"

namespace llm_bricks::detail {

inline Status add_ref(AddParams& params) {
    if (!params.lhs.ptr() || !params.rhs.ptr() || !params.output.ptr() || params.lhs.dtype() != DType::f32 ||
        params.rhs.dtype() != DType::f32 || params.output.dtype() != DType::f32 ||
        params.lhs.shape() != params.rhs.shape() || params.lhs.shape() != params.output.shape()) {
        return Status::invalid_argument;
    }
    const auto* lhs = static_cast<const float*>(params.lhs.ptr());
    const auto* rhs = static_cast<const float*>(params.rhs.ptr());
    auto* output = static_cast<float*>(params.output.ptr());
    for (size_t i = 0; i < params.output.numel(); ++i) {
        output[i] = lhs[i] + rhs[i];
    }
    return Status::ok;
}

}  // namespace llm_bricks::detail