#pragma once

#include "micro/gemv.hpp"

namespace llm_bricks {
namespace detail {
Status gemv_ref(const GemvParams &params) {
    const Tensor &input_a = params.input_a;
    const float *input_b = params.input_b;
    Tensor &output = params.output;

    if (input_a.ptr() == nullptr || input_a.dtype() != DType::f32 || input_a.ndims() != 2) {
        return Status::invalid_argument;
    }
    if (input_b == nullptr) {
        return Status::invalid_argument;
    }
    if (output.ptr() == nullptr || output.dtype() != DType::f32 || output.ndims() != 1) {
        return Status::invalid_argument;
    }

    const size_t M = input_a.dim(0);
    const size_t N = input_a.dim(1);

    if (output.dim(0) != M) {
        return Status::invalid_argument;
    }

    const float *a_data = static_cast<const float *>(input_a.ptr());
    float *out_data = static_cast<float *>(output.ptr());

    for (size_t i = 0; i < M; ++i) {
        out_data[i] = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            out_data[i] += a_data[i * N + j] * input_b[j];
        }
    }

    return Status::ok; 
}

}  // namespace detail
}  // namespace llm_bricks