#pragma once

#include "micro/scale.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

#include <immintrin.h>

namespace llm_bricks::detail {

__attribute__((target("avx2"))) inline Status scale_avx2(ScaleParams& params) {
    const auto& input = params.input;
    auto& output = params.output;
    if (!input.ptr() || !output.ptr() || input.dtype() != DType::f32 || output.dtype() != DType::f32 ||
        input.shape() != output.shape()) {
        return Status::invalid_argument;
    }

    const auto* source = static_cast<const float*>(input.ptr());
    auto* target = static_cast<float*>(output.ptr());
    const size_t element_num = input.numel();
    const __m256 factor = _mm256_set1_ps(params.factor);

    size_t i = 0;
    for (; i + 8 <= element_num; i += 8) {
        _mm256_storeu_ps(target + i, _mm256_mul_ps(_mm256_loadu_ps(source + i), factor));
    }
    for (; i < element_num; ++i) {
        target[i] = source[i] * params.factor;
    }

    return Status::ok;
}

}  // namespace llm_bricks::detail

#else

namespace llm_bricks::detail {

inline Status scale_avx2(ScaleParams&) {
    return Status::unsupported;
}

}  // namespace llm_bricks::detail

#endif
