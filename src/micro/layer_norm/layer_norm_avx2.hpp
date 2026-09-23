#pragma once

#include "micro/layer_norm.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

#include <cmath>
#include <immintrin.h>

namespace llm_bricks::detail {

__attribute__((target("avx2"))) inline float horizontal_sum_avx2(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    float sum = 0.0F;
    for (float lane : lanes) {
        sum += lane;
    }
    return sum;
}

__attribute__((target("avx2"))) inline Status layer_norm_avx2(LayerNormParams& params) {
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
    const size_t row_num = input.numel() / width;

    for (size_t row = 0; row < row_num; ++row) {
        const float* x = src + row * width;
        float* y = dst + row * width;

        __m256 sum_vector = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= width; i += 8) {
            sum_vector = _mm256_add_ps(sum_vector, _mm256_loadu_ps(x + i));
        }
        float sum = horizontal_sum_avx2(sum_vector);
        for (; i < width; ++i) {
            sum += x[i];
        }
        const float mean = sum / static_cast<float>(width);
        const __m256 mean_vector = _mm256_set1_ps(mean);

        __m256 squared_sum_vector = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= width; i += 8) {
            const __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(x + i), mean_vector);
            squared_sum_vector = _mm256_add_ps(squared_sum_vector, _mm256_mul_ps(delta, delta));
        }
        float squared_sum = horizontal_sum_avx2(squared_sum_vector);
        for (; i < width; ++i) {
            const float delta = x[i] - mean;
            squared_sum += delta * delta;
        }
        const float inverse_std = 1.0F / std::sqrt(squared_sum / static_cast<float>(width) + params.epsilon);
        const __m256 inverse_std_vector = _mm256_set1_ps(inverse_std);

        i = 0;
        for (; i + 8 <= width; i += 8) {
            const __m256 normalized = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(x + i), mean_vector), inverse_std_vector);
            const __m256 scaled = _mm256_mul_ps(normalized, _mm256_loadu_ps(g + i));
            _mm256_storeu_ps(y + i, _mm256_add_ps(scaled, _mm256_loadu_ps(b + i)));
        }
        for (; i < width; ++i) {
            y[i] = (x[i] - mean) * inverse_std * g[i] + b[i];
        }
    }

    return Status::ok;
}

}  // namespace llm_bricks::detail

#else

namespace llm_bricks::detail {

inline Status layer_norm_avx2(LayerNormParams&) {
    return Status::unsupported;
}

}  // namespace llm_bricks::detail

#endif
