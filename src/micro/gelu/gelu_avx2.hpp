#pragma once

#include "micro/gelu.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

#include <cmath>
#include <immintrin.h>

namespace llm_bricks::detail {

__attribute__((target("avx2"))) inline Status gelu_avx2(GeluParams& params) {
    auto& input = params.input;
    if (!input.ptr() || input.dtype() != DType::f32) {
        return Status::invalid_argument;
    }

    constexpr float sqrt_2_over_pi = 0.7978845608028654F;
    constexpr float coefficient = 0.044715F;
    auto* data = static_cast<float*>(input.ptr());
    const size_t element_num = input.numel();

    const __m256 half = _mm256_set1_ps(0.5F);
    const __m256 one = _mm256_set1_ps(1.0F);
    const __m256 sqrt_2_over_pi_vector = _mm256_set1_ps(sqrt_2_over_pi);
    const __m256 coefficient_vector = _mm256_set1_ps(coefficient);
    const __m256 lower_bound = _mm256_set1_ps(-5.0F);
    const __m256 upper_bound = _mm256_set1_ps(5.0F);
    const __m256 numerator_0 = _mm256_set1_ps(135135.0F);
    const __m256 numerator_1 = _mm256_set1_ps(17325.0F);
    const __m256 numerator_2 = _mm256_set1_ps(378.0F);
    const __m256 denominator_0 = _mm256_set1_ps(135135.0F);
    const __m256 denominator_1 = _mm256_set1_ps(62370.0F);
    const __m256 denominator_2 = _mm256_set1_ps(3150.0F);
    const __m256 denominator_3 = _mm256_set1_ps(28.0F);

    size_t i = 0;
    for (; i + 8 <= element_num; i += 8) {
        const __m256 x = _mm256_loadu_ps(data + i);
        const __m256 x_squared = _mm256_mul_ps(x, x);
        const __m256 x_cubed = _mm256_mul_ps(x, x_squared);
        const __m256 tanh_input = _mm256_mul_ps(
            sqrt_2_over_pi_vector,
            _mm256_add_ps(x, _mm256_mul_ps(coefficient_vector, x_cubed)));
        const __m256 clamped_input = _mm256_min_ps(_mm256_max_ps(tanh_input, lower_bound), upper_bound);
        const __m256 input_squared = _mm256_mul_ps(clamped_input, clamped_input);

        const __m256 numerator = _mm256_mul_ps(
            clamped_input,
            _mm256_add_ps(numerator_0,
                          _mm256_mul_ps(input_squared,
                                        _mm256_add_ps(numerator_1,
                                                      _mm256_mul_ps(input_squared,
                                                                    _mm256_add_ps(numerator_2, input_squared))))));
        const __m256 denominator = _mm256_add_ps(
            denominator_0,
            _mm256_mul_ps(input_squared,
                          _mm256_add_ps(denominator_1,
                                        _mm256_mul_ps(input_squared,
                                                      _mm256_add_ps(denominator_2,
                                                                    _mm256_mul_ps(denominator_3, input_squared))))));
        const __m256 tanh_approximation = _mm256_min_ps(
            _mm256_max_ps(_mm256_div_ps(numerator, denominator), _mm256_set1_ps(-1.0F)),
            one);
        const __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_approximation)));
        _mm256_storeu_ps(data + i, result);
    }

    for (; i < element_num; ++i) {
        const float x = data[i];
        data[i] = 0.5F * x * (1.0F + std::tanh(sqrt_2_over_pi * (x + coefficient * x * x * x)));
    }

    return Status::ok;
}

}  // namespace llm_bricks::detail

#else

namespace llm_bricks::detail {

inline Status gelu_avx2(GeluParams&) {
    return Status::unsupported;
}

}  // namespace llm_bricks::detail

#endif
