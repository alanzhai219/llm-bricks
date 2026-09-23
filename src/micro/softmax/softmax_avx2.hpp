#pragma once

#include "micro/softmax.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace llm_bricks::detail {

__attribute__((target("avx2"))) inline __m256 exp_avx2(__m256 values) {
    const __m256 lower_bound = _mm256_set1_ps(-87.33654475F);
    const __m256 upper_bound = _mm256_set1_ps(88.37626266F);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341F);
    const __m256 ln2_high = _mm256_set1_ps(0.693359375F);
    const __m256 ln2_low = _mm256_set1_ps(-2.12194440e-4F);

    values = _mm256_min_ps(_mm256_max_ps(values, lower_bound), upper_bound);
    const __m256 exponent = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(values, log2e), _mm256_set1_ps(0.5F)));
    const __m256 reduced = _mm256_sub_ps(
        _mm256_sub_ps(values, _mm256_mul_ps(exponent, ln2_high)),
        _mm256_mul_ps(exponent, ln2_low));
    const __m256 reduced_squared = _mm256_mul_ps(reduced, reduced);

    __m256 polynomial = _mm256_set1_ps(1.9875691500e-4F);
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced), _mm256_set1_ps(1.3981999507e-3F));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced), _mm256_set1_ps(8.3334519073e-3F));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced), _mm256_set1_ps(4.1665795894e-2F));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced), _mm256_set1_ps(1.6666665459e-1F));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced), _mm256_set1_ps(5.0000001201e-1F));
    polynomial = _mm256_add_ps(_mm256_mul_ps(polynomial, reduced_squared), reduced);
    polynomial = _mm256_add_ps(polynomial, _mm256_set1_ps(1.0F));

    __m256i exponent_bits = _mm256_cvttps_epi32(exponent);
    exponent_bits = _mm256_add_epi32(exponent_bits, _mm256_set1_epi32(127));
    exponent_bits = _mm256_slli_epi32(exponent_bits, 23);
    return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent_bits));
}

__attribute__((target("avx2"))) inline float horizontal_sum_avx2_softmax(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    float sum = 0.0F;
    for (float lane : lanes) {
        sum += lane;
    }
    return sum;
}

__attribute__((target("avx2"))) inline float horizontal_max_avx2(__m256 values) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, values);
    float maximum = lanes[0];
    for (size_t i = 1; i < 8; ++i) {
        maximum = std::max(maximum, lanes[i]);
    }
    return maximum;
}

__attribute__((target("avx2"))) inline Status softmax_avx2(SoftmaxParams& params) {
    const auto& input = params.input;
    auto& output = params.output;
    if (!input.ptr() || !output.ptr() || input.dtype() != DType::f32 || output.dtype() != DType::f32 ||
        input.ndims() == 0 || input.shape() != output.shape()) {
        return Status::invalid_argument;
    }

    const size_t width = input.dim(input.ndims() - 1);
    const auto* src = static_cast<const float*>(input.ptr());
    auto* dst = static_cast<float*>(output.ptr());
    const size_t row_num = input.numel() / width;

    for (size_t row = 0; row < row_num; ++row) {
        const float* x = src + row * width;
        float* y = dst + row * width;
        __m256 maximum_vector = _mm256_set1_ps(x[0]);
        size_t i = 0;
        for (; i + 8 <= width; i += 8) {
            maximum_vector = _mm256_max_ps(maximum_vector, _mm256_loadu_ps(x + i));
        }
        float maximum = horizontal_max_avx2(maximum_vector);
        for (; i < width; ++i) {
            maximum = std::max(maximum, x[i]);
        }

        const __m256 maximum_broadcast = _mm256_set1_ps(maximum);
        __m256 sum_vector = _mm256_setzero_ps();
        i = 0;
        for (; i + 8 <= width; i += 8) {
            const __m256 exponent = exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x + i), maximum_broadcast));
            _mm256_storeu_ps(y + i, exponent);
            sum_vector = _mm256_add_ps(sum_vector, exponent);
        }
        float sum = horizontal_sum_avx2_softmax(sum_vector);
        for (; i < width; ++i) {
            y[i] = std::exp(x[i] - maximum);
            sum += y[i];
        }

        const __m256 inverse_sum = _mm256_set1_ps(1.0F / sum);
        i = 0;
        for (; i + 8 <= width; i += 8) {
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), inverse_sum));
        }
        for (; i < width; ++i) {
            y[i] /= sum;
        }
    }

    return Status::ok;
}

}  // namespace llm_bricks::detail

#else

namespace llm_bricks::detail {

inline Status softmax_avx2(SoftmaxParams&) {
    return Status::unsupported;
}

}  // namespace llm_bricks::detail

#endif
