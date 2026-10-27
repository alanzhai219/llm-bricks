#pragma once

#include <thread>
#include <immintrin.h>

#include "micro/gemv.hpp"
#include "parallel.hpp"

namespace llm_bricks {
namespace detail {

Status gemv_avx2(const GemvParams &params) {
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
    const float *a_data = input_a.ptr();
    float *out_data = output.ptr();

    // #pragma omp parallel for schedule(static)

    int num_threads = 12;
    parallel_nt(num_threads, [&](int ithr, int nthr) {
        size_t start, end;
        splitter(M, nthr, ithr, start, end);
        for (size_t i = start; i < end; ++i) {
            const float* row = a_data + i * N;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            size_t j = 0;

            for (; j + 63 < N; j += 64) {
                __m256 load_0 = _mm256_loadu_ps(row + j);
                __m256 load_1 = _mm256_loadu_ps(input_b + j);
                acc0 = _mm256_fmadd_ps(load_0, load_1, acc0);

                __m256 load_2 = _mm256_loadu_ps(row + j + 8);
                __m256 load_3 = _mm256_loadu_ps(input_b + j + 8);
                acc1 = _mm256_fmadd_ps(load_2, load_3, acc1);

                __m256 load_4 = _mm256_loadu_ps(row + j + 16);
                __m256 load_5 = _mm256_loadu_ps(input_b + j + 16);
                acc2 = _mm256_fmadd_ps(load_4, load_5, acc2);

                __m256 load_6 = _mm256_loadu_ps(row + j + 24);
                __m256 load_7 = _mm256_loadu_ps(input_b + j + 24);
                acc3 = _mm256_fmadd_ps(load_6, load_7, acc3);

                __m256 load_8 = _mm256_loadu_ps(row + j + 32);
                __m256 load_9 = _mm256_loadu_ps(input_b + j + 32);
                acc4 = _mm256_fmadd_ps(load_8, load_9, acc4);

                __m256 load_10 = _mm256_loadu_ps(row + j + 40);
                __m256 load_11 = _mm256_loadu_ps(input_b + j + 40);
                acc5 = _mm256_fmadd_ps(load_10, load_11, acc5);

                __m256 load_12 = _mm256_loadu_ps(row + j + 48);
                __m256 load_13 = _mm256_loadu_ps(input_b + j + 48);
                acc6 = _mm256_fmadd_ps(load_12, load_13, acc6);

                __m256 load_14 = _mm256_loadu_ps(row + j + 56);
                __m256 load_15 = _mm256_loadu_ps(input_b + j + 56);
                acc7 = _mm256_fmadd_ps(load_14, load_15, acc7);
            }

            __m256 sum0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            __m256 sum1 = _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7));
            __m256 sum = _mm256_add_ps(sum0, sum1);

            // 水平归约：8 个 float 合并成一个 float
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 v = _mm_add_ps(lo, hi);
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);

            float result = _mm_cvtss_f32(v);

            // 处理不能被 32 整除的尾部元素
            for (; j < N; ++j) {
                result += row[j] * input_b[j];
            }

            out_data[i] = result;
        }
    });
    return Status::ok;
}
} // namespace detail
} // namespace llm_bricks
