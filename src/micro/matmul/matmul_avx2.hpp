#pragma once

#include <algorithm>
#include <immintrin.h>
#include <vector>

#include "micro/matmul.hpp"

namespace llm_bricks {
namespace detail {

// cache blocking for better performance
constexpr size_t MC = 128;  // block size for M dimension
constexpr size_t KC = 256;  // block size for K dimension
constexpr size_t NC = 768;  // block size for N dimension

// register blocking for better performance
constexpr size_t MR = 6;  // register block size for M dimension
constexpr size_t NR = 16;  // register block size for N dimension

// A is packed [kc, mc], B is packed [kc, nc]; C retains its original row stride.
__attribute__((target("avx2,fma")))
inline void micro_kernel_avx2_6x16(const float* Ap, const float* Bp, float* C,
                                    size_t kc, size_t mc, size_t nc, size_t ldc) {
    // Two YMM accumulators per row: columns 0..7 and 8..15.
    __m256 ymm_c00 = _mm256_loadu_ps(C + 0 * ldc); __m256 ymm_c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 ymm_c10 = _mm256_loadu_ps(C + 1 * ldc); __m256 ymm_c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 ymm_c20 = _mm256_loadu_ps(C + 2 * ldc); __m256 ymm_c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 ymm_c30 = _mm256_loadu_ps(C + 3 * ldc); __m256 ymm_c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 ymm_c40 = _mm256_loadu_ps(C + 4 * ldc); __m256 ymm_c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 ymm_c50 = _mm256_loadu_ps(C + 5 * ldc); __m256 ymm_c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    for (size_t p = 0; p < kc; ++p) {
        const float* a = Ap + p * mc;
        const float* b = Bp + p * nc;
        const __m256 ymm_b0 = _mm256_loadu_ps(b);
        const __m256 ymm_b1 = _mm256_loadu_ps(b + 8);

        __m256 ymm_a = _mm256_set1_ps(a[0]);
        ymm_c00 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c00);
        ymm_c01 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c01);

        ymm_a = _mm256_set1_ps(a[1]);
        ymm_c10 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c10);
        ymm_c11 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c11);

        ymm_a = _mm256_set1_ps(a[2]);
        ymm_c20 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c20);
        ymm_c21 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c21);

        ymm_a = _mm256_set1_ps(a[3]);
        ymm_c30 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c30);
        ymm_c31 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c31);

        ymm_a = _mm256_set1_ps(a[4]);
        ymm_c40 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c40);
        ymm_c41 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c41);

        ymm_a = _mm256_set1_ps(a[5]);
        ymm_c50 = _mm256_fmadd_ps(ymm_a, ymm_b0, ymm_c50);
        ymm_c51 = _mm256_fmadd_ps(ymm_a, ymm_b1, ymm_c51);
    }

    _mm256_storeu_ps(C + 0 * ldc, ymm_c00); _mm256_storeu_ps(C + 0 * ldc + 8, ymm_c01);
    _mm256_storeu_ps(C + 1 * ldc, ymm_c10); _mm256_storeu_ps(C + 1 * ldc + 8, ymm_c11);
    _mm256_storeu_ps(C + 2 * ldc, ymm_c20); _mm256_storeu_ps(C + 2 * ldc + 8, ymm_c21);
    _mm256_storeu_ps(C + 3 * ldc, ymm_c30); _mm256_storeu_ps(C + 3 * ldc + 8, ymm_c31);
    _mm256_storeu_ps(C + 4 * ldc, ymm_c40); _mm256_storeu_ps(C + 4 * ldc + 8, ymm_c41);
    _mm256_storeu_ps(C + 5 * ldc, ymm_c50); _mm256_storeu_ps(C + 5 * ldc + 8, ymm_c51);
}

// Scalar fringe kernel for a tile smaller than 6x16, with the same packed layout.
inline void micro_kernel_generic(const float* Ap, const float* Bp, float* C,
                                 size_t mr, size_t nr, size_t kc,
                                 size_t mc, size_t nc, size_t ldc) {
    for (size_t p = 0; p < kc; ++p) {
        const float* a = Ap + p * mc;
        const float* b = Bp + p * nc;
        for (size_t i = 0; i < mr; ++i) {
            float* c_row = C + i * ldc;
            const float av = a[i];
            for (size_t j = 0; j < nr; ++j) {
                c_row[j] += av * b[j];
            }
        }
    }
}

// Decode (M=1): stream the original row-major B without packing either operand.
__attribute__((target("avx2,fma")))
inline void matmul_row_avx2(const float* a, const float* b, float* c,
                            size_t k, size_t n) {
    const size_t vector_end = n & ~size_t{7};
    for (size_t p = 0; p < k; ++p) {
        const __m256 av = _mm256_set1_ps(a[p]);
        const float* b_row = b + p * n;
        size_t j = 0;
        for (; j < vector_end; j += 8) {
            const __m256 cv = _mm256_loadu_ps(c + j);
            const __m256 bv = _mm256_loadu_ps(b_row + j);
            _mm256_storeu_ps(c + j, _mm256_fmadd_ps(av, bv, cv));
        }
        for (; j < n; ++j) {
            c[j] += a[p] * b_row[j];
        }
    }
}

void macro_kernel(const float* Ap, const float* Bp, float* C, size_t mc, size_t nc, size_t kc, size_t ldc) {
    for (size_t i = 0; i < mc; i += MR) {
        for (size_t j = 0; j < nc; j += NR) {
            size_t ib = std::min(MR, mc - i);
            size_t jb = std::min(NR, nc - j);

            const float* A_block = Ap + i;
            const float* B_block = Bp + j;
            float* C_block = C + i * ldc + j;

            if (ib == MR && jb == NR) {
                // Use the optimized kernel for full blocks
                micro_kernel_avx2_6x16(A_block, B_block, C_block, kc, mc, nc, ldc);
            } else {
                // Use the generic kernel for edge blocks
                micro_kernel_generic(A_block, B_block, C_block, ib, jb, kc, mc, nc, ldc);
            }
        }
    } 
}

Status matmul_2d_avx2(const MatMulParams& params) {
    if (params.transpose_a || params.transpose_b) {
        return Status::invalid_argument;  // Transpose not supported in this implementation
    }

    const auto& a = params.input_a;
    const auto& b = params.input_b;
    auto& c = params.output;

    if (a.ndims() != 2 || b.ndims() != 2 || c.ndims() != 2) {
        return Status::invalid_argument;
    }

    size_t M = a.dim(0);
    size_t K = a.dim(1);
    size_t N = b.dim(1);

    if (b.dim(0) != K || c.dim(0) != M || c.dim(1) != N) {
        return Status::invalid_argument;
    }

    size_t lda = K;
    size_t ldb = N;
    size_t ldc = N;


    const float* a_ptr = a.ptr();
    const float* b_ptr = b.ptr();
    float* c_ptr = c.ptr();

    if (M == 1) {
        matmul_row_avx2(a_ptr, b_ptr, c_ptr, K, N);
        return Status::ok;
    }

    // pack_a is used to pack a block of A matrix into a contiguous buffer for better cache performance
    auto pack_a_func = [](const float* a, size_t lda, float* packed_a, size_t M, size_t K) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                packed_a[k * M + m] = a[m * lda + k];
            }

        }
    };

    auto pack_b_func = [](const float* b, size_t ldb, float* packed_b, size_t K, size_t N) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                packed_b[k * N + n] = b[k * ldb + n];
            }
        }
    };

    for (size_t jc = 0; jc < N; jc += NC) {
        size_t nc = std::min(NC, N - jc); // block size for N dimension
        for (size_t pc = 0; pc < K; pc += KC) {
            size_t kc = std::min(KC, K - pc); // block size for K dimension
            // Pack B block
            std::vector<float> packed_b(kc * nc);
            const float* b_block_ptr = b_ptr + pc * ldb + jc;
            pack_b_func(b_block_ptr, ldb, packed_b.data(), kc, nc);

            for (size_t ic = 0; ic < M; ic += MC) {
                size_t mc = std::min(MC, M - ic); // block size for M dimension
                // Pack A block
                std::vector<float> packed_a(mc * kc);
                const float* a_block_ptr = a_ptr + ic * lda + pc;
                pack_a_func(a_block_ptr, lda, packed_a.data(), mc, kc);

                // Compute C block
                float* c_block_ptr = c_ptr + ic * ldc + jc;
                macro_kernel(packed_a.data(), packed_b.data(), c_block_ptr, mc, nc, kc, ldc);
            }
        }
    }

    return Status::ok;
}

Status matmul_3d_avx2(const MatMulParams& params) {
    const auto& a = params.input_a;
    const auto& b = params.input_b;
    auto& c = params.output;

    if (a.ndims() != 3 || b.ndims() != 3 || c.ndims() != 3) {
        return Status::invalid_argument;
    }

    size_t B = a.dim(0);
    size_t M = a.dim(1);
    size_t K = a.dim(2);
    size_t N = b.dim(2);

    if (b.dim(0) != B || b.dim(1) != K || c.dim(0) != B || c.dim(1) != M || c.dim(2) != N) {
        return Status::invalid_argument;
    }

    for (size_t b_idx = 0; b_idx < B; ++b_idx) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t m = 0; m < M; ++m) {
                float a_bmk = a.at(b_idx, m, k);
                for (size_t n = 0; n < N; ++n) {
                    c.at(b_idx, m, n) += a_bmk * b.at(b_idx, k, n);
                }
            }
        }
    }

    return Status::ok;
}

Status matmul_avx2(const MatMulParams& params) {
    const auto& a = params.input_a;
    const auto& b = params.input_b;
    auto& c = params.output;

    if (a.ndims() == 2 && b.ndims() == 2 && c.ndims() == 2) {
        return matmul_2d_avx2(params);
    } else if (a.ndims() == 3 && b.ndims() == 3 && c.ndims() == 3) {
        return matmul_3d_avx2(params);
    } else {
        return Status::invalid_argument;
    }
}
} // namespace detail
} // namespace llm_bricks