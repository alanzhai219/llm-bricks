#pragma once

#include "micro/matmul.hpp"

namespace llm_bricks {
namespace detail {

Status matmul_2d_avx2(const MatMulParams& params) {
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

    for (size_t k = 0; k < K; ++k) {
        for (size_t m = 0; m < M; ++m) {
            float a_mk = a.at(m, k);
            for (size_t n = 0; n < N; ++n) {
                c.at(m, n) += a_mk * b.at(k, n);
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