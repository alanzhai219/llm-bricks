#pragma once

#include "micro/matmul.hpp"

namespace llm_bricks {
namespace detail {

// pre definition of matmul_2d_ref and matmul_3d_ref
Status matmul_2d_ref(const MatMulParams& params);
Status matmul_3d_ref(const MatMulParams& params);

Status matmul_ref(const MatMulParams& params) {
    const auto& a_shape = params.input_a.shape();
    const auto& b_shape = params.input_b.shape();

    bool is_2d_matmul = (a_shape.size() == 2 && b_shape.size() == 2);
    bool is_3d_matmul = (a_shape.size() == 3 && b_shape.size() == 3);

    if (is_2d_matmul) {
        return matmul_2d_ref(params);
    }

    if (is_3d_matmul) {
        return matmul_3d_ref(params);
    }
    return Status::invalid_argument;
}

Status matmul_2d_ref(const MatMulParams& params) {
    const auto& a = params.input_a;
    const auto& b = params.input_b;
    auto& c = params.output;

    if (a.ndims() != 2 || b.ndims() != 2 || c.ndims() != 2) {
        return Status::invalid_argument;
    }

    size_t m = a.dim(0);
    size_t k = a.dim(1);
    size_t n = b.dim(1);

    if (b.dim(0) != k || c.dim(0) != m || c.dim(1) != n) {
        return Status::invalid_argument;
    }

    const float* a_ptr = a.ptr();
    const float* b_ptr = b.ptr();
    float* c_ptr = c.ptr();
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            const float av = a_ptr[i * k + p];
            const float* b_row = b_ptr + p * n;
            float* c_row = c_ptr + i * n;
            for (size_t j = 0; j < n; ++j) {
                c_row[j] += av * b_row[j];
            }
        }
    }

    return Status::ok;
}

Status matmul_3d_ref(const MatMulParams& params) {
    const auto& a = params.input_a;
    const auto& b = params.input_b;
    auto& c = params.output;

    if (a.ndims() != 3 || b.ndims() != 3 || c.ndims() != 3) {
        return Status::invalid_argument;
    }

    size_t batch = a.dim(0);
    size_t m = a.dim(1);
    size_t k = a.dim(2);
    size_t n = b.dim(2);

    if (b.dim(0) != batch || b.dim(1) != k || c.dim(0) != batch || c.dim(1) != m || c.dim(2) != n) {
        return Status::invalid_argument;
    }

    const float* a_ptr = a.ptr();
    const float* b_ptr = b.ptr();
    float* c_ptr = c.ptr();
    for (size_t h = 0; h < batch; ++h) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const float av = a_ptr[(h * m + i) * k + p];
                const float* b_row = b_ptr + (h * k + p) * n;
                float* c_row = c_ptr + (h * m + i) * n;
                for (size_t j = 0; j < n; ++j) {
                    c_row[j] += av * b_row[j];
                }
            }
        }
    }

    return Status::ok;
}

}  // namespace detail
}  // namespace llm_bricks