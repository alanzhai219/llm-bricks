#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {
struct MatMulParams : public BaseParams {
    MatMulParams(const Tensor& a, const Tensor& b, Tensor& c, bool transpose_a = false, bool transpose_b = false) :
        input_a(a), input_b(b), output(c), transpose_a(transpose_a), transpose_b(transpose_b) {}
    const Tensor& input_a;
    const Tensor& input_b;
    Tensor& output;
    bool transpose_a = false;
    bool transpose_b = false;
};

class MatMul : public BaseOp {
public:
    explicit MatMul(Context& context) : m_ctx(context) {}
    void set_input(MatMulParams& params) { m_params = &params; }
    Status infer() override;
private:
    Context& m_ctx;
    MatMulParams* m_params = nullptr;
};
}  // namespace llm_bricks