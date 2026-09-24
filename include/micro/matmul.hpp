#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {
struct MatMulParams : public BaseParams {
    MatMulParams(const Tensor& a, const Tensor& b, Tensor& c) :
        input_a(a), input_b(b), output(c) {}
    const Tensor& input_a;
    const Tensor& input_b;
    Tensor& output;
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