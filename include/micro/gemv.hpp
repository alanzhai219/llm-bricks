#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {
struct GemvParams : public BaseParams {
    explicit GemvParams(const Tensor& in_a, const float* in_b, Tensor& out) : input_a(in_a), input_b(in_b), output(out) {}
    const Tensor& input_a;
    const float* input_b;
    Tensor& output;
};

class Gemv : public BaseOp {
public:
    explicit Gemv(Context& context) : m_ctx(context) {}
    void set_input(GemvParams& params) { m_params = &params; }
    Status infer() override;
private:
    Context m_ctx;
    GemvParams* m_params = nullptr;
};
} // namespace llm_bricks
