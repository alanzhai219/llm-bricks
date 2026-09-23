#pragma once

#include "base_op.hpp"
#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct LayerNormParams : public BaseParams {
    LayerNormParams(const Tensor& source, const Tensor& scale, const Tensor& shift, float eps, Tensor& result)
        : input(source), gamma(scale), beta(shift), epsilon(eps), output(result) {}
    const Tensor& input;
    const Tensor& gamma;
    const Tensor& beta;
    float epsilon;
    Tensor& output;
};

class LayerNorm : public BaseOp {
public:
    explicit LayerNorm(Context& context) : m_ctx(context) {}
    void set_input(LayerNormParams& params) { m_params = &params; }
    Status infer() override;
private:
    Context m_ctx;
    LayerNormParams* m_params = nullptr;
};

}  // namespace llm_bricks