#pragma once

#include <cstddef>

#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"
#include "base_op.hpp"

namespace llm_bricks {

struct RopeParams : public BaseParams {
        RopeParams(Tensor& input,
                   float rope_base = 10000.0F,
                   std::size_t rope_rotary_dim = 0,
                   bool rope_interleaved = true)
                : x(input),
                  base(rope_base),
                  rotary_dim(rope_rotary_dim),
                  interleaved(rope_interleaved) {}
    // inputs
    Tensor& x;
    // parameters
    float base = 10000.0F;
    std::size_t rotary_dim = 0; // 0 means the complete final dimension.
    bool interleaved = true;
};

class Rope : public BaseOp {
public:
    Rope(Context& context) : m_ctx(context) {}

    void set_input(RopeParams& params) {
        m_params = &params;
    }

    Status infer();
private:
    Context m_ctx;
    RopeParams* m_params = nullptr;
};

} // namespace llm_bricks
