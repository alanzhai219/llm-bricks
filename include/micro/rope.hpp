#pragma once

#include <cstddef>

#include "context.hpp"
#include "status.hpp"
#include "tensor.hpp"

namespace llm_bricks {

struct RopeParams {
    // inputs
    Tensor& x;
    // outputs
    // parameters
    float base = 10000.0F;
    std::size_t rotary_dim = 0; // 0 means the complete final dimension.
    bool interleaved = true;
};

// Applies in-place rotary position embeddings to a contiguous f32 [B, H, S, D] tensor.
[[nodiscard]] Status rope(Context& context, const RopeParams& params);

} // namespace llm_bricks
